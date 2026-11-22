"""Run one experiment and write results/<experiment>/<timestamp>/.

run.mode = "moe":    train the Big MoE for every router x seed and evaluate the portfolios
run.mode = "router": fit the regime routers only and analyse their regimes

    uv run python src/run_experiment.py src/configs/experiments/synthetic_sanity.toml
    uv run python src/run_experiment.py src/configs/experiments/router_only.toml
"""

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from regime_moe.config import RESULTS_DIR, Config, load_config
from regime_moe.data import load_market, market_seeds, split_positions
from regime_moe.evaluate import equal_weight_metrics, evaluate
from regime_moe.features import build_features
from regime_moe.models import BigMoE, build_experts
from regime_moe.objectives import build_objective
from regime_moe.router_analysis import analyze_router
from regime_moe.routers import build_router
from regime_moe.seeding import set_seed
from regime_moe.train import train_moe


@dataclass
class Prepared:
    feats: pd.DataFrame  # raw features at t
    target: pd.DataFrame  # asset returns at t+1
    x_all: np.ndarray  # scaled features (scaler fit on train)
    y_all: np.ndarray
    splits: dict[str, np.ndarray]
    true_regimes: np.ndarray | None


def prepare(cfg: Config, market_seed: int | None) -> Prepared:
    market = load_market(cfg, market_seed)

    # features at t, target = asset returns at t+1 (last period has no target)
    feats = build_features(market.returns, cfg.features, market.extra_features)
    target = market.returns.shift(-1).loc[feats.index]
    valid = target.notna().all(axis=1).to_numpy()
    feats, target = feats[valid], target[valid]

    splits = split_positions(feats.index, cfg)
    if cfg.eval.split not in splits:
        raise ValueError(f"eval split '{cfg.eval.split}' not available, have {sorted(splits)}")

    scaler = StandardScaler().fit(feats.to_numpy()[splits["train"]])
    true_regimes = None
    if market.regimes is not None:
        true_regimes = pd.Series(market.regimes, index=market.returns.index).loc[feats.index].to_numpy()
    return Prepared(
        feats=feats,
        target=target,
        x_all=scaler.transform(feats.to_numpy()).astype(np.float32),
        y_all=target.to_numpy().astype(np.float32),
        splits=splits,
        true_regimes=true_regimes,
    )


def run_moe(cfg: Config, out: Path, device: str) -> None:
    """Every market path x router x model seed; plus an equal-weight benchmark per market path."""
    objective = build_objective(cfg.train)
    rows = []
    for market_seed in market_seeds(cfg):
        data = prepare(cfg, market_seed)
        x_all, y_all = data.x_all, data.y_all
        train_pos, eval_pos = data.splits["train"], data.splits[cfg.eval.split]
        n_features, n_assets = x_all.shape[1], y_all.shape[1]

        def tensors(pos: np.ndarray):
            return torch.from_numpy(x_all[pos]), torch.from_numpy(y_all[pos]), torch.from_numpy(pos)

        x_tr, y_tr, i_tr = tensors(train_pos)
        x_ev, y_ev, i_ev = tensors(eval_pos)
        bench = equal_weight_metrics(y_all[eval_pos], cfg.eval.periods_per_year)
        rows.append({"market_seed": market_seed, "router": "equal_weight", "seed": None, **bench})
        print(f"market {market_seed}: equal-weight benchmark sharpe {bench['sharpe']:+.2f}")

        for router_name in cfg.router.types:
            for seed in cfg.train.seeds:
                set_seed(seed)
                # experts first: identical initial experts for every router at a given seed
                moe = BigMoE(build_experts(cfg.model, n_features, n_assets)).to(device)
                router = build_router(router_name, cfg.model.n_experts, n_features, cfg.router)
                router.fit(x_all[train_pos])
                router.prepare(x_all)
                router.to(device)

                history = train_moe(moe, router, objective, x_tr, y_tr, i_tr, cfg.train, seed, device)
                metrics = evaluate(moe, router, x_ev, y_ev, i_ev, cfg.eval.periods_per_year, device)
                rows.append({"market_seed": market_seed, "router": router_name, "seed": seed,
                             "final_train_loss": history[-1], **metrics})
                print(f"  {router_name:9s} seed {seed}: sharpe {metrics['sharpe']:+.2f}")

    results = pd.DataFrame(rows)
    # excess over the equal-weight benchmark of the same market path
    bench = results[results.router == "equal_weight"].set_index("market_seed")["sharpe"]
    results["sharpe_vs_equal_weight"] = results.sharpe - results.market_seed.map(bench).fillna(bench.iloc[0])
    results.to_csv(out / "runs.csv", index=False)
    metrics = ["sharpe", "sharpe_vs_equal_weight", "ann_return", "ann_vol", "max_drawdown", "turnover"]
    summary = results.groupby("router", sort=False)[metrics].agg(["mean", "std"])
    summary.to_csv(out / "summary.csv")
    print(f"\n{len(market_seeds(cfg))} market path(s) x {len(cfg.train.seeds)} model seed(s)")
    print(summary.round(3).to_string())


def run_router(cfg: Config, data: Prepared, out: Path) -> None:
    analysed = set()
    for router_name in cfg.router.types:
        router = build_router(router_name, cfg.model.n_experts, data.x_all.shape[1], cfg.router)
        if router.regime_model is None:
            print(f"skipping '{router_name}': no explicit regime model to analyse")
            continue
        if router.regime_model in analysed:
            continue  # e.g. hmm_hard and hmm_soft share the same fitted HMM
        analysed.add(router.regime_model)
        router.fit(data.x_all[data.splits["train"]])
        router.prepare(data.x_all)
        print(f"\n===== router: {router.regime_model}")
        analyze_router(
            router, data.x_all, data.feats, data.target, data.splits, cfg.eval.split,
            cfg.eval.periods_per_year, out / f"router_{router.regime_model}", data.true_regimes,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", nargs="?", help="experiment TOML (default: defaults only)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.experiment)
    first = prepare(cfg, market_seeds(cfg)[0])
    n_train, n_eval = len(first.splits["train"]), len(first.splits[cfg.eval.split])
    print(f"{cfg.name} [{cfg.run.mode}]: {cfg.data.source} data, {first.y_all.shape[1]} assets, "
          f"{first.x_all.shape[1]} features, train {n_train} / {cfg.eval.split} {n_eval} periods")

    out = RESULTS_DIR / cfg.name / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    if cfg.run.mode == "moe":
        run_moe(cfg, out, args.device)
    elif cfg.run.mode == "router":
        run_router(cfg, first, out)  # first market path only
    else:
        raise ValueError(f"unknown run.mode '{cfg.run.mode}', choose 'moe' or 'router'")
    print(f"\nresults: {out}")


if __name__ == "__main__":
    main()
