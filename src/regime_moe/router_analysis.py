"""Standalone analysis of a regime router: which regimes it finds and how they behave.

Regime statistics use the returns realised *after* the router's call (next period),
i.e. what routing on the regime would have experienced.
"""

import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from regime_moe.routers import Router  # noqa: E402


def episodes(states: np.ndarray) -> pd.DataFrame:
    """One row per run of the same state: state, start position, length."""
    starts = np.flatnonzero(np.r_[True, states[1:] != states[:-1]])
    return pd.DataFrame({"state": states[starts], "start": starts, "length": np.diff(np.r_[starts, len(states)])})


def regime_summary(
    states: np.ndarray, next_returns: pd.DataFrame, feats_raw: pd.DataFrame, n_states: int, periods_per_year: int
) -> pd.DataFrame:
    ep = episodes(states)
    rows = {}
    for k in range(n_states):
        m = states == k
        lengths = ep.length[ep.state == k]
        row = {
            "share": m.mean(),
            "episodes": len(lengths),
            "mean_duration": lengths.mean() if len(lengths) else np.nan,
            "feat_win_std": feats_raw["win_std"].to_numpy()[m].mean() if m.any() else np.nan,
        }
        for asset in next_returns.columns:
            r = next_returns[asset].to_numpy()[m]
            row[f"{asset}_ann_ret"] = r.mean() * periods_per_year if m.any() else np.nan
            row[f"{asset}_sharpe"] = r.mean() / r.std() * np.sqrt(periods_per_year) if m.sum() > 1 else np.nan
        rows[f"state_{k}"] = row
    return pd.DataFrame(rows)


def switches_per_year(states: np.ndarray, periods_per_year: int) -> float:
    return (len(episodes(states)) - 1) / (len(states) / periods_per_year)


def detection_delays(insample: np.ndarray, online: np.ndarray, max_lag: int = 60) -> pd.DataFrame:
    """Periods until the online states reach the state of each in-sample switch."""
    rows = []
    for pos in np.flatnonzero(insample[1:] != insample[:-1]) + 1:
        hit = np.flatnonzero(online[pos : pos + max_lag + 1] == insample[pos])
        rows.append({"position": int(pos), "to_state": int(insample[pos]), "delay": int(hit[0]) if len(hit) else np.nan})
    return pd.DataFrame(rows, columns=["position", "to_state", "delay"])


def best_permutation_accuracy(pred: np.ndarray, true: np.ndarray, n_states: int) -> tuple[float, tuple[int, ...]]:
    """State labels are arbitrary: accuracy under the best mapping pred state -> true regime."""
    best = max(itertools.permutations(range(n_states)), key=lambda p: np.mean(np.asarray(p)[pred] == true))
    return float(np.mean(np.asarray(best)[pred] == true)), best


def plot_states(price: pd.Series, states: np.ndarray, eval_start, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(price.index, price.to_numpy(), color="lightgrey", lw=1, zorder=1)
    for k in np.unique(states):
        m = states == k
        ax.scatter(price.index[m], price.to_numpy()[m], s=4, label=f"state {k}", zorder=2)
    ax.axvspan(eval_start, price.index[-1], color="tab:blue", alpha=0.08, label="eval split")
    ax.set_title(title)
    ax.legend(loc="upper left", markerscale=3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def analyze_router(
    router: Router,
    x_all: np.ndarray,
    feats_raw: pd.DataFrame,
    next_returns: pd.DataFrame,
    splits: dict[str, np.ndarray],
    eval_split: str,
    periods_per_year: int,
    out_dir: Path,
    true_regimes: np.ndarray | None = None,
) -> dict[str, pd.DataFrame]:
    """Print and save the regime analysis of an already fitted and prepared router."""
    out_dir.mkdir(parents=True, exist_ok=True)
    probs = router.state_probabilities()
    n_states = probs.shape[1]
    online = probs.argmax(axis=1)
    train, ev = splits["train"], splits[eval_split]
    insample = router.insample_states(x_all[train])

    tables = {
        "insample_train": regime_summary(insample, next_returns.iloc[train], feats_raw.iloc[train], n_states, periods_per_year),
        "online_train": regime_summary(online[train], next_returns.iloc[train], feats_raw.iloc[train], n_states, periods_per_year),
        f"online_{eval_split}": regime_summary(online[ev], next_returns.iloc[ev], feats_raw.iloc[ev], n_states, periods_per_year),
    }
    trans = router.transition_matrix()
    tables["transition_matrix"] = pd.DataFrame(trans, index=[f"from_{k}" for k in range(n_states)],
                                               columns=[f"to_{k}" for k in range(n_states)])
    tables["expected_duration"] = pd.DataFrame({"periods": 1 / (1 - np.diag(trans))}, index=[f"state_{k}" for k in range(n_states)])
    delays = detection_delays(insample, online[train])
    tables["delays_train"] = delays

    for name, table in tables.items():
        table.to_csv(out_dir / f"{name}.csv")
        if name != "delays_train":
            print(f"\n--- {name}\n{table.round(3).to_string()}")

    overview = pd.DataFrame({
        "switches_per_year": {
            "insample_train": switches_per_year(insample, periods_per_year),
            "online_train": switches_per_year(online[train], periods_per_year),
            f"online_{eval_split}": switches_per_year(online[ev], periods_per_year),
        }
    })
    overview.to_csv(out_dir / "overview.csv")
    tables["overview"] = overview
    print(f"\n--- switches per year\n{overview.round(2).to_string()}")

    agreement = float(np.mean(online[train] == insample))
    print(f"\nonline vs in-sample agreement (train): {agreement:.1%}")
    print(f"detection delay (train): median {delays.delay.median():.0f}, mean {delays.delay.mean():.1f} periods, "
          f"{delays.delay.isna().sum()} of {len(delays)} switches not reached within 60")
    if true_regimes is not None:
        for name, pos in [("train", train), (eval_split, ev)]:
            acc, mapping = best_permutation_accuracy(online[pos], true_regimes[pos], n_states)
            print(f"accuracy vs true regimes ({name}, online, mapping state->regime {mapping}): {acc:.1%}")

    price = (1 + next_returns.iloc[:, 0].shift(1).fillna(0)).cumprod()
    plot_states(price, online, next_returns.index[ev[0]], f"Online states ({next_returns.columns[0]})", out_dir / "online_states.png")
    return tables
