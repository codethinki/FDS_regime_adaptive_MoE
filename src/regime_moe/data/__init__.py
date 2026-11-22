import numpy as np
import pandas as pd

from regime_moe.config import Config
from regime_moe.data.market import MarketData
from regime_moe.data.real import load_real_market
from regime_moe.data.synthetic import make_synthetic_market, synthetic_index


def market_seeds(cfg: Config) -> list[int | None]:
    """One entry per market path: the synthetic seeds, or [None] for real data."""
    return list(cfg.synthetic.seeds) if cfg.data.source == "synthetic" else [None]


def load_market(cfg: Config, market_seed: int | None = None, allow_test: bool = False) -> MarketData:
    if cfg.data.source == "real":
        return load_real_market(cfg.data, allow_test)
    if cfg.data.source == "synthetic":
        return make_synthetic_market(cfg.synthetic, cfg.synthetic.seeds[0] if market_seed is None else market_seed)
    raise ValueError(f"unknown data source '{cfg.data.source}'")


def split_positions(index: pd.DatetimeIndex, cfg: Config) -> dict[str, np.ndarray]:
    """Integer positions of each split within ``index``."""
    if cfg.data.source == "synthetic":
        # cut by market period (not feature row), so dropped warm-up rows don't shift the split
        s = cfg.synthetic
        cutoff = synthetic_index(s.n_days)[int(s.n_days * s.train_fraction)]
        return {"train": np.flatnonzero(index < cutoff), "val": np.flatnonzero(index >= cutoff)}

    d = cfg.data
    bounds = {"train": (d.train_start, d.train_end), "val": (d.val_start, d.val_end), "test": (d.test_start, None)}
    out = {}
    for name, (start, end) in bounds.items():
        mask = index >= pd.Timestamp(start)
        if end is not None:
            mask &= index <= pd.Timestamp(end)
        if mask.any():
            out[name] = np.flatnonzero(mask)
    return out
