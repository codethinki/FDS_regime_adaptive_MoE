"""Toy market with known regimes (from the big MoE demo).

Regimes follow a Markov chain; returns are Gaussian with regime-dependent means
and volatility, independent across assets. The true regimes are returned for
analysis only and must not be used as model input.

Uses the legacy NumPy RNG with the demo's draw order (regimes, then returns, then the
optional noise feature), so seed 42 reproduces the demo's market exactly.
"""

import numpy as np
import pandas as pd

from regime_moe.config import SyntheticConfig
from regime_moe.data.market import MarketData

START = "2000-01-03"


def synthetic_index(n_days: int) -> pd.DatetimeIndex:
    return pd.bdate_range(START, periods=n_days)


def make_synthetic_market(cfg: SyntheticConfig, seed: int) -> MarketData:
    rng = np.random.RandomState(seed)
    transition = np.asarray(cfg.transition)
    means = np.asarray(cfg.means)
    vols = np.asarray(cfg.vols)
    n_regimes, n_assets = means.shape

    regimes = np.zeros(cfg.n_days, dtype=int)
    for t in range(1, cfg.n_days):
        regimes[t] = rng.choice(n_regimes, p=transition[regimes[t - 1]])

    returns = np.zeros((cfg.n_days, n_assets))
    for t in range(cfg.n_days):
        returns[t] = means[regimes[t]] + rng.normal(0, vols[regimes[t]], n_assets)

    index = synthetic_index(cfg.n_days)
    extra = None
    if cfg.noise_feature:
        noise = [rng.lognormal(mean=0.0, sigma=0.25) for _ in range(cfg.n_days)]
        extra = pd.DataFrame({"noise_volume": noise}, index=index)

    return MarketData(
        returns=pd.DataFrame(returns, index=index, columns=[f"asset_{i}" for i in range(n_assets)]),
        regimes=regimes,
        extra_features=extra,
    )
