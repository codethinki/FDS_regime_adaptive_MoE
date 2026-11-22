"""Causal features: the row for period t uses returns up to and including t."""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from regime_moe.config import FeaturesConfig


def build_features(returns: pd.DataFrame, cfg: FeaturesConfig, extra: pd.DataFrame | None = None) -> pd.DataFrame:
    """Current returns plus std / mean / mean |r| over the last ``window`` periods (all assets pooled),
    plus any ``extra`` columns (aligned by index).

    The first ``window - 1`` periods are dropped (incomplete window).
    """
    w = cfg.window
    windows = sliding_window_view(returns.to_numpy(), w, axis=0)  # [T - w + 1, assets, w]
    index = returns.index[w - 1 :]

    feats = pd.DataFrame(returns.to_numpy()[w - 1 :], index=index, columns=[f"ret_{c}" for c in returns.columns])
    feats["win_std"] = windows.std(axis=(1, 2))
    feats["win_mean"] = windows.mean(axis=(1, 2))
    feats["win_abs_mean"] = np.abs(windows).mean(axis=(1, 2))
    if extra is not None:
        feats = feats.join(extra)
    return feats
