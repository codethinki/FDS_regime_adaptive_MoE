from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MarketData:
    """Common output of all data sources."""

    returns: pd.DataFrame  # tradable assets, one row per period, date index
    regimes: np.ndarray | None = None  # true regime per period (synthetic data only)
    extra_features: pd.DataFrame | None = None  # source-specific model inputs (synthetic noise feature)
