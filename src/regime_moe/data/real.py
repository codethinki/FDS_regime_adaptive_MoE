"""Real market data from yfinance with a guarded test period.

The test period (lockbox) is cut off unless a caller explicitly passes
``allow_test=True``, which should only happen in the final evaluation.
"""

import pandas as pd
import yfinance as yf

from regime_moe.config import DataConfig
from regime_moe.data.market import MarketData


def load_prices(tickers: list[str], cfg: DataConfig, allow_test: bool = False) -> pd.DataFrame:
    """Adjusted close prices from cfg.warmup_start; cut at cfg.val_end unless allow_test."""
    px = yf.download(tickers, start=cfg.warmup_start, auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame(tickers[0])
    if not allow_test:
        px = px.loc[: cfg.val_end]
    return px[tickers]


def load_returns(tickers: list[str], cfg: DataConfig, allow_test: bool = False) -> pd.DataFrame:
    """Simple daily returns; the first row (NaN from pct_change) is dropped."""
    return load_prices(tickers, cfg, allow_test).pct_change().iloc[1:]


def load_real_market(cfg: DataConfig, allow_test: bool = False) -> MarketData:
    tickers = cfg.assets + ([cfg.cash] if cfg.cash else [])
    px = load_prices(tickers, cfg, allow_test).dropna()
    return MarketData(returns=px.pct_change().iloc[1:])


def split(df: pd.DataFrame | pd.Series, name: str, cfg: DataConfig):
    """Slice a date-indexed object to 'train', 'val' or 'test'."""
    bounds = {
        "train": (cfg.train_start, cfg.train_end),
        "val": (cfg.val_start, cfg.val_end),
        "test": (cfg.test_start, None),
    }
    start, end = bounds[name]
    out = df.loc[start:end]
    if name == "test" and len(out) == 0:
        raise ValueError("no test data loaded; load with allow_test=True (final evaluation only)")
    return out
