import numpy as np
import pandas as pd
import pytest

from regime_moe.config import load_config
from regime_moe.data import split_positions
from regime_moe.data.real import split
from regime_moe.features import build_features


def test_split_guard_without_test_data():
    cfg = load_config().data
    df = pd.DataFrame({"a": 1.0}, index=pd.bdate_range("2013-01-01", cfg.val_end))
    with pytest.raises(ValueError):
        split(df, "test", cfg)


def test_split_positions_do_not_overlap():
    cfg = load_config()
    index = pd.bdate_range("2012-01-02", "2026-01-01")
    pos = split_positions(index, cfg)
    assert set(pos) == {"train", "val", "test"}
    assert pos["train"].max() < pos["val"].min() < pos["val"].max() < pos["test"].min()


def test_features_are_causal():
    cfg = load_config().features
    rng = np.random.default_rng(0)
    r = pd.DataFrame(rng.normal(size=(100, 3)), index=pd.bdate_range("2020-01-01", periods=100))
    changed = r.copy()
    changed.iloc[60:] *= 10
    a, b = build_features(r, cfg), build_features(changed, cfg)
    t = r.index[59]
    pd.testing.assert_frame_equal(a.loc[:t], b.loc[:t])
