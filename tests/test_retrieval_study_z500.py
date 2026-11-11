"""retrieval_study/z500_direct.py: the 500 hPa scores are appended to the right input days (offline, synthetic)."""
import numpy as np
import pandas as pd
import pytest

from retrieval_study import z500_direct as zd
from training.folds import FORECAST_DAYS, INPUT_DAYS, SplitArrays


def split(query_dates: pd.DatetimeIndex, n_features: int = 3) -> SplitArrays:
    n = len(query_dates)
    zeros5 = np.zeros((n, FORECAST_DAYS), dtype=np.float32)
    return SplitArrays(X=np.zeros((n, INPUT_DAYS, n_features), dtype=np.float32), y=zeros5, y_raw=zeros5,
                       clim_target=zeros5.astype(np.float64), hot=zeros5.astype(bool),
                       stratum=np.full((n, FORECAST_DAYS), "normal"), query_dates=pd.Series(query_dates))


def test_each_window_gets_the_scores_of_its_own_14_input_days():
    days = pd.date_range("2010-01-01", periods=60, freq="D")
    pcs = pd.DataFrame(np.arange(60)[:, None] + np.arange(zd.N_PCS)[None, :] / 100, index=days)
    s = split(days[[20, 40]])
    out = zd.with_z500(s, pcs)
    assert out.X.shape == (2, INPUT_DAYS, 3 + zd.N_PCS) and out.X.dtype == np.float32
    np.testing.assert_allclose(out.X[0, :, 3], np.arange(6, 20))  # days 6..19 for the window starting on day 20
    np.testing.assert_allclose(out.X[1, -1, 3:], 39 + np.arange(zd.N_PCS) / 100, rtol=1e-6)
    np.testing.assert_array_equal(out.X[:, :, :3], 0)  # the original channels are untouched
    with pytest.raises(ValueError):
        zd.with_z500(split(days[[5]]), pcs)  # not enough input days
