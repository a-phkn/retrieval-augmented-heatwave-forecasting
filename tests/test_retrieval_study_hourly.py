"""retrieval_study/hourly.py: the hourly split, windows and phase features (offline, synthetic series)."""
import numpy as np
import pandas as pd

from retrieval_study import hourly as hr


def test_the_split_keeps_fitting_choosing_and_validation_apart():
    times = pd.date_range("2012-06-01", "2018-12-31 23:00", freq="h")
    pl = hr.plan(times, 12)
    t = {k: times[v] for k, v in pl.items()}
    horizon = pd.Timedelta(hours=hr.HORIZON)
    assert (t["fit"] + horizon <= hr.INNER_START).all()  # fitting forecasts end before the choosing block
    assert (t["stop"] >= hr.INNER_START).all() and (t["stop"] + horizon <= hr.TRAIN_END).all()
    assert (t["val"] >= hr.TRAIN_END).all() and (t["val"] + horizon <= hr.TEST_START).all()
    assert (t["cand"] + horizon <= hr.TRAIN_END).all() and set(t["cand"].hour) <= {0, 3, 6, 9, 12, 15, 18, 21}
    assert set(t["val"].hour) == set(hr.ISSUE_HOURS) and len(pl["fit"]) == min(hr.N_TRAIN_QUERIES, len(pl["fit"]))


def test_windows_and_phase_features():
    times = pd.date_range("2015-01-01", periods=300, freq="h")
    x = np.arange(300, dtype=float)
    w = hr.make_windows(x, times, np.array([100, 200]), 12)
    np.testing.assert_allclose(w.inputs[0], np.arange(88, 100))
    assert w.last[1] == 199 and w.future.shape == (2, hr.HORIZON) and w.future[0, 0] == 100
    assert hr.base(w, "H0").shape == (2, 12) and hr.base(w, "H1").shape == (2, 16)
    ph = hr.phase(pd.DatetimeIndex(["2015-01-01 00:00", "2015-01-01 06:00"]))
    np.testing.assert_allclose(ph[:, :2], [[0.0, 1.0], [1.0, 0.0]], atol=1e-12)
    df = hr.long_frame(w, w.future + 0.5, "m")
    assert len(df) == 2 * hr.HORIZON and (df["error"] == 0.5).all()
    assert df.loc[hr.HORIZON - 1, "target_date"] == times[100] + pd.Timedelta(hours=hr.HORIZON - 1)
