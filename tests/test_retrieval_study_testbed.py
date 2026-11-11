"""retrieval_study/testbed.py and diagnostics.py: windows, ridge selection, layouts (offline), plus one check against the
project's own fold arrays (development fold f4; no 2019+ data)."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from retrieval_study import diagnostics as dg
from retrieval_study import raft
from retrieval_study import testbed as tb


def daily(n_days: int = 120, start: str = "2010-03-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n_days, freq="D")
    return pd.DataFrame({"t_max": np.arange(n_days, dtype=float), "stratum": "normal"}, index=idx)


def test_windows_take_14_input_days_and_the_next_5():
    d = daily()
    w = tb.make_windows(d, "t_max", d.index[[20, 50]])
    np.testing.assert_allclose(w.inputs[0], np.arange(6, 20))
    assert w.last[0] == 19 and list(w.future[1]) == [50, 51, 52, 53, 54]
    assert w.day[1] - w.day[0] == 30
    with pytest.raises(ValueError):
        tb.make_windows(d, "t_max", d.index[[5]])  # not enough input days


def test_the_ridge_strength_is_chosen_on_the_stop_rows_then_refitted_on_all():
    rng = np.random.default_rng(0)
    n = 400
    x = rng.normal(size=(n, 30))
    last = rng.normal(30, 3, size=n)
    future = last[:, None] + x[:, :1] @ np.ones((1, 5)) + 0.5 * rng.normal(size=(n, 5))
    w = SimpleNamespace(future=future, last=last)
    fit, stop = np.arange(n) < 300, np.arange(n) >= 300
    use = np.ones(n, dtype=bool)
    pred, alpha, score = tb.select_fit_predict(x, x[:50], w, SimpleNamespace(last=last[:50]), use, fit, stop)
    y = future - last[:, None]
    scores = [tb.rmse(tb.ridge_predict(m, x[stop]) + last[stop, None], future[stop]) for m in tb.ridge_path(x[fit], y[fit])]
    assert alpha == tb.ALPHAS[int(np.argmin(scores))] and score == pytest.approx(min(scores))
    full = tb.ridge_path(x, y, (alpha,))[0]
    np.testing.assert_allclose(pred, tb.ridge_predict(full, x[:50]) + last[:50, None])
    exact = tb.ridge_path(x[:, :1], y, (1e-9,))[0]  # near-OLS recovers the slope 1 on every lead
    np.testing.assert_allclose(exact[2] / exact[1][:, None], np.ones((1, 5)), atol=0.05)


def test_long_rows_follow_the_prediction_file_layout():
    d = daily()
    d.loc[d.index[52], "stratum"] = "extreme"
    val = tb.make_windows(d, "t_max", d.index[[50, 60]])
    fd = SimpleNamespace(val=val, d=d, fold="f4")
    pred = val.future + 1.0
    df = tb.long_rows(fd, pred, "x", seed=3)
    assert len(df) == 10 and list(df["lead"][:5]) == [1, 2, 3, 4, 5]
    assert df.loc[2, "target_date"] == d.index[52] and df.loc[2, "stratum"] == "extreme"
    assert (df["error"] == 1.0).all() and (df["seed"] == 3).all() and (df["fold"] == "f4").all()


def test_value_similarity_and_percentile_ranks():
    rng = np.random.default_rng(3)
    c, q = rng.normal(size=(30, 5)), rng.normal(size=(4, 5))
    direct = -np.sqrt(((c[None, :, :] - q[:, None, :]) ** 2).mean(axis=2))
    np.testing.assert_allclose(dg._value_similarity(c, q), direct, atol=1e-12)
    values = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    ok = np.ones_like(values, dtype=bool)
    np.testing.assert_allclose(dg._percentile_of(values, np.array([[4, 0]]), ok), [[100.0, 0.0]])
    ok[0, 3] = False  # an ineligible candidate does not count
    np.testing.assert_allclose(dg._percentile_of(values, np.array([[4]]), ok), [[100.0]])


@pytest.mark.slow
def test_fold_data_matches_the_projects_fold_arrays_and_rules():
    from training.folds import build_fold

    fd = tb.FoldData.build("f4", "Tmax")
    ref = build_fold("f4", "t_max", "v2")
    np.testing.assert_allclose(fd.train.future, ref.train.y_raw, atol=1e-4)
    np.testing.assert_allclose(fd.val.future, ref.val.y_raw, atol=1e-4)
    assert fd.val.q.min() == pd.Timestamp("2016-01-01") and fd.train.q.max() <= pd.Timestamp("2015-12-31")
    assert fd.stop.sum() > 600 and not (fd.fit & fd.stop).any()
    widths = {b: fd.base(b, "val").shape[1] for b in tb.BASES}
    assert widths["B0"] == 14 and widths["B1"] == 21 and widths["B3"] > widths["B2"] > widths["B1"]
    nb = fd.raft_neighbours()[1]
    chosen = nb.idx[1][nb.idx[1] >= 0]
    assert len(chosen) == nb.idx[1].size  # every validation query gets its 20 neighbours
    rows = np.repeat(np.arange(len(fd.val.q)), nb.idx[1].shape[1])
    assert np.all(np.abs(fd.train.day[nb.idx[1].ravel()] - fd.val.day[rows]) >= raft.OVERLAP_DAYS)
    r_tr, r_va, full = fd.r0_values()
    idx = fd.r0_idx()[1]
    i = 100
    np.testing.assert_allclose(r_va[i], fd.train.future[idx[i]].mean(axis=0) - fd.val.last[i])
    assert full.mean() > 0.99
