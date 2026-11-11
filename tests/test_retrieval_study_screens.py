"""retrieval_study/screens.py: the add-on correction, nearest-window search and interval score (offline)."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from retrieval_study import screens as sc


def network(n_inner: int = 730, n_val: int = 50, bias: float = 1.0, seed: int = 0) -> sc.NetworkForecasts:
    rng = np.random.default_rng(seed)
    cols = sc.PRED_COLS + sc.ACT_COLS

    def frame(dates):
        truth = rng.normal(35, 3, size=(len(dates), 5))
        pred = truth - bias + rng.normal(0, 1, size=truth.shape)  # forecasts too cold by `bias`
        return pd.DataFrame(np.column_stack([pred, truth]), index=dates, columns=cols)

    inner = pd.date_range("2014-01-01", periods=n_inner, freq="D")
    val = pd.date_range("2016-01-01", periods=n_val, freq="D")
    return sc.NetworkForecasts("net", frame(inner), frame(val))


def test_the_addon_learns_the_constant_shift_and_uses_an_informative_feature():
    nf = network()
    noise_in, noise_val = np.random.default_rng(1).normal(size=(730, 3)), np.random.default_rng(2).normal(size=(50, 3))
    pred, alpha = sc.addon_correction(nf, noise_in, noise_val)
    shift = (pred - nf.val[sc.PRED_COLS].to_numpy()).mean()
    assert shift == pytest.approx(1.0, abs=0.15) and alpha in sc.ALPHAS  # the bias correction alone
    informative = nf.error("inner")  # a feature that is the error itself
    pred2, _ = sc.addon_correction(nf, informative, nf.error("val"))
    assert np.sqrt(np.mean((pred2 - nf.val[sc.ACT_COLS].to_numpy()) ** 2)) < 0.2
    x_val = np.full((50, 3), np.nan)
    pred3, _ = sc.addon_correction(nf, noise_in, x_val, alpha=10.0)
    np.testing.assert_allclose(pred3, nf.val[sc.PRED_COLS].to_numpy())  # no features -> no correction


def test_nearest_windows_respect_the_overlap_rule_and_random_draws_are_eligible():
    keys = np.arange(40, dtype=float)[:, None] * np.ones((1, 3))
    days = np.arange(40) * 5
    nb = sc._nearest(keys[[20]], days[[20]], keys, days, overlap=19, m=3, seed=None)
    assert set(nb[0]) == {16, 24, 15} or set(nb[0]) == {16, 24, 25}  # nearest keys outside +-19 days
    assert all(abs(days[j] - days[20]) >= 19 for j in nb[0])
    rnd = sc._nearest(keys[[20]], days[[20]], keys, days, overlap=19, m=5, seed=3)
    assert len(set(rnd[0])) == 5 and all(abs(days[j] - days[20]) >= 19 for j in rnd[0])


def test_interval_score_rewards_narrow_ranges_and_penalises_misses():
    y = np.array([10.0, 10.0, 10.0])
    lo, hi = np.array([9.0, 9.5, 11.0]), np.array([11.0, 10.5, 12.0])
    s = sc.interval_score(lo, hi, y, alpha=0.2)
    np.testing.assert_allclose(s, [2.0, 1.0, 1.0 + 10.0 * 1.0])


def test_the_range_methods_use_inner_errors():
    nf = network(bias=0.0)
    fr = SimpleNamespace(query_vectors=lambda dates: np.ones((len(dates), 2)) / np.sqrt(2))
    fd = SimpleNamespace(fr=fr)
    g = sc.knn_ranges(fd, nf, "global")
    err = nf.error("inner")
    np.testing.assert_allclose(g["hi"] - nf.val[sc.PRED_COLS].to_numpy(),
                               np.broadcast_to(np.quantile(err, 0.9, axis=0), (50, 5)))
    k = sc.knn_ranges(fd, nf, "knn", k=50)
    assert k["lo"].shape == (50, 5) and np.all(k["hi"] > k["lo"])
    r1, r2 = sc.knn_ranges(fd, nf, "random", seed=1), sc.knn_ranges(fd, nf, "random", seed=1)
    np.testing.assert_allclose(r1["lo"], r2["lo"])
    with pytest.raises(ValueError):
        sc.knn_ranges(fd, nf, "nope")
