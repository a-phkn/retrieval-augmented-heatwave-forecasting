"""
Correctness and calibration tests for evaluation/stats.py.

Calibration uses a realistic simulator: AR(1) daily errors, larger errors in the
Mar 15-Jul 31 season and inside multi-day episodes, one 5-day-ahead window per
day (consecutive windows share target days), year x season clusters of unequal
size. Sizes match the project: 3 years (val, G=9) and 8 years (test, G=24).
With 300 replications and a true 5% rate, the binomial SD is 1.26 pp; the
upper bound of 9% is ~3 SD above nominal.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.signal import lfilter

from evaluation.stats import (
    MIN_CLUSTERS,
    andrews_lag,
    dm_test,
    make_cluster_ids,
    newey_west_variance,
    paired_cluster_test,
)

H = 5


# ---------------------------------------------------------------- simulator


def _ar1(rng, n, phi):
    e = rng.normal(0, np.sqrt(1 - phi**2), n)
    e[0] = rng.normal()
    return lfilter([1.0], [1.0, -phi], e)


def _simulate(rng, n_years, phi=0.7, scale_child=1.0, n_seeds=1):
    """Returns (target dates per instance, episode mask per instance,
    parent errors (seeds, n), child errors (seeds, n))."""
    days = pd.date_range("2016-01-01", periods=365 * n_years + n_years // 4, freq="D")
    n = len(days)
    md = days.month * 100 + days.day
    hot = (md >= 315) & (md <= 731)
    episode = np.zeros(n, bool)
    for year in np.unique(days.year):
        idx = np.where(hot & (days.year == year))[0]
        for _ in range(rng.poisson(1.0)):
            start = rng.choice(idx[:-15])
            episode[start : start + rng.integers(5, 13)] = True
    amp = np.where(hot, 1.5, 1.0) * np.where(episode, 3.0, 1.0)
    shared = _ar1(rng, n, phi) * amp
    q = np.arange(n - H)
    tgt = q[:, None] + np.arange(1, H + 1)[None, :]

    def model_errors(scale):
        own = _ar1(rng, n, phi) * amp
        rows = []
        for _ in range(n_seeds):
            daily = shared + own + 0.3 * _ar1(rng, n, phi) * amp
            e = np.sqrt(np.arange(1, H + 1) / H)[None, :] * daily[tgt] + 0.3 * rng.normal(size=tgt.shape)
            rows.append(scale * e.ravel())
        return np.stack(rows)

    return days[tgt.ravel()], episode[tgt.ravel()], model_errors(1.0), model_errors(scale_child)


# ---------------------------------------------------------------- point estimates


def test_point_estimate_matches_direct_rmse_and_mae():
    rng = np.random.default_rng(1)
    dates, _, p, c = _simulate(rng, 3)
    cid = make_cluster_ids(dates)
    for metric, fn in (("rmse", lambda e: np.sqrt(np.mean(e**2))), ("mae", lambda e: np.mean(np.abs(e)))):
        r = paired_cluster_test(parent=p[0], child=c[0], cluster_ids=cid, metric=metric)
        assert r.parent == pytest.approx(fn(p[0]))
        assert r.child == pytest.approx(fn(c[0]))
        assert r.delta == pytest.approx(fn(c[0]) - fn(p[0]))
        assert r.n_clusters == 9 and r.df == 8


def test_seed_matrix_point_estimate_is_mean_of_per_seed_metric():
    rng = np.random.default_rng(2)
    parent = rng.normal(0, 1, (5, 300))
    child = rng.normal(0, 1, (3, 300))  # different seed counts are allowed
    cid = np.repeat(np.arange(15), 20)
    r = paired_cluster_test(parent=list(parent), child=list(child), cluster_ids=cid)
    assert r.parent == pytest.approx(np.mean(np.sqrt(np.mean(parent**2, axis=1))))
    assert r.child == pytest.approx(np.mean(np.sqrt(np.mean(child**2, axis=1))))
    assert (r.n_seeds_parent, r.n_seeds_child) == (5, 3)


def test_jackknife_se_matches_brute_force():
    """Independent re-implementation: recompute delta with each cluster removed."""
    rng = np.random.default_rng(3)
    dates, _, p, c = _simulate(rng, 3, n_seeds=2)
    cid = make_cluster_ids(dates)
    r = paired_cluster_test(parent=list(p), child=list(c), cluster_ids=cid, include_seed_variance=False)
    labels = np.unique(cid)
    rmse = lambda e: np.sqrt(np.mean(e**2, axis=1)).mean()  # noqa: E731
    jack = np.array([rmse(c[:, cid != g]) - rmse(p[:, cid != g]) for g in labels])
    G = labels.size
    se = np.sqrt((G - 1) / G * np.sum((jack - jack.mean()) ** 2))
    assert r.se == pytest.approx(se, rel=1e-10)


def test_seed_variance_term_increases_se():
    rng = np.random.default_rng(4)
    dates, _, p, c = _simulate(rng, 3, n_seeds=5)
    cid = make_cluster_ids(dates)
    with_seed = paired_cluster_test(parent=list(p), child=list(c), cluster_ids=cid)
    without = paired_cluster_test(parent=list(p), child=list(c), cluster_ids=cid, include_seed_variance=False)
    assert with_seed.se > without.se


def test_identical_models():
    rng = np.random.default_rng(5)
    dates, _, p, _ = _simulate(rng, 3)
    r = paired_cluster_test(parent=list(p), child=list(p.copy()), cluster_ids=make_cluster_ids(dates))
    assert r.delta == 0.0 and r.se == 0.0 and r.p_value == 1.0 and not r.significant


def test_cluster_labels_any_type_and_order():
    rng = np.random.default_rng(6)
    dates, _, p, c = _simulate(rng, 3)
    cid = make_cluster_ids(dates)
    mixed = np.array([f"b{x}" if x % 2 else int(x) for x in cid], dtype=object)
    r1 = paired_cluster_test(parent=list(p), child=list(c), cluster_ids=cid)
    r2 = paired_cluster_test(parent=list(p), child=list(c), cluster_ids=mixed)
    assert r1.delta == pytest.approx(r2.delta) and r1.se == pytest.approx(r2.se)


def test_sign_count():
    cid = np.repeat(np.arange(6), 10)
    parent = np.ones(60)
    child = np.where(cid < 4, 0.5, 2.0)  # child better in 4 of 6 clusters
    assert paired_cluster_test(parent=parent, child=child, cluster_ids=cid).clusters_child_better == 4


# ---------------------------------------------------------------- calibration


@pytest.mark.parametrize("n_years, expected_g", [(3, 9), (8, 24)])
def test_false_positive_rate_at_project_cluster_counts(n_years, expected_g):
    rng = np.random.default_rng(100 + n_years)
    reps, rejections = 300, 0
    for _ in range(reps):
        dates, _, p, c = _simulate(rng, n_years)
        r = paired_cluster_test(parent=list(p), child=list(c), cluster_ids=make_cluster_ids(dates))
        assert r.n_clusters == expected_g
        rejections += r.p_value < 0.05
    rate = rejections / reps
    assert 0.01 <= rate <= 0.09, f"FPR {rate:.3f} at G={expected_g}"


def test_false_positive_rate_extreme_subset():
    """Extreme-only instances: few, unequal clusters -- the hardest case."""
    rng = np.random.default_rng(200)
    reps = rejections = 0
    while reps < 300:
        dates, ep, p, c = _simulate(rng, 12)
        cid = make_cluster_ids(dates)[ep]
        if np.unique(cid).size < MIN_CLUSTERS:
            continue
        reps += 1
        rejections += paired_cluster_test(parent=list(p[:, ep]), child=list(c[:, ep]), cluster_ids=cid).p_value < 0.05
    rate = rejections / reps
    assert 0.01 <= rate <= 0.09, f"extreme-subset FPR {rate:.3f}"


def test_power_for_real_effect():
    rng = np.random.default_rng(300)
    detected = 0
    for _ in range(100):
        dates, _, p, c = _simulate(rng, 8, scale_child=0.8)  # child errors 20% smaller
        r = paired_cluster_test(parent=list(p), child=list(c), cluster_ids=make_cluster_ids(dates))
        detected += r.significant and r.delta < 0
    assert detected >= 80


def test_too_few_clusters_returns_nan_and_warns():
    cid = np.repeat(np.arange(4), 25)
    rng = np.random.default_rng(7)
    with pytest.warns(UserWarning, match="no CI or p-value"):
        r = paired_cluster_test(parent=rng.normal(size=100), child=rng.normal(size=100), cluster_ids=cid)
    assert np.isnan(r.p_value) and np.isnan(r.ci_low) and not r.significant
    assert np.isfinite(r.delta) and r.n_clusters == 4


# ---------------------------------------------------------------- input validation


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(parent=np.ones(10), child=np.ones(9), cluster_ids=np.arange(10)), "same instances"),
        (dict(parent=np.ones(10), child=np.ones(10), cluster_ids=np.arange(9)), "same instances"),
        (dict(parent=np.r_[np.nan, np.ones(9)], child=np.ones(10), cluster_ids=np.arange(10)), "NaN"),
        (dict(parent=np.full(10, 1e200), child=np.ones(10), cluster_ids=np.arange(10)), "overflows"),
        (dict(parent=np.ones(10), child=np.ones(10), cluster_ids=np.zeros(10)), "at least 2 clusters"),
        (dict(parent=np.ones(10), child=np.ones(10), cluster_ids=np.arange(10), metric="mse"), "metric"),
        (dict(parent=np.ones((2, 2, 2)), child=np.ones(10), cluster_ids=np.arange(10)), "ravel"),
        # (lead days x windows) or (windows x lead days): 2-D is always rejected
        (dict(parent=np.ones((5, 40)), child=np.ones((5, 40)), cluster_ids=np.arange(40)), "ravel"),
        (dict(parent=np.ones((40, 5)), child=np.ones((40, 5)), cluster_ids=np.arange(5)), "ravel"),
        (dict(parent=[np.ones(10), np.ones(9)], child=np.ones(10), cluster_ids=np.arange(10)), "same number"),
        (dict(parent=[np.ones((2, 5))], child=np.ones(10), cluster_ids=np.arange(10)), "1-D arrays"),
        (dict(parent=np.ones(4), child=np.ones(4), cluster_ids=np.array([1, None, 2, 3], dtype=object)), "missing"),
    ],
)
def test_rejects_bad_input(kwargs, match):
    with pytest.raises(ValueError, match=match):
        paired_cluster_test(**kwargs)


def test_warns_on_date_cluster_ids():
    rng = np.random.default_rng(8)
    dates = np.repeat(pd.date_range("2016-01-01", periods=60).to_numpy(), 5)
    # Both warnings fire: dates as clusters, and clusters too small (5 instances each).
    with pytest.warns(UserWarning, match="this small"), pytest.warns(UserWarning, match="cluster_ids are dates"):
        paired_cluster_test(parent=rng.normal(size=300), child=rng.normal(size=300), cluster_ids=dates)


def test_warns_on_many_tiny_clusters():
    rng = np.random.default_rng(9)
    with pytest.warns(UserWarning, match="too small|this small"):
        paired_cluster_test(parent=rng.normal(size=300), child=rng.normal(size=300), cluster_ids=np.repeat(np.arange(60), 5))


def test_warns_on_lone_series_with_custom_index():
    idx = pd.date_range("2020-01-01", periods=100)
    s = pd.Series(np.random.default_rng(10).normal(size=100), index=idx)
    with pytest.warns(UserWarning, match="cannot be checked"):
        paired_cluster_test(parent=s, child=np.ones(100), cluster_ids=np.repeat(np.arange(10), 10))


def test_seed_variance_included_flag():
    rng = np.random.default_rng(11)
    cid = np.repeat(np.arange(10), 20)
    one = rng.normal(size=200)
    two = [rng.normal(size=200), rng.normal(size=200)]
    # Multi-seed model vs single-seed baseline: the multi-seed model's variance IS added.
    mixed = paired_cluster_test(parent=one, child=two, cluster_ids=cid)
    mixed_off = paired_cluster_test(parent=one, child=two, cluster_ids=cid, include_seed_variance=False)
    assert mixed.seed_variance_included and mixed.se > mixed_off.se
    assert not paired_cluster_test(parent=one, child=one * 1.1, cluster_ids=cid).seed_variance_included
    assert paired_cluster_test(parent=two, child=two[::-1], cluster_ids=cid).seed_variance_included
    assert not paired_cluster_test(
        parent=two, child=two[::-1], cluster_ids=cid, include_seed_variance=False
    ).seed_variance_included


def test_constant_offset_gives_nan_not_zero_p():
    cid = np.repeat(np.arange(10), 20)
    parent = np.ones(200)
    r = paired_cluster_test(parent=parent, child=2 * parent, cluster_ids=cid)
    assert r.delta == pytest.approx(1.0) and r.se == 0.0 and np.isnan(r.p_value)


def test_rejects_misaligned_series_inside_seed_list():
    rng = np.random.default_rng(12)
    s1 = pd.Series(rng.normal(size=100))
    s2 = pd.Series(rng.normal(size=100)).sample(frac=1, random_state=0)  # shuffled index
    with pytest.raises(ValueError, match="different indexes"):
        paired_cluster_test(parent=[s1, s2], child=np.ones(100), cluster_ids=np.repeat(np.arange(10), 10))


def test_warns_on_reversed_lone_series():
    """iloc[::-1] keeps a RangeIndex (step -1); it must still be flagged."""
    s = pd.Series(np.random.default_rng(13).normal(size=100)).iloc[::-1]
    with pytest.warns(UserWarning, match="cannot be checked"):
        paired_cluster_test(parent=s, child=np.ones(100), cluster_ids=np.repeat(np.arange(10), 10))


def test_plain_python_list_message_is_helpful():
    with pytest.raises(ValueError, match="np.asarray"):
        paired_cluster_test(parent=[1.0, 2.0, 3.0], child=np.ones(3), cluster_ids=np.arange(3))


def test_rejects_misaligned_pandas_series():
    idx = pd.date_range("2020-01-01", periods=10)
    a = pd.Series(np.arange(10.0), index=idx)
    with pytest.raises(ValueError, match="different indexes"):
        paired_cluster_test(parent=a, child=a[::-1], cluster_ids=np.repeat(np.arange(5), 2))


def test_arguments_are_keyword_only():
    with pytest.raises(TypeError):
        paired_cluster_test(np.ones(10), np.ones(10), np.arange(10))  # type: ignore[misc]


# ---------------------------------------------------------------- cluster ids


def test_make_cluster_ids_season_boundaries():
    ids = make_cluster_ids(["2019-03-14", "2019-03-15", "2019-07-31", "2019-08-01", "2019-12-31", "2020-01-01"])
    assert ids.tolist() == [20190, 20191, 20191, 20192, 20192, 20200]


def test_make_cluster_ids_leap_day_and_timestamps():
    ids = make_cluster_ids(pd.DatetimeIndex([pd.Timestamp("2020-02-29"), pd.Timestamp("2024-06-15 13:00")]))
    assert ids.tolist() == [20200, 20241]


def test_make_cluster_ids_rejects_nat():
    with pytest.raises(ValueError, match="NaT"):
        make_cluster_ids(pd.DatetimeIndex([pd.Timestamp("2020-01-01"), pd.NaT]))


# ---------------------------------------------------------------- Diebold-Mariano


def test_newey_west_lag0_is_plain_variance():
    x = np.random.default_rng(20).normal(size=500)
    assert newey_west_variance(x, 0) == pytest.approx(np.var(x))


def test_newey_west_matches_statsmodels():
    sw = pytest.importorskip("statsmodels.stats.sandwich_covariance")
    x = _ar1(np.random.default_rng(21), 800, 0.6)
    theirs = sw.S_hac_simple(x - x.mean(), nlags=4)[0, 0] / x.size  # statsmodels: unnormalised sum
    assert newey_west_variance(x, 4) == pytest.approx(theirs, rel=1e-10)


def test_andrews_lag_floor_cap_and_growth():
    rng = np.random.default_rng(22)
    assert andrews_lag(rng.normal(size=500), horizon=5) == 4  # white noise -> floor h-1
    assert andrews_lag(_ar1(rng, 1000, 0.9)) > andrews_lag(_ar1(rng, 1000, 0.3))
    assert andrews_lag(_ar1(rng, 40, 0.97)) <= 10  # capped at n // 4


def test_dm_false_positive_rate_with_persistent_errors():
    rng = np.random.default_rng(23)
    reps, rejections = 300, 0
    for _ in range(reps):
        _, _, p, c = _simulate(rng, 8)
        la = (p[0].reshape(-1, H) ** 2).mean(axis=1)  # per-window MSE, time-ordered
        lb = (c[0].reshape(-1, H) ** 2).mean(axis=1)
        rejections += dm_test(la, lb).p_value < 0.05
    rate = rejections / reps
    assert 0.01 <= rate <= 0.09, f"DM FPR {rate:.3f}"


def test_dm_detects_planted_effect_and_sign():
    rng = np.random.default_rng(24)
    _, _, p, c = _simulate(rng, 8, scale_child=1.3)  # B clearly worse
    la, lb = (p[0].reshape(-1, H) ** 2).mean(axis=1), (c[0].reshape(-1, H) ** 2).mean(axis=1)
    r = dm_test(la, lb)
    assert r.statistic < 0 and r.p_value < 0.01 and r.mean_diff < 0 and r.lag >= H - 1


def test_dm_explicit_lag_is_used():
    x = np.random.default_rng(25).random(200)
    assert dm_test(x, x[::-1].copy(), lag=7).lag == 7


def test_dm_degenerate_series():
    x = np.random.default_rng(26).random(100)
    same = dm_test(x, x.copy())
    assert same.statistic == 0.0 and same.p_value == 1.0
    offset = dm_test(x, x + 1.0)  # deterministic difference: nothing to test
    assert np.isnan(offset.p_value) and offset.mean_diff == pytest.approx(-1.0)


def test_dm_checks_dates_are_ordered():
    rng = np.random.default_rng(27)
    a, b = rng.random(100), rng.random(100)
    dates = pd.date_range("2016-01-01", periods=100)
    assert dm_test(a, b, dates=dates).n == 100
    with pytest.raises(ValueError, match="strictly increasing"):
        dm_test(a, b, dates=dates[::-1])


def test_dm_rejects_bad_input():
    with pytest.raises(ValueError):
        dm_test(np.ones(30), np.ones(29))
    with pytest.raises(ValueError):
        dm_test(np.ones(10), np.ones(10))  # too short
    with pytest.raises(ValueError):
        dm_test(np.r_[np.inf, np.ones(99)], np.ones(100))
    with pytest.raises(ValueError):
        dm_test(np.ones(100), np.ones(100), lag=-1)
