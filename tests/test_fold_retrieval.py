"""retrieval/fold_retrieval.py: per-fold candidate pool, v1 eligibility rules, the three
modes (sim / rand / time) and the analogue-ensemble forecast. Pre-2019 data only."""
import numpy as np
import pandas as pd
import pytest

import retrieval.fold_retrieval as fr_mod
from retrieval.fold_retrieval import BUFFER_DAYS, DOY_WINDOW, MIN_DAYS_APART, FoldRetriever, anen_forecast
from scripts.make_manifest import REPO_ROOT
from training.folds import TEST_START, fold_bounds

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(scope="module")
def f1():
    return FoldRetriever("f1", "v2")


def _analogue_dates(fr, r):
    return np.where(r.idx >= 0, fr.cand_dates.values[np.maximum(r.idx, 0)], np.datetime64("NaT"))


@pytest.mark.skipif(not (REPO_ROOT / "retrieval/analogues_top20.parquet").exists(),
                    reason="v1 analogue table is gitignored; rebuild with retrieval.precompute_analogues")
def test_primary_fold_with_v1_labels_reproduces_v1_analogues():
    """f4's training years are v1's, so with v1 labels the R0 top-5 must equal v1's table.
    Validation queries: identical. Training queries: v1's precompute screened only the 500
    most similar windows without query.py's full-scan fallback, so some 1980s queries got
    fewer analogues; v1's list must then be the start of ours, never different."""
    fr = FoldRetriever("f4", "v1")
    v1 = pd.read_parquet(REPO_ROOT / "retrieval/analogues_top20.parquet")
    v1 = {q: list(g.sort_values("rank")["analogue_query_date"][:5]) for q, g in v1.groupby("query_date")}
    r = fr.retrieve(fr.val_w["query_date"], "sim")
    for q, row in zip(fr.val_w["query_date"], r.idx):
        assert list(fr.cand_dates[row[row >= 0]]) == v1.get(q, []), q
    r = fr.retrieve(fr.train_w["query_date"], "sim")
    n_short = 0
    for q, row in zip(fr.train_w["query_date"], r.idx):
        mine, old = list(fr.cand_dates[row[row >= 0]]), v1.get(q, [])
        assert mine[: len(old)] == old, q
        n_short += len(old) < len(mine)
        assert len(old) == len(mine) or q < pd.Timestamp("1989-01-01"), q
    # Regression pin: deterministic for the frozen v1 data and window index (measured 2026-10-06).
    # If it changes, either the frozen inputs or the eligibility rules changed; find out which.
    assert n_short == 881


@pytest.mark.parametrize("mode", ["sim", "rand", "time", "time_rand"])
def test_analogues_are_eligible_training_windows(f1, mode):
    train_end, _, _ = fold_bounds("f1")
    for windows in (f1.val_w, f1.train_w.iloc[::7]):
        q = pd.DatetimeIndex(windows["query_date"])
        r = f1.retrieve(q, mode, seed=3)
        a = _analogue_dates(f1, r)
        ok = r.idx >= 0
        cutoff = (q - pd.Timedelta(days=BUFFER_DAYS)).values[:, None]
        assert (a[ok] <= np.broadcast_to(cutoff, a.shape)[ok]).all()  # rule 1 (chronological)
        assert (a[ok] + np.timedelta64(4, "D") <= np.datetime64(train_end)).all()  # outcome ends in training years
        assert (a[ok] < np.datetime64(TEST_START)).all()
        q_ep = fr_mod.window_episode(f1.d, q)
        for i in range(len(q)):
            row = r.idx[i][r.idx[i] >= 0]
            days = np.sort(f1.cand_dates.values[row].astype("datetime64[D]").astype(np.int64))
            assert (np.diff(days) >= MIN_DAYS_APART).all()  # dedup: spacing
            eps = f1.cand_episode[row]
            assert all((eps == e).sum() <= 2 for e in set(eps[eps > 0]))  # dedup: <= 2 per episode
            if q_ep[i] > 0:
                assert q_ep[i] not in eps  # rule 3: not from the query's own episode
        if mode in ("time", "time_rand"):
            qd = (q - pd.Timedelta(days=1)).dayofyear.to_numpy()[:, None]
            cd = f1.cand_doy[np.maximum(r.idx, 0)]
            diff = np.abs(cd - qd)
            assert (np.minimum(diff, 365 - diff)[ok] <= DOY_WINDOW).all()


def test_validation_queries_get_full_analogue_sets(f1):
    for mode in ("sim", "rand", "time", "time_rand"):
        r = f1.retrieve(f1.val_w["query_date"], mode, seed=0)
        assert (r.idx >= 0).all(), mode


def test_earliest_training_queries_have_no_analogue(f1):
    r = f1.retrieve(f1.train_w["query_date"].iloc[:5], "sim")
    assert (r.idx == -1).all() and (r.n_eligible == 0).all() and np.isnan(r.sim).all()


def test_random_mode_is_seeded_and_differs_from_similarity(f1):
    q = f1.val_w["query_date"]
    a, b, c = (f1.retrieve(q, "rand", seed=s).idx for s in (1, 1, 2))
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    sim = f1.retrieve(q, "sim")
    assert (a != sim.idx).any(axis=1).mean() > 0.9
    assert np.nanmean(f1.retrieve(q, "rand", seed=1).sim) < np.nanmean(sim.sim)
    with pytest.raises(ValueError, match="seed"):
        f1.retrieve(q, "rand")


def test_random_control_sees_exactly_the_same_eligible_pool(f1):
    """R0-rand differs from R0 only in how it orders the eligible candidates."""
    for windows in (f1.val_w, f1.train_w.iloc[::11]):
        q = windows["query_date"]
        assert np.array_equal(f1.retrieve(q, "sim").n_eligible, f1.retrieve(q, "rand", seed=4).n_eligible)


def test_time_random_control_matches_r1_pool_and_is_seeded(f1):
    """R1-rand differs from R1 only in how it orders R1's eligible candidates."""
    for windows in (f1.val_w, f1.train_w.iloc[::11]):
        q = windows["query_date"]
        assert np.array_equal(f1.retrieve(q, "time").n_eligible, f1.retrieve(q, "time_rand", seed=4).n_eligible)
    q = f1.val_w["query_date"]
    a, b, c = (f1.retrieve(q, "time_rand", seed=s).idx for s in (1, 1, 2))
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert (a != f1.retrieve(q, "time").idx).any(axis=1).mean() > 0.9
    with pytest.raises(ValueError, match="seed"):
        f1.retrieve(q, "time_rand")


def test_similarity_order_is_descending(f1):
    r = f1.retrieve(f1.val_w["query_date"], "sim")
    assert (np.diff(r.sim, axis=1) <= 1e-12).all()


def test_no_leakage_from_after_the_training_years(monkeypatch):
    """Changing every value after train_end must not change the pool, the normalisation or
    any retrieval for validation queries (features use fold climatology and training data)."""
    import training.folds as tf

    train_end, _, _ = fold_bounds("f2")
    before = FoldRetriever("f2", "v2")
    original = tf._load_daily_pre_test

    def altered():
        d = original().copy()
        cols = d.select_dtypes("number").columns.drop(["doy_sin", "doy_cos", "years_since_1980"])
        d.loc[d.index > train_end, cols] += 7.0
        return d

    monkeypatch.setattr(tf, "_load_daily_pre_test", altered)
    after = FoldRetriever("f2", "v2")
    assert np.array_equal(before.vectors, after.vectors) and np.array_equal(before.mean, after.mean)
    # validation inputs reach into the altered years, but candidates and their order must
    # only depend on training data; the chosen set may change because the QUERY changed.
    q = before.val_w["query_date"].iloc[:1]  # first val query: its inputs are all training days
    assert np.array_equal(before.retrieve(q, "sim").idx, after.retrieve(q, "sim").idx)


def test_unknown_mode_and_fold_rejected(f1):
    with pytest.raises(ValueError):
        f1.retrieve(f1.val_w["query_date"], "nearest")
    with pytest.raises(ValueError):
        FoldRetriever("f9", "v2")


def test_anen_forecast_averages_standardised_analogue_outcomes(f1):
    d = f1.d
    q = pd.DatetimeIndex(f1.val_w["query_date"].iloc[:3])
    an = np.array([[np.datetime64("2001-05-10"), np.datetime64("1999-06-01")],
                   [np.datetime64("1995-04-20"), np.datetime64("NaT")],
                   [np.datetime64("NaT"), np.datetime64("NaT")]])
    out = anen_forecast(d, "t_max", q, an)
    z = (d["t_max_anomaly"] / d["clim_std_t_max"])
    zbar0 = (z.loc["2001-05-10":"2001-05-14"].to_numpy() + z.loc["1999-06-01":"1999-06-05"].to_numpy()) / 2
    days0 = pd.date_range(q[0], periods=5)
    assert np.allclose(out[0], d.loc[days0, "clim_mean_t_max"] + d.loc[days0, "clim_std_t_max"] * zbar0)
    days2 = pd.date_range(q[2], periods=5)
    assert np.allclose(out[2], d.loc[days2, "clim_mean_t_max"])  # no analogue -> climatology
