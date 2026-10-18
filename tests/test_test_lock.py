"""The test lock (training/test_lock.py) and the "test" fold. No real 2019+ value is read here: the unlocked
path is checked on SYNTHETIC test-period rows (copies of 2011-2018 rows shifted forward, plus an offset)."""
import subprocess

import numpy as np
import pandas as pd
import pytest

import training.folds as folds
import training.graph_data as gd
from training import test_lock

TAG_EXISTS = subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"refs/tags/{test_lock.TAG}"],
                            cwd=test_lock.REPO_ROOT, capture_output=True).returncode == 0


@pytest.mark.skipif(TAG_EXISTS, reason="the protocol is frozen; the locked state no longer applies")
def test_everything_refuses_the_test_period_while_locked(monkeypatch):
    from retrieval.fold_retrieval import FoldRetriever

    monkeypatch.setattr(test_lock, "_unlocked", False)
    for call in (test_lock.require, lambda: folds.fold_bounds("test"), lambda: folds.fold_windows("test"),
                 lambda: folds.build_fold("test", "t_max", "v2"), lambda: gd.load_upstream(include_test=True),
                 lambda: FoldRetriever("test", "v2", "t_max")):
        with pytest.raises(test_lock.TestLocked):
            call()
    assert "test" not in folds.FOLDS  # loops over development folds never include it


def test_development_folds_never_consult_the_lock(monkeypatch):
    def boom():
        raise AssertionError("a development fold asked for the test period")
    monkeypatch.setattr(test_lock, "require", boom)
    data = folds.build_fold("f4", "t_max", "v2")
    assert pd.DatetimeIndex(data.val.query_dates).max() < folds.TEST_START
    assert gd.load_upstream().index.max() < folds.TEST_START


def _synthetic(tmp_path, offset: float):
    """Real pre-2019 rows + synthetic test-period rows (2011+ rows moved forward by 8 years, + offset)."""
    end = folds.TEST_END
    new_dates = pd.date_range(folds.TEST_START, end, freq="D")

    def extend(df: pd.DataFrame) -> pd.DataFrame:
        src = df[df["date"] >= pd.Timestamp("2011-01-01")].head(len(new_dates)).copy()
        assert len(src) == len(new_dates)
        num = src.drop(columns="date").select_dtypes("number").columns
        src[num] = src[num].astype(float) + offset
        src["date"] = new_dates.values
        return pd.concat([df, src], ignore_index=True)

    daily = pd.read_parquet(folds.DAILY_V2_PATH, filters=[("date", "<", folds.TEST_START)])
    lj = pd.read_parquet(folds.LILJEGREN_DAILY_PATH, filters=[("date", "<", folds.TEST_START)])
    p_daily, p_lj = tmp_path / f"daily_{offset}.parquet", tmp_path / f"lj_{offset}.parquet"
    extend(daily).to_parquet(p_daily, index=False)
    extend(lj).to_parquet(p_lj, index=False)
    return p_daily, p_lj


@pytest.mark.parametrize("target,labels,form", [("t_max", "v2", "raw"), ("wbgt_lj_max", "wbgt", "anomaly")])
def test_unlocked_test_fold_reads_test_rows_but_no_statistic_depends_on_them(tmp_path, monkeypatch, target, labels, form):
    monkeypatch.setattr(test_lock, "require", lambda: None)
    built = {}
    for offset in (100.0, 200.0):
        p_daily, p_lj = _synthetic(tmp_path, offset)
        monkeypatch.setattr(folds, "DAILY_V2_PATH", p_daily)
        monkeypatch.setattr(folds, "LILJEGREN_DAILY_PATH", p_lj)
        built[offset] = folds.build_fold("test", target, labels, target_form=form)
    a, b = built[100.0], built[200.0]
    # the test period is the validation block and it is read: values differ exactly by the offset change
    q = pd.DatetimeIndex(a.val.query_dates)
    assert q.min() >= folds.TEST_START and (q + pd.Timedelta(days=4)).max() <= folds.TEST_END
    w = folds._load_windows()
    assert len(q) == int(((w["query_date"] >= folds.TEST_START) & (w["forecast_end"] <= folds.TEST_END)).sum())
    np.testing.assert_allclose(b.val.y_raw - a.val.y_raw, 100.0, atol=1e-3)
    # nothing fitted depends on the test rows: training windows, normalisation, target scaling, climatology
    assert pd.DatetimeIndex(a.train.query_dates).max() <= folds.TEST_START - pd.Timedelta(days=5)
    np.testing.assert_array_equal(a.feature_mean, b.feature_mean)
    np.testing.assert_array_equal(a.feature_std, b.feature_std)
    assert a.y_mean == b.y_mean and a.y_std == b.y_std
    np.testing.assert_array_equal(a.train.X, b.train.X)
    np.testing.assert_array_equal(a.train.hot, b.train.hot)
    np.testing.assert_allclose(a.val.clim_target, b.val.clim_target)


def test_unlocked_upstream_table_extends_to_the_test_end_only_on_request(tmp_path, monkeypatch):
    monkeypatch.setattr(test_lock, "require", lambda: None)
    up = pd.read_parquet(gd.UPSTREAM_PATH, filters=[("date", "<", folds.TEST_START)])
    new = pd.date_range(folds.TEST_START, folds.TEST_END, freq="D")
    tail = up.loc["2011-01-01":].head(len(new)).copy()
    tail.index = pd.DatetimeIndex(new, name=up.index.name)
    path = tmp_path / "up.parquet"
    pd.concat([up, tail]).to_parquet(path)
    monkeypatch.setattr(gd, "UPSTREAM_PATH", path)
    assert gd.load_upstream().index.max() < folds.TEST_START
    assert gd.load_upstream(include_test=True).index.max() == folds.TEST_END
