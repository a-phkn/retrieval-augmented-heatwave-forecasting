"""The test lock (training/test_lock.py) and the "test" fold. No real 2019+ value is read here: the unlocked
path is checked on SYNTHETIC test-period rows (copies of 2011-2018 rows shifted forward, plus an offset)."""
import os
import subprocess

import numpy as np
import pandas as pd
import pytest

import training.folds as folds
import training.graph_data as gd
from training import test_lock


def test_everything_refuses_the_test_period_while_locked(monkeypatch):
    from retrieval.fold_retrieval import FoldRetriever

    # Simulate "not frozen yet" by pointing the lock at a tag that does not exist (works with or without the real tag)
    monkeypatch.setattr(test_lock, "TAG", "no-such-tag-for-this-test")
    monkeypatch.setattr(test_lock, "_unlocked", False)
    for call in (test_lock.require, lambda: folds.fold_bounds("test"), lambda: folds.fold_windows("test"),
                 lambda: folds.build_fold("test", "t_max", "v2"), lambda: gd.load_upstream(include_test=True),
                 lambda: FoldRetriever("test", "v2", "t_max")):
        with pytest.raises(test_lock.TestLocked):
            call()
    assert "test" not in folds.FOLDS  # loops over development folds never include it


# ---- the four lock conditions, on a throwaway git repo (not this repo) ----

def _git(repo, *args):
    # Isolated from the user's and the system git config (no signing hooks, no aliases); this repo is thrown away.
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
    subprocess.run(["git", "-c", "user.name=lock-test", "-c", "user.email=lock-test@example.invalid", *args],
                   cwd=repo, env=env, check=True, capture_output=True)


@pytest.fixture
def frozen_repo(tmp_path, monkeypatch):
    """A repo with one locked file, committed and tagged; test_lock is pointed at it."""
    (tmp_path / "training").mkdir()
    (tmp_path / "training" / "code.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("notes\n", encoding="utf-8")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "frozen")
    _git(tmp_path, "tag", test_lock.TAG)  # lightweight tag: no tag message or signature involved
    monkeypatch.setattr(test_lock, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(test_lock, "_unlocked", False)
    return tmp_path


def test_lock_opens_on_the_tagged_commit_and_after_later_non_code_commits(frozen_repo):
    test_lock.require()
    (frozen_repo / "README.md").write_text("results added\n", encoding="utf-8")
    _git(frozen_repo, "commit", "-q", "-am", "results")
    test_lock._unlocked = False
    test_lock.require()  # later commits outside the locked folders are allowed
    (frozen_repo / "training" / "results.parquet").write_bytes(b"")  # ignored files do not count ...
    (frozen_repo / ".gitignore").write_text("*.parquet\n", encoding="utf-8")
    test_lock._unlocked = False
    test_lock.require()


def test_lock_refuses_a_committed_code_change(frozen_repo):
    (frozen_repo / "training" / "code.py").write_text("x = 2\n", encoding="utf-8")
    _git(frozen_repo, "commit", "-q", "-am", "edit frozen code")
    with pytest.raises(test_lock.TestLocked, match="changed after"):
        test_lock.require()


def test_lock_refuses_an_uncommitted_code_change(frozen_repo):
    (frozen_repo / "training" / "code.py").write_text("x = 3\n", encoding="utf-8")
    with pytest.raises(test_lock.TestLocked, match="uncommitted"):
        test_lock.require()


def test_lock_refuses_an_untracked_file_in_a_locked_folder(frozen_repo):
    (frozen_repo / "training" / "new_module.py").write_text("y = 1\n", encoding="utf-8")
    with pytest.raises(test_lock.TestLocked, match="uncommitted"):
        test_lock.require()


def test_lock_refuses_when_head_does_not_contain_the_tag(frozen_repo):
    _git(frozen_repo, "checkout", "-q", "--orphan", "elsewhere")
    _git(frozen_repo, "commit", "-q", "-m", "unrelated history")
    with pytest.raises(test_lock.TestLocked, match="does not contain"):
        test_lock.require()


def test_lock_refuses_without_the_tag(frozen_repo):
    _git(frozen_repo, "tag", "-d", test_lock.TAG)
    with pytest.raises(test_lock.TestLocked, match="does not exist"):
        test_lock.require()


def test_failed_check_is_not_cached(frozen_repo):
    (frozen_repo / "training" / "code.py").write_text("x = 4\n", encoding="utf-8")
    with pytest.raises(test_lock.TestLocked):
        test_lock.require()
    assert test_lock._unlocked is False


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
