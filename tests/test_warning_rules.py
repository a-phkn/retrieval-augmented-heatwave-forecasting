"""post_test/warning_rules.py: the warning-level choice, frequency bias and the cluster bootstrap, on hand-made data."""
import numpy as np
import pandas as pd
import pytest

import post_test.warning_rules as wr


def test_choose_picks_the_highest_ets_and_breaks_ties_towards_the_current_rule():
    o = np.array([1, 1, 0, 0, 0, 0], bool)
    p = np.array([0.9, 0.4, 0.35, 0.1, 0.05, 0.0])
    t, ets = wr.choose(o, p, np.round(np.arange(0.01, 1.0, 0.01), 2), "prob")
    # every t in (0.35, 0.40] warns exactly the two events: perfect; the largest such t is kept
    assert t == 0.4 and ets == pytest.approx(1.0)
    # shift: forecast - threshold; events at -0.3 and +0.2 -> the smallest delta catching both without false alarms
    gap = np.array([0.2, -0.3, -1.0, -2.0, -2.0, -3.0])
    d, ets = wr.choose(o, gap, np.round(np.arange(0.0, 3.0001, 0.05), 2), "shift")
    assert d == 0.3 and ets == pytest.approx(1.0)


def test_scores_add_frequency_bias():
    s = wr.scores(np.array([1, 1, 0, 0], bool), np.array([1, 0, 1, 1], bool))
    assert s["frequency_bias"] == pytest.approx(3 / 2) and s["hit_rate"] == pytest.approx(0.5)


def test_bootstrap_counts_and_paired_differences():
    rng = np.random.default_rng(1)
    clusters = np.repeat(np.arange(20), 30)
    o = rng.random(600) < 0.2
    good, bad = o.copy(), np.zeros(600, bool)
    good[rng.random(600) < 0.05] = True  # a few false alarms
    out = wr.bootstrap(clusters, o, {"good": good, "bad": bad}, [("good", "bad")], n_boot=300, seed=0)
    lo, hi = out["rules"]["good"]["hit_rate_ci"]
    assert lo == pytest.approx(1.0) and hi == pytest.approx(1.0)  # every event is warned
    assert out["pairs"]["good minus bad"]["share_ets_above_0"] == pytest.approx(1.0)
    c = wr.cluster_counts(clusters, o, good)
    assert int(c["n"].sum()) == 600 and int(c["h"].sum()) == int(o.sum())


def test_persistence_uses_the_day_before_the_query(monkeypatch):
    hot = pd.Series([False, True, False], index=pd.to_datetime(["2020-05-01", "2020-05-02", "2020-05-03"]), name="hot")
    import training.folds as folds
    monkeypatch.setattr(folds, "fold_daily", lambda fold, target, labels: (hot.to_frame(), None))
    rows = pd.DataFrame({"query_date": pd.to_datetime(["2020-05-02", "2020-05-03", "2020-05-03", "2020-05-04"]),
                         "lead": [1, 1, 2, 1], "fold": "test"})
    assert wr.persistence_warning(rows, "t_max", "v2").tolist() == [False, True, True, False]
