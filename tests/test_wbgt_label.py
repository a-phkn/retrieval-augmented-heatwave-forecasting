"""WBGT heatwave label (pipeline/labels_v2.py) and its pre-registered percentile rule
(evaluation/select_wbgt_label.py). Pre-2019 data only."""
import json

import numpy as np
import pandas as pd
import pytest

import evaluation.select_wbgt_label as sel
from pipeline.labels_v2 import WBGT_PERCENTILES, in_wbgt_season, label_frame_wbgt, wbgt_threshold
from scripts.make_manifest import REPO_ROOT


def test_wbgt_season_runs_mar15_to_sep30():
    s = in_wbgt_season(["2010-03-14", "2010-03-15", "2010-09-30", "2010-10-01", "2010-07-15"])
    assert s.tolist() == [False, True, True, False, True]


def test_threshold_uses_only_in_season_training_days():
    dates = pd.date_range("2000-01-01", "2003-12-31")
    rng = np.random.default_rng(0)
    values = rng.normal(30, 3, len(dates))
    train = dates.year <= 2001
    a = wbgt_threshold(dates, values, train, 95)
    altered = values.copy()
    altered[~train] += 10  # later years change
    altered[train & ~in_wbgt_season(dates)] += 10  # off-season training days change
    assert wbgt_threshold(dates, altered, train, 95) == a


def test_label_frame_wbgt_gates_season_and_builds_episodes():
    dates = pd.date_range("2010-09-27", periods=8)  # Sep 27 .. Oct 4
    values = np.full(8, 40.0)
    lab = label_frame_wbgt(dates, values, threshold=35.0)
    assert lab["hot_wbgt"].tolist() == [True] * 4 + [False] * 4  # October is out of season
    assert (lab["stratum_wbgt"].iloc[:4] == "extreme").all() and lab["episode_id_wbgt"].iloc[0] == 1


def test_choose_takes_the_highest_percentile_meeting_the_episode_rule():
    rows = [{"percentile": p, "fold": "f1", "episodes": e}
            for p, e in zip(WBGT_PERCENTILES, (10, 24, 25, 40, 60, 80))]
    assert sel.choose(pd.DataFrame(rows)) == 97.5
    none = pd.DataFrame([{"percentile": p, "fold": "f1", "episodes": 1} for p in WBGT_PERCENTILES])
    assert sel.choose(none) is None


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_shipped_label_config_is_reproduced_by_the_rule():
    cfg = json.loads((REPO_ROOT / "configs/wbgt_label.json").read_text(encoding="utf-8"))
    d = sel.load_pre_test()
    assert d.index.max() < pd.Timestamp("2019-01-01")
    assert cfg["primary_variable"] == sel.PRIMARY == "wbgt_lj_max"  # decision 2026-10-05
    table = sel.count_table(d, sel.PRIMARY)
    p = sel.choose(table)
    assert p == cfg["chosen_percentile"] == cfg[sel.PRIMARY]["chosen_percentile"]
    pooled = table.groupby("percentile")["episodes"].sum()
    assert pooled[p] >= 25 and all(pooled[q] < 25 for q in WBGT_PERCENTILES if q > p)
