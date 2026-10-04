"""
pipeline/climatology.py and pipeline/labels_v2.py.

Integration tests prove the shared code reproduces v1 exactly (climatology and the
episode catalogue: 85 of v1's 102 episodes fall before 2019) when given v1's settings.
The acceptance test checks labels v2 on known pre-2019 Delhi/north-India heatwaves.
Every test reads pre-2019 rows only: the test period (2019+) stays locked, so nothing is
checked or tuned on test-period data.
"""
import numpy as np
import pandas as pd
import pytest

from pipeline.climatology import apply_climatology, doy_climatology
from pipeline.labels_v2 import episodes, hot_days_tmax, in_season, label_frame
from scripts.make_manifest import REPO_ROOT

TRAIN_END = pd.Timestamp("2015-12-31")
TEST_START = pd.Timestamp("2019-01-01")


@pytest.fixture(scope="module")
def daily():
    """v1 daily table, pre-2019 rows only (filtered at read time)."""
    d = pd.read_parquet(REPO_ROOT / "datasets/all_daily.parquet", filters=[("date", "<", TEST_START)])
    d = d.set_index("date").sort_index()
    assert d.index.max() < TEST_START
    return d


# ---------------------------------------------------------------- v1 reproduction


def test_climatology_reproduces_v1_exactly(daily):
    clim = doy_climatology(daily["t_max"], daily.index <= TRAIN_END)
    mean, std = apply_climatology(daily.index, clim)
    assert np.allclose(mean, daily["clim_mean_t_max"].to_numpy(), atol=1e-9)
    assert np.allclose(std, daily["clim_std_t_max"].to_numpy(), atol=1e-9)
    assert np.allclose(daily["t_max"].to_numpy() - mean, daily["t_max_anomaly"].to_numpy(), atol=1e-9)


def test_episode_code_reproduces_v1_catalogue(daily):
    """Same episode convention as v1: given v1's hot days and v1's 3-day minimum span,
    the shared function must rebuild events/event_catalogue.parquet exactly."""
    v1_hot = (daily["t_max_anomaly"] > 1.5 * daily["clim_std_t_max"]).to_numpy()
    ids, table = episodes(daily.index, v1_hot, min_span_days=3)
    v1 = pd.read_parquet(REPO_ROOT / "events/event_catalogue.parquet")
    v1 = v1[pd.to_datetime(v1["episode_end"]) < TEST_START].reset_index(drop=True)
    assert len(table) == len(v1) == 85
    assert (table["episode_start"].to_numpy() == pd.to_datetime(v1["episode_start"]).to_numpy()).all()
    assert (table["episode_end"].to_numpy() == pd.to_datetime(v1["episode_end"]).to_numpy()).all()
    v1_ids = daily["heatwave_episode_id"].fillna(0).astype(int).to_numpy()
    assert (ids == v1_ids).all()


def test_climatology_uses_only_training_years(daily):
    """Changing a non-training value must not change the climatology."""
    mask = daily.index <= TRAIN_END
    altered = daily["t_max"].copy()
    altered[~mask] = altered[~mask] + 10.0
    a = doy_climatology(daily["t_max"], mask)
    b = doy_climatology(altered, mask)
    pd.testing.assert_frame_equal(a, b)


# ---------------------------------------------------------------- rule units


def test_season_gate_boundaries():
    s = in_season(["2010-03-14", "2010-03-15", "2010-07-31", "2010-08-01", "2010-01-20"])
    assert s.tolist() == [False, True, True, False, False]


def test_hot_rule():
    dates = ["2010-05-01"] * 5 + ["2010-01-15"]
    t = [40.0, 39.9, 41.0, 45.0, 44.9, 46.0]
    a = [3.0, 5.0, 2.9, 0.0, 2.0, 9.0]
    assert hot_days_tmax(dates, t, a).tolist() == [True, False, False, True, False, False]


def test_episode_convention_on_synthetic_days():
    dates = pd.date_range("2010-05-01", periods=12)
    hot = np.array([1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0], dtype=bool)
    ids, table = episodes(dates, hot)
    # days 0-2 (gap of one day) -> one episode spanning 3 days; days 5-6 -> 2-day episode;
    # day 10 alone -> span 1, not an episode.
    assert table["duration_days"].tolist() == [3, 2]
    assert ids.tolist() == [1, 1, 1, 0, 0, 2, 2, 0, 0, 0, 0, 0]
    lab = label_frame(dates, np.where(hot, 41.0, 35.0), np.where(hot, 4.0, 0.0))
    assert lab["stratum_v2"].tolist()[:3] == ["extreme", "normal", "extreme"]  # gap day not extreme
    assert lab["stratum_v2"].iloc[10] == "unusual"


def test_rejects_non_contiguous_dates():
    with pytest.raises(ValueError, match="contiguous"):
        episodes(pd.DatetimeIndex(["2010-05-01", "2010-05-03"]), [True, True])


# ---------------------------------------------------------------- acceptance (pre-2019 only)

# Widely reported Delhi / north-India heatwaves before 2019 (to be confirmed against IMD
# records; see docs/PHASE6_SOURCE_RESEARCH_BRIEF.md). Each window must overlap a v2 episode.
KNOWN_EVENTS = [
    ("1998-05-24", "1998-06-01"),  # late May - early June 1998
    ("2002-05-10", "2002-05-20"),  # May 2002
    ("2010-04-10", "2010-04-20"),  # April 2010
    ("2015-05-22", "2015-05-26"),  # late May 2015
]


@pytest.fixture(scope="module")
def labels_pre2019(daily):
    dev = daily  # already pre-2019 only
    clim = doy_climatology(dev["t_max"], dev.index <= TRAIN_END)
    mean, _ = apply_climatology(dev.index, clim)
    return label_frame(dev.index, dev["t_max"], dev["t_max"].to_numpy() - mean)


@pytest.mark.parametrize("start, end", KNOWN_EVENTS)
def test_known_pre2019_heatwaves_are_flagged(labels_pre2019, start, end):
    window = labels_pre2019.loc[start:end]
    assert (window["stratum_v2"] == "extreme").sum() >= 2, f"no v2 episode in {start}..{end}"


def test_no_heatwave_labels_outside_season(labels_pre2019):
    off = ~in_season(labels_pre2019.index)
    assert not labels_pre2019.loc[off, "hot_v2"].any()
    jan_feb = labels_pre2019.index.month.isin([1, 2])
    assert not labels_pre2019.loc[jan_feb, "hot_v2"].any()


def test_episode_counts_are_usable(labels_pre2019):
    """Pre-registered minimum-sample rule: >= 25 episodes over the rolling-fold years."""
    ids = labels_pre2019["episode_id_v2"]
    fold_years = (labels_pre2019.index.year >= 2007) & (labels_pre2019.index.year <= 2018)
    assert ids[fold_years & (ids > 0)].nunique() >= 25
