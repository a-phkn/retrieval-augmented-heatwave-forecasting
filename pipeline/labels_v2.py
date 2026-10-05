"""
Heatwave labels v2 (plan v5, decision 2; context/decisions.md).

Hot day (Tmax definition), only inside the heat season Mar 15 - Jul 31:
    (Tmax >= 40 C AND anomaly >= 3.0 C)  OR  Tmax >= 45 C      [IMD-style absolute criterion]
    anomaly = Tmax - train-only day-of-year climatology (pipeline/climatology.py), computed
    per fold from that fold's training years.
    The 3.0 C anomaly (instead of IMD's 4.5 C) follows the pre-registered minimum-sample
    rule: ERA5 9-cell means damp station peaks, and 4.5 C leaves the validation split with
    ~1 episode. IMD's 4.5 C is kept as the 'severe' sub-stratum.
Severe hot day: hot AND anomaly >= 4.5 C.
Episode: v1 convention (prepare_datasets.py) -- hot days at most 2 dates apart (one
    non-hot day in between) join the same episode; an episode spans first..last hot day
    (gap days included) and must span >= 2 days (v1 used >= 3 and a year-round
    relative-anomaly hot day).
Stratum per day (for evaluation): 'extreme' = hot day inside an episode, 'unusual' = hot
    but not in an episode, 'normal' otherwise (as in v1, a non-hot gap day inside an
    episode is not 'extreme'). Off-season days are always 'normal' for v2.

WBGT label (decision 2026-10-05; label_frame_wbgt):
    Hot WBGT day: inside the WBGT season Mar 15 - Sep 30 (longer than the Tmax season
    because humid heat peaks in the Jul-Sep monsoon) AND daily max WBGT >= the P-th
    percentile of in-season daily max WBGT over the fold's TRAINING years.
    P is the highest of WBGT_PERCENTILES that gives >= 25 episodes pooled over the four
    validation blocks, each labelled with its own fold's threshold (the same pre-registered
    minimum-sample rule as for Tmax; evaluation/select_wbgt_label.py). Primary variable
    (decision 2026-10-05): the physical Liljegren WBGT (95th percentile, 29 episodes); the BoM
    index is reported as a sensitivity check.
    Episodes and strata use the same convention as the Tmax label.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEASON_START_MD, SEASON_END_MD = 315, 731  # Mar 15 .. Jul 31 inclusive
TMAX_FLOOR_C = 40.0
ANOMALY_HOT_C = 3.0
ANOMALY_SEVERE_C = 4.5
TMAX_ABSOLUTE_C = 45.0
MAX_GAP_DATES = 2  # hot days at most 2 dates apart join one episode (v1 convention)
MIN_EPISODE_SPAN_DAYS = 2


def in_season(dates) -> np.ndarray:
    d = pd.DatetimeIndex(pd.to_datetime(dates))
    md = d.month * 100 + d.day
    return np.asarray((md >= SEASON_START_MD) & (md <= SEASON_END_MD))


def hot_days_tmax(dates, t_max, anomaly) -> np.ndarray:
    t_max, anomaly = np.asarray(t_max, dtype=np.float64), np.asarray(anomaly, dtype=np.float64)
    rule = ((t_max >= TMAX_FLOOR_C) & (anomaly >= ANOMALY_HOT_C)) | (t_max >= TMAX_ABSOLUTE_C)
    return in_season(dates) & rule


def episodes(dates, hot, min_span_days: int = MIN_EPISODE_SPAN_DAYS) -> tuple[np.ndarray, pd.DataFrame]:
    """Episode id per date (0 = no episode) and an episode table. `dates` must be
    contiguous daily dates (one row per day), as in all_daily."""
    d = pd.DatetimeIndex(pd.to_datetime(dates))
    if len(d) > 1 and not (np.diff(d.values).astype("timedelta64[D]").astype(int) == 1).all():
        raise ValueError("dates must be contiguous daily")
    hot = np.asarray(hot, dtype=bool)
    ids = np.zeros(len(d), dtype=np.int64)
    rows, start, last = [], None, None

    def close(start, last):
        span = (d[last] - d[start]).days + 1
        if span >= min_span_days:
            ep = len(rows) + 1
            ids[start:last + 1] = ep
            rows.append({"episode_id": ep, "episode_start": d[start], "episode_end": d[last], "duration_days": span})

    for i in np.flatnonzero(hot):
        if start is None:
            start = last = i
        elif (d[i] - d[last]).days <= MAX_GAP_DATES:
            last = i
        else:
            close(start, last)
            start = last = i
    if start is not None:
        close(start, last)
    return ids, pd.DataFrame(rows, columns=["episode_id", "episode_start", "episode_end", "duration_days"])


def label_frame(dates, t_max, anomaly) -> pd.DataFrame:
    """Per-day v2 labels: hot_v2, severe_v2, episode_id_v2 (0 = none), stratum_v2."""
    hot = hot_days_tmax(dates, t_max, anomaly)
    severe = hot & (np.asarray(anomaly, dtype=np.float64) >= ANOMALY_SEVERE_C)
    ids, _ = episodes(dates, hot)
    # As in v1: only actually-hot days count as extreme (gap days inside an episode do not).
    stratum = np.where(hot & (ids > 0), "extreme", np.where(hot, "unusual", "normal"))
    return pd.DataFrame({"hot_v2": hot, "severe_v2": severe, "episode_id_v2": ids, "stratum_v2": stratum},
                        index=pd.DatetimeIndex(pd.to_datetime(dates), name="date"))


# ---------------------------------------------------------------- WBGT label

WBGT_SEASON_START_MD, WBGT_SEASON_END_MD = 315, 930  # Mar 15 .. Sep 30 inclusive
WBGT_PERCENTILES = (99.0, 98.0, 97.5, 95.0, 92.5, 90.0)  # highest first (pre-registered)
MIN_POOLED_EPISODES = 25


def in_wbgt_season(dates) -> np.ndarray:
    d = pd.DatetimeIndex(pd.to_datetime(dates))
    md = d.month * 100 + d.day
    return np.asarray((md >= WBGT_SEASON_START_MD) & (md <= WBGT_SEASON_END_MD))


def wbgt_threshold(dates, values, train_mask, percentile: float) -> float:
    """P-th percentile of in-season daily values over the training dates only."""
    values = np.asarray(values, dtype=np.float64)
    sel = in_wbgt_season(dates) & np.asarray(train_mask, dtype=bool)
    if not sel.any():
        raise ValueError("no in-season training days")
    return float(np.percentile(values[sel], percentile))


def label_frame_wbgt(dates, values, threshold: float) -> pd.DataFrame:
    """Per-day WBGT labels: hot_wbgt, episode_id_wbgt (0 = none), stratum_wbgt."""
    hot = in_wbgt_season(dates) & (np.asarray(values, dtype=np.float64) >= threshold)
    ids, _ = episodes(dates, hot)
    stratum = np.where(hot & (ids > 0), "extreme", np.where(hot, "unusual", "normal"))
    return pd.DataFrame({"hot_wbgt": hot, "episode_id_wbgt": ids, "stratum_wbgt": stratum},
                        index=pd.DatetimeIndex(pd.to_datetime(dates), name="date"))
