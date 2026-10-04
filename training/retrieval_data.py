"""
Builds training tensors for the retrieval-augmented model: for each query
window, its own (X, y) PLUS its top-K analogues' own (X, y) pairs, using
retrieval/analogues_top20.parquet (already eligibility-filtered and
deduplicated -- no leakage/dedup logic needed here, just data assembly).

Speed: uses integer date-offset indexing into a single contiguous numpy
array (all_daily.parquet has zero gaps, so date -> row index is exact
arithmetic) instead of repeated pandas .loc[] date-range slicing -- the
naive per-window pandas approach that training/data.py uses is fine for
ONE window per sample, but here every sample needs up to K+1 window
slices, so the naive approach would be K+1x slower at exactly the point
where speed starts to matter (13,131 train windows x up to 21 slices each).

X: (N, 14, 13) query input windows, raw units (normalize with
   training.data's existing NormalizationStats before use)
y: (N, 5) query targets, raw units
X_analogues: (N, K, 14, 13) each analogue's own input window, raw units
y_analogues: (N, K, 5) each analogue's own actual outcome, raw units
analogue_mask: (N, K) bool -- True where a real analogue exists (queries
   early in the train period may have fewer than K eligible analogues;
   padded slots are zero-filled and masked out of attention)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from training.data import FEATURE_COLUMNS, FORECAST_DAYS, INPUT_DAYS, TARGET_COLUMN, _load_windows

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = REPO_ROOT / "datasets"
RETRIEVAL_DIR = REPO_ROOT / "retrieval"


def _load_daily_arrays():
    """Returns (feature_array (T,13), target_array (T,), date_to_idx dict).
    all_daily.parquet has zero gaps (verified by P1), so this is a safe,
    exact arithmetic mapping -- not an approximation."""
    daily = pd.read_parquet(DATASETS_DIR / "all_daily.parquet").sort_values("date").reset_index(drop=True)
    feature_array = daily[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    target_array = daily[TARGET_COLUMN].to_numpy(dtype=np.float32)
    date_to_idx = {d: i for i, d in enumerate(daily["date"])}
    return feature_array, target_array, date_to_idx


def _window_for(query_date: pd.Timestamp, feature_array, target_array, date_to_idx):
    """Returns (X: (14,13), y: (5,)) for the window whose OWN query_date is
    `query_date` -- works identically for a real query or an analogue,
    since both are just rows of the frozen window index
    (splits/window_index_v1.parquet, identical to datasets/forecast_windows.parquet)
    with the same query_date -> [query_date-14, query_date-1] input /
    [query_date, query_date+4] target relationship (verified against the
    window file directly)."""
    idx = date_to_idx[query_date]
    X = feature_array[idx - INPUT_DAYS: idx]
    y = target_array[idx: idx + FORECAST_DAYS]
    return X, y


def build_split_arrays_with_analogues(split: str, k: int = 5) -> dict:
    """Main entry point. Returns a dict with X, y, X_analogues, y_analogues,
    analogue_mask, query_dates -- everything needed to train/evaluate the
    retrieval-augmented model for one split."""
    windows = _load_windows()
    split_windows = windows.loc[windows["split"] == split].reset_index(drop=True)

    analogues = pd.read_parquet(RETRIEVAL_DIR / "analogues_top20.parquet")
    analogues = analogues[analogues["rank"] <= k]

    feature_array, target_array, date_to_idx = _load_daily_arrays()

    n = len(split_windows)
    X = np.empty((n, INPUT_DAYS, len(FEATURE_COLUMNS)), dtype=np.float32)
    y = np.empty((n, FORECAST_DAYS), dtype=np.float32)
    X_analogues = np.zeros((n, k, INPUT_DAYS, len(FEATURE_COLUMNS)), dtype=np.float32)
    y_analogues = np.zeros((n, k, FORECAST_DAYS), dtype=np.float32)
    analogue_mask = np.zeros((n, k), dtype=bool)

    # Group analogues by query_date once, avoid re-filtering the full
    # analogues DataFrame inside the per-query loop.
    analogues_by_query = {qd: g.sort_values("rank") for qd, g in analogues.groupby("query_date")}

    n_short = 0  # queries with fewer than k eligible analogues, for reporting
    for i, row in enumerate(split_windows.itertuples(index=False)):
        X[i], y[i] = _window_for(row.query_date, feature_array, target_array, date_to_idx)

        group = analogues_by_query.get(row.query_date)
        if group is None:
            n_short += 1
            continue
        if len(group) < k:
            n_short += 1
        for j, analogue_row in enumerate(group.itertuples(index=False)):
            if j >= k:
                break
            aX, ay = _window_for(analogue_row.analogue_query_date, feature_array, target_array, date_to_idx)
            X_analogues[i, j] = aX
            y_analogues[i, j] = ay
            analogue_mask[i, j] = True

    if n_short > 0:
        print(f"  [{split}] {n_short}/{n} queries had fewer than k={k} eligible analogues (padded/masked)")

    return {
        "X": X, "y": y,
        "X_analogues": X_analogues, "y_analogues": y_analogues,
        "analogue_mask": analogue_mask,
        "query_dates": split_windows["query_date"],
    }


if __name__ == "__main__":
    for split in ("train", "val", "test"):
        d = build_split_arrays_with_analogues(split, k=5)
        print(f"{split}: X={d['X'].shape}, X_analogues={d['X_analogues'].shape}, "
              f"mean analogues/query={d['analogue_mask'].sum(axis=1).mean():.2f}")
