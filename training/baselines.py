"""
Persistence and climatology sanity baselines for the 5-day T_max forecast.

Persistence: forecast day t+k = last observed t_max (day t, i.e. the final
day of the 14-day input window), repeated for all 5 forecast days.

Climatology: forecast day t+k = train-only +/-7-day-of-year climatology mean
for that specific calendar date (reuses P1's precomputed clim_mean_t_max
column directly -- not recomputed here).

These exist purely as a sanity floor: any real model (the LSTM baseline)
must beat both on val MAE/RMSE, per Execution_Pipeline.md Step 3's
validation criterion.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from training.data import DATASETS_DIR, FEATURE_COLUMNS, TARGET_COLUMN, _load_daily, _load_windows

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO_ROOT / "evaluation"

T_MAX_IDX = FEATURE_COLUMNS.index("t_max")
CLIM_MEAN_IDX = FEATURE_COLUMNS.index("clim_mean_t_max")


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def persistence_predict(X_raw: np.ndarray) -> np.ndarray:
    """X_raw: (N, 14, 13). Returns (N, 5) -- last day's t_max repeated."""
    last_t_max = X_raw[:, -1, T_MAX_IDX]  # (N,)
    return np.repeat(last_t_max[:, None], repeats=5, axis=1)


def climatology_predict(daily: pd.DataFrame, windows: pd.DataFrame, split: str) -> np.ndarray:
    """Looks up clim_mean_t_max directly for each of the 5 forecast dates,
    per window, for the given split. This uses the already-computed
    train-only climatology column -- no leakage, since clim_mean_t_max was
    fit on train-only data by prepare_datasets.py (P1)."""
    split_windows = windows.loc[windows["split"] == split].reset_index(drop=True)
    clim_block = daily["clim_mean_t_max"]

    n = len(split_windows)
    preds = np.empty((n, 5), dtype=np.float32)
    for i, row in enumerate(split_windows.itertuples(index=False)):
        preds[i] = clim_block.loc[row.forecast_start : row.forecast_end].to_numpy(dtype=np.float32)
    return preds


def evaluate_baselines() -> dict:
    daily = _load_daily()
    windows = _load_windows()

    results = {}
    for split in ("train", "val", "test"):
        split_windows = windows.loc[windows["split"] == split].reset_index(drop=True)
        feature_block = daily[FEATURE_COLUMNS]
        target_block = daily[TARGET_COLUMN]

        n = len(split_windows)
        X_raw = np.empty((n, 14, len(FEATURE_COLUMNS)), dtype=np.float32)
        y_raw = np.empty((n, 5), dtype=np.float32)
        for i, row in enumerate(split_windows.itertuples(index=False)):
            X_raw[i] = feature_block.loc[row.input_start : row.input_end].to_numpy(dtype=np.float32)
            y_raw[i] = target_block.loc[row.forecast_start : row.forecast_end].to_numpy(dtype=np.float32)

        persistence_pred = persistence_predict(X_raw)
        climatology_pred = climatology_predict(daily, windows, split)

        results[split] = {
            "persistence": {"mae": mae(persistence_pred, y_raw), "rmse": rmse(persistence_pred, y_raw)},
            "climatology": {"mae": mae(climatology_pred, y_raw), "rmse": rmse(climatology_pred, y_raw)},
            "n_windows": n,
        }
    return results


if __name__ == "__main__":
    results = evaluate_baselines()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVAL_DIR / "sanity_baselines.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    for split, metrics in results.items():
        print(f"\n[{split}] n={metrics['n_windows']}")
        for name in ("persistence", "climatology"):
            m = metrics[name]
            print(f"  {name:12s}  MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}")
    print(f"\nSaved to {out_path}")
