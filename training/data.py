"""
Builds (X, y) arrays for the baseline/retrieval-augmented forecasters from
the already-prepared datasets/forecast_windows.parquet + datasets/all_daily.parquet.

Do NOT re-derive windowing or split logic here — this file only reshapes
what P1's pipeline already produced. See context/data.md and
context/architecture.md for the contract this relies on.

X: (N, 14, 13) float32 — 14 days of input, 13 features per day (raw units).
y: (N, 5) float32 — next 5 days of t_max (raw degC).

Normalization: z-score stats fit on TRAIN split only (on the daily feature
columns), applied unchanged to val/test. Saved to
evaluation/normalization_stats.json so downstream eval can de-normalize
predictions consistently.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = REPO_ROOT / "datasets"
EVAL_DIR = REPO_ROOT / "evaluation"

FEATURE_COLUMNS = [
    "t_max",
    "t_min",
    "t_mean",
    "relative_humidity_mean",
    "wind_speed_mean",
    "surface_pressure_mean",
    "shortwave_radiation_sum",
    "doy_sin",
    "doy_cos",
    "years_since_1980",
    "clim_mean_t_max",
    "clim_std_t_max",
    "t_max_anomaly",
]
TARGET_COLUMN = "t_max"
INPUT_DAYS = 14
FORECAST_DAYS = 5


@dataclass
class NormalizationStats:
    feature_mean: np.ndarray  # (13,)
    feature_std: np.ndarray  # (13,)
    target_mean: float
    target_std: float

    def to_json(self) -> dict:
        return {
            "feature_columns": FEATURE_COLUMNS,
            "feature_mean": self.feature_mean.tolist(),
            "feature_std": self.feature_std.tolist(),
            "target_column": TARGET_COLUMN,
            "target_mean": self.target_mean,
            "target_std": self.target_std,
        }

    @classmethod
    def from_json(cls, d: dict) -> "NormalizationStats":
        return cls(
            feature_mean=np.array(d["feature_mean"], dtype=np.float32),
            feature_std=np.array(d["feature_std"], dtype=np.float32),
            target_mean=d["target_mean"],
            target_std=d["target_std"],
        )


def _load_daily() -> pd.DataFrame:
    df = pd.read_parquet(DATASETS_DIR / "all_daily.parquet")
    df = df.set_index("date").sort_index()
    return df


def _load_windows() -> pd.DataFrame:
    return pd.read_parquet(DATASETS_DIR / "forecast_windows.parquet")


def fit_normalization_stats(daily: pd.DataFrame, windows: pd.DataFrame) -> NormalizationStats:
    """Fit z-score stats on TRAIN split's daily feature values only."""
    train_dates = windows.loc[windows["split"] == "train"]
    # Use the full set of TRAIN daily rows (not just window subsets) for a
    # stable estimate of mean/std, restricted to dates actually in-range.
    train_daily = daily.loc[daily["split"] == "train"]
    feat = train_daily[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    feature_mean = feat.mean(axis=0)
    feature_std = feat.std(axis=0)
    feature_std[feature_std == 0] = 1.0  # guard against constant columns

    target = train_daily[TARGET_COLUMN].to_numpy(dtype=np.float64)
    target_mean = float(target.mean())
    target_std = float(target.std())
    if target_std == 0:
        target_std = 1.0

    return NormalizationStats(
        feature_mean=feature_mean.astype(np.float32),
        feature_std=feature_std.astype(np.float32),
        target_mean=target_mean,
        target_std=target_std,
    )


def save_normalization_stats(stats: NormalizationStats, path: Path | None = None) -> None:
    path = path or (EVAL_DIR / "normalization_stats.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats.to_json(), f, indent=2)


def load_normalization_stats(path: Path | None = None) -> NormalizationStats:
    path = path or (EVAL_DIR / "normalization_stats.json")
    with open(path) as f:
        return NormalizationStats.from_json(json.load(f))


def build_split_arrays(
    split: str,
    daily: pd.DataFrame,
    windows: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    """Returns X (N,14,13) raw, y (N,5) raw t_max, and the query_date Series
    (for traceability / joining to episode labels later)."""
    split_windows = windows.loc[windows["split"] == split].reset_index(drop=True)
    n = len(split_windows)
    X = np.empty((n, INPUT_DAYS, len(FEATURE_COLUMNS)), dtype=np.float32)
    y = np.empty((n, FORECAST_DAYS), dtype=np.float32)

    feature_block = daily[FEATURE_COLUMNS]
    target_block = daily[TARGET_COLUMN]

    for i, row in enumerate(split_windows.itertuples(index=False)):
        in_slice = feature_block.loc[row.input_start : row.input_end]
        out_slice = target_block.loc[row.forecast_start : row.forecast_end]
        if len(in_slice) != INPUT_DAYS or len(out_slice) != FORECAST_DAYS:
            raise ValueError(
                f"Window at query_date={row.query_date} has malformed slice "
                f"lengths (in={len(in_slice)}, out={len(out_slice)}); "
                "forecast_windows.parquet may be inconsistent with all_daily.parquet."
            )
        X[i] = in_slice.to_numpy(dtype=np.float32)
        y[i] = out_slice.to_numpy(dtype=np.float32)

    return X, y, split_windows["query_date"]


def normalize_X(X: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    return (X - stats.feature_mean) / stats.feature_std


def normalize_y(y: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    return (y - stats.target_mean) / stats.target_std


def denormalize_y(y_norm: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    return y_norm * stats.target_std + stats.target_mean


def build_split_target_climatology(
    split: str,
    daily: pd.DataFrame,
    windows: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each window's 5 forecast days, returns the train-only climatology
    mean/std for that calendar date (already computed by P1, no leakage) and
    the ground-truth hot_day label. Used to derive a heatwave-detection
    confusion matrix from continuous forecasts -- NOT used for training.

    Returns: clim_mean (N,5), clim_std (N,5), actual_hot (N,5) bool.
    """
    split_windows = windows.loc[windows["split"] == split].reset_index(drop=True)
    clim_mean_block = daily["clim_mean_t_max"]
    clim_std_block = daily["clim_std_t_max"]
    hot_block = daily["hot_day"]

    n = len(split_windows)
    clim_mean = np.empty((n, FORECAST_DAYS), dtype=np.float32)
    clim_std = np.empty((n, FORECAST_DAYS), dtype=np.float32)
    actual_hot = np.empty((n, FORECAST_DAYS), dtype=bool)
    for i, row in enumerate(split_windows.itertuples(index=False)):
        clim_mean[i] = clim_mean_block.loc[row.forecast_start : row.forecast_end].to_numpy(dtype=np.float32)
        clim_std[i] = clim_std_block.loc[row.forecast_start : row.forecast_end].to_numpy(dtype=np.float32)
        actual_hot[i] = hot_block.loc[row.forecast_start : row.forecast_end].to_numpy(dtype=bool)
    return clim_mean, clim_std, actual_hot

def build_split_target_stratum(
    split: str,
    daily: pd.DataFrame,
    windows: pd.DataFrame,
) -> np.ndarray:
    """Per-forecast-day-instance stratum label:
      - normal:  anomaly <= 1.0 sigma
      - unusual: 1.0 < anomaly <= 1.5 sigma, OR anomaly > 1.5 sigma but not
                 part of a qualifying >=3-day episode (isolated spike)
      - extreme: anomaly > 1.5 sigma AND part of a qualifying episode
    """
    split_windows = windows.loc[windows["split"] == split].reset_index(drop=True)
    anomaly_block = daily["t_max_anomaly"]
    std_block = daily["clim_std_t_max"]
    episode_block = daily["heatwave_episode_id"]

    n = len(split_windows)
    stratum = np.empty((n, FORECAST_DAYS), dtype="<U7")
    for i, row in enumerate(split_windows.itertuples(index=False)):
        anomaly = anomaly_block.loc[row.forecast_start : row.forecast_end].to_numpy(dtype=np.float64)
        std = std_block.loc[row.forecast_start : row.forecast_end].to_numpy(dtype=np.float64)
        episode = episode_block.loc[row.forecast_start : row.forecast_end].to_numpy()
        sigma = anomaly / std
        in_episode = ~pd.isnull(episode)

        labels = np.full(FORECAST_DAYS, "normal", dtype="<U7")
        unusual_mask = (sigma > 1.0) & (sigma <= 1.5)
        isolated_spike_mask = (sigma > 1.5) & (~in_episode)
        extreme_mask = (sigma > 1.5) & in_episode
        labels[unusual_mask | isolated_spike_mask] = "unusual"
        labels[extreme_mask] = "extreme"
        stratum[i] = labels
    return stratum

def load_all_splits(refit_stats: bool = True) -> dict:
    """Convenience entry point: loads daily+windows, fits/loads normalization
    stats, and returns raw + normalized arrays for all three splits."""
    daily = _load_daily()
    windows = _load_windows()

    if refit_stats:
        stats = fit_normalization_stats(daily, windows)
        save_normalization_stats(stats)
    else:
        stats = load_normalization_stats()

    out = {"stats": stats}
    for split in ("train", "val", "test"):
        X, y, query_dates = build_split_arrays(split, daily, windows)
        out[split] = {
            "X_raw": X,
            "y_raw": y,
            "X_norm": normalize_X(X, stats),
            "y_norm": normalize_y(y, stats),
            "query_dates": query_dates,
        }
    return out


if __name__ == "__main__":
    data = load_all_splits(refit_stats=True)
    for split in ("train", "val", "test"):
        d = data[split]
        print(f"{split}: X_raw={d['X_raw'].shape}, y_raw={d['y_raw'].shape}, "
              f"X_norm mean/std sanity: {d['X_norm'].mean():.4f}/{d['X_norm'].std():.4f}")
    print("Normalization stats saved to evaluation/normalization_stats.json")
