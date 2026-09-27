"""
Statistical feature representation for retrieval: each 14-day input window
is summarized into a ~17-dim hand-engineered vector capturing level, trend,
variability, and end-of-window state -- the representation FAISS similarity
search operates on. Roadmap spec (Retrieval_Augmented_Forecasting_Roadmap_
Updated.md Section on retrieval) asks for "~15-20 dim summary vector"
covering things like mean/slope/std of T_max, end-of-window anomaly,
days-above-threshold-so-far, humidity trend -- this is one reasonable,
documented instantiation of that spec, not a literal transcription (the
exact feature list wasn't re-confirmed against the original doc this
session; if the roadmap's own file specifies an exact list, reconcile
against it before treating this as final).

ASSUMPTION (flagging since this wasn't re-verified against the roadmap
file directly): `years_since_1980` is deliberately EXCLUDED from the
similarity vector, even though it's one of P1's daily features. Including
it would bias retrieval toward chronologically nearby windows purely
because they're nearby in time, conflating "similar weather" with "similar
era" -- the non-stationarity ablation (Execution_Pipeline.md Step 7:
"full history vs last-15-years") is meant to test this deliberately, by
restricting the CANDIDATE POOL by year range, not by baking recency into
the feature vector itself. Keeping them separate makes that ablation
possible; folding recency into the features would confound it.

Feature vector (17 dims), computed from the 14-day input window's daily
values in all_daily.parquet:
  0. anomaly_sigma_mean   -- mean(t_max_anomaly / clim_std_t_max) over window
  1. anomaly_sigma_end    -- same, last day of window only
  2. anomaly_sigma_slope  -- linear slope of anomaly_sigma over the 14 days
  3. anomaly_sigma_std    -- std of anomaly_sigma over window
  4. anomaly_sigma_max    -- max anomaly_sigma in window
  5. days_above_1sigma    -- count of days with anomaly_sigma > 1.0
  6. days_above_1_5sigma  -- count of days with anomaly_sigma > 1.5 (project's own hot-day rule)
  7. t_max_level_mean     -- mean raw T_max (absolute regime, not just anomaly)
  8. t_max_slope          -- linear slope of raw T_max
  9. rh_mean              -- mean relative humidity
  10. rh_slope            -- linear slope of relative humidity
  11. wind_mean           -- mean wind speed
  12. pressure_mean       -- mean surface pressure
  13. pressure_slope      -- linear slope of surface pressure (building ridge signal)
  14. radiation_mean      -- mean shortwave radiation (clear-sky signal)
  15. doy_sin_end         -- seasonal position, end of window
  16. doy_cos_end         -- seasonal position, end of window

Normalization: z-score stats fit on TRAIN-split windows only (same
methodology as training/data.py), then each vector L2-normalized so FAISS
IndexFlatIP's inner product equals cosine similarity.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = REPO_ROOT / "datasets"
RETRIEVAL_DIR = REPO_ROOT / "retrieval"

FEATURE_NAMES = [
    "anomaly_sigma_mean", "anomaly_sigma_end", "anomaly_sigma_slope",
    "anomaly_sigma_std", "anomaly_sigma_max", "days_above_1sigma",
    "days_above_1_5sigma", "t_max_level_mean", "t_max_slope",
    "rh_mean", "rh_slope", "wind_mean", "pressure_mean", "pressure_slope",
    "radiation_mean", "doy_sin_end", "doy_cos_end",
]
DAY_INDEX = np.arange(14, dtype=np.float64)


def _slope(values: np.ndarray) -> float:
    """Simple linear regression slope over a 14-point window."""
    return float(np.polyfit(DAY_INDEX, values, 1)[0])


def compute_window_features(window_df: pd.DataFrame) -> np.ndarray:
    """window_df: 14 rows (one per input day) of all_daily.parquet columns,
    already sliced to [input_start, input_end]. Returns a (17,) vector."""
    anomaly_sigma = (window_df["t_max_anomaly"] / window_df["clim_std_t_max"]).to_numpy(dtype=np.float64)
    t_max = window_df["t_max"].to_numpy(dtype=np.float64)
    rh = window_df["relative_humidity_mean"].to_numpy(dtype=np.float64)
    wind = window_df["wind_speed_mean"].to_numpy(dtype=np.float64)
    pressure = window_df["surface_pressure_mean"].to_numpy(dtype=np.float64)
    radiation = window_df["shortwave_radiation_sum"].to_numpy(dtype=np.float64)

    return np.array([
        anomaly_sigma.mean(),
        anomaly_sigma[-1],
        _slope(anomaly_sigma),
        anomaly_sigma.std(),
        anomaly_sigma.max(),
        float(np.sum(anomaly_sigma > 1.0)),
        float(np.sum(anomaly_sigma > 1.5)),
        t_max.mean(),
        _slope(t_max),
        rh.mean(),
        _slope(rh),
        wind.mean(),
        pressure.mean(),
        _slope(pressure),
        radiation.mean(),
        window_df["doy_sin"].iloc[-1],
        window_df["doy_cos"].iloc[-1],
    ], dtype=np.float64)


def build_all_feature_vectors(daily: pd.DataFrame, windows: pd.DataFrame) -> np.ndarray:
    """Computes the (N, 17) raw (un-normalized) feature matrix for EVERY
    window in forecast_windows.parquet, across all splits -- eligibility
    filtering happens at query time, not by building separate per-split
    indices, so every window needs a feature vector regardless of split."""
    cols = ["t_max_anomaly", "clim_std_t_max", "t_max", "relative_humidity_mean",
            "wind_speed_mean", "surface_pressure_mean", "shortwave_radiation_sum",
            "doy_sin", "doy_cos"]
    feature_block = daily[cols]

    n = len(windows)
    X = np.empty((n, len(FEATURE_NAMES)), dtype=np.float64)
    for i, row in enumerate(windows.itertuples(index=False)):
        window_df = feature_block.loc[row.input_start : row.input_end]
        if len(window_df) != 14:
            raise ValueError(f"query_date={row.query_date} has {len(window_df)} input days, expected 14")
        X[i] = compute_window_features(window_df)
    return X


def fit_feature_normalization(X: np.ndarray, windows: pd.DataFrame) -> dict:
    """Z-score stats fit on TRAIN-split windows only."""
    train_mask = (windows["split"] == "train").to_numpy()
    mean = X[train_mask].mean(axis=0)
    std = X[train_mask].std(axis=0)
    std[std == 0] = 1.0
    return {"feature_names": FEATURE_NAMES, "mean": mean.tolist(), "std": std.tolist()}


def apply_normalization(X: np.ndarray, stats: dict) -> np.ndarray:
    mean = np.array(stats["mean"], dtype=np.float64)
    std = np.array(stats["std"], dtype=np.float64)
    return (X - mean) / std


def l2_normalize(X: np.ndarray) -> np.ndarray:
    """So FAISS IndexFlatIP's inner product equals cosine similarity."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def save_normalization_stats(stats: dict, path: Path | None = None) -> None:
    path = path or (RETRIEVAL_DIR / "feature_normalization_stats.json")
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)


def load_normalization_stats(path: Path | None = None) -> dict:
    path = path or (RETRIEVAL_DIR / "feature_normalization_stats.json")
    with open(path) as f:
        return json.load(f)
