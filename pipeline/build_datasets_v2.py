"""
Builds datasets_v2/ (plan v5, Week 2) from the frozen v1 daily table plus hourly-derived
heat-stress features.

Outputs:
  datasets_v2/all_daily_v2.parquet   one row per day (17,051; 1980-01-01 .. 2026-09-06):
      v1 weather columns (t_max, t_min, t_mean, relative_humidity_mean, wind_speed_mean,
      surface_pressure_mean, shortwave_radiation_sum, doy_sin, doy_cos, years_since_1980)
      + 9-cell domain means of the hourly features (pipeline/hourly_features.py):
      wbgt_bom_max, wbgt_bom_mean_12_18, tw_max, hi_max, rh_at_tmax, e_at_tmax
      + split (the primary v1 split, for reference only).
  datasets_v2/per_cell_daily.parquet one row per (cell, day) with the hourly features, for
      the spatial models later.

Deliberately NOT stored: climatology, anomalies and heatwave labels. They depend on which
years are training years, so they are computed per rolling-origin fold at training time
(training/folds.py); storing one version would invite leakage across folds.

Windows are not rebuilt: every model reads the frozen splits/window_index_v1.parquet.
This script only reads data/ and datasets/ and writes datasets_v2/.

Run from repo root:  python -m pipeline.build_datasets_v2
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.hourly_features import domain_daily

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "datasets_v2"
V1_WEATHER_COLUMNS = [
    "t_max", "t_min", "t_mean", "relative_humidity_mean", "wind_speed_mean",
    "surface_pressure_mean", "shortwave_radiation_sum", "doy_sin", "doy_cos", "years_since_1980",
]
HOURLY_COLUMNS = ["wbgt_bom_max", "wbgt_bom_mean_12_18", "tw_max", "hi_max", "rh_at_tmax", "e_at_tmax"]


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    v1 = pd.read_parquet(REPO_ROOT / "datasets/all_daily.parquet").set_index("date").sort_index()
    cells, domain = domain_daily()
    if not domain.index.equals(v1.index):
        raise ValueError("hourly domain dates do not match the frozen v1 daily table")
    if not np.allclose(domain["t_max"].to_numpy(), v1["t_max"].to_numpy(), atol=1e-9):
        raise ValueError("hourly-derived t_max does not reproduce the frozen v1 t_max")
    daily = v1[V1_WEATHER_COLUMNS + ["split"]].join(domain[HOURLY_COLUMNS])
    if daily.isna().any().any():
        raise ValueError("NaN in datasets_v2 daily table")
    per_cell = cells[["date", "cell"] + ["t_max"] + HOURLY_COLUMNS].sort_values(["date", "cell"]).reset_index(drop=True)
    return daily.reset_index(), per_cell


def main() -> None:
    daily, per_cell = build()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(OUT_DIR / "all_daily_v2.parquet", index=False)
    per_cell.to_parquet(OUT_DIR / "per_cell_daily.parquet", index=False)
    print(f"Wrote datasets_v2/all_daily_v2.parquet {daily.shape} and per_cell_daily.parquet {per_cell.shape}")


if __name__ == "__main__":
    main()
