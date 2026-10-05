"""
Builds the physical (Liljegren) WBGT dataset for the 9 Delhi cells (plan v5, Week 3).

Inputs (both gitignored raw folders):
    data/raw/era5_monthly/  v1: hourly temperature, relative humidity, 10 m wind, pressure
    data/raw/era5_v2/       v2: hourly shortwave (global horizontal), diffuse radiation
Radiation is the mean of the PRECEDING hour (Open-Meteo convention), so each hour is paired
with the mean sunlit cosine zenith of the hour that ends at its timestamp. Direct horizontal
= global - diffuse. Hourly WBGT via pipeline/wbgt_liljegren.py (Liljegren 2008, adapted to
reanalysis following Kong & Huber 2022 -- see that module for credits and licence).

Outputs (a separate file, so the training table datasets_v2/all_daily_v2.parquet is
unchanged):
    datasets_v2/wbgt_liljegren_daily.parquet   one row per day: 9-cell means of
        wbgt_lj_max (daily max of hourly WBGT), wbgt_lj_max_hour (hour of the max, domain
        median), wbgt_lj_mean_12_18 (mean 12:00-18:00 IST), tg_max, tnwb_max
    datasets_v2/wbgt_liljegren_per_cell.parquet the same per (cell, day)
Only complete days (24 hours) are kept, as in pipeline/hourly_features.py.

Run from repo root:  python -m pipeline.build_wbgt_liljegren
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.hourly_features import RAW_DIR, load_cell_hourly
from pipeline.wbgt_liljegren import mean_sunlit_cosz, wbgt

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_V2_DIR = REPO_ROOT / "data" / "raw" / "era5_v2"
OUT_DIR = REPO_ROOT / "datasets_v2"
IST_OFFSET = pd.Timedelta(hours=5, minutes=30)
END_DATE = pd.Timestamp("2026-09-06")


def load_cell_radiation(cell: int, raw_dir: Path = RAW_V2_DIR) -> tuple[pd.DataFrame, tuple[float, float]]:
    """Hourly global and diffuse radiation for one cell (local IST time), plus the ERA5 grid
    point (lat, lon) the API snapped to."""
    files = sorted((raw_dir / f"cell_{cell}").glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no v2 raw files for cell {cell} in {raw_dir}")
    frames, coords = [], set()
    for path in files:
        p = json.loads(path.read_text(encoding="utf-8"))
        coords.add((p["latitude"], p["longitude"]))
        h = p["hourly"]
        frames.append(pd.DataFrame({"time": h["time"], "ghi": h["shortwave_radiation"], "diffuse": h["diffuse_radiation"]}))
    if len(coords) != 1:
        raise ValueError(f"cell {cell}: files disagree on location {coords}")
    df = pd.concat(frames, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"])
    return df.drop_duplicates("time").sort_values("time").reset_index(drop=True), coords.pop()


def cell_hourly_wbgt(cell: int, raw_dir: Path = RAW_DIR, raw_v2_dir: Path = RAW_V2_DIR) -> pd.DataFrame:
    """Hourly Liljegren WBGT and components for one cell."""
    met = load_cell_hourly(cell, raw_dir)
    rad, (lat, lon) = load_cell_radiation(cell, raw_v2_dir)
    df = met.merge(rad, on="time", how="inner").dropna().reset_index(drop=True)
    utc = (df["time"] - IST_OFFSET).to_numpy(dtype="datetime64[ns]")
    cosz = np.concatenate([mean_sunlit_cosz(chunk, lat, lon) for chunk in np.array_split(utc, max(1, len(utc) // 50_000))])
    direct = np.clip(df["ghi"].to_numpy() - df["diffuse"].to_numpy(), 0.0, None)
    out = wbgt(df["t"], df["rh"], df["pressure"], df["wind"], df["ghi"], direct, cosz, urban=True)
    for k in ("wbgt", "tg", "tnwb"):
        df[k] = out[k]
    df["cosz"] = cosz
    if df[["wbgt", "tg", "tnwb"]].isna().any().any():
        raise ValueError(f"cell {cell}: solver failed for {int(df['wbgt'].isna().sum())} hours")
    return df


def daily_from_hourly(h: pd.DataFrame) -> pd.DataFrame:
    """Daily features from hourly WBGT; incomplete days dropped."""
    h = h.assign(date=h["time"].dt.normalize(), hour=h["time"].dt.hour)
    g = h.groupby("date")
    idx = g["wbgt"].idxmax()
    out = pd.DataFrame({
        "wbgt_lj_max": g["wbgt"].max(),
        "wbgt_lj_max_hour": h.loc[idx, "hour"].to_numpy(),
        "wbgt_lj_mean_12_18": h[h["hour"].between(12, 18)].groupby("date")["wbgt"].mean(),
        "tg_max": g["tg"].max(),
        "tnwb_max": g["tnwb"].max(),
    })
    return out[g.size() == 24]


def build(cells=range(1, 10)) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_cell = []
    for cell in cells:
        d = daily_from_hourly(cell_hourly_wbgt(cell))
        per_cell.append(d.assign(cell=cell).reset_index())
        print(f"  cell {cell}: {len(d)} days, mean daily max WBGT {d['wbgt_lj_max'].mean():.2f} C", flush=True)
    cells_df = pd.concat(per_cell, ignore_index=True)
    cells_df = cells_df[cells_df["date"] <= END_DATE]
    counts = cells_df.groupby("date")["cell"].nunique()
    full = counts[counts == len(list(cells))].index
    domain = (cells_df[cells_df["date"].isin(full)].groupby("date")
              .agg(wbgt_lj_max=("wbgt_lj_max", "mean"), wbgt_lj_max_hour=("wbgt_lj_max_hour", "median"),
                   wbgt_lj_mean_12_18=("wbgt_lj_mean_12_18", "mean"), tg_max=("tg_max", "mean"),
                   tnwb_max=("tnwb_max", "mean")))
    return domain.reset_index(), cells_df.sort_values(["date", "cell"]).reset_index(drop=True)


def main() -> None:
    domain, cells = build()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    domain.to_parquet(OUT_DIR / "wbgt_liljegren_daily.parquet", index=False)
    cells.to_parquet(OUT_DIR / "wbgt_liljegren_per_cell.parquet", index=False)
    print(f"Wrote wbgt_liljegren_daily.parquet {domain.shape} and wbgt_liljegren_per_cell.parquet {cells.shape}")


if __name__ == "__main__":
    main()
