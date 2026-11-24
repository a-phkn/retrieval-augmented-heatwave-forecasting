"""
Hourly physical (Liljegren) WBGT for the v2 extra cities (PROTOCOL_V2 Section 7; H-E).

Input: data/raw/cities/<city>/cell_N/YYYY-MM.json from v2/download_cities.py (Open-Meteo ERA5, local time
Asia/Kolkata; temperature, RH, 10 m wind and pressure instantaneous at the stamp, radiation the mean of the
PRECEDING hour). Same physics and conventions as Delhi (pipeline/build_wbgt_liljegren.py): each stamp is
paired with the mean sunlit cos(zenith) of the hour ending at it, direct horizontal = global - diffuse,
urban wind profile, numpy reference pipeline/wbgt_liljegren.wbgt.

Output (gitignored, under data/raw/cities/_wbgt/):
    <city>_hourly_dev.parquet    1980-2018, one row per (cell, hour): t, rh, wind, pressure, ghi, diffuse,
                                 cosz, wbgt, tg, tnwb, plus the grid point Open-Meteo returned
    <city>_hourly_test.parquet   2019-2025, same columns. Written here but read ONLY through
                                 v2/test_lock_v2.require() (PROTOCOL_V2 Section 5).
Only complete days (24 hours) are kept.

Run from repo root:  python -m v2.build_city_wbgt --city ahmedabad
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.wbgt_liljegren import mean_sunlit_cosz, wbgt

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW = REPO_ROOT / "data" / "raw" / "cities"
OUT = RAW / "_wbgt"
IST_OFFSET = pd.Timedelta(hours=5, minutes=30)
LAST_DEV_YEAR = 2018
COLS = {"temperature_2m": "t", "relative_humidity_2m": "rh", "wind_speed_10m": "wind", "surface_pressure": "pressure",
        "shortwave_radiation": "ghi", "diffuse_radiation": "diffuse"}


def load_cell(city: str, cell: int, raw: Path = RAW) -> tuple[pd.DataFrame, tuple[float, float]]:
    files = sorted((raw / city / f"cell_{cell}").glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no files for {city} cell {cell} in {raw}")
    frames, coords = [], set()
    for path in files:
        p = json.loads(path.read_text(encoding="utf-8"))
        if p.get("timezone") != "Asia/Kolkata":
            raise ValueError(f"{path}: timezone {p.get('timezone')}, expected Asia/Kolkata")
        coords.add((p["latitude"], p["longitude"]))
        frames.append(pd.DataFrame({"time": p["hourly"]["time"], **{v: p["hourly"][k] for k, v in COLS.items()}}))
    if len(coords) != 1:
        raise ValueError(f"{city} cell {cell}: files disagree on location {coords}")
    df = pd.concat(frames, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"])
    return df.drop_duplicates("time").sort_values("time").reset_index(drop=True), coords.pop()


def cell_hourly_wbgt(city: str, cell: int, raw: Path = RAW) -> pd.DataFrame:
    df, (lat, lon) = load_cell(city, cell, raw)
    df = df.dropna().reset_index(drop=True)
    utc = (df["time"] - IST_OFFSET).to_numpy(dtype="datetime64[ns]")
    cosz = np.concatenate([mean_sunlit_cosz(c, lat, lon) for c in np.array_split(utc, max(1, len(utc) // 50_000))])
    direct = np.clip(df["ghi"].to_numpy() - df["diffuse"].to_numpy(), 0.0, None)
    out = wbgt(df["t"], df["rh"], df["pressure"], df["wind"], df["ghi"], direct, cosz, urban=True)
    for k in ("wbgt", "tg", "tnwb"):
        df[k] = out[k]
    df["cosz"] = cosz
    if df[["wbgt", "tg", "tnwb"]].isna().any().any():
        raise ValueError(f"{city} cell {cell}: solver failed for {int(df['wbgt'].isna().sum())} hours")
    date = df["time"].dt.normalize()
    df = df[date.map(date.value_counts()) == 24]
    return df.assign(cell=cell, grid_lat=lat, grid_lon=lon)


def build(city: str, cells=range(1, 10), raw: Path = RAW) -> pd.DataFrame:
    parts = []
    for cell in cells:
        h = cell_hourly_wbgt(city, cell, raw)
        print(f"  {city} cell {cell}: {h['time'].dt.normalize().nunique()} days, "
              f"grid ({h['grid_lat'].iat[0]}, {h['grid_lon'].iat[0]})", flush=True)
        parts.append(h)
    return pd.concat(parts, ignore_index=True).sort_values(["time", "cell"]).reset_index(drop=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Hourly Liljegren WBGT for the v2 extra cities.")
    ap.add_argument("--city", choices=("ahmedabad", "chennai"), required=True)
    a = ap.parse_args(argv)
    h = build(a.city)
    OUT.mkdir(parents=True, exist_ok=True)
    dev = h[h["time"].dt.year <= LAST_DEV_YEAR]
    test = h[h["time"].dt.year > LAST_DEV_YEAR]
    dev.to_parquet(OUT / f"{a.city}_hourly_dev.parquet", index=False)
    test.to_parquet(OUT / f"{a.city}_hourly_test.parquet", index=False)
    print(f"Wrote {a.city}: dev {dev.shape}, test {test.shape} (test file is read only via the v2 lock)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
