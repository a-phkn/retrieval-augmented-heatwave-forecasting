"""
Build the upstream daily dataset from the verified raw download (plan v5, Week 3; input to
gate G-D0 and the regional DSTGNN).

Input:  data/raw/era5_upstream/<node>/<year>.json   (pipeline/download_era5_upstream.py;
        27 points x 1980-01-01..2026-09-06, Delhi local time, verified complete)
Output: datasets_v2/upstream_daily.parquet   one row per date, one column per
        <variable>__<node>, float32 (e.g. temperature_2m_max__n28e074)
        datasets_v2/upstream_daily.meta.json  fingerprint of the raw files, node list,
        variables, date range and the cleaning applied

Cleaning (documented, minimal):
  - soil_moisture_0_to_7cm_mean: ERA5 reports tiny negative values (-0.001..-0.003 m3/m3) on
    29 days at 3 very dry desert points; clipped to 0.
  - wind: speed + "dominant" direction (meteorological, the direction the wind comes FROM)
    are kept as they are AND converted to components, because 359 and 1 degrees are almost
    the same direction but far apart as numbers:
        u = -speed * sin(dir)   (towards the east, m/s)
        v = -speed * cos(dir)   (towards the north, m/s)
    Approximation: the dominant direction paired with the MEAN speed of the day.

The full period is stored (like datasets_v2/all_daily_v2.parquet); every consumer must drop
2019+ before use (test lock), as training/folds.py does.

Run from repo root:  python -m pipeline.build_upstream_daily
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.download_era5_upstream import DAILY_VARIABLES, NODES, START, UPSTREAM_DIR, node_id
from pipeline.download_era5_v2 import END_DATE

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "datasets_v2" / "upstream_daily.parquet"
META = REPO_ROOT / "datasets_v2" / "upstream_daily.meta.json"


def raw_fingerprint(root: Path = UPSTREAM_DIR) -> str:
    """SHA-256 over (relative path, file SHA-256) of every raw file, in sorted order."""
    h = hashlib.sha256()
    for f in sorted(root.rglob("*.json")):
        h.update(f.relative_to(root).as_posix().encode() + b"\0" + hashlib.sha256(f.read_bytes()).hexdigest().encode())
    return h.hexdigest()


def load_node(nid: str, root: Path = UPSTREAM_DIR) -> pd.DataFrame:
    """All years of one node, indexed by date, the 10 raw variables as float64."""
    frames = [pd.DataFrame(json.loads(f.read_text(encoding="utf-8"))["daily"]) for f in sorted((root / nid).glob("*.json"))]
    df = pd.concat(frames, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time")[DAILY_VARIABLES].apply(pd.to_numeric, errors="raise").astype(np.float64)


def clean_node(df: pd.DataFrame) -> pd.DataFrame:
    """Clip negative soil moisture to 0 and add wind components (see module docstring)."""
    out = df.copy()
    out["soil_moisture_0_to_7cm_mean"] = out["soil_moisture_0_to_7cm_mean"].clip(lower=0.0)
    rad = np.radians(out["wind_direction_10m_dominant"])
    out["wind_u_10m"] = -out["wind_speed_10m_mean"] * np.sin(rad)
    out["wind_v_10m"] = -out["wind_speed_10m_mean"] * np.cos(rad)
    return out


def build(root: Path = UPSTREAM_DIR) -> tuple[pd.DataFrame, dict]:
    dates = pd.date_range(pd.Timestamp(START), pd.Timestamp(END_DATE), name="date")
    cols, n_clipped = {}, 0
    for lat, lon in NODES:
        nid = node_id(lat, lon)
        raw = load_node(nid, root)
        if not raw.index.equals(pd.DatetimeIndex(dates, name="time")):
            raise ValueError(f"{nid}: dates are not exactly {dates[0].date()}..{dates[-1].date()}")
        if raw.isna().any().any():
            raise ValueError(f"{nid}: missing values in the raw download")
        n_clipped += int((raw["soil_moisture_0_to_7cm_mean"] < 0).sum())
        for var, series in clean_node(raw).items():
            cols[f"{var}__{nid}"] = series.to_numpy(dtype=np.float32)
    table = pd.DataFrame(cols, index=dates)
    meta = {
        "source": "data/raw/era5_upstream (Open-Meteo ERA5, daily, Asia/Kolkata)",
        "raw_sha256": raw_fingerprint(root),
        "nodes": [node_id(a, o) for a, o in NODES],
        "variables": DAILY_VARIABLES + ["wind_u_10m", "wind_v_10m"],
        "first_date": str(dates[0].date()), "last_date": str(dates[-1].date()), "n_days": len(dates),
        "cleaning": {"soil_moisture_negative_values_clipped_to_0": n_clipped,
                     "wind_components": "u = -speed*sin(dir_from), v = -speed*cos(dir_from); dominant direction with mean speed"},
        "test_lock": "contains 2019+; consumers must drop dates >= 2019-01-01 before use",
    }
    return table, meta


def main() -> None:
    table, meta = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(OUT)
    META.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"{OUT.relative_to(REPO_ROOT)}: {table.shape[0]} days x {table.shape[1]} columns; "
          f"{meta['cleaning']['soil_moisture_negative_values_clipped_to_0']} soil-moisture values clipped")


if __name__ == "__main__":
    main()
