"""
Build the humid-air daily dataset (screen S2, pre-registered 2026-10-18) from the verified raw download of
pipeline/download_era5_humid.py, exactly as pipeline/build_upstream_daily.py builds the 27-point table
(same loading, cleaning and wind components; its functions are reused).

Input:  data/raw/era5_humid/<node>/<year>.json   (12 points x 1980-01-01..2026-09-06)
Output: datasets_v2/humid_daily.parquet   one row per date, one column per <variable>__<node>, float32
        datasets_v2/humid_daily.meta.json  fingerprint of the raw files, nodes, variables, cleaning
The full period is stored; every consumer must drop 2019+ before use (test lock).

Run from repo root:  python -m pipeline.build_humid_daily
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.build_upstream_daily import clean_node, load_node, raw_fingerprint
from pipeline.download_era5_humid import HUMID_DIR, HUMID_NODES
from pipeline.download_era5_upstream import DAILY_VARIABLES, START, node_id
from pipeline.download_era5_v2 import END_DATE

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "datasets_v2" / "humid_daily.parquet"
META = REPO_ROOT / "datasets_v2" / "humid_daily.meta.json"


def build(root: Path = HUMID_DIR) -> tuple[pd.DataFrame, dict]:
    dates = pd.date_range(pd.Timestamp(START), pd.Timestamp(END_DATE), name="date")
    cols, n_clipped = {}, 0
    for lat, lon in HUMID_NODES:
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
    meta = {"source": "data/raw/era5_humid (Open-Meteo ERA5, daily, Asia/Kolkata)", "raw_sha256": raw_fingerprint(root),
            "nodes": [node_id(a, o) for a, o in HUMID_NODES], "variables": DAILY_VARIABLES + ["wind_u_10m", "wind_v_10m"],
            "first_date": str(dates[0].date()), "last_date": str(dates[-1].date()), "n_days": len(dates),
            "cleaning": {"soil_moisture_negative_values_clipped_to_0": n_clipped,
                         "wind_components": "as pipeline/build_upstream_daily.py"},
            "test_lock": "contains 2019+; consumers must drop dates >= 2019-01-01 before use"}
    return table, meta


def main() -> None:
    table, meta = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(OUT)
    META.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"{OUT.relative_to(REPO_ROOT)}: {table.shape[0]} days x {table.shape[1]} columns")


if __name__ == "__main__":
    main()
