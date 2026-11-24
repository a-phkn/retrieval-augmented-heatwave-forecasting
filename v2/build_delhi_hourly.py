"""
Hourly physical (Liljegren) WBGT for Delhi's 9 cells, kept hourly (the v1 builder
pipeline/build_wbgt_liljegren.py keeps only daily summaries). Uses that builder's cell_hourly_wbgt unchanged
(same physics and conventions; frozen code imported, not edited).

Output (gitignored, data/raw/era5_v2/_wbgt/):
    delhi_hourly_dev.parquet    1980-2018, one row per (cell, hour)
    delhi_hourly_test.parquet   2019+, read ONLY through v2/test_lock_v2.require()
Only complete days (24 hours) are kept.

Run from repo root:  python -m v2.build_delhi_hourly
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipeline.build_wbgt_liljegren import cell_hourly_wbgt

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "data" / "raw" / "era5_v2" / "_wbgt"
LAST_DEV_YEAR = 2018


def build() -> pd.DataFrame:
    parts = []
    for cell in range(1, 10):
        h = cell_hourly_wbgt(cell)
        date = h["time"].dt.normalize()
        h = h[date.map(date.value_counts()) == 24].assign(cell=cell)
        print(f"  cell {cell}: {h['time'].dt.normalize().nunique()} complete days", flush=True)
        parts.append(h)
    return pd.concat(parts, ignore_index=True).sort_values(["time", "cell"]).reset_index(drop=True)


def main() -> int:
    h = build()
    OUT.mkdir(parents=True, exist_ok=True)
    dev, test = h[h["time"].dt.year <= LAST_DEV_YEAR], h[h["time"].dt.year > LAST_DEV_YEAR]
    dev.to_parquet(OUT / "delhi_hourly_dev.parquet", index=False)
    test.to_parquet(OUT / "delhi_hourly_test.parquet", index=False)
    print(f"Wrote delhi: dev {dev.shape}, test {test.shape} (test file is read only via the v2 lock)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
