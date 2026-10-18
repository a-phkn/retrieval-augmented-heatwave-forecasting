"""
Large-scale weather pattern for retrieval rung Rz (pre-registered 2026-10-17, context/decisions.md):
ERA5 500 hPa geopotential height over South Asia, one value per day at 12 UTC, 1980-2018.

Why: both backbones see only surface weather; Delhi heatwaves are driven by upper-level
anticyclones (Ratnam et al. 2016, Scientific Reports 6: 24395). Rz matches past days on this pattern.

Source: WeatherBench 2 (Rasp et al. 2024), public bucket, read anonymously over plain HTTPS (the
gcsfs client hung on this machine; see evaluation/graphcast_yardstick.py):
  datasets/era5/1959-2022-6h-128x64_equiangular_conservative.zarr  (2.8125 deg, 6-hourly, 13 levels)
Each zarr chunk holds 40 time steps x 13 levels x the whole globe (~11.6 MB compressed), so ~1,425
chunks (~16.5 GB) are streamed through memory and only the 12 x 21-cell box at 500 hPa, 12 UTC is
kept (~15 MB). Nothing else is written to disk.

Box: cell centres in 10-45 N x 40-100 E. Values: geopotential / g0 (m). 2019+ is never read here
(the locked test period); it would be fetched only if Rz passes its rules.

Layout: data/external/era5_z500_southasia_1980_2018.parquet (index date, one column per cell) and a
.meta.json beside it. Resumable: finished blocks of chunks are kept in data/external/z500_parts/
(gitignored) until the final file is written; re-running skips them. Ctrl+C is safe.

Test years (the Rz side-check, docs/CONFIRMATORY_PROTOCOL.md): --test-years streams 2019-01-01 .. 2021-12-31
(this file's last year is 2021) into OUT_TEST. It refuses unless the test lock is open (training.test_lock).

Run from repo root:
    python -m pipeline.download_era5_z500 --dry-run
    python -m pipeline.download_era5_z500
    python -m pipeline.download_era5_z500 --test-years      # only after the protocol freeze
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from training.folds import TEST_START

REPO_ROOT = Path(__file__).resolve().parents[1]
URL = "https://storage.googleapis.com/weatherbench2/datasets/era5/1959-2022-6h-128x64_equiangular_conservative.zarr"
OUT = REPO_ROOT / "data" / "external" / "era5_z500_southasia_1980_2018.parquet"
OUT_TEST = REPO_ROOT / "data" / "external" / "era5_z500_southasia_2019_2021.parquet"
PARTS = REPO_ROOT / "data" / "external" / "z500_parts"
BOX = (10.0, 45.0, 40.0, 100.0)  # lat_min, lat_max, lon_min, lon_max (cell centres)
LEVEL_HPA = 500
HOUR_UTC = 12
START, END = pd.Timestamp("1980-01-01"), pd.Timestamp("2018-12-31")
START_TEST, END_TEST = pd.Timestamp("2019-01-01"), pd.Timestamp("2021-12-31")
ORIGIN = pd.Timestamp("1959-01-01")  # the file's "hours since 1959-01-01"
STEP_HOURS = 6
G0 = 9.80665
CHUNKS_PER_PART = 50
WORKERS = 6


def time_index(dates: pd.DatetimeIndex) -> np.ndarray:
    """Position on the file's 6-hourly time axis of HOUR_UTC on each date."""
    hours = (pd.DatetimeIndex(dates) - ORIGIN) / pd.Timedelta(hours=1) + HOUR_UTC
    if not np.allclose(hours % STEP_HOURS, 0):
        raise ValueError("requested hour is not on the 6-hourly axis")
    return (hours // STEP_HOURS).astype(np.int64).to_numpy()


def chunk_plan(t_idx: np.ndarray, t_chunk: int) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """{chunk number: (positions inside the chunk, rows of the output)}."""
    ci, pos = t_idx // t_chunk, t_idx % t_chunk
    return {int(c): (pos[ci == c], np.flatnonzero(ci == c)) for c in np.unique(ci)}


def box_indices(lat: np.ndarray, lon: np.ndarray, box=BOX) -> tuple[np.ndarray, np.ndarray]:
    """Indices of the cell centres inside the box (ascending latitude and longitude)."""
    return (np.flatnonzero((lat >= box[0]) & (lat <= box[1])), np.flatnonzero((lon >= box[2]) & (lon <= box[3])))


class Store:
    """Anonymous HTTPS reads of one zarr v2 store (consolidated metadata, blosc chunks)."""

    def __init__(self, url: str = URL):
        import requests

        self.url, self.sess = url, requests.Session()
        self.meta = json.loads(self.get(".zmetadata"))["metadata"]

    def get(self, path: str) -> bytes:
        import requests

        for attempt in range(6):
            try:
                r = self.sess.get(f"{self.url}/{path}", timeout=120)
                r.raise_for_status()
                return r.content
            except requests.RequestException:
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)
        raise AssertionError

    def coord(self, name: str) -> np.ndarray:
        import numcodecs

        z = self.meta[f"{name}/.zarray"]
        n_chunks = -(-z["shape"][0] // z["chunks"][0])
        parts = [np.frombuffer(numcodecs.Blosc().decode(self.get(f"{name}/{c}")), dtype=z["dtype"]) for c in range(n_chunks)]
        return np.concatenate(parts)[: z["shape"][0]]


def check_layout(store: Store) -> dict:
    za, attrs = store.meta["geopotential/.zarray"], store.meta["geopotential/.zattrs"]
    if attrs["_ARRAY_DIMENSIONS"] != ["time", "level", "longitude", "latitude"] or attrs.get("units") != "m**2 s**-2":
        raise ValueError(f"unexpected geopotential layout {attrs}")
    if store.meta["time/.zattrs"]["units"] != "hours since 1959-01-01":
        raise ValueError("unexpected time units")
    t = store.coord("time")
    if not (np.all(np.diff(t) == STEP_HOURS) and t[0] == 0):
        raise ValueError("time axis is not a gap-free 6-hourly series from the origin")
    lat, lon, lev = store.coord("latitude"), store.coord("longitude"), store.coord("level")
    if not (np.all(np.diff(lat) > 0) and np.all(np.diff(lon) > 0)):
        raise ValueError("expected ascending latitude and longitude")
    ilat, ilon = box_indices(lat, lon)
    return {"za": za, "n_time": len(t), "lat": lat, "lon": lon, "ilat": ilat, "ilon": ilon,
            "ilev": int(np.flatnonzero(lev == LEVEL_HPA)[0])}


def fetch_chunk(store: Store, lay: dict, ci: int, positions: np.ndarray) -> np.ndarray:
    """(n, n_lat, n_lon) geopotential height (m) at 500 hPa in the box for the given chunk positions."""
    import numcodecs

    za = lay["za"]
    a = np.frombuffer(numcodecs.Blosc().decode(store.get(f"geopotential/{ci}.0.0.0")), dtype=za["dtype"])
    a = a.reshape(za["chunks"])  # zarr v2 stores edge chunks at full size
    sub = a[positions, lay["ilev"]][:, lay["ilon"]][:, :, lay["ilat"]]  # (n, lon, lat)
    return np.transpose(sub, (0, 2, 1)).astype(np.float64) / G0


def columns(lat: np.ndarray, lon: np.ndarray) -> list[str]:
    return [f"z500_{a:.4f}N_{o:.4f}E" for a in lat for o in lon]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Stream ERA5 z500 over South Asia (Rz).")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--test-years", action="store_true", help="2019-2021 for the Rz side-check (needs the test lock)")
    args = ap.parse_args(argv)
    if args.test_years:
        from training import test_lock

        test_lock.require()
        start, end, out_path, parts = START_TEST, END_TEST, OUT_TEST, PARTS.with_name("z500_parts_test")
    else:
        if END >= TEST_START:
            raise ValueError("the locked test period must not be read")
        start, end, out_path, parts = START, END, OUT, PARTS
    dates = pd.date_range(start, end, freq="D")
    store = Store()
    lay = check_layout(store)
    t_idx = time_index(dates)
    if t_idx.max() >= lay["n_time"]:
        raise ValueError("requested dates beyond the file")
    plan = chunk_plan(t_idx, lay["za"]["chunks"][0])
    keys = sorted(plan)
    blocks = [keys[i:i + CHUNKS_PER_PART] for i in range(0, len(keys), CHUNKS_PER_PART)]
    lat, lon = lay["lat"][lay["ilat"]], lay["lon"][lay["ilon"]]
    print(f"{len(dates)} days, {len(keys)} chunks in {len(blocks)} blocks, box {len(lat)} x {len(lon)} cells "
          f"(lat {lat[0]:.2f}-{lat[-1]:.2f}, lon {lon[0]:.2f}-{lon[-1]:.2f}), level index {lay['ilev']}", flush=True)
    if args.dry_run:
        return
    parts.mkdir(parents=True, exist_ok=True)
    cols = columns(lat, lon)
    for b, block in enumerate(blocks, 1):
        part = parts / f"part_{block[0]:05d}_{block[-1]:05d}.parquet"
        if part.exists():
            continue
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            arrays = list(pool.map(lambda c: fetch_chunk(store, lay, c, plan[c][0]), block))
        rows = np.concatenate([plan[c][1] for c in block])
        vals = np.concatenate(arrays).reshape(len(rows), -1)
        df = pd.DataFrame(vals.astype(np.float32), index=pd.DatetimeIndex(dates[rows], name="date"), columns=cols)
        tmp = part.with_suffix(".tmp")
        df.to_parquet(tmp)
        tmp.replace(part)  # atomic: a part file exists only when complete
        print(f"  block {b}/{len(blocks)} ({dates[rows].min().date()} .. {dates[rows].max().date()}): "
              f"{time.time() - t0:.0f}s", flush=True)

    out = pd.concat([pd.read_parquet(p) for p in sorted(parts.glob("part_*.parquet"))]).sort_index()
    if not out.index.equals(dates.rename("date")) or out.isna().any().any():
        raise ValueError("extract is incomplete or has missing values")
    if not ((out.to_numpy() > 5000) & (out.to_numpy() < 6200)).all():
        raise ValueError("z500 heights outside a plausible range (5000-6200 m)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path)
    meta = {"source": URL, "variable": "geopotential / 9.80665 (m)", "level_hpa": LEVEL_HPA, "hour_utc": HOUR_UTC,
            "start": str(start.date()), "end": str(end.date()), "box_cell_centres": BOX,
            "lat": lat.tolist(), "lon": lon.tolist(), "n_days": len(out),
            "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    for p in parts.glob("part_*.parquet"):
        p.unlink()
    parts.rmdir()
    print(f"wrote {out_path.relative_to(REPO_ROOT)} ({out_path.stat().st_size / 1e6:.1f} MB, {len(out)} days x "
          f"{out.shape[1]} cells)")


if __name__ == "__main__":
    main()
