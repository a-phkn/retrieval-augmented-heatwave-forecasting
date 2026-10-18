"""
Humid-air points for the heatwave-information screen S2 (pre-registered 2026-10-18, context/decisions.md).

Why: WBGT heatwaves are humid heat. The 27 upstream points (pipeline/download_era5_upstream.py) all lie to
the north-west (dry advection); moisture reaches Delhi from the Gangetic plain / Bay of Bengal (east) and
the Arabian Sea (south-west). These 12 points cover those directions, on the same 2-degree ERA5 lattice.
All are below 600 m (Open-Meteo elevation API, probe 2026-10-18).

Data, request format, validation, atomic writes, resumability and rate-limit handling are exactly those of
pipeline/download_era5_upstream.py (its functions are reused): daily ERA5, Asia/Kolkata days,
1980-01-01 .. 2026-09-06, the same 10 variables. 2019+ is downloaded (as for the 27 points) but stays locked
at read time (training/folds.py).

Layout: data/raw/era5_humid/<node>/<year>.json (gitignored).
Quota: 12 points x ~17,050 days / 14 ~ 14,600 counted calls, ~1.5 days at the free 10,000/day limit.
Stop and re-run any time.

Run from repo root:
    python -m pipeline.download_era5_humid --dry-run
    python -m pipeline.download_era5_humid            # resumable; Ctrl+C is safe
    python -m pipeline.download_era5_humid --verify
"""
from __future__ import annotations

import argparse
import time
from datetime import date
from pathlib import Path

import requests

from pipeline.download_era5_upstream import (
    DAILY_VARIABLES, START, download_one, estimated_calls, is_complete, node_id, out_path, years,
)
from pipeline.download_era5_v2 import END_DATE, FREE_DAILY_LIMIT, RateLimitExhausted

REPO_ROOT = Path(__file__).resolve().parents[1]
HUMID_DIR = REPO_ROOT / "data" / "raw" / "era5_humid"
EAST = [(28, 80), (26, 80), (26, 82), (26, 84), (26, 86), (24, 84), (24, 88)]
SOUTH_WEST = [(22, 70), (22, 72), (22, 74), (22, 76), (20, 72)]
HUMID_NODES: list[tuple[int, int]] = EAST + SOUTH_WEST


def plan(nodes: list[tuple[int, int]] = HUMID_NODES) -> list[tuple[tuple[int, int], date, date]]:
    return [(n, f, l) for n in nodes for f, l in years()]


def todo(root: Path = HUMID_DIR) -> list[tuple[tuple[int, int], date, date]]:
    return [(n, f, l) for n, f, l in plan() if not is_complete(out_path(n, f, root), n, f, l)]


def run(sleep_s: float, root: Path = HUMID_DIR) -> int:
    pending = todo(root)
    print(f"{len(pending)} node-year files to download into {root}", flush=True)
    failed = []
    with requests.Session() as session:
        for i, (n, f, l) in enumerate(pending, 1):
            try:
                ok = download_one(session, n, f, l, root)
            except RateLimitExhausted as err:
                print(f"\n{err}.\nThe daily API quota is probably used up: stopping now. Run again tomorrow to resume.",
                      flush=True)
                return 2
            if not ok:
                failed.append((n, f))
            if i % 25 == 0 or i == len(pending):
                print(f"  {i}/{len(pending)} processed (last: {node_id(*n)} {f.year}), {len(failed)} failed", flush=True)
            time.sleep(sleep_s)
    if failed:
        print(f"\n{len(failed)} file(s) failed; run again later to retry: "
              + ", ".join(f"{node_id(*n)} {f.year}" for n, f in failed[:20]))
        return 1
    print("All requested files present.")
    return 0


def verify(root: Path = HUMID_DIR) -> int:
    bad = todo(root)
    total = len(plan())
    print(f"{total - len(bad)}/{total} node-year files complete in {root}")
    for n, f, _ in bad[:10]:
        print(f"  missing/incomplete: {node_id(*n)} {f.year}")
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Download daily ERA5 at the 12 humid-air points (S2).")
    ap.add_argument("--sleep", type=float, default=2.0, help="seconds between requests")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args(argv)
    if args.verify:
        return verify()
    if args.dry_run:
        pending = todo()
        calls = estimated_calls(pending)
        print(f"{len(HUMID_NODES)} nodes, years {START.year}..{END_DATE.year}; requests still needed: {len(pending)} "
              f"-> ~{calls:,.0f} counted API calls (~{calls / FREE_DAILY_LIMIT:.1f} days at the free limit)")
        print(f"variables: {', '.join(DAILY_VARIABLES)}")
        return 0
    return run(args.sleep)


if __name__ == "__main__":
    raise SystemExit(main())
