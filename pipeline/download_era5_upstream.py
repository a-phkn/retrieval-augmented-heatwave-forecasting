"""
Upstream downloader: daily ERA5 at ~2-degree points over north-west India and Pakistan
(plan v5, Week 3; Option 1, the regional "upstream" DSTGNN).

Why: Delhi heatwaves are largely hot, dry air advected from the north-west (Thar desert,
Rajasthan, Punjab, Pakistan) over 1-3 days. These points are the graph nodes that can see
it coming; check G-D0 (does upstream heat lead Delhi?) before building on them.

Points: a 2-degree lattice, lat 24-32 N x lon 68-78 E (30 candidates). Points above 1,000 m
were dropped (unrepresentative mountain temperatures), using Open-Meteo's elevation API
(probe 2026-10-05): 32N 68E (2,298 m), 30N 70E (1,721 m), 32N 78E (5,339 m). 27 remain;
the list below is fixed so the node set cannot change between runs. Lattice points sit
exactly on the ERA5 0.25-degree grid.

Data: DAILY values (aggregated by Open-Meteo from hourly ERA5 in Delhi local time, so day
boundaries match the Delhi data), 1980-01-01 .. 2026-09-06 (the v1/v2 end date). The 10
variables were probed on 2026-10-05 and all return values for 1980. Default elevation
downscaling, as in the v1 and v2 downloads; each file records the elevation used.

Layout: data/raw/era5_upstream/<node>/<year>.json (gitignored), node id e.g. n26e072.
Same safeguards as pipeline/download_era5_v2.py: validated in memory, atomic writes,
resumable, --dry-run, --verify, and a persistent rate limit stops the run cleanly.

Quota: Open-Meteo counts a request covering more than 2 weeks as several calls (about
days / 14). 27 points x ~17,050 days / 14 ~ 33,000 counted calls, ~3-4 days at the free
10,000/day limit. Stop and re-run any time. Non-commercial use only (Open-Meteo terms).

Run from repo root:
    python -m pipeline.download_era5_upstream --dry-run
    python -m pipeline.download_era5_upstream            # resumable; Ctrl+C is safe
    python -m pipeline.download_era5_upstream --verify
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

import requests

from pipeline.download_era5_v2 import END_DATE, FREE_DAILY_LIMIT, URL, RateLimitExhausted, write_atomic

REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_DIR = REPO_ROOT / "data" / "raw" / "era5_upstream"
START = date(1980, 1, 1)

# (lat, lon) of the 27 nodes kept; see the module docstring for the 3 dropped.
NODES: list[tuple[int, int]] = [
    (24, 68), (24, 70), (24, 72), (24, 74), (24, 76), (24, 78),
    (26, 68), (26, 70), (26, 72), (26, 74), (26, 76), (26, 78),
    (28, 68), (28, 70), (28, 72), (28, 74), (28, 76), (28, 78),
    (30, 68), (30, 72), (30, 74), (30, 76), (30, 78),
    (32, 70), (32, 72), (32, 74), (32, 76),
]
DAILY_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "dew_point_2m_mean",
    "relative_humidity_2m_mean",
    "wind_speed_10m_mean",
    "wind_direction_10m_dominant",
    "shortwave_radiation_sum",
    "soil_moisture_0_to_7cm_mean",
    "surface_pressure_mean",
]  # 10 variables: the most one request can carry before Open-Meteo counts extra calls
MAX_GRID_OFFSET_DEG = 0.125  # response must snap to the requested ERA5 grid point


def node_id(lat: int, lon: int) -> str:
    return f"n{lat:02d}e{lon:03d}"


def years(start: date = START, end: date = END_DATE) -> list[tuple[date, date]]:
    """(first_day, last_day) per calendar year, the last year clamped to `end`."""
    return [(date(y, 1, 1), min(date(y, 12, 31), end)) for y in range(start.year, end.year + 1)]


def out_path(node: tuple[int, int], first: date, root: Path = UPSTREAM_DIR) -> Path:
    return root / node_id(*node) / f"{first.year}.json"


def params_for(node: tuple[int, int], first: date, last: date) -> dict:
    lat, lon = node
    return {
        "latitude": lat, "longitude": lon,
        "start_date": first.isoformat(), "end_date": last.isoformat(),
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": "Asia/Kolkata", "wind_speed_unit": "ms",
        "cell_selection": "nearest", "models": "era5",
    }


def payload_complete(payload: dict, node: tuple[int, int], first: date, last: date) -> bool:
    """Every variable present with exactly one non-null value per day from `first` to
    `last`, at the requested grid point (within MAX_GRID_OFFSET_DEG)."""
    if not isinstance(payload, dict):
        return False
    daily = payload.get("daily")
    if not daily or "time" not in daily:
        return False
    lat, lon = node
    try:
        if abs(payload["latitude"] - lat) > MAX_GRID_OFFSET_DEG or abs(payload["longitude"] - lon) > MAX_GRID_OFFSET_DEG:
            return False
    except (KeyError, TypeError):
        return False
    times = daily["time"]
    n_days = (last - first).days + 1
    if len(times) != n_days or times[0] != first.isoformat() or times[-1] != last.isoformat():
        return False
    return all(var in daily and len(daily[var]) == n_days and all(v is not None for v in daily[var])
               for var in DAILY_VARIABLES)


def is_complete(path: Path, node: tuple[int, int], first: date, last: date) -> bool:
    """File exists, parses, and passes payload_complete."""
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return False
    return payload_complete(payload, node, first, last)


def download_one(session: requests.Session, node: tuple[int, int], first: date, last: date,
                 root: Path = UPSTREAM_DIR, retries: int = 4, max_rate_waits: int = 10) -> bool:
    """Fetch one node-year; validate in memory, then write atomically. Rate-limit (429)
    waits are bounded separately from retry attempts."""
    path = out_path(node, first, root)
    if is_complete(path, node, first, last):
        return True
    label = f"{node_id(*node)} {first.year}"
    attempts = rate_waits = 0
    while attempts < retries:
        try:
            r = session.get(URL, params=params_for(node, first, last), timeout=120)
            if r.status_code == 429:
                rate_waits += 1
                if rate_waits > max_rate_waits:
                    raise RateLimitExhausted(f"{label}: still rate limited after {max_rate_waits} waits")
                wait = min(60 * rate_waits, 600)
                print(f"  {label}: rate limited (429), waiting {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            payload = r.json()
            if not payload_complete(payload, node, first, last):
                raise ValueError("response incomplete (missing variables/days, wrong range or location, or nulls)")
            write_atomic(path, payload)
            return True
        except (requests.RequestException, ValueError) as err:
            attempts += 1
            print(f"  {label}: attempt {attempts}/{retries} failed ({err})")
            if attempts < retries:
                time.sleep(15 * attempts)
    return False


def plan(nodes: list[tuple[int, int]] = NODES) -> list[tuple[tuple[int, int], date, date]]:
    return [(n, f, l) for n in nodes for f, l in years()]


def todo(root: Path = UPSTREAM_DIR) -> list[tuple[tuple[int, int], date, date]]:
    return [(n, f, l) for n, f, l in plan() if not is_complete(out_path(n, f, root), n, f, l)]


def run(sleep_s: float, root: Path = UPSTREAM_DIR) -> int:
    """Download everything still missing. A year that keeps failing is skipped and
    reported at the end (re-running retries it); the rest of the run continues."""
    pending = todo(root)
    print(f"{len(pending)} node-year files to download into {root}")
    failed = []
    with requests.Session() as session:
        for i, (n, f, l) in enumerate(pending, 1):
            try:
                ok = download_one(session, n, f, l, root)
            except RateLimitExhausted as err:
                print(f"\n{err}.\nThe daily API quota is probably used up: stopping now. Run again tomorrow to resume.")
                return 2
            if not ok:
                failed.append((n, f))
            if i % 25 == 0 or i == len(pending):
                print(f"  {i}/{len(pending)} processed (last: {node_id(*n)} {f.year}), {len(failed)} failed", flush=True)
            time.sleep(sleep_s)
    if failed:
        print(f"\n{len(failed)} file(s) failed; run again later to retry:")
        for n, f in failed[:20]:
            print(f"  {node_id(*n)} {f.year}")
        return 1
    print("All requested files present.")
    return 0


def verify(root: Path = UPSTREAM_DIR) -> int:
    bad = todo(root)
    total = len(plan())
    print(f"{total - len(bad)}/{total} node-year files complete in {root}")
    for n, f, _ in bad[:10]:
        print(f"  missing/incomplete: {node_id(*n)} {f.year}")
    return 1 if bad else 0


def estimated_calls(items: list[tuple[tuple[int, int], date, date]]) -> float:
    """Open-Meteo weighting estimate: max(1, days / 14) per request (<= 10 variables)."""
    return sum(max(1.0, ((l - f).days + 1) / 14) for _, f, l in items)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Download daily ERA5 at the upstream (NW India / Pakistan) nodes.")
    ap.add_argument("--sleep", type=float, default=2.0, help="seconds between requests")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args(argv)
    if args.verify:
        return verify()
    if args.dry_run:
        pending = todo()
        calls = estimated_calls(pending)
        print(f"{len(NODES)} nodes, years {START.year}..{END_DATE.year} (last year ends {END_DATE})")
        print(f"requests still needed: {len(pending)} -> ~{calls:,.0f} counted API calls "
              f"(~{calls / FREE_DAILY_LIMIT:.1f} days at the free {FREE_DAILY_LIMIT:,}/day limit), "
              f"~{len(pending) * (args.sleep + 1.5) / 3600:.1f} h of wall time")
        print(f"variables: {', '.join(DAILY_VARIABLES)}")
        return 0
    return run(args.sleep)


if __name__ == "__main__":
    raise SystemExit(main())
