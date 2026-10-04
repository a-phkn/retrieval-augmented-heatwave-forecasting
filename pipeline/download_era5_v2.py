"""
Downloader v2: extra hourly ERA5 variables for the 9 Delhi cells (plan v5, Week 1).

Adds what v1 (download_era5.py) did not fetch, for Phase 1A/4 and the physics head:
  dew_point_2m, wind_direction_10m, wind_speed_10m (paired with direction, for u/v),
  shortwave_radiation, direct_normal_irradiance, diffuse_radiation, cloud_cover,
  soil_moisture_0_to_7cm.
Verified available from Open-Meteo's ERA5 archive for 1980 and 2019 (probe, 2026-10-04).

Differences from v1 (download_era5.py is kept unchanged as the frozen v1 recipe):
  * fixed END_DATE (2026-09-06, the v1 end date) instead of date.today(), so the dataset
    cannot drift on re-download;
  * atomic writes (temp file + rename): an interrupted run can never leave a truncated
    file that a resume would skip as "done";
  * separate folder data/raw/era5_v2/ (gitignored), same cell_N/YYYY-MM.json layout;
  * --dry-run prints the request count and estimated API usage before downloading;
  * --verify checks every file is present and complete through END_DATE.

Open-Meteo's free tier counts a request with <= 10 variables and > 2 weeks of data as
several calls (about 31 days / 14 days ~ 2.2 per monthly request). ~5,050 monthly requests
~ 11,000 counted calls, above the 10,000/day free limit, so a full run takes ~1-2 days;
it is resumable -- just run it again. Non-commercial use only (Open-Meteo terms).

Run from repo root:
    python -m pipeline.download_era5_v2 --dry-run
    python -m pipeline.download_era5_v2            # resumable; Ctrl+C is safe
    python -m pipeline.download_era5_v2 --verify
"""
from __future__ import annotations

import argparse
import json
import os
import time
from calendar import monthrange
from datetime import date
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
URL = "https://archive-api.open-meteo.com/v1/archive"
RAW_V2_DIR = REPO_ROOT / "data" / "raw" / "era5_v2"

# Same requested coordinates as download_era5.py (v1); each snaps to a distinct ERA5
# 0.25-degree grid point (lat 28.25/28.5/28.75 x lon 77.0/77.25/77.5).
CELLS = [
    (28.295254, 76.93878),
    (28.365553, 77.23042),
    (28.295254, 77.44898),
    (28.576448, 76.98177),
    (28.576448, 77.18678),
    (28.576448, 77.4943),
    (28.857643, 76.9222),
    (28.857643, 77.231125),
    (28.857643, 77.43707),
]
HOURLY_VARIABLES = [
    "dew_point_2m",
    "wind_direction_10m",
    "wind_speed_10m",
    "shortwave_radiation",
    "direct_normal_irradiance",
    "diffuse_radiation",
    "cloud_cover",
    "soil_moisture_0_to_7cm",
]
START = date(1980, 1, 1)
END_DATE = date(2026, 9, 6)
CALLS_PER_REQUEST = 31 / 14  # Open-Meteo weighting estimate for one month, <= 10 variables
FREE_DAILY_LIMIT = 10_000


def months(start: date = START, end: date = END_DATE) -> list[tuple[date, date]]:
    """(first_day, last_day) per month, the last month clamped to `end`."""
    out, y, m = [], start.year, start.month
    while (y, m) <= (end.year, end.month):
        first = date(y, m, 1)
        last = min(date(y, m, monthrange(y, m)[1]), end)
        out.append((first, last))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def out_path(cell: int, first: date, root: Path = RAW_V2_DIR) -> Path:
    return root / f"cell_{cell}" / f"{first.year}-{first.month:02d}.json"


def params_for(cell: int, first: date, last: date) -> dict:
    lat, lon = CELLS[cell - 1]
    return {
        "latitude": lat, "longitude": lon,
        "start_date": first.isoformat(), "end_date": last.isoformat(),
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "Asia/Kolkata", "wind_speed_unit": "ms",
        "cell_selection": "nearest", "models": "era5",
    }


def payload_complete(payload: dict, first: date, last: date) -> bool:
    """Response/file content is complete: every variable present, exactly the expected
    hours from {first}T00:00 to {last}T23:00 (checked by first/last timestamp and
    count), and no nulls."""
    hourly = payload.get("hourly") if isinstance(payload, dict) else None
    if not hourly or "time" not in hourly:
        return False
    times = hourly["time"]
    n_hours = ((last - first).days + 1) * 24
    if len(times) != n_hours or times[0] != f"{first.isoformat()}T00:00" or times[-1] != f"{last.isoformat()}T23:00":
        return False
    return all(var in hourly and len(hourly[var]) == n_hours and all(v is not None for v in hourly[var])
               for var in HOURLY_VARIABLES)


def is_complete(path: Path, first: date, last: date) -> bool:
    """File exists, parses, and passes payload_complete."""
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return False
    return payload_complete(payload, first, last)


class RateLimitExhausted(RuntimeError):
    """Rate limit persisted through all waits -- most likely the daily quota is used up."""


def write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, path)  # atomic on the same filesystem


def download_one(session: requests.Session, cell: int, first: date, last: date,
                 root: Path = RAW_V2_DIR, retries: int = 4, max_rate_waits: int = 10) -> bool:
    """Fetch one cell-month. The payload is validated IN MEMORY and only then written
    atomically, so a bad response never reaches disk. Rate-limit (429) waits do not use
    up retry attempts (bounded separately by max_rate_waits)."""
    path = out_path(cell, first, root)
    if is_complete(path, first, last):
        return True
    attempts = rate_waits = 0
    while attempts < retries:
        try:
            r = session.get(URL, params=params_for(cell, first, last), timeout=120)
            if r.status_code == 429:
                rate_waits += 1
                if rate_waits > max_rate_waits:
                    raise RateLimitExhausted(f"cell {cell} {first:%Y-%m}: still rate limited after {max_rate_waits} waits")
                wait = min(60 * rate_waits, 600)
                print(f"  cell {cell} {first:%Y-%m}: rate limited (429), waiting {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            payload = r.json()
            if not payload_complete(payload, first, last):
                raise ValueError("response incomplete (missing variables/hours, wrong time range, or nulls)")
            write_atomic(path, payload)
            return True
        except (requests.RequestException, ValueError) as err:
            attempts += 1
            print(f"  cell {cell} {first:%Y-%m}: attempt {attempts}/{retries} failed ({err})")
            if attempts < retries:
                time.sleep(15 * attempts)
    return False


def plan(cells: list[int]) -> list[tuple[int, date, date]]:
    return [(c, f, l) for c in cells for f, l in months()]


def run(cells: list[int], sleep_s: float, root: Path = RAW_V2_DIR) -> int:
    """Download everything still missing. A month that keeps failing is skipped and
    reported at the end (re-running retries it); the rest of the run continues."""
    todo = [(c, f, l) for c, f, l in plan(cells) if not is_complete(out_path(c, f, root), f, l)]
    print(f"{len(todo)} monthly files to download for cells {cells} into {root}")
    failed = []
    with requests.Session() as session:
        for i, (c, f, l) in enumerate(todo, 1):
            try:
                ok = download_one(session, c, f, l, root)
            except RateLimitExhausted as err:
                print(f"\n{err}.\nThe daily API quota is probably used up: stopping now. Run again tomorrow to resume.")
                return 2
            if not ok:
                failed.append((c, f))
            if i % 50 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} processed (last: cell {c} {f:%Y-%m}), {len(failed)} failed", flush=True)
            time.sleep(sleep_s)
    if failed:
        print(f"\n{len(failed)} month(s) failed; run again later to retry:")
        for c, f in failed[:20]:
            print(f"  cell {c} {f:%Y-%m}")
        return 1
    print("All requested files present.")
    return 0


def verify(cells: list[int], root: Path = RAW_V2_DIR) -> int:
    bad = [(c, f) for c, f, l in plan(cells) if not is_complete(out_path(c, f, root), f, l)]
    total = len(plan(cells))
    print(f"{total - len(bad)}/{total} monthly files complete in {root}")
    for c, f in bad[:10]:
        print(f"  missing/incomplete: cell {c} {f:%Y-%m}")
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Download extra hourly ERA5 variables (v2).")
    ap.add_argument("--cells", type=int, nargs="+", default=list(range(1, 10)), choices=range(1, 10))
    ap.add_argument("--sleep", type=float, default=2.0, help="seconds between requests")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args(argv)
    if args.verify:
        return verify(args.cells)
    if args.dry_run:
        todo = [p for p in plan(args.cells) if not is_complete(out_path(*p[:2]), p[1], p[2])]
        calls = len(todo) * CALLS_PER_REQUEST
        print(f"months {START:%Y-%m}..{END_DATE:%Y-%m} (last month ends {END_DATE}), cells {args.cells}")
        print(f"requests still needed: {len(todo)} -> ~{calls:,.0f} counted API calls "
              f"(~{calls / FREE_DAILY_LIMIT:.1f} days at the free {FREE_DAILY_LIMIT:,}/day limit), "
              f"~{len(todo) * (args.sleep + 1.5) / 3600:.1f} h of wall time")
        print(f"variables: {', '.join(HOURLY_VARIABLES)}")
        return 0
    return run(args.cells, args.sleep)


if __name__ == "__main__":
    raise SystemExit(main())
