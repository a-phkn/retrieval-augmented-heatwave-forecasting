"""
ERA5 hourly downloader for the v2 extra cities (docs/PROTOCOL_V2.md, Section 7): Chennai (humid) and
Ahmedabad (dry). Exploratory replication of H-A; the cities are the study's only unseen data.

Same source and settings as the Delhi downloaders (download_era5.py and pipeline/download_era5_v2.py):
Open-Meteo ERA5 archive, models=era5, timezone Asia/Kolkata, wind in m/s. One request fetches all six hourly
inputs that the Liljegren WBGT needs (pipeline/build_wbgt_liljegren.py): temperature, relative humidity, 10 m
wind, surface pressure, global (shortwave) and diffuse radiation.

Grid: a 3 x 3 block of ERA5 0.25-degree points around each city's IMD airport station, as Delhi's 9 cells.
    Ahmedabad (airport ~23.07 N, 72.63 E): lat 22.75 / 23.00 / 23.25, lon 72.50 / 72.75 / 73.00
    Chennai   (Meenambakkam ~12.99 N, 80.18 E): lat 12.75 / 13.00 / 13.25, lon 79.75 / 80.00 / 80.25
cell_selection = "land" (Open-Meteo's default: "finds a suitable grid-cell on land with similar elevation to
the requested coordinates"), so a coastal point (Chennai's east column) is not a sea cell. Delhi used "nearest";
inland the two normally coincide. Disclosed in the paper. The coordinates Open-Meteo
actually returns are kept in every saved file ("latitude", "longitude").

Months: March-September only (the WBGT season of labels v2 is Mar 15 - Sep 30), 1980-2025. Cuts the API load
to 7/12; about 5,800 requests, about 17,000 counted calls at the pricing page's weighting, so with the free
10,000 calls/day limit a full run takes about two days. It is resumable: run it again and it skips finished files.

Writes data/raw/cities/<city>/cell_N/YYYY-MM.json (data/raw/ is gitignored).

Run from repo root:
    python -m v2.download_cities --dry-run
    python -m v2.download_cities                  # both cities; Ctrl+C is safe
    python -m v2.download_cities --city chennai
    python -m v2.download_cities --verify
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
ROOT = REPO_ROOT / "data" / "raw" / "cities"

CITIES = {
    "ahmedabad": [(lat, lon) for lat in (22.75, 23.00, 23.25) for lon in (72.50, 72.75, 73.00)],
    "chennai": [(lat, lon) for lat in (12.75, 13.00, 13.25) for lon in (79.75, 80.00, 80.25)],
}
HOURLY = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "surface_pressure",
          "shortwave_radiation", "diffuse_radiation"]
YEARS = range(1980, 2026)
MONTHS = range(3, 10)  # March..September
CALLS_PER_REQUEST = 3.0  # conservative: open-meteo.com/en/pricing counts "4 weeks of data" as 3.0 calls
FREE_DAILY_LIMIT = 10_000


def months() -> list[tuple[date, date]]:
    return [(date(y, m, 1), date(y, m, monthrange(y, m)[1])) for y in YEARS for m in MONTHS]


def out_path(city: str, cell: int, first: date) -> Path:
    return ROOT / city / f"cell_{cell}" / f"{first.year}-{first.month:02d}.json"


def params_for(city: str, cell: int, first: date, last: date) -> dict:
    lat, lon = CITIES[city][cell - 1]
    return {"latitude": lat, "longitude": lon, "start_date": first.isoformat(), "end_date": last.isoformat(),
            "hourly": ",".join(HOURLY), "timezone": "Asia/Kolkata", "wind_speed_unit": "ms",
            "cell_selection": "land", "models": "era5"}


def payload_complete(payload: dict, first: date, last: date) -> bool:
    """Every variable present, exactly the hours {first}T00:00 .. {last}T23:00, no nulls."""
    hourly = payload.get("hourly") if isinstance(payload, dict) else None
    if not hourly or "time" not in hourly:
        return False
    n = ((last - first).days + 1) * 24
    t = hourly["time"]
    if len(t) != n or t[0] != f"{first.isoformat()}T00:00" or t[-1] != f"{last.isoformat()}T23:00":
        return False
    return all(v in hourly and len(hourly[v]) == n and None not in hourly[v] for v in HOURLY)


def is_complete(path: Path, first: date, last: date) -> bool:
    if not path.exists():
        return False
    try:
        return payload_complete(json.loads(path.read_text(encoding="utf-8")), first, last)
    except ValueError:
        return False


def write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, path)


class RateLimitExhausted(RuntimeError):
    pass


def download_one(session, city, cell, first, last, retries=4, max_rate_waits=10) -> bool:
    path = out_path(city, cell, first)
    if is_complete(path, first, last):
        return True
    attempts = rate_waits = 0
    while attempts < retries:
        try:
            r = session.get(URL, params=params_for(city, cell, first, last), timeout=120)
            if r.status_code == 429:
                rate_waits += 1
                if rate_waits > max_rate_waits:
                    raise RateLimitExhausted(f"{city} cell {cell} {first:%Y-%m}: still rate limited")
                wait = min(60 * rate_waits, 600)
                print(f"  {city} cell {cell} {first:%Y-%m}: rate limited, waiting {wait}s", flush=True)
                time.sleep(wait)
                continue
            r.raise_for_status()
            payload = r.json()
            if not payload_complete(payload, first, last):
                raise ValueError("response incomplete")
            write_atomic(path, payload)
            return True
        except (requests.RequestException, ValueError) as err:
            attempts += 1
            print(f"  {city} cell {cell} {first:%Y-%m}: attempt {attempts}/{retries} failed ({err})", flush=True)
            if attempts < retries:
                time.sleep(15 * attempts)
    return False


def plan(cities):
    return [(c, k, f, l) for c in cities for k in range(1, 10) for f, l in months()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ERA5 hourly for the v2 extra cities.")
    ap.add_argument("--city", choices=sorted(CITIES), nargs="+", default=sorted(CITIES))
    ap.add_argument("--sleep", type=float, default=2.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args(argv)
    todo = [p for p in plan(a.city) if not is_complete(out_path(p[0], p[1], p[2]), p[2], p[3])]
    if a.verify:
        total = len(plan(a.city))
        print(f"{total - len(todo)}/{total} monthly files complete under {ROOT}")
        return 1 if todo else 0
    if a.dry_run:
        calls = len(todo) * CALLS_PER_REQUEST
        print(f"cities {a.city}; {len(todo)} requests still needed -> ~{calls:,.0f} counted calls "
              f"(~{calls / FREE_DAILY_LIMIT:.1f} days at the free limit)")
        return 0
    print(f"{len(todo)} monthly files to download into {ROOT}", flush=True)
    failed = []
    with requests.Session() as s:
        for i, (c, k, f, l) in enumerate(todo, 1):
            try:
                if not download_one(s, c, k, f, l):
                    failed.append((c, k, f))
            except RateLimitExhausted as err:
                print(f"\n{err}. Daily quota probably used up: stopping. Run again tomorrow to resume.")
                return 2
            if i % 50 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} done, {len(failed)} failed", flush=True)
            time.sleep(a.sleep)
    if failed:
        print(f"{len(failed)} month(s) failed; run again to retry.")
        return 1
    print("All requested files present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
