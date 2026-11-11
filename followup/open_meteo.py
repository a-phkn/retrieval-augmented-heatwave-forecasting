"""
Archived global-model forecasts for the 9 Delhi cells, from Open-Meteo (P3-1, context/decisions_phase3.md).

- Previous Runs API: `<variable>_previous_dayL` is the value predicted L x 24 h before its valid time; it is our
  lead-L forecast. Models: ECMWF IFS 0.25° (`ecmwf_ifs025`), ECMWF AIFS 0.25° (`ecmwf_aifs025`, the experimental
  series, which ends 2025-03-10), NCEP GFS (`gfs_seamless`), and the operational AIFS Single
  (`ecmwf_aifs025_single`), added by the deviation of 2026-11-10.
- Single Runs API: one model run by its start time. Only for the IFS timing check: IFS HRES 9 km (`ecmwf_ifs`), the run
  started at 12 UTC on q - 1.
Requests use the ERA5 download's coordinates and settings (pipeline/download_era5_v2.py), so a 0.25° model snaps to
the same 9 grid points as the ERA5 truth and Open-Meteo's elevation adjustment is the same.

Replies are cached as JSON under data/raw/open_meteo_runs/ (gitignored, like every raw download), validated in
memory before writing, and never fetched twice, so the download can be stopped and re-run. Free, non-commercial API:
at most 10 variables per request and a pause between requests keep within its limits; when the daily quota is used
up the run stops, and running it again later resumes it.

Run from repo root:  python -m followup.open_meteo [--single-runs] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from pipeline.download_era5_v2 import CELLS, months, write_atomic
from training.folds import FORECAST_DAYS

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw" / "open_meteo_runs"
PREVIOUS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
SINGLE_URL = "https://single-runs-api.open-meteo.com/v1/forecast"
END = date(2026, 9, 6)  # the test period's last day
WBGT_VARS = ("temperature_2m", "relative_humidity_2m", "surface_pressure", "wind_speed_10m",
             "shortwave_radiation", "diffuse_radiation")
MAX_COLUMNS = 10  # Open-Meteo counts a request with more variables as several calls
# name: (Open-Meteo model id, variables, first day requested, last day requested); first days are the month of each
# model's first archived value (probe 2026-11-05), the GFS 2 m temperature archive starts earlier
PULLS = {
    "IFS": ("ecmwf_ifs025", WBGT_VARS, date(2024, 2, 1), END),
    "AIFS": ("ecmwf_aifs025", ("temperature_2m",), date(2024, 3, 1), END),
    "GFS": ("gfs_seamless", WBGT_VARS, date(2024, 1, 1), END),
    "GFS early": ("gfs_seamless", ("temperature_2m",), date(2021, 3, 1), date(2023, 12, 31)),
    # added by the deviation of 2026-11-10; its first archived value is 2025-02-18 (probe 2026-11-10)
    "AIFS Single": ("ecmwf_aifs025_single", WBGT_VARS, date(2025, 2, 1), END),
}
SINGLE_MODEL = "ecmwf_ifs"  # ECMWF IFS HRES 9 km; Single Runs archive from 2024-03-14
SINGLE_FIRST_RUN = date(2024, 3, 14)
SINGLE_RUN_HOUR_UTC = 12
SINGLE_DIR = "single_ecmwf_ifs"
FREE_DAILY_LIMIT = 10_000
COMMON = {"timezone": "Asia/Kolkata", "wind_speed_unit": "ms", "cell_selection": "nearest"}


class DailyLimit(RuntimeError):
    """The API quota is used up (or rate limiting persisted): stop now and run again later to resume."""


class BadReply(RuntimeError):
    """The API kept sending a body that is not JSON (an empty or cut-off reply)."""


# ------------------------------------------------------------------ request plans


def column_groups(variables: tuple[str, ...]) -> list[list[str]]:
    """Every variable's `_previous_dayL` columns (L = 1..5), packed into requests of at most MAX_COLUMNS columns;
    the leads of one variable always stay in one request."""
    groups, cur = [], []
    for v in variables:
        cols = [f"{v}_previous_day{L}" for L in range(1, FORECAST_DAYS + 1)]
        if cur and len(cur) + len(cols) > MAX_COLUMNS:
            groups.append(cur)
            cur = []
        cur = cur + cols
    if cur:
        groups.append(cur)
    return groups


def group_name(cols: list[str]) -> str:
    return "+".join(sorted({c.split("_previous_day")[0] for c in cols}))


def previous_path(model: str, cell: int, first: date, cols: list[str], root: Path = RAW_DIR) -> Path:
    return root / model / f"cell_{cell}" / f"{first:%Y-%m}_{group_name(cols)}.json"


def previous_params(model: str, cell: int, first: date, last: date, cols: list[str]) -> dict:
    lat, lon = CELLS[cell - 1]
    return {"latitude": lat, "longitude": lon, "start_date": first.isoformat(), "end_date": last.isoformat(),
            "hourly": ",".join(cols), "models": model, **COMMON}


def previous_plan(pulls: dict = PULLS) -> list[tuple[str, int, date, date, list[str]]]:
    """(model, cell, first day, last day, columns) for every request, month by month."""
    return [(model, cell, first, last, cols)
            for model, variables, start, end in pulls.values()
            for first, last in months(start, end)
            for cols in column_groups(variables)
            for cell in range(1, 10)]


def single_run_dates(first: date = SINGLE_FIRST_RUN, end: date = END) -> list[date]:
    """Run dates r = q - 1 for every query date q from the first run onwards whose 5 target days end by `end`."""
    last = end - timedelta(days=FORECAST_DAYS)  # q + 4 <= end  <=>  r = q - 1 <= end - 5
    return [first + timedelta(days=i) for i in range((last - first).days + 1)]


def single_path(run: date, root: Path = RAW_DIR) -> Path:
    return root / SINGLE_DIR / f"{run.isoformat()}.json"


def single_params(run: date) -> dict:
    return {"latitude": ",".join(str(c[0]) for c in CELLS), "longitude": ",".join(str(c[1]) for c in CELLS),
            "hourly": ",".join(WBGT_VARS), "models": SINGLE_MODEL, "run": f"{run.isoformat()}T{SINGLE_RUN_HOUR_UTC:02d}:00",
            "forecast_days": 7, **COMMON}


# ------------------------------------------------------------------ validation (in memory, before writing)


def previous_reply_ok(payload: dict, first: date, last: date, cols: list[str]) -> bool:
    """Every requested column, with exactly the hours {first}T00:00 .. {last}T23:00. Nulls are allowed: the archive
    has gaps, and days with gaps are dropped in the analysis, not here."""
    hourly = payload.get("hourly") if isinstance(payload, dict) else None
    if not hourly or "time" not in hourly:
        return False
    t, n = hourly["time"], ((last - first).days + 1) * 24
    if len(t) != n or t[0] != f"{first.isoformat()}T00:00" or t[-1] != f"{last.isoformat()}T23:00":
        return False
    return all(c in hourly and len(hourly[c]) == n for c in cols)


def single_reply_ok(payload, run: date) -> bool:
    """A list of 9 locations, each with every variable and hours covering all 5 target days (IST days r+1 .. r+5)."""
    if isinstance(payload, dict) and "unavailable" in payload:
        return True
    if not isinstance(payload, list) or len(payload) != len(CELLS):
        return False
    need_last = f"{(run + timedelta(days=FORECAST_DAYS)).isoformat()}T23:30"
    for p in payload:
        hourly = p.get("hourly") if isinstance(p, dict) else None
        if not hourly or "time" not in hourly or not hourly["time"] or hourly["time"][-1] < need_last:
            return False
        if not all(v in hourly and len(hourly[v]) == len(hourly["time"]) for v in WBGT_VARS):
            return False
    return True


def cached(path: Path, check) -> bool:
    if not path.exists():
        return False
    try:
        return bool(check(json.loads(path.read_text(encoding="utf-8"))))
    except ValueError:
        return False


# ------------------------------------------------------------------ HTTP


def _reason(r: requests.Response) -> str:
    try:
        return str(r.json().get("reason", r.text[:200]))
    except ValueError:
        return r.text[:200]


def get_json(session, url: str, params: dict, retries: int = 4, max_rate_waits: int = 10, sleep=time.sleep):
    """GET with retries for timeouts, connection errors, server errors and bodies that are not JSON (BadReply after
    the last try). A 429 waits (60 s, 120 s, ... up to 10 min) unless the reason names the daily limit; DailyLimit
    is raised when the quota is gone. Any other 4xx raises ValueError with Open-Meteo's reason (for example a model
    run that is not in the archive)."""
    attempts = rate_waits = 0
    while True:
        try:
            r = session.get(url, params=params, timeout=120)
        except requests.RequestException:
            attempts += 1
            if attempts >= retries:
                raise
            sleep(15 * attempts)
            continue
        if r.status_code == 429:
            reason = _reason(r)
            rate_waits += 1
            if "daily" in reason.lower() or rate_waits > max_rate_waits:
                raise DailyLimit(reason)
            sleep(min(60 * rate_waits, 600))
            continue
        if r.status_code >= 500:
            attempts += 1
            if attempts >= retries:
                r.raise_for_status()
            sleep(15 * attempts)
            continue
        if r.status_code >= 400:
            raise ValueError(f"Open-Meteo refused the request ({r.status_code}): {_reason(r)}")
        try:
            return r.json()
        except ValueError:  # an empty or cut-off body: retried like a server error
            attempts += 1
            if attempts >= retries:
                raise BadReply(f"the reply was not JSON in {retries} tries") from None
            sleep(15 * attempts)


# ------------------------------------------------------------------ downloads


def _report_failed(failed: list[str]) -> int:
    """0 if nothing failed; otherwise list the failures (a re-run retries them) and return 1."""
    if not failed:
        return 0
    print(f"\n{len(failed)} request(s) failed after their retries; run again to retry them:", flush=True)
    print("\n".join(f"  {f}" for f in failed[:20]), flush=True)
    return 1


def download_previous(session, root: Path = RAW_DIR, pause: float = 2.0, sleep=time.sleep) -> int:
    """Fetch every Previous Runs request not cached yet. Returns 0 when all are present, 1 when some failed after
    their retries (skipped, the rest continues), 2 when the quota stopped it."""
    todo = [p for p in previous_plan() if not cached(previous_path(*p[:3], p[4], root=root),
                                                     lambda d, p=p: previous_reply_ok(d, p[2], p[3], p[4]))]
    print(f"Previous Runs: {len(todo)} requests to download into {root}", flush=True)
    failed = []
    for i, (model, cell, first, last, cols) in enumerate(todo, 1):
        try:
            payload = get_json(session, PREVIOUS_URL, previous_params(model, cell, first, last, cols), sleep=sleep)
        except DailyLimit as err:
            print(f"\nStopped at {i - 1}/{len(todo)}: {err}\nRun again later to resume.", flush=True)
            return 2
        except (BadReply, requests.RequestException) as err:
            failed.append(f"{model} cell {cell} {first:%Y-%m} {group_name(cols)}: {err}")
            sleep(pause)
            continue
        if not previous_reply_ok(payload, first, last, cols):
            raise ValueError(f"{model} cell {cell} {first:%Y-%m}: reply has missing columns or hours")
        write_atomic(previous_path(model, cell, first, cols, root), payload)
        if i % 50 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)} (last: {model} cell {cell} {first:%Y-%m})", flush=True)
        sleep(pause)
    return _report_failed(failed)


def download_single(session, root: Path = RAW_DIR, pause: float = 6.0, sleep=time.sleep) -> int:
    """Fetch every IFS HRES run (12 UTC on q - 1) not cached yet; a run the archive does not have is cached as
    {"unavailable": reason} so it is not asked for again."""
    todo = [r for r in single_run_dates() if not cached(single_path(r, root), lambda d, r=r: single_reply_ok(d, r))]
    print(f"Single Runs: {len(todo)} runs to download into {root / SINGLE_DIR}", flush=True)
    failed = []
    for i, run in enumerate(todo, 1):
        try:
            payload = get_json(session, SINGLE_URL, single_params(run), sleep=sleep)
        except DailyLimit as err:
            print(f"\nStopped at {i - 1}/{len(todo)}: {err}\nRun again later to resume.", flush=True)
            return 2
        except (BadReply, requests.RequestException) as err:
            failed.append(f"run {run}: {err}")
            sleep(pause)
            continue
        except ValueError as err:
            payload = {"unavailable": str(err)}
        if not single_reply_ok(payload, run):
            raise ValueError(f"run {run}: reply has missing locations, variables or hours")
        write_atomic(single_path(run, root), payload)
        if i % 25 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)} (last run: {run})", flush=True)
        sleep(pause)
    return _report_failed(failed)


# ------------------------------------------------------------------ reading the cache


def load_previous(model: str, variables: tuple[str, ...], root: Path = RAW_DIR
                  ) -> tuple[dict[int, pd.DataFrame], dict[int, dict]]:
    """Cached Previous Runs values of one model: {lead: frame(cell, time, <variables>)} (hours with no file are
    absent, archive gaps are NaN) and each cell's grid point {cell: {latitude, longitude, elevation}}."""
    per_cell, where = [], {}
    for cell in range(1, 10):
        frames = []
        for path in sorted((root / model / f"cell_{cell}").glob("*.json")):
            p = json.loads(path.read_text(encoding="utf-8"))
            spot = {k: p[k] for k in ("latitude", "longitude", "elevation")}
            if where.setdefault(cell, spot) != spot:
                raise ValueError(f"{model} cell {cell}: files disagree on the grid point ({where[cell]} vs {spot})")
            frames.append(pd.DataFrame(p["hourly"]).set_index("time"))
        if not frames:
            continue
        wide = pd.concat(frames).groupby(level=0).first()  # one row per hour; columns come from different files
        wide = wide.astype("float64")  # a month with no values at all arrives as object columns; a string would raise
        wide.index = pd.to_datetime(wide.index)
        per_cell.append(wide.assign(cell=cell))
    if not per_cell:
        raise FileNotFoundError(f"no cached replies for {model} in {root}; run python -m followup.open_meteo")
    wide = pd.concat(per_cell).rename_axis("time").reset_index()
    out = {}
    for L in range(1, FORECAST_DAYS + 1):
        cols = {f"{v}_previous_day{L}": v for v in variables}
        missing = [c for c in cols if c not in wide]
        if missing:
            raise ValueError(f"{model}: no cached column {missing[0]}")
        out[L] = wide[["cell", "time", *cols]].rename(columns=cols)
    return out, where


def load_single(root: Path = RAW_DIR) -> tuple[pd.DataFrame, dict[int, dict], list[str]]:
    """Cached IFS HRES runs: frame(run, cell, time, <WBGT_VARS>), each cell's grid point, and the runs the archive
    did not have."""
    rows, where, unavailable = [], {}, []
    for path in sorted((root / SINGLE_DIR).glob("*.json")):
        run = pd.Timestamp(path.stem)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "unavailable" in payload:
            unavailable.append(path.stem)
            continue
        for cell, p in enumerate(payload, 1):
            spot = {k: p[k] for k in ("latitude", "longitude", "elevation")}
            if where.setdefault(cell, spot) != spot:
                raise ValueError(f"IFS HRES cell {cell}: runs disagree on the grid point")
            h = pd.DataFrame(p["hourly"])
            h[list(WBGT_VARS)] = h[list(WBGT_VARS)].astype("float64")  # all-null columns arrive as object
            rows.append(h.assign(run=run, cell=cell, time=pd.to_datetime(h["time"])))
    if not rows:
        raise FileNotFoundError(f"no cached IFS HRES runs in {root / SINGLE_DIR}; run python -m followup.open_meteo --single-runs")
    return pd.concat(rows, ignore_index=True), where, unavailable


# ------------------------------------------------------------------ command line


def main(argv: list[str] | None = None) -> int:
    from training import test_lock

    ap = argparse.ArgumentParser(description="Download archived global-model forecasts for Delhi (P3-1).")
    ap.add_argument("--single-runs", action="store_true", help="the IFS HRES 9 km timing check instead")
    ap.add_argument("--pause", type=float, default=None, help="seconds between requests (default 2, single runs 6)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    test_lock.require()  # these are 2019+ data (pre-registration: every script checks the lock first)
    if args.dry_run:
        prev = [p for p in previous_plan() if not previous_path(*p[:3], p[4]).exists()]
        calls = sum(max(1.0, ((last - first).days + 1) / 14) for _, _, first, last, _ in prev)
        runs = [r for r in single_run_dates() if not single_path(r).exists()]
        print(f"Previous Runs: {len(prev)} requests left, ~{calls:,.0f} counted calls; "
              f"Single Runs: {len(runs)} runs left, ~{9 * len(runs):,} counted calls "
              f"(free limit {FREE_DAILY_LIMIT:,} a day)")
        return 0
    with requests.Session() as session:
        if args.single_runs:
            return download_single(session, pause=6.0 if args.pause is None else args.pause)
        return download_previous(session, pause=2.0 if args.pause is None else args.pause)


if __name__ == "__main__":
    raise SystemExit(main())
