"""
P3-7 step 4, as changed by the deviation of 2026-11-27 (context/decisions_phase3.md): Delhi's ERA5 dew point against
the Safdarjung and Palam stations, heat seasons of 1980-2018. Descriptive: no pass rule and no prediction.

Why: evaluation_v2/premise_report.md found ERA5's vapour pressure at Tmax stepping around 2000-01. A station record
shows whether such a step is in the air or in the reanalysis.

- Station: the mean of its dew-point readings at the 8 synoptic times of the IST day (21 UTC of the day before, then
  00-18 UTC every 3 hours; 02:30-23:30 IST), read from the raw ISD reports with followup/isd_rebuild.py's rules.
- ERA5: its own 2 m dew point (data/raw/era5_v2/cell_<n>/<YYYY-MM>.json, hourly, IST), at the same instants. The
  value at half past an IST hour is the mean of that hour and the next. Files of 1980-2018 only.
- A day needs at least 7 of its 8 instants with both values; both means use exactly those instants.
- A heat season (15 Mar - 31 Jul) counts if at least 70 of its days have a value.

Run from repo root:  python -m followup.station_humidity  -> followup/results/station_humidity.{md,json}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from followup import isd_rebuild as ir
from pipeline.labels_v2 import in_season

REPO_ROOT = Path(__file__).resolve().parents[1]
ERA5_DIR = REPO_ROOT / "data" / "raw" / "era5_v2"
OUT_DIR = REPO_ROOT / "followup" / "results"
YEARS = (1980, 2018)  # station and ERA5 values outside these years are not read
SYNOPTIC_UTC = (21, 0, 3, 6, 9, 12, 15, 18)  # 21 UTC belongs to the next IST day (02:30 IST)
MIN_INSTANTS = 7
MIN_SEASON_DAYS = 70  # of the 139 days of 15 Mar - 31 Jul
DECADES = {"1990-1999": (1990, 1999), "2002-2011": (2002, 2011)}
N_BOOT, SEED = 10_000, 0
COORDS = {"safdarjung": (28.585, 77.206), "palam": (28.567, 77.103)}  # NOAA's station list (isd-history.csv)
NAMES = {"safdarjung": "Safdarjung", "palam": "Palam (Indira Gandhi International)"}
RECORDS = {"station": "Station", "era5_cells": "ERA5, 9 cells", "era5_nearest": "ERA5, nearest cell"}
DIFFS = {"diff_cells": "ERA5 - station, 9 cells", "diff_nearest": "ERA5 - station, nearest cell"}


# ------------------------------------------------------------------ the station


def _year(path: Path) -> int:
    """The year of a raw ISD file named <USAF>-<WBAN>-<YEAR>.gz."""
    return int(path.name.split("-")[-1].split(".")[0])


def station_instants(rep: pd.DataFrame) -> pd.Series:
    """The station's dew point (°C) at the synoptic times, indexed by UTC time: the minute-00 reports at SYNOPTIC_UTC
    hours among the selected reports (one per time), usable values only."""
    at = rep[(rep["time"].dt.minute == 0) & rep["time"].dt.hour.isin(SYNOPTIC_UTC)]
    return at.set_index("time")["td"].dropna().astype("float64").rename("station")


def read_station(name: str, years: tuple[int, int] = YEARS, root: Path = ir.RAW_DIR) -> tuple[pd.Series, int]:
    """The station's synoptic dew points from its raw files of `years` only, and the number of files read."""
    paths = [p for p in ir.paths_for(ir.STATIONS[name], root) if years[0] <= _year(p) <= years[1]]
    if not paths:
        raise FileNotFoundError(f"no raw ISD files for {name} in {root} (python -m followup.isd_rebuild --download)")
    return station_instants(ir.read_reports(paths)), len(paths)


# ------------------------------------------------------------------ ERA5


def load_era5_dew(cell: int, years: tuple[int, int] = YEARS, root: Path = ERA5_DIR) -> pd.Series:
    """ERA5's hourly 2 m dew point (°C) for one cell, indexed by the IST hour, every hour, from the monthly files of
    `years` only. Every file must be in IST."""
    files = sorted(p for p in (root / f"cell_{cell}").glob("*.json") if years[0] <= int(p.stem[:4]) <= years[1])
    if not files:
        raise FileNotFoundError(f"no ERA5 files for cell {cell} in {root}")
    parts = []
    for p in files:
        j = json.loads(p.read_text(encoding="utf-8"))
        if j.get("timezone") != "Asia/Kolkata" or j.get("utc_offset_seconds") != 19800:
            raise ValueError(f"{p}: the times are not IST")
        h = j["hourly"]
        parts.append(pd.Series(h["dew_point_2m"], index=pd.to_datetime(h["time"]), dtype="float64"))
    s = pd.concat(parts)
    s = s[~s.index.duplicated()].sort_index()
    return s.reindex(pd.date_range(s.index[0], s.index[-1], freq="h")).rename(f"cell_{cell}")


def cell_coords(root: Path = ERA5_DIR) -> dict[int, tuple[float, float]]:
    """Each ERA5 cell's centre (latitude, longitude), from its first file."""
    out = {}
    for folder in sorted(root.glob("cell_*")):
        first = min(folder.glob("*.json"), default=None)
        if first is not None:
            j = json.loads(first.read_text(encoding="utf-8"))
            out[int(folder.name.split("_")[1])] = (float(j["latitude"]), float(j["longitude"]))
    return out


def nearest_cell(lat: float, lon: float, cells: dict[int, tuple[float, float]]) -> int:
    """The cell whose centre is nearest, in degrees, with longitude scaled by cos(latitude)."""
    return min(cells, key=lambda c: np.hypot(lat - cells[c][0], (lon - cells[c][1]) * np.cos(np.radians(lat))))


def era5_instants(hourly: pd.Series) -> pd.Series:
    """ERA5 at the synoptic instants, indexed by UTC time. `hourly` is indexed by the IST hour, every hour; the value at
    half past an IST hour is the mean of that hour and the next, as in evaluation/graphcast_yardstick.py."""
    if not hourly.index.equals(pd.date_range(hourly.index[0], hourly.index[-1], freq="h")):
        raise ValueError("the ERA5 series must be hourly, with no gap in its index")
    half = (hourly + hourly.shift(-1)) / 2  # at t (IST): the value at t + 30 min
    half.index = half.index + pd.Timedelta(minutes=30) - ir.IST  # the UTC instant
    return half[half.index.hour.isin(SYNOPTIC_UTC)].dropna()


# ------------------------------------------------------------------ days and seasons


def instant_frame(station: pd.Series, cells: dict[int, pd.Series], nearest: int) -> pd.DataFrame:
    """The synoptic instants with a station reading and an ERA5 value in every cell: the station, the 9-cell mean and
    the nearest cell, indexed by UTC time."""
    e = pd.concat({c: era5_instants(s) for c, s in cells.items()}, axis=1)
    f = pd.concat([station.rename("station"), e.mean(axis=1, skipna=False).rename("era5_cells"),
                   e[nearest].rename("era5_nearest")], axis=1, join="inner")
    return f.dropna()


def daily_means(inst: pd.DataFrame, min_instants: int = MIN_INSTANTS) -> pd.DataFrame:
    """Per IST day: each record's mean over the instants they share, and their number. Days with fewer than
    min_instants are left out."""
    g = inst.groupby((inst.index + ir.IST).normalize())
    out = g.mean()
    out["instants"] = g.size()
    out.index.name = "date"
    return out[out["instants"] >= min_instants]


def season_table(d: pd.DataFrame, min_days: int = MIN_SEASON_DAYS) -> pd.DataFrame:
    """Per heat season (15 Mar - 31 Jul): the days with a value, each record's mean dew point, the mean of ERA5 -
    station (9 cells and nearest cell), and whether the season counts (at least min_days days)."""
    s = d[in_season(d.index)]
    y = s.index.year
    out = pd.DataFrame({"days": s.groupby(y).size(), **{k: s[k].groupby(y).mean() for k in RECORDS},
                        "diff_cells": (s["era5_cells"] - s["station"]).groupby(y).mean(),
                        "diff_nearest": (s["era5_nearest"] - s["station"]).groupby(y).mean()})
    out.index.name = "year"
    out["counts"] = out["days"] >= min_days
    return out


def decade_change(seasons: pd.DataFrame, col: str, decades: dict = DECADES, n_boot: int = N_BOOT,
                  seed: int = SEED) -> dict | None:
    """The change of `col` from the first decade to the second, over the seasons that count, with a 95% percentile CI
    by bootstrap (the seasons of each decade resampled with replacement). None if a decade has no season."""
    ok = seasons[seasons["counts"]]
    (n1, (a1, b1)), (n2, (a2, b2)) = decades.items()
    x = ok.loc[(ok.index >= a1) & (ok.index <= b1), col].to_numpy(dtype=np.float64)
    y = ok.loc[(ok.index >= a2) & (ok.index <= b2), col].to_numpy(dtype=np.float64)
    if not len(x) or not len(y):
        return None
    rng = np.random.default_rng(seed)
    bx = rng.choice(x, (n_boot, len(x))).mean(axis=1)
    by = rng.choice(y, (n_boot, len(y))).mean(axis=1)
    lo, hi = np.percentile(by - bx, [2.5, 97.5])
    return {"first": n1, "second": n2, "seasons_first": int(len(x)), "seasons_second": int(len(y)),
            "mean_first": float(x.mean()), "mean_second": float(y.mean()), "change": float(y.mean() - x.mean()),
            "ci_low": float(lo), "ci_high": float(hi)}


def trend(seasons: pd.DataFrame, col: str) -> dict | None:
    """The least-squares slope of `col` over the seasons that count, per decade. None with fewer than two seasons."""
    ok = seasons[seasons["counts"]]
    if len(ok) < 2:
        return None
    slope = np.polyfit(ok.index.to_numpy(dtype=np.float64), ok[col].to_numpy(dtype=np.float64), 1)[0]
    return {"per_decade": float(10 * slope), "seasons": int(len(ok)), "first": int(ok.index.min()),
            "last": int(ok.index.max())}


def _records(seasons: pd.DataFrame) -> list[dict]:
    """The season table as plain JSON values."""
    return [{"year": int(y), "days": int(r["days"]), **{k: float(r[k]) for k in (*RECORDS, *DIFFS)},
             "counts": bool(r["counts"])} for y, r in seasons.iterrows()]


def run_station(name: str, cells: dict[int, pd.Series], coords: dict[int, tuple[float, float]],
                root: Path = ir.RAW_DIR) -> dict:
    """Step 4 for one station."""
    station, n_files = read_station(name, root=root)
    near = nearest_cell(*COORDS[name], coords)
    d = daily_means(instant_frame(station, cells, near))
    seasons = season_table(d)
    return {"files": n_files, "nearest_cell": near, "nearest_centre": list(coords[near]),
            "days": int(len(d)), "in_season_days": int(in_season(d.index).sum()),
            "seasons": _records(seasons),
            "change": {k: decade_change(seasons, k) for k in DIFFS},
            "trend": {k: trend(seasons, k) for k in RECORDS}}


# ------------------------------------------------------------------ report


def _num(x, fmt: str = "+.2f") -> str:
    return "n/a" if x is None or not np.isfinite(x) else format(x, fmt)


def _change(c: dict | None) -> str:
    if c is None:
        return "n/a (no season in one of the decades)"
    return (f"{_num(c['change'])} °C [{_num(c['ci_low'])}, {_num(c['ci_high'])}] "
            f"({c['seasons_first']} and {c['seasons_second']} seasons)")


def _trend(t: dict | None) -> str:
    return "n/a" if t is None else f"{_num(t['per_decade'])} ({t['first']}-{t['last']}, {t['seasons']} seasons)"


def write_report(res: dict) -> None:
    (first, second) = res["decades"]
    L = ["# Delhi's ERA5 dew point against the stations, heat seasons 1980-2018 (P3-7 step 4)", "",
         "Pre-registered 2026-11-25 (P3-7 step 4) and changed by the deviation of 2026-11-27, before any dew-point value "
         "was read (`context/decisions_phase3.md`). Descriptive: no pass rule and no prediction. Why: "
         "`evaluation_v2/premise_report.md` found ERA5's vapour pressure at Tmax stepping around 2000-01, and a "
         "station record shows whether such a step is in the air or in the reanalysis.", "",
         "- Station: the mean of the readings at the 8 synoptic times of the IST day (02:30 to 23:30 IST, every 3 "
         "hours).",
         "- ERA5: its own 2 m dew point at the same instants (the mean of the hourly values either side of each half "
         "hour), as the mean of the 9 cells and at the cell nearest the station.",
         f"- A day needs at least {res['min_instants']} of the 8 instants with both values, and both means use those "
         f"instants. A heat season (15 Mar - 31 Jul) counts if at least {res['min_season_days']} of its days have a "
         "value.", "",
         "## Summary", "",
         f"| Station | Seasons that count | Change of ERA5 - station, {first} to {second}: 9 cells [95% CI] | "
         "Nearest cell [95% CI] | Trend per decade (°C): station | ERA5, 9 cells | ERA5, nearest cell |",
         "|---|---|---|---|---|---|---|"]
    for name, r in res["stations"].items():
        ok = [s["year"] for s in r["seasons"] if s["counts"]]
        span = f"{len(ok)} ({min(ok)}-{max(ok)})" if ok else "0"
        L.append(f"| {NAMES[name]} | {span} | {_change(r['change']['diff_cells'])} | "
                 f"{_change(r['change']['diff_nearest'])} | "
                 + " | ".join(_trend(r["trend"][k]) for k in RECORDS) + " |")
    L += ["", f"The change is the mean over the {second} seasons minus the mean over the {first} seasons. Its CI "
          f"resamples the seasons within each decade ({res['n_boot']:,} resamples, seed {res['seed']}). 2000 and 2001, "
          "the years of the step, are in neither decade.", ""]
    for name, r in res["stations"].items():
        lat, lon = r["nearest_centre"]
        L += [f"## {NAMES[name]}", "",
              f"Raw files read: {r['files']} (1980-2018). Days with a value: {r['days']}, of which {r['in_season_days']} "
              f"in the heat season. Nearest ERA5 cell: cell {r['nearest_cell']} ({lat:.2f}N {lon:.2f}E).", ""]
        for key, c in r["change"].items():
            if c is not None:
                L.append(f"- {DIFFS[key]}: {_num(c['mean_first'])} °C in {c['first']}, {_num(c['mean_second'])} °C in "
                         f"{c['second']}; change {_change(c)}.")
        L += ["", "| Season | Days | Station (°C) | ERA5, 9 cells | ERA5, nearest cell | ERA5 - station, 9 cells | "
              "ERA5 - station, nearest cell |", "|---|---|---|---|---|---|---|"]
        for s in r["seasons"]:
            flag = "" if s["counts"] else f" (not counted: fewer than {res['min_season_days']} days)"
            L.append(f"| {s['year']}{flag} | {s['days']} | " + " | ".join(_num(s[k], ".2f") for k in RECORDS)
                     + " | " + " | ".join(_num(s[k]) for k in DIFFS) + " |")
        L.append("")
    L += ["## Notes", "",
          "- Only station and ERA5 values of 1980-2018 are read; the test years are not used.",
          "- The 8 synoptic readings miss part of the daily cycle, so neither mean is a full-day mean. ERA5 is taken at "
          "the same instants, so the difference is like-for-like.",
          "- Palam has no dew-point readings at these times before 2002, so its 1990s cannot be compared.",
          "- A single station's dew point also reflects its surroundings, such as irrigation, green space and building "
          "around it. A change in ERA5 - station can come from the station's side as well as from ERA5's."]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "station_humidity.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    coords = cell_coords()
    if sorted(coords) != list(range(1, 10)):
        raise ValueError(f"expected ERA5 cells 1-9 in {ERA5_DIR}, found {sorted(coords)}")
    cells = {c: load_era5_dew(c) for c in coords}
    res = {"years": list(YEARS), "synoptic_utc": list(SYNOPTIC_UTC), "min_instants": MIN_INSTANTS,
           "min_season_days": MIN_SEASON_DAYS, "decades": DECADES, "n_boot": N_BOOT, "seed": SEED, "stations": {}}
    for name in COORDS:
        res["stations"][name] = run_station(name, cells, coords)
        print(f"{name}: done")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "station_humidity.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    write_report(res)
    print(f"wrote {OUT_DIR / 'station_humidity.md'}")


if __name__ == "__main__":
    main()
