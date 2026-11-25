"""
P3-7 (context/decisions_phase3.md): station records rebuilt from NOAA's Integrated Surface Database (ISD).

It downloads the yearly raw files and writes their manifest, counts what the reports contain before any value is
read (step 1, the census), builds each station's daily series, and runs the validation of step 2. Steps 3 and 4
(the P3-3 rerun, the humidity check) read the daily files this module writes.

- Source: https://www.ncei.noaa.gov/pub/data/noaa/<YEAR>/<USAF>-<WBAN>-<YEAR>.gz, ISD's raw archive. A made-up
  station number there returns 404; the "global-hourly access" path returns an HTML page for any number, so it is not
  used.
- Stations, 1980-2025:
  - Safdarjung 421820-99999 and Palam 421810-99999 (Delhi);
  - Ahmedabad 426470-99999 and Chennai 432790-99999 (added by the P3-7 deviation of 2026-11-25).
- Storage: files go to data/raw/noaa_isd/<USAF>-<WBAN>/ (gitignored, like every raw download).
- A reply is kept only if it is gzip data.
- The manifest, followup/data/isd_manifest.json (committed), gives each file's URL, size and SHA-256. It also lists
  the two GHCN files received from the experiments-v2 study.
- A file already on disk with its manifest hash is not fetched again, so an interrupted run resumes.

Run from repo root:
    python -m followup.isd_rebuild --download
    python -m followup.isd_rebuild --census   -> followup/results/station_rebuild_census.json (no value read)
    python -m followup.isd_rebuild --build    -> followup/data/isd_daily_<station>.csv, followup/results/station_rebuild.{md,json}
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw" / "noaa_isd"
MANIFEST = REPO_ROOT / "followup" / "data" / "isd_manifest.json"
BASE_URL = "https://www.ncei.noaa.gov/pub/data/noaa"
STATIONS = {"safdarjung": "421820-99999", "palam": "421810-99999",
            "ahmedabad": "426470-99999", "chennai": "432790-99999"}
YEARS = range(1980, 2026)
GZIP_MAGIC = b"\x1f\x8b"
# The GHCN-Daily files the experiments-v2 study used, received from it on 2026-11-25 (P3-7 part B), kept here.
GHCN_DIR = REPO_ROOT / "data" / "raw" / "ghcn_experiments_v2"
GHCN_RECEIVED = {
    "IN005010600": ("Ahmedabad", "0f09273e57ed7a74b9315309511eb6412605508e0e936115a60894498fbccdf8"),
    "IN020040900": ("Chennai, Minambakkam", "e2f6e651c8aca0332b99a6097a5b75e4a388b694b8cda994c6c133f878feb381"),
}


def url(station: str, year: int) -> str:
    """The raw-archive URL of one station-year."""
    return f"{BASE_URL}/{year}/{station}-{year}.gz"


def raw_path(station: str, year: int, root: Path = RAW_DIR) -> Path:
    """Where one station-year is stored."""
    return root / station / f"{station}-{year}.gz"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)  # atomic on the same filesystem


def fetch(session, u: str, retries: int = 4, sleep=time.sleep) -> bytes | None:
    """The file's bytes, or None if the archive has no such file (404). Timeouts, connection errors, 429 and server
    errors are retried; a 200 reply that is not gzip data (an HTML page) raises ValueError."""
    attempts = 0
    while True:
        try:
            r = session.get(u, timeout=120)
        except requests.RequestException:
            attempts += 1
            if attempts >= retries:
                raise
            sleep(10 * attempts)
            continue
        if r.status_code == 404:
            return None
        if r.status_code == 429 or r.status_code >= 500:
            attempts += 1
            if attempts >= retries:
                r.raise_for_status()
            sleep(10 * attempts)
            continue
        r.raise_for_status()
        if not r.content.startswith(GZIP_MAGIC):
            raise ValueError(f"{u}: the reply is not gzip data")
        return r.content


def _ghcn_entries(root: Path = GHCN_DIR) -> dict:
    """The received GHCN files with their expected hash, and whether the copy on disk matches it."""
    out = {}
    for sid, (name, expected) in GHCN_RECEIVED.items():
        p = root / f"{sid}.csv"
        out[sid] = {"name": name, "sha256": expected, "received": "2026-11-25, from the experiments-v2 study",
                    "on_disk_matches": p.exists() and sha256(p.read_bytes()) == expected}
    return out


def download(session, stations: dict[str, str] = STATIONS, years=YEARS, root: Path = RAW_DIR,
             manifest_path: Path = MANIFEST, ghcn_root: Path = GHCN_DIR, pause: float = 0.5, sleep=time.sleep) -> dict:
    """Fetch every station-year not on disk with its manifest hash, then write the manifest (also when stopped
    part-way, so a re-run resumes). A year the archive does not have is recorded as missing. Returns the manifest."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    files = manifest.setdefault("isd_files", {})
    try:
        for name, st in stations.items():
            for y in years:
                key, p = f"{st}-{y}", raw_path(st, y, root)
                if p.exists() and files.get(key, {}).get("sha256") == sha256(p.read_bytes()):
                    continue
                data = fetch(session, url(st, y), sleep=sleep)
                if data is None:
                    files[key] = {"station": name, "url": url(st, y), "missing": True}
                else:
                    _write_atomic(p, data)
                    files[key] = {"station": name, "url": url(st, y), "bytes": len(data), "sha256": sha256(data)}
                sleep(pause)
    finally:
        manifest["source"] = BASE_URL + "/<YEAR>/<USAF>-<WBAN>-<YEAR>.gz"
        manifest["ghcn_received"] = _ghcn_entries(ghcn_root)
        manifest["isd_files"] = dict(sorted(files.items()))
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


# ------------------------------------------------------------------ reading the reports
# Positions from the ISD format document (1-based): date 16-23, UTC time 24-27, report type 42-46, air temperature
# 88-92 with its quality code 93, dew point 94-98 with its quality code 99; the additional data section starts with
# "ADD" at 106-108 and ends where the remarks (REM), element-quality (EQD) or original-observation (QNN) part begins.

USE_QUALITY = frozenset("01459ACIMPRU")  # passed or accepted; 2, 3, 6, 7 (suspect, erroneous) are dropped
TYPE_RANK = {"FM-12": 0, "FM-15": 1, "FM-16": 2}  # SYNOP, METAR, SPECI; any other type ranks 3
KA_RE = re.compile(r"KA[1-4](\d{3})([MNOP9])([+-]\d{4})(.)")  # period (tenths of h), code, temperature, quality
IST = pd.Timedelta(hours=5, minutes=30)
FIRST_DAY, LAST_DAY = pd.Timestamp("1980-01-01"), pd.Timestamp("2025-08-24")
DAILY_DIR = REPO_ROOT / "followup" / "data"
RESULTS = REPO_ROOT / "followup" / "results"


def additional(line: str) -> str:
    """The additional data section of a report line, without remarks and quality parts; '' if there is none."""
    if line[105:108] != "ADD":
        return ""
    add = line[108:]
    for marker in ("REM", "EQD", "QNN"):
        i = add.find(marker)
        if i != -1:
            add = add[:i]
    return add


def value(field: str, quality: str) -> float:
    """A signed temperature field in tenths of °C, as °C; NaN if missing or its quality code is not usable."""
    if field in ("+9999", "-9999") or quality not in USE_QUALITY:
        return float("nan")
    return int(field) / 10.0


def extremes(line: str) -> list[tuple[float, str, str, str]]:
    """The KA groups of a report: (period in hours or NaN, code, raw temperature field, quality code). The fields
    stay raw, so the census can count groups without converting a value."""
    return [(int(p) / 10.0 if p != "999" else float("nan"), c, t, q) for p, c, t, q in KA_RE.findall(additional(line))]


def _structure(paths: list[Path]) -> pd.DataFrame:
    """One row per report line, structure only (no value): UTC time, type, file order, KA codes and periods."""
    rows, order = [], 0
    for path in paths:
        with gzip.open(path, "rt", encoding="latin-1") as f:
            for line in f:
                groups = extremes(line)
                rows.append((line[15:27], line[41:46].strip(), order,
                             tuple(p for p, c, _, _ in groups if c == "M"), tuple(p for p, c, _, _ in groups if c == "N")))
                order += 1
    df = pd.DataFrame(rows, columns=["stamp", "type", "order", "m_periods", "n_periods"])
    df["time"] = pd.to_datetime(df["stamp"], format="%Y%m%d%H%M")
    return df.drop(columns="stamp")


def select(df: pd.DataFrame) -> pd.DataFrame:
    """One report per UTC time: FM-12 first, then FM-15, then FM-16, then any other type; within a type the first in
    file order."""
    rank = df["type"].map(TYPE_RANK).fillna(3)
    return (df.assign(_rank=rank).sort_values(["time", "_rank", "order"]).drop_duplicates("time")
            .drop(columns="_rank").reset_index(drop=True))


def census(paths: list[Path]) -> dict:
    """P3-7 step 1, before any value is read: what the reports contain (types, minutes, hours per day, KA groups)."""
    raw = _structure(paths)
    sel = select(raw)
    full_hours = sel[sel["time"].dt.minute == 0]
    per_day = full_hours.groupby(full_hours["time"].dt.normalize()).size()
    ka = Counter()
    for t, ms, ns in zip(sel["time"], sel["m_periods"], sel["n_periods"]):
        for code, periods in (("M", ms), ("N", ns)):
            for p in periods:
                ka[f"{t.hour:02d} UTC {code} {p:g}h"] += 1
    at = {h: sel[(sel["time"].dt.hour == h) & (sel["time"].dt.minute == 0)] for h in (3, 12)}
    years = sorted(raw["time"].dt.year.unique())
    return {
        "reports": int(len(raw)), "selected": int(len(sel)), "years": [int(y) for y in years],
        "types": {k: int(v) for k, v in raw["type"].value_counts().items()},
        "minutes": {f"{k:02d}": int(v) for k, v in raw["time"].dt.minute.value_counts().head(6).items()},
        "full_hours_per_day_median_by_year": {int(y): float(g.median())
                                              for y, g in per_day.groupby(per_day.index.year)},
        "ka_groups": dict(ka.most_common(12)),
        "12utc_reports": {"n": int(len(at[12])), "with_M": int(at[12]["m_periods"].map(len).gt(0).sum()),
                          "with_2plus_M": int(at[12]["m_periods"].map(len).gt(1).sum())},
        "03utc_reports": {"n": int(len(at[3])), "with_N": int(at[3]["n_periods"].map(len).gt(0).sum()),
                          "with_2plus_N": int(at[3]["n_periods"].map(len).gt(1).sum())},
    }


def read_reports(paths: list[Path]) -> pd.DataFrame:
    """The selected reports with their values: UTC time, type, t and td (°C, NaN unless usable), and the first usable
    code-M and code-N KA values with their periods. Estimated extremes (codes O and P) are not used."""
    rows, order = [], 0
    for path in paths:
        with gzip.open(path, "rt", encoding="latin-1") as f:
            for line in f:
                m = n = (float("nan"), float("nan"))
                for period, code, field, q in extremes(line):
                    v = value(field, q)
                    if code == "M" and np.isnan(m[0]) and not np.isnan(v):
                        m = (v, period)
                    elif code == "N" and np.isnan(n[0]) and not np.isnan(v):
                        n = (v, period)
                rows.append((line[15:27], line[41:46].strip(), order, value(line[87:92], line[92]),
                             value(line[93:98], line[98]), *m, *n))
                order += 1
    df = pd.DataFrame(rows, columns=["stamp", "type", "order", "t", "td", "tmax_ka", "tmax_period", "tmin_ka",
                                     "tmin_period"])
    df["time"] = pd.to_datetime(df["stamp"], format="%Y%m%d%H%M")
    return select(df.drop(columns="stamp"))


def daily(rep: pd.DataFrame, first: pd.Timestamp = FIRST_DAY, last: pd.Timestamp = LAST_DAY) -> pd.DataFrame:
    """P3-7's daily values per IST day (see the entry): reported and hourly Tmax and Tmin, the dew point mean, and the
    hourly counts behind them. No gap is filled."""
    days = pd.date_range(first, last, freq="D")
    hours = pd.date_range(first - pd.Timedelta(hours=6), last + pd.Timedelta(hours=18), freq="h")  # 18 UTC D-1 on
    h = rep[rep["time"].dt.minute == 0].set_index("time")[["t", "td"]].reindex(hours)
    ist_day = (h.index + IST).normalize()
    g = h.groupby(ist_day)
    n_t, n_td = g["t"].count().reindex(days), g["td"].count().reindex(days)
    after = h["t"][(h.index.hour >= 6) & (h.index.hour <= 11)]
    after_n = after.groupby(after.index.normalize()).count().reindex(days)  # 06-11 UTC of D = 11:30-16:30 IST of D
    night = h["t"][(h.index.hour >= 21) | (h.index.hour <= 3)]
    night_day = (night.index + pd.Timedelta(hours=3)).normalize()  # 21 UTC D-1 .. 03 UTC D -> D
    gn = night.groupby(night_day)
    at = rep[rep["time"].dt.minute == 0].set_index("time")
    rep12, rep03 = at[at.index.hour == 12].copy(), at[at.index.hour == 3].copy()
    rep12.index, rep03.index = rep12.index.normalize(), rep03.index.normalize()  # 17:30 and 08:30 IST of that date
    out = pd.DataFrame(index=days.rename("date"))
    out["tmax_reported"] = rep12["tmax_ka"].reindex(days)
    out["tmax_reported_period_h"] = rep12["tmax_period"].reindex(days)
    out["tmax_hourly"] = g["t"].max().reindex(days).where((n_t >= 18) & (after_n == 6))
    out["tmin_reported"] = rep03["tmin_ka"].reindex(days)
    out["tmin_reported_period_h"] = rep03["tmin_period"].reindex(days)
    out["tmin_hourly"] = gn.min().reindex(days).where(gn.count().reindex(days) >= 6)
    out["td_mean"] = g["td"].mean().reindex(days).where(n_td >= 18)
    out["hours_t"], out["hours_td"] = n_t.fillna(0).astype(int), n_td.fillna(0).astype(int)
    return out


# ------------------------------------------------------------------ step 2: validation


def lag_mad(a: pd.Series, b: pd.Series) -> dict[int, float]:
    """Mean absolute difference between a on day D and b on day D + lag (lags -1, 0, +1), over days both have."""
    return {lag: float((a - b.shift(-lag)).abs().mean()) for lag in (-1, 0, 1)}


def self_check(d: pd.DataFrame) -> dict:
    """Step 2(a): the reported Tmax must match the station's own hourly Tmax best at lag 0, and may fall more than
    0.5 °C below it on at most 5% of the days both have."""
    both = d["tmax_reported"].notna() & d["tmax_hourly"].notna()
    if both.sum() == 0:
        return {"days": 0, "passes": None, "note": "no day has both values"}
    mad = lag_mad(d["tmax_reported"], d["tmax_hourly"])
    below = float((d.loc[both, "tmax_reported"] < d.loc[both, "tmax_hourly"] - 0.5).mean())
    return {"days": int(both.sum()), "mad_by_lag": mad, "share_below_hourly_by_0.5": below,
            "mean_reported_minus_hourly": float((d.loc[both, "tmax_reported"] - d.loc[both, "tmax_hourly"]).mean()),
            "passes": bool(mad[0] < mad[-1] and mad[0] < mad[1] and below <= 0.05)}


def ghcn_check(rebuilt: pd.Series, ghcn: pd.Series, years: tuple[int, int] = (1980, 2008),
               rule: bool = True) -> dict:
    """Step 2(b): the rebuilt series against GHCN on the years given, days both have. Passes (if rule) when the lag-0
    mean absolute difference is the smallest, the mean difference is within ±0.3 °C and at least 80% of days agree
    within 0.5 °C."""
    span = lambda s: s[(s.index.year >= years[0]) & (s.index.year <= years[1])]  # noqa: E731
    a, b = span(rebuilt), span(ghcn.reindex(rebuilt.index))
    both = a.notna() & b.notna()
    diff = (a - b)[both]
    mad = lag_mad(a, b)
    out = {"years": list(years), "days": int(both.sum()), "mad_by_lag": mad, "mean_rebuilt_minus_ghcn": float(diff.mean()),
           "share_within_0.5": float((diff.abs() <= 0.5).mean())}
    if rule:
        out["passes"] = bool(mad[0] < mad[-1] and mad[0] < mad[1] and abs(out["mean_rebuilt_minus_ghcn"]) <= 0.3
                             and out["share_within_0.5"] >= 0.80)
    return out


def ghcn_safdarjung() -> pd.DataFrame:
    """Safdarjung's GHCN-Daily tidy file (P3-3), TMAX and TMIN in °C; quality-flagged TMAX values dropped."""
    from followup.station_check import STATION_CSV, STATION_SHA256

    if hashlib.sha256(STATION_CSV.read_bytes()).hexdigest() != STATION_SHA256:
        raise ValueError("the Safdarjung GHCN file is not the one P3-3 registered")
    s = pd.read_csv(STATION_CSV, parse_dates=["date"]).set_index("date").sort_index()
    return pd.DataFrame({"tmax": s["tmax_c"].where(s["tmax_qflag"].isna()), "tmin": s["tmin_c"]})


def paths_for(station: str, root: Path = RAW_DIR) -> list[Path]:
    return sorted((root / station).glob(f"{station}-*.gz"))


def coverage(d: pd.DataFrame) -> dict:
    """Days with each daily value, per block of years."""
    blocks = {"1980-2008": (1980, 2008), "2009-2010": (2009, 2010), "2011-2015": (2011, 2015), "2016-2025": (2016, 2025)}
    cols = ["tmax_reported", "tmax_hourly", "tmin_reported", "tmin_hourly", "td_mean"]
    return {k: {c: int(d.loc[(d.index.year >= a) & (d.index.year <= b), c].notna().sum()) for c in cols}
            | {"days": int(((d.index.year >= a) & (d.index.year <= b)).sum())} for k, (a, b) in blocks.items()}


def build() -> dict:
    """Read every station, write its daily file, and run step 2. Returns the results (also written as JSON)."""
    res = {}
    for name, st in STATIONS.items():
        d = daily(read_reports(paths_for(st)))
        DAILY_DIR.mkdir(parents=True, exist_ok=True)
        d.round(2).to_csv(DAILY_DIR / f"isd_daily_{name}.csv", date_format="%Y-%m-%d")
        res[name] = {"station": st, "coverage": coverage(d), "self_check": self_check(d)}
        if name == "safdarjung":
            g = ghcn_safdarjung()
            res[name]["ghcn_check_tmax"] = ghcn_check(d["tmax_reported"], g["tmax"])
            res[name]["ghcn_check_tmin"] = ghcn_check(d["tmin_reported"], g["tmin"], rule=False)
        print(f"{name}: written; self-check {res[name]['self_check'].get('passes')}", flush=True)
    return res


def write_report(res: dict, cen: dict | None = None) -> None:
    def f(x, fmt="{:.2f}"):
        return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else fmt.format(x)

    L = ["# P3-7: station records rebuilt from NOAA ISD", "",
         "Pre-registered in `context/decisions_phase3.md` (P3-7, 2026-11-25). Daily files: "
         "`followup/data/isd_daily_<station>.csv` (IST days). Step 1 (the census) is in "
         "`followup/results/station_rebuild_census.json`.", "",
         "## Days with each value", "",
         "| Station | Years | Days | Tmax reported | Tmax hourly | Tmin reported | Tmin hourly | Dew point mean |",
         "|---|---|---|---|---|---|---|---|"]
    for name, r in res.items():
        for block, c in r["coverage"].items():
            L.append(f"| {name} | {block} | {c['days']} | {c['tmax_reported']} | {c['tmax_hourly']} | "
                     f"{c['tmin_reported']} | {c['tmin_hourly']} | {c['td_mean']} |")
    L += ["", "## Step 2(a): each station against itself", "",
          "Reported Tmax against the station's own hourly Tmax, on days both exist. Passes if the lag-0 mean absolute "
          "difference is the smallest and the reported value is more than 0.5 °C below the hourly one on at most 5% "
          "of days.", "",
          "| Station | Days | MAD lag -1 / 0 / +1 (°C) | Reported minus hourly (°C) | Share > 0.5 °C below | Passes |",
          "|---|---|---|---|---|---|"]
    for name, r in res.items():
        s = r["self_check"]
        if not s.get("days"):
            L.append(f"| {name} | 0 | - | - | - | {s.get('note')} |")
            continue
        m = s["mad_by_lag"]
        L.append(f"| {name} | {s['days']} | {f(m[-1])} / {f(m[0])} / {f(m[1])} | "
                 f"{f(s['mean_reported_minus_hourly'], '{:+.2f}')} | {f(s['share_below_hourly_by_0.5'], '{:.1%}')} | "
                 f"{s['passes']} |")
    g, gn = res["safdarjung"].get("ghcn_check_tmax", {}), res["safdarjung"].get("ghcn_check_tmin", {})
    L += ["", "## Step 2(b): Safdarjung against GHCN, 1980-2008", "",
          "Passes if the lag-0 mean absolute difference is the smallest, the mean difference is within ±0.3 °C and at "
          "least 80% of days agree within 0.5 °C. Tmin is descriptive.", "",
          "| Series | Days | MAD lag -1 / 0 / +1 (°C) | Rebuilt minus GHCN (°C) | Within 0.5 °C | Passes |",
          "|---|---|---|---|---|---|"]
    for label, t in (("Tmax reported", g), ("Tmin reported", gn)):
        if t:
            m = t["mad_by_lag"]
            L.append(f"| {label} | {t['days']} | {f(m[-1])} / {f(m[0])} / {f(m[1])} | "
                     f"{f(t['mean_rebuilt_minus_ghcn'], '{:+.2f}')} | {f(t['share_within_0.5'], '{:.1%}')} | "
                     f"{t.get('passes', 'descriptive')} |")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "station_rebuild.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (RESULTS / "station_rebuild.json").write_text(json.dumps(res, indent=1, default=float) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="P3-7: station records rebuilt from NOAA ISD.")
    ap.add_argument("--download", action="store_true", help="fetch the yearly raw files and write the manifest")
    ap.add_argument("--census", action="store_true", help="step 1: count what the reports contain (no value read)")
    ap.add_argument("--build", action="store_true", help="daily files and the step 2 checks")
    args = ap.parse_args(argv)
    if args.download:
        with requests.Session() as s:
            m = download(s)
        got = [v for v in m["isd_files"].values() if not v.get("missing")]
        print(f"{len(got)} files on disk ({sum(v['bytes'] for v in got) / 1e6:.1f} MB), "
              f"{sum(1 for v in m['isd_files'].values() if v.get('missing'))} missing in the archive; "
              f"manifest: {MANIFEST.relative_to(REPO_ROOT).as_posix()}", flush=True)
    if args.census:
        cen = {name: census(paths_for(st)) for name, st in STATIONS.items()}
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / "station_rebuild_census.json").write_text(json.dumps(cen, indent=1) + "\n", encoding="utf-8")
        for name, c in cen.items():
            print(f"{name}: {c['reports']} reports; types {c['types']}; 12 UTC {c['12utc_reports']}; "
                  f"03 UTC {c['03utc_reports']}; KA {c['ka_groups']}", flush=True)
    if args.build:
        res = build()
        write_report(res)
        print(f"wrote {(RESULTS / 'station_rebuild.md').relative_to(REPO_ROOT).as_posix()}", flush=True)


if __name__ == "__main__":
    main()
