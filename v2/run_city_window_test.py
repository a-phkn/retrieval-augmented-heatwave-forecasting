"""
H-A and H-A1 for the extra cities (PROTOCOL_V2 Section 7, amendment A4; exploratory). Same definitions as Delhi
except the alert population, which comes from the 9-cell ERA5 Tmax (no station series for these cities in the
repo) and the city's own rule (A4.2). Reads only <city>_hourly_dev.parquet (1980-2018).
Writes results_v2/h_a_<city>.json.

Run from repo root (after python -m v2.build_city_wbgt --city <city>):
    python -m v2.run_city_window_test --city chennai
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from v2.build_city_wbgt import OUT as WBGT_DIR
from v2.exposure import PRIMARY_LIMIT
from v2.graded import rest_hour_table, wcoverage
from v2.run_window_test import nine_cell_mean, nine_cell_tmax
from v2.window_test import (DEV_END, SEARCH_END, WINDOWS, WORK_HOURS, _classify, coverage, covers, departure,
                            doy_normal, era5_proxy_alert_days, jackknife, unsafe_hour_table)

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results_v2"
NORMAL_YEARS = (1981, 2010)
# A4.3: the window each city's own guidance names; compared against the best past window.
CITY_WINDOW = {"chennai": "W3 12-3 pm (IMD public)", "ahmedabad": "W1 12-4 pm (HAP p.155)"}


def candidates(length: int) -> tuple[tuple[int, int], ...]:
    """A4.3: windows of the comparator's length inside the working day, so a longer window never wins by length."""
    return tuple((s, s + length) for s in range(WORK_HOURS[0] - 1, WORK_HOURS[-1] - length + 1))


def best(u: pd.DataFrame, length: int, weighted: bool = False) -> tuple[int, int]:
    c = candidates(length)
    if weighted:
        return c[int(np.argmax([wcoverage(u, w) for w in c]))]
    hours = u["hour"].to_numpy()
    return c[int(np.argmax([int(covers(w, hours).sum()) for w in c]))]   # first maximum: earliest start


def two_day(met: pd.Series) -> pd.Series:
    """Criterion met on the day and the day before (IMD declaration); a known 'not met' day settles 'no'."""
    today, prev = met.astype(float), met.shift(1).astype(float)
    alert = pd.Series(np.nan, index=met.index)
    alert[(today == 1) & (prev == 1)] = 1.0
    alert[(today == 0) | (prev == 0)] = 0.0
    return alert.map({1.0: True, 0.0: False})


def alert_rules(city: str, tmax: pd.Series) -> dict:
    """A4.2. tmax: daily 9-cell ERA5 Tmax. Primary rule first."""
    t = tmax.reindex(pd.date_range(tmax.index.min(), tmax.index.max()))
    dep = departure(t, doy_normal(t, NORMAL_YEARS))
    known = t.notna()
    rules = {}
    if city == "chennai":
        rules["primary: IMD coastal (Tmax>=37 & dep>=4.5), 2 days"] = two_day((((t >= 37.0) & (dep >= 4.5))).where(known))
    else:
        rules["primary: Ahmedabad HAP Yellow (Tmax>=41.1)"] = (t >= 41.1).where(known)
        rules["IMD plains (Tmax>=40 & dep>=4.5 or Tmax>=45), 2 days"] = two_day(
            (((t >= 40.0) & (dep >= 4.5)) | (t >= 45.0)).where(known))
    rules["departure only (dep>=4.5), 2 days"] = two_day((dep >= 4.5).where(known))
    out = {k: _classify(t, v) for k, v in rules.items()}
    out["Delhi ERA5 proxy rule"] = era5_proxy_alert_days(tmax)
    return out


def h_a_city(mean: pd.DataFrame, alerts, comparator: str) -> dict:
    days, missing = alerts
    unsafe = unsafe_hour_table(mean[mean["time"].dt.year <= DEV_END], days, PRIMARY_LIMIT)
    out = {"alert_days": int(len(days)), "unclassified_days": int(missing["unclassified_days"].sum()),
           "alert_days_2007_2018": int((pd.DatetimeIndex(days).year > SEARCH_END).sum())}
    if unsafe.empty or unsafe[unsafe["date"].dt.year <= SEARCH_END].empty or unsafe[unsafe["date"].dt.year > SEARCH_END].empty:
        out["note"] = "no unsafe hours on alert days in the search or hold-out years; H-A not computable"
        return out
    search, hold = unsafe[unsafe["date"].dt.year <= SEARCH_END], unsafe[unsafe["date"].dt.year > SEARCH_END]
    w = WINDOWS[comparator]
    length = w[1] - w[0]
    bst = best(search, length)
    out.update({"comparator": comparator, "best_past_window": f"{bst[0]}-{bst[1]}",
                "primary_diff": asdict(jackknife(hold, lambda u: coverage(u, bst) - coverage(u, w))),
                "unsafe_hours_per_alert_day": float(len(unsafe) / max(1, len(days)))})
    for name, win in WINDOWS.items():
        out[f"share_outside {name}"] = asdict(jackknife(unsafe, lambda u, win=win: 1.0 - coverage(u, win)))
    # H-A1: rest-hours, moderate work, TLV, sun (A1 primary)
    r = rest_hour_table(mean[mean["time"].dt.year <= DEV_END], days, "moderate", "TLV")
    rs, rh = r[r["date"].dt.year <= SEARCH_END], r[r["date"].dt.year > SEARCH_END]
    if len(rs) and len(rh):
        bw = best(rs, length, weighted=True)
        out["rest"] = {"rest_hours_per_alert_day": float(r["weight"].sum() / max(1, len(days))),
                       "best_past_window": f"{bw[0]}-{bw[1]}",
                       "diff_best_minus_comparator": asdict(jackknife(rh, lambda x: wcoverage(x, bw) - wcoverage(x, w))),
                       **{f"share_rest_outside {n}": asdict(jackknife(r, lambda x, win=win: 1.0 - wcoverage(x, win)))
                          for n, win in WINDOWS.items()}}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", choices=tuple(CITY_WINDOW), required=True)
    a = ap.parse_args(argv)
    hourly = pd.read_parquet(WBGT_DIR / f"{a.city}_hourly_dev.parquet", columns=["time", "cell", "wbgt", "t"])
    if hourly["time"].dt.year.max() > DEV_END:
        raise RuntimeError("development file contains 2019+ rows; refusing (PROTOCOL_V2 Section 5)")
    mean = nine_cell_mean(hourly)
    tmax = nine_cell_tmax(hourly)
    out = {"city": a.city, "limit": PRIMARY_LIMIT, "alert_source": "9-cell ERA5 Tmax (A4.1)",
           "era5_tmax_in_season_mean_1981_2010": float(tmax[(tmax.index.year >= 1981) & (tmax.index.year <= 2010)
                                                             & (tmax.index.month.isin([4, 5, 6]))].mean())}
    for name, al in alert_rules(a.city, tmax).items():
        out[name] = h_a_city(mean, al, CITY_WINDOW[a.city])
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"h_a_{a.city}.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    for name in out:
        v = out[name]
        if isinstance(v, dict) and "primary_diff" in v:
            d = v["primary_diff"]
            print(f"{name}: {v['alert_days']} alert days; best {v['best_past_window']} vs {v['comparator']}: "
                  f"{d['estimate']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]; "
                  f"outside comparator {v['share_outside ' + v['comparator']]['estimate']:.2f}"
                  + (f"; rest-hours/day {v['rest']['rest_hours_per_alert_day']:.1f}" if "rest" in v else ""))
        elif isinstance(v, dict):
            print(f"{name}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
