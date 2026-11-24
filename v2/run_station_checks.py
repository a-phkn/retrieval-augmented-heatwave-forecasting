"""
PROTOCOL_V2 amendment A5 (exploratory): station checks for the ERA5-based city results.
  1. ERA5 vs station Tmax (in-season, 1980-2018, days both exist): mean difference, and the share of days above
     each insurance threshold, station vs 9-cell ERA5 vs single-cell ERA5 average (what the audits used).
  2. Chennai H-A on station alert days (Minambakkam, IMD coastal rule, 2 days), A4 definitions otherwise.
  3. Insurance audits (rest-hours, moderate TLV sun) with the trigger evaluated on STATION Tmax, for the
     station-years with >= 90% of trigger-period days present; the ERA5 per-cell audit on the same years beside it.
Reads only development years (<= 2018). Writes results_v2/station_checks.json.

Run from repo root:  python -m v2.run_station_checks
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from v2.build_city_wbgt import OUT as CITY_DIR
from v2.build_delhi_hourly import OUT as DELHI_DIR
from v2.calibration_v2 import in_season
from v2.graded import audit_graded
from v2.insurance_triggers import TRIGGERS, in_period
from v2.run_city_window_test import CITY_WINDOW, h_a_city, two_day
from v2.run_window_test import STATION as SAFDARJUNG, nine_cell_mean, nine_cell_tmax
from v2.window_test import DEV_END, _classify, departure, doy_normal
from v2.work_rest import daily_rest

REPO_ROOT = Path(__file__).resolve().parents[1]
GHCN = REPO_ROOT / "sources" / "data" / "ghcn"
RESULTS = REPO_ROOT / "results_v2"
STATIONS = {"ahmedabad": "IN005010600", "chennai": "IN020040900"}   # A5.1; Nungambakkam IN020070300 has no TMAX
MIN_PERIOD_COVERAGE = 0.90


def ghcn_tmax(station_id: str) -> pd.Series:
    """GHCN-D access CSV: TMAX in tenths of deg C; attributes 'mflag,qflag,sflag'; quality-flagged -> NaN."""
    d = pd.read_csv(GHCN / f"{station_id}.csv", usecols=["DATE", "TMAX", "TMAX_ATTRIBUTES"], dtype=str)
    d["date"] = pd.to_datetime(d["DATE"])
    d = d[d["date"].dt.year <= DEV_END]
    t = pd.to_numeric(d["TMAX"], errors="coerce") / 10.0
    q = d["TMAX_ATTRIBUTES"].fillna(",,").str.split(",").str[1].fillna("").str.strip()
    return pd.Series(t.where(q == "").to_numpy(), index=d["date"]).dropna().sort_index()


def safdarjung_tmax() -> pd.Series:
    s = pd.read_csv(SAFDARJUNG, parse_dates=["date"]).set_index("date").sort_index()
    t = s["tmax_c"].where(s["tmax_qflag"].isna()).dropna()
    return t[t.index.year <= DEV_END]


def load(city: str) -> pd.DataFrame:
    src = DELHI_DIR / "delhi_hourly_dev.parquet" if city == "delhi" else CITY_DIR / f"{city}_hourly_dev.parquet"
    h = pd.read_parquet(src, columns=["time", "cell", "wbgt", "t"])
    if h["time"].dt.year.max() > DEV_END:
        raise RuntimeError("development file contains 2019+ rows; refusing")
    return h


def bias(hourly: pd.DataFrame, st: pd.Series, thresholds) -> dict:
    nine = nine_cell_tmax(hourly)
    per_cell = hourly.assign(date=hourly["time"].dt.normalize()).groupby(["date", "cell"])["t"].max().unstack()
    j = pd.DataFrame({"station": st, "era5_9cell": nine, "era5_cell_mean_of_exceed": np.nan}).dropna(subset=["station", "era5_9cell"])
    j = j[in_season(j.index)]
    pc = per_cell.reindex(j.index)
    out = {"days": int(len(j)), "station_mean": float(j["station"].mean()), "era5_9cell_mean": float(j["era5_9cell"].mean()),
           "mean_diff_era5_minus_station": float((j["era5_9cell"] - j["station"]).mean()),
           "q95_station": float(j["station"].quantile(0.95)), "q95_era5_9cell": float(j["era5_9cell"].quantile(0.95))}
    for thr in thresholds:
        out[f"share_days_above_{thr}"] = {"station": float((j["station"] > thr).mean()),
                                          "era5_9cell": float((j["era5_9cell"] > thr).mean()),
                                          "era5_single_cell_avg": float((pc > thr).mean(axis=1).mean())}
    return out


def chennai_station_h_a(hourly: pd.DataFrame, st: pd.Series) -> dict:
    t = st.reindex(pd.date_range("1980-01-01", f"{DEV_END}-12-31"))
    dep = departure(t, doy_normal(t, (1981, 2010)))
    known = t.notna()
    mean = nine_cell_mean(hourly)
    rules = {"primary: IMD coastal on Minambakkam (Tmax>=37 & dep>=4.5), 2 days":
             two_day(((t >= 37.0) & (dep >= 4.5)).where(known)),
             "departure only (dep>=4.5), 2 days": two_day((dep >= 4.5).where(known))}
    return {k: h_a_city(mean, _classify(t, v), CITY_WINDOW["chennai"]) for k, v in rules.items()}


def station_years(st: pd.Series, trig) -> list[int]:
    yrs = []
    for y in range(1980, DEV_END + 1):
        days = pd.date_range(f"{y}-{trig.start}", f"{y}-{trig.end}")
        if st.reindex(days).notna().mean() >= MIN_PERIOD_COVERAGE:
            yrs.append(y)
    return yrs


def insurance_station(city: str, hourly: pd.DataFrame, st: pd.Series) -> list[dict]:
    mean = nine_cell_mean(hourly).assign(cell=0, t=np.nan)
    dr_mean = daily_rest(mean.assign(t=0.0), "moderate", "TLV", "wbgt").drop(columns="tmax")
    dr_cells = daily_rest(hourly, "moderate", "TLV", "wbgt")
    out = []
    for trig in (x for x in TRIGGERS if x.city == city):
        yrs = station_years(st, trig)
        d = dr_mean.assign(tmax=st.reindex(dr_mean["date"]).to_numpy())
        d = d[d["date"].dt.year.isin(yrs)]
        d = d[~(in_period(d["date"], trig.start, trig.end) & d["tmax"].isna())]   # drop missing period days
        e = dr_cells[dr_cells["date"].dt.year.isin(yrs)]
        out.append({"trigger": trig.name, "station_years": len(yrs), "years": yrs,
                    "station_trigger_9cell_rest": audit_graded(d.fillna({"tmax": -99.0}), trig),
                    "era5_per_cell_same_years": audit_graded(e, trig)})
    return out


def main() -> int:
    out = {}
    thr = {"delhi": (40.0, 43.6, 45.27, 46.53, 47.0), "ahmedabad": (41.1, 43.2, 43.6, 44.1), "chennai": (37.0, 40.0)}
    st = {"delhi": safdarjung_tmax(), **{c: ghcn_tmax(i) for c, i in STATIONS.items()}}
    for city in ("delhi", "ahmedabad", "chennai"):
        h = load(city)
        out[city] = {"bias": bias(h, st[city], thr[city])}
        if city == "chennai":
            out[city]["h_a_station"] = chennai_station_h_a(h, st[city])
        else:
            out[city]["insurance_station"] = insurance_station(city, h, st[city])
        b = out[city]["bias"]
        print(f"{city}: ERA5-station {b['mean_diff_era5_minus_station']:+.2f} C over {b['days']} days; "
              + "; ".join(f">{k.split('_')[-1]}: st {v['station']:.3f} / 9c {v['era5_9cell']:.3f} / cell {v['era5_single_cell_avg']:.3f}"
                          for k, v in b.items() if k.startswith("share_days_above")))
        for r in out[city].get("insurance_station", []):
            s, e = r["station_trigger_9cell_rest"], r["era5_per_cell_same_years"]
            print(f"  {r['trigger']} ({r['station_years']} yrs): paid station {s['share_seasons_paid']:.2f} vs ERA5 {e['share_seasons_paid']:.2f}; "
                  f"top-quartile unpaid {s['share_top_quartile_rest_seasons_unpaid']:.2f} vs {e['share_top_quartile_rest_seasons_unpaid']:.2f}; "
                  f"rho {s['spearman_season_rest_vs_paid_days']:.2f} {np.round(s['spearman_ci95_year_jackknife'], 2)}")
        for k, v in out[city].get("h_a_station", {}).items():
            if "primary_diff" in v:
                d = v["primary_diff"]
                print(f"  {k}: {v['alert_days']} days ({v['alert_days_2007_2018']} in 2007-18); best {v['best_past_window']} vs "
                      f"{v['comparator']}: {d['estimate']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]; outside comparator "
                      f"{v['share_outside ' + v['comparator']]['estimate']:.2f}")
            else:
                print(f"  {k}: {v}")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "station_checks.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
