"""
Post-test exploration (PROTOCOL_V2 Section 18, written after test-lock-v2; descriptive only). Development years
1980-2018, Mar 15 - Sep 30. Per city: how many working days carry heat stress, and how many of those had a heat
alert. Heat-stress day = at least one working hour (06-20) with 9-cell mean WBGT >= 28 C (ISO 7243 moderate,
acclimatised), and separately rest-hours >= 9.0 (A1.6). Alert = the station rule where a station series exists
(Delhi S1 on Safdarjung; Chennai IMD coastal on Minambakkam), else the A4 ERA5 rule (Ahmedabad HAP Yellow).
ERA5 runs cool against the stations (A5), so heat-stress counts here are if anything low.
Writes results_v2/alert_gap.json. Run from repo root:  python -m v2_post.alert_gap
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from v2.build_city_wbgt import OUT as CITY_DIR
from v2.build_delhi_hourly import OUT as DELHI_DIR
from v2.calibration_v2 import in_season
from v2.run_city_window_test import alert_rules, two_day
from v2.run_station_checks import ghcn_tmax, safdarjung_tmax
from v2.run_window_test import nine_cell_mean, nine_cell_tmax
from v2.window_test import DEV_END, _classify, departure, doy_normal, s1_alert_days
from v2.work_rest import daily_rest
from v2.exposure import WORK_HOURS

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "results_v2" / "alert_gap.json"
R = 9.0


def alerts(city: str, hourly: pd.DataFrame) -> pd.DatetimeIndex:
    if city == "delhi":
        st = safdarjung_tmax().rename("tmax_c").rename_axis("date").reset_index()
        return s1_alert_days(st)[0]
    if city == "chennai":
        t = ghcn_tmax("IN020040900").reindex(pd.date_range("1980-01-01", f"{DEV_END}-12-31"))
        dep = departure(t, doy_normal(t, (1981, 2010)))
        return _classify(t, two_day(((t >= 37.0) & (dep >= 4.5)).where(t.notna())))[0]
    return alert_rules("ahmedabad", nine_cell_tmax(hourly))["primary: Ahmedabad HAP Yellow (Tmax>=41.1)"][0]


def main() -> int:
    out = {}
    for city in ("delhi", "ahmedabad", "chennai"):
        src = DELHI_DIR / "delhi_hourly_dev.parquet" if city == "delhi" else CITY_DIR / f"{city}_hourly_dev.parquet"
        h = pd.read_parquet(src, columns=["time", "cell", "wbgt", "t"])
        if h["time"].dt.year.max() > DEV_END:
            raise RuntimeError("development file contains 2019+ rows; refusing")
        m = nine_cell_mean(h)
        hh = m.assign(date=m["time"].dt.normalize(), hour=m["time"].dt.hour)
        w = hh[hh["hour"].isin(WORK_HOURS)]
        unsafe_hours = (w["wbgt"] >= 28.0).groupby(w["date"]).sum()
        rest = daily_rest(m.assign(cell=0, t=0.0), "moderate", "TLV", "wbgt").set_index("date")["rest_hours"]
        d = pd.DataFrame({"unsafe_hours": unsafe_hours, "rest": rest}).dropna()
        d = d[in_season(d.index) & (d.index.year <= DEV_END)]
        al = d.index.isin(alerts(city, h))
        res = {"season_days": int(len(d)), "alert_days": int(al.sum())}
        for name, mask in (("any_unsafe_hour", d["unsafe_hours"] >= 1), ("rest_hours_ge_R", d["rest"] >= R)):
            res[name] = {"days": int(mask.sum()), "share_of_season": float(mask.mean()),
                         "share_with_alert": float(al[mask.to_numpy()].mean()) if mask.any() else float("nan"),
                         "unsafe_hours_on_unalerted_days_share": float(d.loc[mask & ~al, "unsafe_hours"].sum() / max(1, d.loc[mask, "unsafe_hours"].sum()))}
        out[city] = res
        a, b = res["any_unsafe_hour"], res["rest_hours_ge_R"]
        print(f"{city}: {res['season_days']} season days, {res['alert_days']} alert days | days with >=1 unsafe hour "
              f"{a['share_of_season']:.0%}, of which alerted {a['share_with_alert']:.1%} | rest-hours >= 9: "
              f"{b['share_of_season']:.0%} of days, alerted {b['share_with_alert']:.1%}; unsafe hours on unalerted "
              f"heavy days {b['unsafe_hours_on_unalerted_days_share']:.0%}")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
