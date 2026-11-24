"""
Runs H-E (PROTOCOL_V2) on the development years of one city: ISO 7243 limits x the city's triggers.
Reads only <city>_hourly_dev.parquet (1980-2018); the test file is never opened here.
Writes results_v2/insurance_audit_<city>.csv and _by_decade.csv.

Run from repo root (after python -m v2.build_city_wbgt --city ahmedabad, or python -m v2.build_delhi_hourly):
    python -m v2.run_insurance_audit --city ahmedabad   (or --city delhi)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from v2.build_city_wbgt import OUT as WBGT_DIR
from v2.build_delhi_hourly import OUT as DELHI_DIR
from v2.exposure import ISO_LIMITS, daily_exposure
from v2.insurance_triggers import TRIGGERS, audit, audit_by_decade

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results_v2"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", choices=("ahmedabad", "delhi"), required=True)
    a = ap.parse_args(argv)
    src = DELHI_DIR / "delhi_hourly_dev.parquet" if a.city == "delhi" else WBGT_DIR / f"{a.city}_hourly_dev.parquet"
    hourly = pd.read_parquet(src)
    if hourly["time"].dt.year.max() > 2018:
        raise RuntimeError("development file contains 2019+ rows; refusing (PROTOCOL_V2 Section 5)")
    rows, decades = [], []
    for work, (acc, unacc) in ISO_LIMITS.items():
        for status, limit in (("acclimatised", acc), ("unacclimatised", unacc)):
            daily = daily_exposure(hourly, limit=limit)
            for trig in (t for t in TRIGGERS if t.city == a.city):
                tag = {"work_class": work, "status": status, "limit": limit}
                rows.append({**tag, **audit(daily, trig)})
                if work == "moderate" and status == "acclimatised":
                    decades.append(audit_by_decade(daily, trig).assign(**tag))
    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS / f"insurance_audit_{a.city}.csv", index=False)
    pd.concat(decades).to_csv(RESULTS / f"insurance_audit_{a.city}_by_decade.csv", index=False)
    primary = pd.DataFrame(rows).query("work_class == 'moderate' and status == 'acclimatised'")
    print(primary[["trigger", "unsafe_days", "share_unsafe_days_not_counted", "share_seasons_paid"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
