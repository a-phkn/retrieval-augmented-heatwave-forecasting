"""
Runs H-A (PROTOCOL_V2) on Delhi development years (1980-2018) for every ISO 7243 limit; the primary result
is moderate work, acclimatised (28 C); S1, S3 and the ERA5 proxy are alert-population sensitivities. Reads only delhi_hourly_dev.parquet and the Safdarjung station file.
Writes results_v2/h_a_delhi.json.

Run from repo root (after python -m v2.build_delhi_hourly):  python -m v2.run_window_test
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from v2.build_delhi_hourly import OUT as WBGT_DIR
from v2.exposure import ISO_LIMITS
from v2.window_test import (WINDOWS, coverage, era5_proxy_alert_days, h_a, s1_alert_days, s2_alert_days,
                             s3_alert_days, unsafe_hour_table)
from v2.exposure import PRIMARY_LIMIT

REPO_ROOT = Path(__file__).resolve().parents[1]
STATION = REPO_ROOT / "sources" / "data" / "safdarjung_daily_tidy.csv"
RESULTS = REPO_ROOT / "results_v2"


def nine_cell_mean(hourly: pd.DataFrame) -> pd.DataFrame:
    g = hourly.groupby("time")
    m = g["wbgt"].mean()[g["cell"].nunique() == 9]
    return m.rename("wbgt").reset_index()


def nine_cell_tmax(hourly: pd.DataFrame) -> pd.Series:
    """Daily 9-cell ERA5 Tmax: mean over cells of each cell's daily maximum hourly 2 m temperature."""
    d = hourly.assign(date=hourly["time"].dt.normalize())
    per_cell = d.groupby(["date", "cell"])["t"].max().unstack()
    return per_cell[per_cell.notna().sum(axis=1) == 9].mean(axis=1)


def main() -> int:
    hourly = pd.read_parquet(WBGT_DIR / "delhi_hourly_dev.parquet", columns=["time", "cell", "wbgt", "t"])
    if hourly["time"].dt.year.max() > 2018:
        raise RuntimeError("development file contains 2019+ rows; refusing (PROTOCOL_V2 Section 5)")
    mean = nine_cell_mean(hourly)
    station = pd.read_csv(STATION)
    out = {}
    for work, (acc, unacc) in ISO_LIMITS.items():
        for status, limit in (("acclimatised", acc), ("unacclimatised", unacc)):
            out[f"{work}/{status}"] = h_a(mean, station, limit=limit)
    # Sensitivities on the alert population (primary limit only), PROTOCOL_V2 H-A
    sens = {"S1 IMD (2 days, Safdarjung)": s1_alert_days(station), "S3 HAP departure >= +4": s3_alert_days(station),
            "ERA5 9-cell Tmax proxy": era5_proxy_alert_days(nine_cell_tmax(hourly))}
    out["sensitivity_alert_population"] = {k: h_a(mean, None, PRIMARY_LIMIT, alerts=v) for k, v in sens.items()}
    # By decade (descriptive trend): share of unsafe hours outside each window, S2 primary
    u = unsafe_hour_table(mean, s2_alert_days(station)[0], PRIMARY_LIMIT)
    dec = (u["date"].dt.year // 10) * 10
    out["by_decade_share_outside"] = {f"{d}s": {"n_unsafe_hours": int((dec == d).sum()),
                                                **{n: 1.0 - coverage(u[dec == d], w) for n, w in WINDOWS.items()}}
                                      for d in sorted(dec.unique())}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "h_a_delhi.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    p = out["moderate/acclimatised"]
    d = p["primary_diff"]
    print(f"Primary (moderate, acclimatised, 28 C): alert days {p['alert_days']}, unclassified {p['unclassified_days']}")
    print(f"  best past window {p['best_past_window']}; coverage(best) - coverage(12-4) on 2007-2018 = "
          f"{d['estimate']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}], {d['n_clusters']} clusters -> "
          f"{'SUPPORTED' if p['supported'] else 'not supported'}")
    for k, v in p.items():
        if k.startswith("share_outside"):
            print(f"  {k[14:]}: {v['estimate']:.3f} [{v['ci_low']:.3f}, {v['ci_high']:.3f}]")
    for k, v in out["sensitivity_alert_population"].items():
        d = v["primary_diff"]
        print(f"  [{k}] alert days {v['alert_days']}, best {v['best_past_window']}, diff {d['estimate']:+.3f} "
              f"[{d['ci_low']:+.3f}, {d['ci_high']:+.3f}] -> {'supported' if v['supported'] else 'not supported'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
