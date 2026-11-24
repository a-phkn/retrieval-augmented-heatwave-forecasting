"""
Runs PROTOCOL_V2 amendment A1 (exploratory) for Delhi development years: H-A1 and H-E1, sun and shade WBGT,
every ACGIH work class at TLV (acclimatised) and Action Limit (unacclimatised). Primary row of the amendment:
moderate, TLV, sun. Writes results_v2/graded_delhi.json.

Run from repo root (after python -m v2.build_shade_wbgt):  python -m v2.run_graded
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from v2.build_delhi_hourly import OUT as WBGT_DIR
from v2.graded import audit_graded, h_a1
from v2.insurance_triggers import TRIGGERS
from v2.run_window_test import STATION
from v2.window_test import s2_alert_days
from v2.work_rest import ACGIH, daily_rest

RESULTS = Path(__file__).resolve().parents[1] / "results_v2"


def main() -> int:
    h = pd.read_parquet(WBGT_DIR / "delhi_hourly_dev_shade.parquet")
    if h["time"].dt.year.max() > 2018:
        raise RuntimeError("development file contains 2019+ rows; refusing")
    g = h.groupby("time")
    full = g["cell"].nunique() == 9
    mean = pd.DataFrame({"wbgt": g["wbgt"].mean(), "wbgt_shade": g["wbgt_shade"].mean()})[full].reset_index()
    alert, _ = s2_alert_days(pd.read_csv(STATION))
    out = {"h_a1": [], "h_e1": []}
    for level, classes in ACGIH.items():
        for work in classes:
            for col in ("wbgt", "wbgt_shade"):
                out["h_a1"].append(h_a1(mean, alert, work, level, col))
                dr = daily_rest(h, work, level, col)
                for trig in (t for t in TRIGGERS if t.city == "delhi"):
                    out["h_e1"].append({"work": work, "level": level, "wbgt": col, **audit_graded(dr, trig)})
    # A1.5: H-D event threshold R = 75th percentile of in-season daily rest-hours, 9-cell mean, 1980-2006
    m9 = mean.assign(cell=0, t=0.0)
    dr9 = daily_rest(m9, "moderate", "TLV", "wbgt")
    md = dr9["date"].dt.month * 100 + dr9["date"].dt.day
    base = dr9[(dr9["date"].dt.year <= 2006) & (md >= 315) & (md <= 930)]
    out["h_d_threshold_R"] = {"value": float(base["rest_hours"].quantile(0.75)), "n_days": int(len(base))}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "graded_delhi.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("H-A1, S2 alert days (moderate; TLV = acclimatised, AL = unacclimatised):")
    for r in out["h_a1"]:
        if r["work"] == "moderate":
            d, o = r["diff_best_minus_W1"], r["share_rest_outside W1 12-4 pm (HAP p.155)"]
            print(f"  {r['level']:3s} {r['wbgt']:10s} rest-h/alert-day {r['rest_hours_per_alert_day']:.1f}  "
                  f"outside 12-4 {o['estimate']:.3f} [{o['ci_low']:.3f}, {o['ci_high']:.3f}]  best {r['best_past_window']} "
                  f"diff {d['estimate']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]")
    print(f"A1.5 H-D threshold R (record in the protocol): {out['h_d_threshold_R']}")
    print("H-E1 (moderate):")
    e = pd.DataFrame(out["h_e1"]).query("work == 'moderate'")
    e = e.assign(rho_ci=e["spearman_ci95_year_jackknife"].map(lambda c: f"[{c[0]:+.2f}, {c[1]:+.2f}]"))
    print(e.drop(columns=["work", "cell_seasons", "spearman_ci95_year_jackknife"]).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
