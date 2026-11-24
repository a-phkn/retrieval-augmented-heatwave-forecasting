"""
Scores the v2 test period ONCE (PROTOCOL_V2 Section 5 and amendment A6). Secondary: 2019+ was opened by v1.
Model: the test-fold WBGT C3-hop (trained 1980-2016, early-stopped on 2017-2018, seeds 0-9), from
predictions_test/A2Lr_hw5_C3hop/: test_stop.parquet = 2017-2018 (calibration block, as each development
fold's inner block), test.parquet = 2019+.
  E5    lead 1: miss < 0.25 one-sided at 0.05 AND false alarms <= 0.40 (A6.2); leads 2-5 reported.
  H-C   leads 1-5: coverage and TOST, reported (power about 7%, A6.1).
  M-P   model minus persistence, Peirce, leads 1-5, Holm across leads (A3.2).
Refuses to run if results_v2/test_v2.json exists. Run from repo root:  python -m v2.run_test_v2
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from v2 import test_lock_v2
from v2.build_delhi_hourly import OUT as WBGT_DIR
from v2.calibration_v2 import ALPHA, e5, h_c, holm, in_season
from v2.e5_vs_persistence import compare
from v2.risk_control import crc_threshold
from v2.run_calibration_v2 import R
from v2.work_rest import daily_rest

REPO_ROOT = Path(__file__).resolve().parents[1]
PRED = REPO_ROOT / "predictions_test" / "A2Lr_hw5_C3hop"
OUT = REPO_ROOT / "results_v2" / "test_v2.json"
CAL_YEARS = (2017, 2018)


def rest_hours(path: Path) -> pd.Series:
    h = pd.read_parquet(path, columns=["time", "cell", "wbgt"])
    g = h.groupby("time")
    m = g["wbgt"].mean()[g["cell"].nunique() == 9].rename("wbgt").reset_index().assign(cell=0, t=0.0)
    return daily_rest(m, "moderate", "TLV", "wbgt").set_index("date")["rest_hours"]


def model_fire(inner: pd.DataFrame, val: pd.DataFrame, ev: pd.Series, lead: int) -> pd.Series:
    out = []
    i, v = inner[inner["lead"] == lead], val[val["lead"] == lead]
    for seed, vs in v.groupby("seed"):
        c = i[(i["seed"] == seed) & in_season(i["target_date"]) & i["target_date"].isin(ev.index)]
        lam = crc_threshold(c["pred"].to_numpy(), ev.reindex(c["target_date"]).to_numpy(), ALPHA)
        vs = vs[in_season(vs["target_date"]) & vs["target_date"].isin(ev.index)]
        out.append(pd.DataFrame({"target_date": vs["target_date"].to_numpy(), "fire": vs["pred"].to_numpy() >= lam}))
    return pd.concat(out).groupby("target_date")["fire"].mean()


def persistence_fire(rest: pd.Series, ev: pd.Series, lead: int) -> pd.Series:
    sc = rest.shift(lead, freq="D").reindex(rest.index)
    ok = in_season(rest.index) & sc.notna().to_numpy()
    cal = ok & (rest.index.year >= CAL_YEARS[0]) & (rest.index.year <= CAL_YEARS[1])
    test = ok & (rest.index.year > CAL_YEARS[1])
    lam = crc_threshold(sc[cal].to_numpy(), ev[cal].to_numpy(), ALPHA)
    return (sc[test] >= lam).astype(float)


def main() -> int:
    test_lock_v2.require()
    if OUT.exists():
        raise SystemExit(f"{OUT} exists: the test period is scored once (PROTOCOL_V2 Section 8)")
    inner = pd.read_parquet(PRED / "test_stop.parquet")
    val = pd.read_parquet(PRED / "test.parquet")
    if inner["target_date"].dt.year.min() < CAL_YEARS[0] or inner["target_date"].dt.year.max() > CAL_YEARS[1]:
        raise RuntimeError("test_stop.parquet is not the 2017-2018 block")
    rest = pd.concat([rest_hours(WBGT_DIR / "delhi_hourly_dev_shade.parquet").loc["2016-12-01":],
                      rest_hours(WBGT_DIR / "delhi_hourly_test.parquet")]).sort_index()
    rest = rest[~rest.index.duplicated()]
    ev = rest >= R
    folds = [(inner, val)]
    out = {"R": R, "test_years": [int(val["target_date"].dt.year.min()), int(val["target_date"].dt.year.max())],
           "event_base_rate_test_in_season": float(ev[(ev.index.year > 2018) & in_season(ev.index)].mean()),
           "e5": {}, "h_c": {}, "model_vs_persistence": {}}
    for lead in range(1, 6):
        out["e5"][lead] = e5(folds, ev, lead)
        out["h_c"][lead] = h_c(folds, lead)
        out["model_vs_persistence"][lead] = compare(rest[rest.index.year > CAL_YEARS[1]],
                                                    model_fire(inner, val, ev, lead), persistence_fire(rest, ev, lead), lead)
    e1 = out["e5"][1]
    out["e5_lead1_pass"] = bool(e1["p_miss_below_alpha_plus_tol"] < 0.05 and e1["false_alarm_ok"])
    out["model_vs_persistence_holm"] = holm({str(k): v["peirce_diff"]["p_two_sided"]
                                             for k, v in out["model_vs_persistence"].items()})
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"test years {out['test_years']}, event base rate {out['event_base_rate_test_in_season']:.3f}")
    for lead in range(1, 6):
        e, h, m = out["e5"][lead], out["h_c"][lead], out["model_vs_persistence"][lead]
        print(f"lead {lead}: E5 miss {e['miss_rate']['estimate']:.3f} (se {e['miss_rate']['se']:.3f}, "
              f"{e['miss_rate']['n_clusters']} cl, {e['event_days']} ev) p {e['p_miss_below_alpha_plus_tol']:.4f} "
              f"FA {e['false_alarm_rate']:.3f} | H-C all {h['all_days']['estimate']:.3f} hot {h['forecast_hot']['estimate']:.3f} "
              f"pTOST {h['p_tost']:.3f} extreme {(h.get('observed_extreme') or {}).get('estimate', float('nan')):.3f} | "
              f"Peirce M-P {m['peirce_diff']['estimate']:+.3f} [{m['peirce_diff']['ci95'][0]:+.3f},{m['peirce_diff']['ci95'][1]:+.3f}] "
              f"Holm p {out['model_vs_persistence_holm'][str(lead)]['p_holm']:.3f}")
    print("E5 lead-1 replication pass (A6.2):", out["e5_lead1_pass"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
