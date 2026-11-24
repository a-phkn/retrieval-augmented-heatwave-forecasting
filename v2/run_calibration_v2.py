"""
Runs H-C and E5 (PROTOCOL_V2 H-C, amendment A2) on the development out-of-fold forecasts 2007-2018, folds f1-f4.
Needs the inner-block forecasts: predictions_v2/inner/<run>/<fold>.parquet (or the model checkpoints, from which
evaluation.calibration_check.inner_forecasts rebuilds and verifies them). Writes results_v2/h_c_e5.json.

Run from repo root:  python -m v2.run_calibration_v2
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy import stats as sp_stats

from v2.build_delhi_hourly import OUT as WBGT_DIR
from v2.calibration_v2 import e5, h_c, holm, in_season
from v2.work_rest import daily_rest

REPO_ROOT = Path(__file__).resolve().parents[1]
PRED = REPO_ROOT / "predictions_v2"
RESULTS = REPO_ROOT / "results_v2"
RUNS = {"WBGT": "A2Lr_hw5_C3hop", "Tmax": "A1prime_hw5_C3hop"}   # WBGT primary
FOLDS = ("f1", "f2", "f3", "f4")
R = 9.0   # PROTOCOL_V2 A1.6
SEVERE_Q = 0.90   # A2.5: severe subset = 90th percentile of in-season rest-hours, 1980-2006 (secondary)


def load(run: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    val, inner = [], []
    for f in FOLDS:
        v = pd.read_parquet(PRED / run / f"{f}.parquet")
        cache = PRED / "inner" / run / f"{f}.parquet"
        if cache.exists():
            i = pd.read_parquet(cache)
        else:
            try:
                from evaluation.calibration_check import inner_forecasts
                i = inner_forecasts(run, f)
            except Exception as exc:  # noqa: BLE001
                raise SystemExit(f"{run} {f}: no inner-block forecasts ({cache}) and no usable checkpoints: {exc}")
        if v["target_date"].dt.year.max() > 2018 or i["target_date"].dt.year.max() > 2018:
            raise RuntimeError("2019+ rows in development forecasts; refusing")
        val.append(v.assign(fold=f))
        inner.append(i.assign(fold=f))
    return pd.concat(val, ignore_index=True), pd.concat(inner, ignore_index=True)


def rest_hours() -> pd.Series:
    h = pd.read_parquet(WBGT_DIR / "delhi_hourly_dev_shade.parquet", columns=["time", "cell", "wbgt"])
    g = h.groupby("time")
    m = g["wbgt"].mean()[g["cell"].nunique() == 9].rename("wbgt").reset_index().assign(cell=0, t=0.0)
    return daily_rest(m, "moderate", "TLV", "wbgt").set_index("date")["rest_hours"]


def h_a_p() -> float:
    """Two-sided jackknife p of H-A's primary difference (A2.2), from results_v2/h_a_delhi.json."""
    d = json.loads((RESULTS / "h_a_delhi.json").read_text())["moderate/acclimatised"]["primary_diff"]
    t = d["estimate"] / d["se"]
    return float(2 * sp_stats.t.sf(abs(t), d["n_clusters"] - 1))


def main() -> int:
    rest = rest_hours()
    event = rest >= R
    base = rest[(rest.index.year <= 2006) & in_season(rest.index)]
    severe_r = float(base.quantile(SEVERE_Q))
    severe = rest >= severe_r
    out = {"R": R, "severe_R": severe_r, "event_base_rate_1980_2018_in_season":
           float(event[(rest.index.year <= 2018) & in_season(rest.index)].mean())}
    for fam, run in RUNS.items():
        if fam != "WBGT" and not (PRED / "inner" / run).exists():
            out[fam] = {"skipped": f"no inner forecasts for {run} (secondary family; H-C primary is WBGT)"}
            print(out[fam]["skipped"])
            continue
        val, inner = load(run)
        folds = [(inner[inner["fold"] == f], val[val["fold"] == f]) for f in FOLDS]
        out[fam] = {"h_c": {lead: h_c(folds, lead) for lead in range(1, 6)}}
        if fam == "WBGT":
            out[fam]["e5"] = {lead: e5(folds, event, lead, severe=severe) for lead in range(1, 6)}
            out[fam]["e5_alpha_0.10"] = {1: e5(folds, event, 1, alpha=0.10)}
    pvals = {"H-A": h_a_p(), "H-C": out["WBGT"]["h_c"][1]["p_tost"],
             "E5": out["WBGT"]["e5"][1]["p_miss_below_alpha_plus_tol"]}
    out["holm"] = holm(pvals)   # A2.5: family = H-A, H-C, E5; H-N tested alone at 0.05
    a = json.loads((RESULTS / "h_a_delhi.json").read_text())["moderate/acclimatised"]["primary_diff"]["estimate"]
    out["supported"] = {"H-A": out["holm"]["H-A"]["reject"] and a > 0,
                        "H-C": out["holm"]["H-C"]["reject"],
                        "E5": out["holm"]["E5"]["reject"] and out["WBGT"]["e5"][1]["false_alarm_ok"]}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "h_c_e5.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    w = out["WBGT"]
    for lead in range(1, 6):
        hc, e = w["h_c"][lead], w["e5"][lead]
        print(f"lead {lead}: H-C coverage all {hc['all_days']['estimate']:.3f}, forecast-hot "
              f"{hc['forecast_hot']['estimate']:.3f} (p_TOST {hc['p_tost']:.4f}); E5 miss "
              f"{e['miss_rate']['estimate']:.3f} (p<0.25: {e['p_miss_below_alpha_plus_tol']:.4f}), "
              f"base rate {e['base_rate']:.2f}, false alarms {e['false_alarm_rate']:.2f}, lift {e['lift']:.2f}, "
              f"event days {e['event_days']}")
    print("Holm (H-A, H-C, E5):", json.dumps(out["holm"], indent=1))
    print("Supported:", out["supported"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
