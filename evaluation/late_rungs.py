"""
The two rungs that trained after the graph queue, judged by the rules written before they ran
(context/decisions.md 2026-10-07):

Physics head (PH_lstm): the control LSTM encoder + exact Liljegren head, one model giving WBGT
  ({fold}.parquet) and Tmax ({fold}_tmax.parquet). KEEP RULE: the head stays unless, on all days,
  the upper end of the 95% CI of delta RMSE (head minus direct model) is >= +0.05 °C for WBGT vs
  A2Lr_hw5 OR for Tmax vs A1prime_hw5. Extreme days, bias and skipped training batches reported.
R2 (region_mmr, {control}_R2): judged by the G3 rule like every retrieval rung
  (evaluation.compare_v2.g3_conditions): (1) not significantly worse on all days vs the control;
  (2) extreme-day CI vs the control entirely below 0; (3) extreme-day CI vs its random control
  (R0-rand) entirely below 0. Delta vs Rg (the rung R2 extends) is reported.

Run from repo root:  python -m evaluation.late_rungs   ->  evaluation_v2/late_rungs.{md,json}
"""
from __future__ import annotations

import json

import pandas as pd

from evaluation.compare_v2 import PRED_DIR, _ci, _fmt_p, compare, extreme_bias, g3_conditions, load_run, mde80, rmse_by
from evaluation.retrieval_information_check import OUT_DIR
from training.folds import FOLDS

KEEP_MARGIN_C = 0.05
FAMILIES = {"Tmax": "A1prime_hw5", "WBGT (physical)": "A2Lr_hw5"}


def load_physics_tmax(run_id: str = "PH_lstm") -> pd.DataFrame:
    return pd.concat([pd.read_parquet(PRED_DIR / run_id / f"{f}_tmax.parquet").assign(fold=f) for f in FOLDS],
                     ignore_index=True)


def physics_head() -> dict:
    out = {"skipped_batches": 0, "finite": True}
    for f in FOLDS:
        for e in json.loads((PRED_DIR / "PH_lstm" / f"{f}_training_log.json").read_text(encoding="utf-8")):
            out["skipped_batches"] += int(e.get("skipped_batches", 0))
            out["finite"] &= bool(e["finite"])
    for fam, ctrl_id, head in (("WBGT (physical)", "A2Lr_hw5", load_run("PH_lstm")),
                               ("Tmax", "A1prime_hw5", load_physics_tmax())):
        ctrl = load_run(ctrl_id)
        r = {"control": ctrl_id, "rmse_head": rmse_by(head), "rmse_ctrl": rmse_by(ctrl),
             "rmse_ext_head": rmse_by(head[head["stratum"] == "extreme"]),
             "rmse_ext_ctrl": rmse_by(ctrl[ctrl["stratum"] == "extreme"]),
             "bias_ext_head": extreme_bias(head), "bias_ext_ctrl": extreme_bias(ctrl),
             "all": compare(ctrl, head, "all"), "extreme": compare(ctrl, head, "extreme")}
        r["fails_keep_rule"] = bool(r["all"]["ci_high"] >= KEEP_MARGIN_C)
        out[fam] = r
    out["keep"] = not any(out[f]["fails_keep_rule"] for f in FAMILIES)
    return out


def r2_rows() -> dict:
    out = {}
    for fam, ctrl_id in FAMILIES.items():
        ctrl, r2 = load_run(ctrl_id), load_run(f"{ctrl_id}_R2")
        rand, rg = load_run(f"{ctrl_id}_R0rand"), load_run(f"{ctrl_id}_Rg")
        a, e, re = compare(ctrl, r2, "all"), compare(ctrl, r2, "extreme"), compare(rand, r2, "extreme")
        row = {"rmse_all": rmse_by(r2), "rmse_extreme": rmse_by(r2[r2["stratum"] == "extreme"]),
               "ctrl_all": a, "ctrl_ext": e, "rand_all": compare(rand, r2, "all"), "rand_ext": re,
               "rg_all": compare(rg, r2, "all"), "rg_ext": compare(rg, r2, "extreme"),
               "ctrl_all_ci_low": a["ci_low"], "ctrl_ext_ci_high": e["ci_high"], "rand_ext_ci_high": re["ci_high"],
               "ext_mde": mde80(e), "ext_bias": extreme_bias(r2), "ctrl_ext_bias": extreme_bias(ctrl)}
        row["g3"] = g3_conditions(row)
        row["passes_g3"] = all(row["g3"])
        out[fam] = row
    return out


def _t(r: dict) -> str:
    return f"{r['delta']:+.3f} {_ci(r)} (p {_fmt_p(r['p_value'])})"


def write_report(ph: dict, r2: dict) -> None:
    L = ["# Physics head and R2 (rules written 2026-10-07, before either ran)", "",
         "Out of fold 2007-2018, 10 seeds x 4 folds, paired cluster-jackknife; Δ = rung minus comparison "
         "(negative = the rung is better).", "",
         f"## Physics head (PH_lstm): **{'KEPT' if ph['keep'] else 'DROPPED'}** under the keep rule",
         f"(dropped only if the all-days CI upper end is >= +{KEEP_MARGIN_C} °C for WBGT or Tmax)", "",
         f"Training: {ph['skipped_batches']} skipped (non-finite) batches; all losses finite: {ph['finite']}.", "",
         "| Target | RMSE head / direct | Δ all days | Extreme RMSE head / direct | Δ extreme | Extreme bias head / direct | Fails rule |",
         "|---|---|---|---|---|---|---|"]
    for fam in ("WBGT (physical)", "Tmax"):
        r = ph[fam]
        L.append(f"| {fam} vs `{r['control']}` | {r['rmse_head']:.3f} / {r['rmse_ctrl']:.3f} | {_t(r['all'])} | "
                 f"{r['rmse_ext_head']:.3f} / {r['rmse_ext_ctrl']:.3f} | {_t(r['extreme'])} | "
                 f"{r['bias_ext_head']:+.2f} / {r['bias_ext_ctrl']:+.2f} | {r['fails_keep_rule']} |")
    L += ["", "## R2 (regional matching + MMR diversity) under G3", "",
          "| Family | All RMSE | Extreme RMSE | Δ vs control, all | Δ vs control, extreme | Δ vs R0-rand, extreme | "
          "Δ vs Rg, all | G3 (1, 2, 3) |", "|---|---|---|---|---|---|---|---|"]
    for fam, r in r2.items():
        L.append(f"| {fam} | {r['rmse_all']:.3f} | {r['rmse_extreme']:.3f} | {_t(r['ctrl_all'])} | {_t(r['ctrl_ext'])} | "
                 f"{_t(r['rand_ext'])} | {_t(r['rg_all'])} | {'pass' if r['passes_g3'] else 'FAIL'} {r['g3']} |")
    L += ["", "R2 was pre-registered and screened after G3 (decisions.md 2026-10-07); like Rg, it is reported as a "
          "post-G3 rung. Extreme-day results are dominated by bias (see `calibration_check.md`)."]
    (OUT_DIR / "late_rungs.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    ph, r2 = physics_head(), r2_rows()
    write_report(ph, r2)
    (OUT_DIR / "late_rungs.json").write_text(json.dumps({"physics_head": ph, "r2": r2}, indent=2, default=float) + "\n",
                                             encoding="utf-8")
    print({"physics_head_kept": ph["keep"], "r2_passes_g3": {f: r["passes_g3"] for f, r in r2.items()}})


if __name__ == "__main__":
    main()
