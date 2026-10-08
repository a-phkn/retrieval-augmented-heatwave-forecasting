"""
Graph-backbone gates G-D2 and G-D3 and the graph-structure claims, exactly as pre-registered
in context/decisions.md ("Graph backbone runs and gates", 2026-10-07; cross-family rule added
before any graph result was looked at). Written before the graph runs finished.

Per family (Tmax: control A1prime_hw5; physical WBGT: control A2Lr_hw5) and graph config
(C2 none, C3 static, C4 dynamic, C4a dynamic + adaptive; U1 = control LSTM + flattened upstream):

G-D2 (trainability), per graph config:
  - every training loss finite (the trainer's per-fold logs);
  - the early-stopping loss at the best epoch lower than after epoch 1 in >= 90% of seed-folds;
  - seed-to-seed SD of the all-days RMSE <= 2 x the control LSTM's.
G-D3, per family: candidate = lowest all-days RMSE among C3 / C4 / C4a that passed G-D2 (ties
  within 0.02 °C -> the simpler, C3 < C4 < C4a). Non-inferiority on all days (paired
  cluster-jackknife, 95% CI of Δ RMSE = candidate minus other): (a) vs the control LSTM and
  (b) vs U1, each passing if the CI's upper end is below +0.05 °C.
Backbone (BB*), across families (the final model is one backbone for both targets):
  graph   if (a) and (b) pass in BOTH families (the same graph config if both families pick it;
          otherwise the family candidates are reported and the WBGT family's candidate is used,
          WBGT being the primary target);
  control if (a) fails in either family, or no candidate passed G-D2 in either family;
  U1      otherwise ((a) passes in both, (b) fails in at least one: the gain is the data).
Structure claims (descriptive unless significant): C3 / C4 / C4a vs U1 and vs C2 on all days;
  a claim about graph structure needs the CI entirely below 0 against BOTH.

Run from repo root (after the graph queue):  python -m evaluation.graph_gates
Writes evaluation_v2/graph_gates.{md,json}.
After the tuning round (decisions.md 2026-10-08):  python -m evaluation.graph_gates --tuned
Writes evaluation_v2/graph_gates_tuned.{md,json}.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from evaluation.compare_v2 import _ci, compare, load_run, rmse_by
from evaluation.retrieval_information_check import OUT_DIR
from training.folds import FOLDS

PRED_DIR = OUT_DIR.parent / "predictions_v2"
CONTROLS = {"Tmax": "A1prime_hw5", "WBGT (physical)": "A2Lr_hw5"}
GRAPHS = ("C2", "C3", "C4", "C4a")
CANDIDATES = ("C3", "C4", "C4a")  # in order of simplicity
MARGIN_C, TIE_C = 0.05, 0.02
IMPROVE_SHARE, SD_RATIO = 0.90, 2.0


def seed_rmses(df: pd.DataFrame) -> np.ndarray:
    """All-days RMSE per seed (pooled over folds)."""
    return df.assign(se=df["error"] ** 2).groupby("seed")["se"].mean().pipe(np.sqrt).to_numpy()


def load_logs(run_id: str) -> list[dict]:
    logs = []
    for fold in FOLDS:
        logs += json.loads((PRED_DIR / run_id / f"{fold}_training_log.json").read_text(encoding="utf-8"))
    return logs


def g_d2(logs: list[dict], run_seed_rmse: np.ndarray, control_seed_rmse: np.ndarray) -> dict:
    finite = all(entry["finite"] for entry in logs)
    improved = np.mean([min(entry["val_curve"]) < entry["val_curve"][0] for entry in logs])
    sd, sd_ctrl = float(np.std(run_seed_rmse, ddof=1)), float(np.std(control_seed_rmse, ddof=1))
    return {"finite": finite, "improved_share": float(improved), "seed_sd": sd, "control_seed_sd": sd_ctrl,
            "pass": bool(finite and improved >= IMPROVE_SHARE and sd <= SD_RATIO * sd_ctrl)}


def choose_candidate(rmse: dict[str, float], passed: dict[str, bool]) -> str | None:
    ok = [c for c in CANDIDATES if passed.get(c)]
    if not ok:
        return None
    best = min(rmse[c] for c in ok)
    return next(c for c in ok if rmse[c] - best <= TIE_C)  # CANDIDATES is ordered simplest first


def non_inferior(test: dict) -> bool:
    return bool(np.isfinite(test["ci_high"]) and test["ci_high"] < MARGIN_C)


def backbone_decision(fam: dict[str, dict]) -> dict:
    """fam: per family {"candidate", "a", "b"} (a/b: bool or None if no candidate)."""
    cands = {f: r["candidate"] for f, r in fam.items()}
    if any(c is None for c in cands.values()) or not all(r["a"] for r in fam.values()):
        return {"backbone": "control LSTM", "reason": "no G-D2-passing graph candidate in a family, or (a) failed"}
    if not all(r["b"] for r in fam.values()):
        return {"backbone": "U1", "reason": "(a) passed in both families, (b) failed in at least one: the gain is the data"}
    same = len(set(cands.values())) == 1
    pick = next(iter(cands.values())) if same else cands["WBGT (physical)"]
    return {"backbone": f"graph {pick}", "reason": "(a) and (b) passed in both families"
            + ("" if same else "; families picked different configs, the WBGT family's is used")}


def evaluate_family(fam: str) -> dict:
    ctrl_id = CONTROLS[fam]
    ctrl = load_run(ctrl_id)
    runs = {g: load_run(f"{ctrl_id}_{g}") for g in (*GRAPHS, "U1")}
    missing = [g for g, df in runs.items() if df is None]
    if missing:
        raise FileNotFoundError(f"{fam}: runs not complete: {missing}")
    ctrl_sd = seed_rmses(ctrl)
    out = {"rmse_all": {g: rmse_by(df) for g, df in runs.items()} | {"control": rmse_by(ctrl)},
           "rmse_extreme": {g: rmse_by(df[df["stratum"] == "extreme"]) for g, df in runs.items()}
           | {"control": rmse_by(ctrl[ctrl["stratum"] == "extreme"])},
           "g_d2": {g: g_d2(load_logs(f"{ctrl_id}_{g}"), seed_rmses(runs[g]), ctrl_sd) for g in GRAPHS}}
    cand = choose_candidate(out["rmse_all"], {g: v["pass"] for g, v in out["g_d2"].items()})
    out["candidate"] = cand
    tests = {}
    for g in GRAPHS:
        for other, name in ((ctrl, "control"), (runs["U1"], "U1"), (runs["C2"], "C2")):
            if g == "C2" and name == "C2":
                continue
            for s in ("all", "extreme"):
                tests[f"{g} vs {name} ({s})"] = compare(other, runs[g], s)
    for s in ("all", "extreme"):
        tests[f"U1 vs control ({s})"] = compare(ctrl, runs["U1"], s)
    out["tests"] = tests
    out["a"] = non_inferior(tests[f"{cand} vs control (all)"]) if cand else None
    out["b"] = non_inferior(tests[f"{cand} vs U1 (all)"]) if cand else None
    out["structure_claims"] = {g: bool(tests[f"{g} vs U1 (all)"]["ci_high"] < 0 and tests[f"{g} vs C2 (all)"]["ci_high"] < 0)
                               for g in CANDIDATES}
    return out


def write_report(results: dict, decision: dict) -> None:
    L = ["# Graph backbone gates G-D2 / G-D3 (pre-registered 2026-10-07)", "",
         f"**Backbone (BB\\*): {decision['backbone']}** ({decision['reason']}).", "",
         "Out of fold 2007-2018, 10 seeds x 4 folds. Δ = first minus second; negative = first better. Non-inferiority "
         f"margin {MARGIN_C} °C on the upper end of the 95% CI.", ""]
    for fam, r in results.items():
        L += [f"## {fam} (control `{CONTROLS[fam]}`)", "",
              f"Candidate: **{r['candidate'] or 'none'}**; (a) vs control: {r['a']}; (b) vs U1: {r['b']}.", "",
              "| Model | All RMSE | Extreme RMSE | G-D2 | finite | improved after epoch 1 | seed SD (control) |",
              "|---|---|---|---|---|---|---|"]
        L.append(f"| control | {r['rmse_all']['control']:.3f} | {r['rmse_extreme']['control']:.3f} | | | | |")
        for g in (*GRAPHS, "U1"):
            d2 = r["g_d2"].get(g)
            d2_txt = (f"{'pass' if d2['pass'] else 'FAIL'} | {d2['finite']} | {d2['improved_share']:.0%} | "
                      f"{d2['seed_sd']:.3f} ({d2['control_seed_sd']:.3f})") if d2 else "| | |"
            L.append(f"| {g} | {r['rmse_all'][g]:.3f} | {r['rmse_extreme'][g]:.3f} | {d2_txt} |")
        L += ["", "| Comparison | Δ RMSE (95% CI) | p |", "|---|---|---|"]
        for k, t in r["tests"].items():
            L.append(f"| {k} | {t['delta']:+.3f} {_ci(t)} | {t['p_value']:.3f} |")
        L += ["", "Structure claim (CI entirely below 0 vs both U1 and C2, all days): "
              + ", ".join(f"{g} {'yes' if v else 'no'}" for g, v in r["structure_claims"].items()), ""]
    (OUT_DIR / "graph_gates.md").write_text("\n".join(L) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ after the tuning round
# decisions.md, 2026-10-08 ("Graph tuning round, with one added arm C3-pool"), written before any
# tuning result existed. Runs: {control}_C3_tuned, {control}_C3pool_tuned, {control}_U1_tuned.

TUNED_GRAPHS = ("C3", "C3pool")


def tuned_decision(fam: dict[str, dict]) -> dict:
    """fam: per family {arm: {"g_d2": bool, "a": bool, "b": bool}} for arm in TUNED_GRAPHS.
    1. tuned C3 passes (G-D2, a, b) in both families -> graph C3;
    2. else tuned C3-pool passes in both -> graph C3-pool (post-result, test run confirms);
    3. else control LSTM if tuned C3's (a) fails in either family, otherwise tuned U1."""
    def passes(arm: str) -> bool:
        return all(r[arm]["g_d2"] and r[arm]["a"] and r[arm]["b"] for r in fam.values())

    if passes("C3"):
        return {"backbone": "graph C3 (tuned)", "reason": "tuned C3 passed G-D2, (a) and (b) in both families"}
    if passes("C3pool"):
        return {"backbone": "graph C3-pool (tuned)",
                "reason": "tuned C3 failed; tuned C3-pool passed G-D2, (a) and (b) in both families. Chosen after a "
                          "post-result change (decisions.md 2026-10-08); the locked test run is its confirmation"}
    if not all(r["C3"]["a"] for r in fam.values()):
        return {"backbone": "control LSTM", "reason": "no tuned graph arm passed; tuned C3's (a) failed in a family"}
    return {"backbone": "U1 (tuned)", "reason": "no tuned graph arm passed; tuned C3's (a) passed in both families"}


def evaluate_tuned_family(fam: str) -> dict:
    ctrl_id = CONTROLS[fam]
    ctrl = load_run(ctrl_id)
    runs = {a: load_run(f"{ctrl_id}_{a}_tuned") for a in (*TUNED_GRAPHS, "U1")} | {"C2": load_run(f"{ctrl_id}_C2")}
    missing = [a for a, df in runs.items() if df is None]
    if missing:
        raise FileNotFoundError(f"{fam}: runs not complete: {missing}")
    ctrl_sd = seed_rmses(ctrl)
    out = {"rmse_all": {a: rmse_by(df) for a, df in runs.items()} | {"control": rmse_by(ctrl)},
           "rmse_extreme": {a: rmse_by(df[df["stratum"] == "extreme"]) for a, df in runs.items()}
           | {"control": rmse_by(ctrl[ctrl["stratum"] == "extreme"])},
           "g_d2": {a: g_d2(load_logs(f"{ctrl_id}_{a}_tuned"), seed_rmses(runs[a]), ctrl_sd) for a in TUNED_GRAPHS}}
    tests = {}
    for a in TUNED_GRAPHS:
        for other, name in ((ctrl, "control"), (runs["U1"], "tuned U1"), (runs["C2"], "C2")):
            for s in ("all", "extreme"):
                tests[f"{a} tuned vs {name} ({s})"] = compare(other, runs[a], s)
    for s in ("all", "extreme"):
        tests[f"C3pool tuned vs C3 tuned ({s})"] = compare(runs["C3"], runs["C3pool"], s)
        tests[f"U1 tuned vs control ({s})"] = compare(ctrl, runs["U1"], s)
    out["tests"] = tests
    out["arms"] = {a: {"g_d2": out["g_d2"][a]["pass"], "a": non_inferior(tests[f"{a} tuned vs control (all)"]),
                       "b": non_inferior(tests[f"{a} tuned vs tuned U1 (all)"])} for a in TUNED_GRAPHS}
    return out


def write_tuned_report(results: dict, decision: dict) -> None:
    L = ["# Graph backbone after the tuning round (decisions.md 2026-10-08)", "",
         f"**Backbone (BB\\*): {decision['backbone']}** ({decision['reason']}).", "",
         "Out of fold 2007-2018, 10 seeds x 4 folds; hyperparameters were selected on the inner 2-year blocks only "
         "(`evaluation_v2/graph_tuning.md`). The control LSTM and C2 are not retuned (comparisons against them favour "
         f"the tuned models; disclosed). Non-inferiority margin {MARGIN_C} °C on the upper end of the 95% CI.", ""]
    for fam, r in results.items():
        L += [f"## {fam} (control `{CONTROLS[fam]}`)", "", "| Arm | G-D2 | (a) vs control | (b) vs tuned U1 |",
              "|---|---|---|---|"]
        L += [f"| {a} tuned | {v['g_d2']} | {v['a']} | {v['b']} |" for a, v in r["arms"].items()]
        L += ["", "| Model | All RMSE | Extreme RMSE |", "|---|---|---|"]
        L += [f"| {m} | {r['rmse_all'][m]:.3f} | {r['rmse_extreme'][m]:.3f} |" for m in r["rmse_all"]]
        L += ["", "| Comparison | Δ RMSE (95% CI) | p |", "|---|---|---|"]
        L += [f"| {k} | {t['delta']:+.3f} {_ci(t)} | {t['p_value']:.3f} |" for k, t in r["tests"].items()]
        L.append("")
    (OUT_DIR / "graph_gates_tuned.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Graph backbone gates (pre-registered).")
    ap.add_argument("--tuned", action="store_true", help="evaluate the tuning round's 10-seed runs")
    args = ap.parse_args(argv)
    if args.tuned:
        results = {fam: evaluate_tuned_family(fam) for fam in CONTROLS}
        decision = tuned_decision({f: r["arms"] for f, r in results.items()})
        write_tuned_report(results, decision)
        out = OUT_DIR / "graph_gates_tuned.json"
    else:
        results = {fam: evaluate_family(fam) for fam in CONTROLS}
        decision = backbone_decision({f: {"candidate": r["candidate"], "a": r["a"], "b": r["b"]}
                                      for f, r in results.items()})
        write_report(results, decision)
        out = OUT_DIR / "graph_gates.json"
    out.write_text(json.dumps({"results": results, "decision": decision}, indent=2, default=float) + "\n",
                   encoding="utf-8")
    print(decision)


if __name__ == "__main__":
    main()
