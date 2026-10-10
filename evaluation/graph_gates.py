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
# decisions.md 2026-10-08: the tuning round (C3, C3-pool, U1 tuned) and the amended backbone rule
# taken on the independent ML review, both written before any tuning result existed.
# Runs: {control}_C3_tuned, {control}_C3pool_tuned, {control}_U1_tuned, {control}_C2 as run, and the
# multi-hop arm {control}_C3hop (tuned C3's hyperparameters; added after the review, disclosed).

TUNED_GRAPHS = ("C3", "C3pool")
GRAPH_MODELS = ("C2", *TUNED_GRAPHS, "C3hop")  # graph-code models (G-D2 applies)
SIMPLICITY = ("control", "U1", "C2", "C3", "C3pool", "C3hop")  # simplest first (tie-break)
PRIMARY = "WBGT (physical)"
SELECTION_FREE_YEARS = (2007, 2010, 2013, 2016, 2017, 2018)  # never an inner (early-stopping / selection) block


def eligible(fam: dict[str, dict]) -> dict[str, bool]:
    """fam: per family {model: {"g_d2": bool, "a": bool, "b": bool}} for the graph-code models present,
    plus {"U1": {"a": bool}, "control": {"b": bool}}. Eligible = non-inferior to the control (a) and to
    tuned U1 (b) in BOTH families; graph-code models must also pass G-D2. A model missing in a family
    is not eligible."""
    ok = {"control": all(r["control"]["b"] for r in fam.values()),
          "U1": all(r["U1"]["a"] for r in fam.values())}
    for m in GRAPH_MODELS:
        ok[m] = all(m in r and r[m]["g_d2"] and r[m]["a"] and r[m]["b"] for r in fam.values())
    return ok


def amended_decision(ok: dict[str, bool], wbgt_rmse: dict[str, float]) -> dict:
    """Lowest WBGT all-days RMSE among the eligible; ties within TIE_C go to the simpler. PROVISIONAL:
    nothing is final until the user has seen the result and approved it."""
    cands = [m for m in SIMPLICITY if ok.get(m)]
    if not cands:
        return {"backbone": "control", "provisional": True, "reason": "no model eligible; the control is kept"}
    best = min(wbgt_rmse[m] for m in cands)
    pick = next(m for m in cands if wbgt_rmse[m] - best <= TIE_C)
    names = {"control": "control LSTM", "U1": "U1 (tuned)", "C2": "C2, regional node model (no edges)",
             "C3": "graph C3 (tuned)", "C3pool": "graph C3-pool (tuned; arm added after the G-D3 result)",
             "C3hop": "graph C3-hop (ring multi-hop; arm added after the ML review)"}
    return {"backbone": names[pick], "model": pick, "provisional": True, "eligible": cands,
            "reason": f"lowest WBGT all-days RMSE among the eligible {cands} (ties within {TIE_C} °C to the simpler). "
                      "PROVISIONAL: awaiting the user's approval"}


def selection_free(df: pd.DataFrame) -> pd.DataFrame:
    """Only forecasts whose target date falls in a year never used as an inner block (M1 sensitivity)."""
    return df[pd.DatetimeIndex(df["target_date"]).year.isin(SELECTION_FREE_YEARS)]


def structure_claims(tests: dict, models) -> dict[str, bool]:
    """The structure-claim rule for the edged models: CI entirely below 0 against BOTH U1 and C2 (all days)."""
    return {m: bool(tests[f"{m} vs U1 (all)"]["ci_high"] < 0 and tests[f"{m} vs C2 (all)"]["ci_high"] < 0)
            for m in models if f"{m} vs C2 (all)" in tests}


def evaluate_tuned_family(fam: str) -> dict:
    """All candidates of the amended rule. The post-review arm C3-hop is optional (absent until it has run);
    the ridge baselines are reported if `evaluation.ridge_baselines` has written them."""
    ctrl_id = CONTROLS[fam]
    runs = {"control": load_run(ctrl_id), "C2": load_run(f"{ctrl_id}_C2")}
    runs |= {m: load_run(f"{ctrl_id}_{m}_tuned") for m in ("U1", *TUNED_GRAPHS)}
    missing = [m for m, df in runs.items() if df is None]
    if missing:
        raise FileNotFoundError(f"{fam}: runs not complete: {missing}")
    hop = load_run(f"{ctrl_id}_C3hop")
    if hop is not None:
        runs["C3hop"] = hop
    ridges = {}
    for r in ("ridge", "ridge_hw5"):
        df = load_run(f"{ctrl_id}_{r}")
        if df is not None:
            ridges[r] = df
    graphs = [m for m in GRAPH_MODELS if m in runs]
    ctrl_sd = seed_rmses(runs["control"])
    logs = {"C2": f"{ctrl_id}_C2", "C3hop": f"{ctrl_id}_C3hop"} | {m: f"{ctrl_id}_{m}_tuned" for m in TUNED_GRAPHS}
    every = runs | ridges
    out = {"present": list(runs), "rmse_all": {m: rmse_by(df) for m, df in every.items()},
           "rmse_extreme": {m: rmse_by(df[df["stratum"] == "extreme"]) for m, df in every.items()},
           "g_d2": {m: g_d2(load_logs(logs[m]), seed_rmses(runs[m]), ctrl_sd) for m in graphs}}
    tests, free = {}, {}
    pairs = [(m, o) for m in graphs for o in ("control", "U1")]
    pairs += [("U1", "control"), ("control", "U1"), ("C3pool", "C3"), ("C3pool", "C2"), ("C3", "C2")]
    pairs += [("C3hop", "C3"), ("C3hop", "C2")] if "C3hop" in runs else []
    pairs += [(m, r) for m in runs for r in ridges]
    for m, o in pairs:
        for s in ("all", "extreme"):
            tests[f"{m} vs {o} ({s})"] = compare(every[o], every[m], s)
        free[f"{m} vs {o} (all)"] = compare(selection_free(every[o]), selection_free(every[m]), "all")
    out["tests"], out["tests_selection_free"] = tests, free
    out["models"] = {m: {"g_d2": out["g_d2"][m]["pass"], "a": non_inferior(tests[f"{m} vs control (all)"]),
                         "b": non_inferior(tests[f"{m} vs U1 (all)"])} for m in graphs}
    out["models"]["U1"] = {"a": non_inferior(tests["U1 vs control (all)"])}
    out["models"]["control"] = {"b": non_inferior(tests["control vs U1 (all)"])}
    out["structure_claims"] = structure_claims(tests, ("C3", "C3pool", "C3hop"))
    return out


def tuned_decisions(results: dict) -> dict:
    """The amended rule with every candidate present, and (disclosure) without the post-review arm C3-hop."""
    fam = {f: r["models"] for f, r in results.items()}
    without = {f: {m: v for m, v in r.items() if m != "C3hop"} for f, r in fam.items()}
    out = {"eligible": eligible(fam), "with_C3hop": None,
           "without_C3hop": amended_decision(eligible(without), results[PRIMARY]["rmse_all"])}
    if all("C3hop" in r for r in fam.values()):
        out["with_C3hop"] = amended_decision(out["eligible"], results[PRIMARY]["rmse_all"])
    out["headline"] = out["with_C3hop"] or out["without_C3hop"]
    return out


def write_tuned_report(results: dict, decisions: dict) -> None:
    head = decisions["headline"]
    with_hop = decisions["with_C3hop"]["backbone"] if decisions["with_C3hop"] else "C3-hop not run yet"
    L = ["# Backbone after the tuning round (amended rule, decisions.md 2026-10-08)", "",
         f"**Provisional backbone: {head['backbone']}** ({head['reason']}).", "",
         f"With C3-hop: {with_hop}. Without C3-hop (the five candidates declared before the review): "
         f"{decisions['without_C3hop']['backbone']}.", "",
         "Out of fold 2007-2018, 10 seeds x 4 folds. U1 / C3 / C3-pool: hyperparameters selected on the inner 2-year "
         "blocks (`evaluation_v2/graph_tuning.md`); C3-hop uses tuned C3's; the control and C2 are not retuned "
         f"(comparisons favour the tuned models; disclosed). Non-inferiority margin {MARGIN_C} °C on the upper end of the "
         "95% CI. U1 below = tuned U1. Ridges: `evaluation/ridge_baselines.py` (reported, not eligible).", "",
         "Eligible in both families: " + ", ".join(f"{m} {'yes' if v else 'no'}" for m, v in decisions["eligible"].items()),
         ""]
    for fam, r in results.items():
        L += [f"## {fam} (control `{CONTROLS[fam]}`)", "", "| Model | All RMSE | Extreme RMSE | G-D2 |", "|---|---|---|---|"]
        L += [f"| {m} | {r['rmse_all'][m]:.3f} | {r['rmse_extreme'][m]:.3f} | "
              f"{('pass' if r['g_d2'][m]['pass'] else 'FAIL') if m in r['g_d2'] else ''} |" for m in r["rmse_all"]]
        claims = ", ".join(f"{m} {'yes' if v else 'no'}" for m, v in r["structure_claims"].items()) or "n/a"
        L += ["", f"Structure claim (CI entirely below 0 vs both U1 and C2, all days): {claims}", ""]
        L += ["| Comparison | Δ RMSE (95% CI) | p | Selection-free years: Δ all (95% CI) |", "|---|---|---|---|"]
        for k, t in r["tests"].items():
            f = r["tests_selection_free"].get(k)
            L.append(f"| {k} | {t['delta']:+.3f} {_ci(t)} | {t['p_value']:.3f} | "
                     + (f"{f['delta']:+.3f} {_ci(f)}" if f else "") + " |")
        L.append("")
    L += [f"Selection-free years = target dates in {SELECTION_FREE_YEARS}: never an inner block of any fold, so never "
          "used for early stopping or tuning selection (the f2-f4 inner blocks 2008-09, 2011-12, 2014-15 lie inside "
          "2007-2018)."]
    (OUT_DIR / "graph_gates_tuned.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Graph backbone gates (pre-registered).")
    ap.add_argument("--tuned", action="store_true", help="evaluate the tuning round's 10-seed runs")
    args = ap.parse_args(argv)
    if args.tuned:
        results = {fam: evaluate_tuned_family(fam) for fam in CONTROLS}
        decision = tuned_decisions(results)
        write_tuned_report(results, decision)
        out = OUT_DIR / "graph_gates_tuned.json"
        shown = decision["headline"]
    else:
        results = {fam: evaluate_family(fam) for fam in CONTROLS}
        decision = backbone_decision({f: {"candidate": r["candidate"], "a": r["a"], "b": r["b"]}
                                      for f, r in results.items()})
        write_report(results, decision)
        out = OUT_DIR / "graph_gates.json"
        shown = decision
    out.write_text(json.dumps({"results": results, "decision": decision}, indent=2, default=float) + "\n",
                   encoding="utf-8")
    print(shown)


if __name__ == "__main__":
    main()
