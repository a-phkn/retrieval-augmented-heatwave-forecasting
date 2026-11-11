"""
RS-1 runner (pre-registered in context/decisions_retrieval.md): the testbed (RS-1a/b), RAFT's add-on mode on the
networks (RS-1c) and the diagnostics (RS-1d), development folds only.

Run from repo root:
    python -m retrieval_study.step1 --family Tmax     -> retrieval_study/results/step1_Tmax.json
    python -m retrieval_study.step1 --family WBGT     -> retrieval_study/results/step1_WBGT.json
    python -m retrieval_study.step1 --report          -> retrieval_study/results/step1.{md,json}
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.compare_v2 import compare, rmse_by
from retrieval_study import diagnostics as dg
from retrieval_study.testbed import (
    BASE_NAMES, BASES, FAMILIES, RANDOM_ARMS, RANDOM_SEEDS, FoldData, addon, ladder, long_rows, network_rows,
)
from training.folds import FOLDS

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "retrieval_study" / "results"
PRED_DIR = REPO_ROOT / "predictions_v2"
SHORT = {"Tmax": "Tmax", "WBGT": "WBGT (physical)"}
STRATA = ("all", "extreme")
STRATUM_NAMES = {"all": "all days", "extreme": "heatwave days"}
NETWORKS = {"C3-hop": "_C3hop", "Delhi-only LSTM": ""}
TEST_KEYS = ("parent", "child", "delta", "se", "ci_low", "ci_high", "p_value", "n_clusters", "few_clusters",
             "n_instances")


def _test(parent: pd.DataFrame, child: pd.DataFrame, stratum: str) -> dict:
    """Paired cluster jackknife of RMSE(child) - RMSE(parent) (evaluation.compare_v2.compare)."""
    r = compare(parent, child, stratum)
    return {k: r[k] for k in TEST_KEYS}


def _rmse(df: pd.DataFrame) -> dict:
    return {s: rmse_by(df if s == "all" else df[df["stratum"] == s]) for s in STRATA}


def run_family(short: str) -> dict:
    family = SHORT[short]
    _, _, ctrl = FAMILIES[family]
    frames: dict[tuple[str, str], list[pd.DataFrame]] = {}
    settings, net_settings, diag_parts = {}, {}, []
    nets = {name: {"alone": [], "+RAFT": [], "+RAFT-rand": []} for name in NETWORKS}
    for fold in FOLDS:
        t0 = time.time()
        fd = FoldData.build(fold, family)
        f_frames, f_set = ladder(fd)
        for key, df in f_frames.items():
            frames.setdefault(key, []).append(df)
        settings[fold] = {f"{b}/{a}": s for (b, a), s in f_set.items()}
        for name, suffix in NETWORKS.items():
            rid = ctrl + suffix
            inner = pd.read_parquet(PRED_DIR / "inner" / rid / f"{fold}.parquet")
            val = pd.read_parquet(PRED_DIR / rid / f"{fold}.parquet")
            nets[name]["alone"].append(network_rows(fd, val, rid))
            pred, setting = addon(fd, inner, val)
            nets[name]["+RAFT"].append(long_rows(fd, pred, f"{rid}+RAFT"))
            rand = []
            for s in RANDOM_SEEDS:
                p, _ = addon(fd, inner, val, seed=s, setting=setting)
                rand.append(long_rows(fd, p, f"{rid}+RAFT-rand", seed=s))
            nets[name]["+RAFT-rand"].append(pd.concat(rand, ignore_index=True))
            net_settings.setdefault(name, {})[fold] = {"alpha": setting[0], "m": setting[1]}
        diag_parts.append(dg.fold_diagnostics(fd, f_set[("B2", "RAFT")]["m"]))
        print(f"{short} {fold}: done in {time.time() - t0:.0f} s", flush=True)

    pooled = {k: pd.concat(v, ignore_index=True) for k, v in frames.items()}
    res = {"family": family, "rmse": {f"{b}/{a}": _rmse(df) for (b, a), df in pooled.items()}, "tests": {},
           "settings": settings}
    for base in BASES:
        none = pooled[(base, "none")]
        for arm in ("RAFT", "RAFT-rand", "R0", "R0-rand"):
            for s in STRATA:
                res["tests"][f"{base}|{arm} vs none|{s}"] = _test(none, pooled[(base, arm)], s)
        for rand, real in RANDOM_ARMS.items():
            for s in STRATA:
                res["tests"][f"{base}|{real} vs {rand}|{s}"] = _test(pooled[(base, rand)], pooled[(base, real)], s)
    res["networks"] = {}
    for name, parts in nets.items():
        p = {k: pd.concat(v, ignore_index=True) for k, v in parts.items()}
        r = {"rmse": {k: _rmse(v) for k, v in p.items()}, "tests": {}, "settings": net_settings[name]}
        for s in STRATA:
            r["tests"][f"+RAFT vs alone|{s}"] = _test(p["alone"], p["+RAFT"], s)
            r["tests"][f"+RAFT vs +RAFT-rand|{s}"] = _test(p["+RAFT-rand"], p["+RAFT"], s)
        res["networks"][name] = r
    res["diagnostics"] = dg.summarise(diag_parts)
    return res


# ------------------------------------------------------------------ report


def _d(t: dict) -> str:
    """Delta with its CI and a marker when the CI excludes 0."""
    if not np.isfinite(t["ci_low"]):
        return f"{t['delta']:+.3f} (no CI)"
    mark = " **(CI < 0)**" if t["ci_high"] < 0 else (" (CI > 0)" if t["ci_low"] > 0 else "")
    return f"{t['delta']:+.3f} [{t['ci_low']:+.3f}, {t['ci_high']:+.3f}]{mark}"


def predictions_check(res: dict) -> dict:
    """The pre-registered P2 rule and the other predictions' plain readings, per family."""
    t = res["tests"]
    p2_b0 = t["B0|R0 vs none|all"]["ci_high"] < 0 and t["B0|R0 vs R0-rand|all"]["ci_high"] < 0
    shrink = abs(t["B1|R0 vs none|all"]["delta"]) < abs(t["B0|R0 vs none|all"]["delta"])
    flat = all(t[f"{b}|R0 vs none|all"]["ci_low"] <= 0 <= t[f"{b}|R0 vs none|all"]["ci_high"]
               or t[f"{b}|R0 vs none|all"]["ci_low"] > 0 for b in ("B2", "B3"))
    p1 = all(not (t[f"{b}|RAFT vs none|all"]["ci_high"] < 0) and not (t[f"{b}|RAFT vs RAFT-rand|all"]["ci_high"] < 0)
             for b in ("B2", "B3"))
    p3 = all(abs(t[f"{b}|{a} vs none|all"]["delta"]) < 0.01 for b in ("B2", "B3") for a in ("RAFT", "R0"))
    n = res["networks"]
    p4 = all(not (r["tests"]["+RAFT vs alone|all"]["ci_high"] < 0) for r in n.values())
    dgn = res["diagnostics"]
    p5 = all(dgn["key_value_spearman"][f"{k}, all"] < 0.6 for k in ("raft", "r0"))
    p6 = all(dgn["oracle_rank_median"][f"{k}, all"] < 90 for k in ("raft", "r0"))
    cold = np.mean([r["r0_bias"] for r in dgn["cold_days_by_lead"]])
    p7 = bool(cold < -1.0 and np.mean([r["raft_bias"] for r in dgn["cold_days_by_lead"]]) > cold)
    return {"P1": p1, "P2 (B0 gain and beats random)": p2_b0, "P2 (shrinks B0 -> B1)": shrink,
            "P2 (no gain on B2, B3)": flat, "P2 holds": bool(p2_b0 and shrink and flat), "P3": p3, "P4": p4,
            "P5": p5, "P6": p6, "P7": p7}


def write_report(results: dict[str, dict]) -> None:
    L = ["# Retrieval study, step 1: why retrieval did not help (RS-1)", "",
         "Pre-registered in `context/decisions_retrieval.md` (RS-1) before anything was run. Development folds only "
         "(validation blocks 2007-2018 pooled; nothing from 2019 on is read). RMSE in °C. Δ = RMSE of the second "
         "minus RMSE of the first; negative = the retrieval helps. 95% CIs: paired cluster jackknife over year x "
         "season clusters (36 on all days, 12 on heatwave days). Random arms: 10 seeds.", "",
         "Bases: " + "; ".join(f"**{b}** {BASE_NAMES[b]}" for b in BASES) + ". Arms: **RAFT** as published (Han et "
         "al. 2025); **R0** our own retrieval as in the runs; each with its random control.", ""]
    for short, res in results.items():
        L += [f"## {res['family']}", ""]
        for s in STRATA:
            L += [f"### The information ladder, {STRATUM_NAMES[s]}", "",
                  "| Base | RMSE, no retrieval | RAFT vs none | RAFT vs its random control | R0 vs none | "
                  "R0 vs its random control |", "|---|---|---|---|---|---|"]
            for b in BASES:
                t = res["tests"]
                L.append(f"| {b} ({BASE_NAMES[b]}) | {res['rmse'][f'{b}/none'][s]:.3f} | {_d(t[f'{b}|RAFT vs none|{s}'])} | "
                         f"{_d(t[f'{b}|RAFT vs RAFT-rand|{s}'])} | {_d(t[f'{b}|R0 vs none|{s}'])} | "
                         f"{_d(t[f'{b}|R0 vs R0-rand|{s}'])} |")
            L.append("")
        L += ["### RAFT's add-on mode on our networks (10-seed ensemble mean)", "",
              "| Network | Days | RMSE alone | + RAFT vs alone | + RAFT vs + RAFT-rand |", "|---|---|---|---|---|"]
        for name, r in res["networks"].items():
            for s in STRATA:
                L.append(f"| {name} | {STRATUM_NAMES[s]} | {r['rmse']['alone'][s]:.3f} | "
                         f"{_d(r['tests'][f'+RAFT vs alone|{s}'])} | {_d(r['tests'][f'+RAFT vs +RAFT-rand|{s}'])} |")
        dgn = res["diagnostics"]
        L += ["", "### Diagnostics", "",
              f"{dgn['n_queries']} validation queries, {dgn['n_heatwave_queries']} with a heatwave day among their 5 "
              "target days.", "",
              "**Does a similar past mean a similar future?** Spearman correlation between key similarity and value "
              "similarity (minus the RMSE between the 5-day changes), over 200 random candidates per query. RAFT "
              "reports 0.60 on its ETTh1 data (definition not given, so the comparison is rough).", "",
              "| Key | All queries | Heatwave queries |", "|---|---|---|"]
        for k, label in (("raft", "RAFT (Pearson of the 14 days)"), ("r0", "R0 (cosine of the 17 features)")):
            L.append(f"| {label} | {dgn['key_value_spearman'][f'{k}, all']:+.3f} | "
                     f"{dgn['key_value_spearman'][f'{k}, heatwave queries']:+.3f} |")
        L += ["", "**Can the key find the right analogues?** The 5 candidates whose actual 5-day changes were closest to "
              "the query's (the oracle analogues): their percentile rank under each key (50 = chance, 100 = ranked "
              "first).", "", "| Key | Median, all | Median, heatwave queries | Mean, all |", "|---|---|---|---|"]
        for k, label in (("raft", "RAFT"), ("r0", "R0")):
            L.append(f"| {label} | {dgn['oracle_rank_median'][f'{k}, all']:.1f} | "
                     f"{dgn['oracle_rank_median'][f'{k}, heatwave queries']:.1f} | {dgn['oracle_rank_mean'][f'{k}, all']:.1f} |")
        L += ["", "**The retrievals' own forecasts** (no model; oracle = last value + the oracle analogues' mean change, "
              "an upper bound that uses the truth):", "",
              "| Forecast | RMSE all | Bias all | RMSE heatwave days | Bias heatwave days |", "|---|---|---|---|---|"]
        names = {"oracle": "Oracle analogues (uses the truth)", "raft": "RAFT, change-based (last value + weighted change)",
                 "r0": "R0, level-based (analogues' mean raw future)"}
        for k, label in names.items():
            f = dgn["forecasts"][k]
            L.append(f"| {label} | {f['all']['rmse']:.3f} | {f['all']['bias']:+.2f} | {f['heatwave days']['rmse']:.3f} | "
                     f"{f['heatwave days']['bias']:+.2f} |")
        L += ["", "**Cold days** (heatwave-day rows):", "",
              "| Lead | Rows | R0 analogues' raw future - truth | RAFT change-based - truth | No candidate as hot as the truth |",
              "|---|---|---|---|---|"]
        for r in dgn["cold_days_by_lead"]:
            L.append(f"| {r['lead']} | {r['rows']} | {r['r0_bias']:+.2f} | {r['raft_bias']:+.2f} | "
                     f"{100 * r['nothing_as_hot_share']:.0f}% |")
        g = dgn["r0_last_gap"]
        L += ["", f"Query's last input value minus its R0 analogues' mean last value: {g['all queries']:+.2f} °C over all "
              f"queries, {g['queries with a heatwave target day']:+.2f} °C for queries with a heatwave target day.", ""]
        pc = predictions_check(res)
        L += ["### The predictions (automatic reading)", "", "| Prediction | Holds |", "|---|---|"]
        L += [f"| {k} | {'yes' if v else 'no'} |" for k, v in pc.items()]
        L.append("")
    L += ["## Settings chosen on the inner block (per fold)", "",
          "Ridge strength and RAFT's m were chosen on each fold's last 2 training years; details in `step1.json`.", ""]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "step1.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Retrieval study, step 1 (RS-1).")
    ap.add_argument("--family", choices=sorted(SHORT))
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.family:
        res = run_family(args.family)
        (OUT_DIR / f"step1_{args.family}.json").write_text(json.dumps(res, indent=2, default=float) + "\n",
                                                           encoding="utf-8")
        print(f"wrote {OUT_DIR / f'step1_{args.family}.json'}")
    if args.report:
        results = {s: json.loads((OUT_DIR / f"step1_{s}.json").read_text(encoding="utf-8")) for s in SHORT}
        for res in results.values():
            res["predictions"] = predictions_check(res)
        (OUT_DIR / "step1.json").write_text(json.dumps(results, indent=2, default=float) + "\n", encoding="utf-8")
        write_report(results)
        print(f"wrote {OUT_DIR / 'step1.md'}")


if __name__ == "__main__":
    main()
