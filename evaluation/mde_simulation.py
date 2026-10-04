"""
Minimum detectable effect (MDE) for the paired cluster test (plan v5, Week 1).

Question: how large an RMSE improvement can the project reliably detect with
evaluation/stats.paired_cluster_test, given the real error structure of this problem?

Method (val split only; test never read):
  1. Error templates (stratum 'all' or 'extreme'), parent = A1 LSTM (5 seeds):
       similar   child = RA-v1 (5 seeds; errors correlate ~0.99 with A1 -- an easy pair)
       less similar child = damped persistence (deterministic). Its errors still correlate
                 ~0.90 with A1 on all days (both are persistence-like), so agreement between
                 the two templates is only moderate evidence of robustness.
  2. Exact null: the child's errors are rescaled so its RMSE (mean over seeds) equals the
     parent's on the full sample.
  3. Replicates: draw G whole year x season clusters with replacement (re-labelled), and
     each model's seeds with replacement. A planted improvement shrinks the child's errors
     by (1 - eps_g), eps_g = eps * exp(SIGMA * z_g - SIGMA^2 / 2), z_g ~ N(0, 1) per drawn
     cluster: an UNEVEN effect (mean eps, coefficient of variation ~0.53 at SIGMA = 0.5;
     every cluster improves, by different amounts), unlike a uniform shrink, which would
     overstate power. SENSITIVITY: a more uneven effect, eps_g = eps * (1 + z_g) (CV = 1;
     some clusters get worse), is also reported. With effects that uneven, power is capped
     by the number of clusters: at G = 9 it may never reach 80% at any effect size.
  4. Power = share of replicates with p < 0.05 AND delta < 0 (a one-sided success, so the
     nominal false-positive rate is 0.025). MDE = smallest eps on the grid with power >= 0.80.
     With N_REP replicates, power has a standard error of up to ~0.03, so MDEs are accurate
     to about one grid step.

Caveats: pseudo-datasets reuse the few observed val clusters (9; 5 for extreme), so true
heterogeneity is probably larger -> MDEs are OPTIMISTIC. The seed-noise floor is estimated
from 5 seeds per model (few degrees of freedom) and is reported with a 95% interval.

Writes evaluation_v2/mde.json and evaluation_v2/mde.md.
Run from repo root:  python -m evaluation.mde_simulation
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from evaluation.stats import paired_cluster_test

REPO_ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = REPO_ROOT / "predictions_v1" / "val"
OUT_DIR = REPO_ROOT / "evaluation_v2"
KEY = ["query_date", "lead"]
EFFECTS = [0.0] + [round(x, 3) for x in np.arange(0.01, 0.151, 0.01)] + [0.175, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
DESIGNS = {"all": [9, 24, 36], "extreme": [5, 10, 15, 20]}
TEMPLATES = {"similar (A1 vs RA-v1)": "ra_v1", "less similar (A1 vs damped persistence)": "damped_persistence"}
N_REP = 300
POWER_TARGET = 0.80
RNG_SEED = 7
SIGMA = 0.5  # log-normal spread of the planted per-cluster effect (primary design)


def seed_matrix(df: pd.DataFrame, ref: pd.DataFrame) -> np.ndarray:
    idx = pd.MultiIndex.from_frame(ref[KEY])
    return np.stack([g.set_index(KEY).loc[idx, "error"].to_numpy() for _, g in df.groupby("seed", sort=True)])


def rmse_by_seed(e: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(e**2, axis=1))


def prepare(child_name: str, stratum: str):
    parent = pd.read_parquet(PRED_DIR / "lstm_a1.parquet")
    child = pd.read_parquet(PRED_DIR / f"{child_name}.parquet")
    ref = parent[parent["seed"] == parent["seed"].min()].sort_values(KEY).reset_index(drop=True)
    if stratum != "all":
        ref = ref[ref["stratum"] == stratum].reset_index(drop=True)
    p, c = seed_matrix(parent, ref), seed_matrix(child, ref)
    corr = float(np.corrcoef(p.mean(axis=0), c.mean(axis=0))[0, 1])
    c = c * (rmse_by_seed(p).mean() / rmse_by_seed(c).mean())
    return p, c, ref["cluster_id"].to_numpy(), float(rmse_by_seed(p).mean()), corr


def power_curve(p, c, clusters, n_clusters_target, rng, effect_model: str = "lognormal") -> dict[float, float]:
    labels = np.unique(clusters)
    members = {g: np.where(clusters == g)[0] for g in labels}
    hits = {eps: 0 for eps in EFFECTS}
    for _ in range(N_REP):
        drawn = rng.choice(labels, size=n_clusters_target, replace=True)
        idx = np.concatenate([members[g] for g in drawn])
        new_ids = np.concatenate([np.full(members[g].size, i) for i, g in enumerate(drawn)])
        z = rng.standard_normal(len(drawn))
        per_cluster = np.exp(SIGMA * z - SIGMA**2 / 2) if effect_model == "lognormal" else 1.0 + z
        mult = np.concatenate([np.full(members[g].size, m) for g, m in zip(drawn, per_cluster)])
        pp = p[rng.integers(0, p.shape[0], size=p.shape[0])][:, idx]
        cc0 = c[rng.integers(0, c.shape[0], size=c.shape[0])][:, idx]
        for eps in EFFECTS:
            cc = cc0 * np.clip(1.0 - eps * mult, 0.0, None)[None, :]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = paired_cluster_test(parent=list(pp), child=list(cc), cluster_ids=new_ids)
            hits[eps] += bool(r.p_value < 0.05 and r.delta < 0)
    return {eps: hits[eps] / N_REP for eps in EFFECTS}


def mde(curve: dict[float, float]) -> float | None:
    return next((e for e in EFFECTS if e > 0 and curve[e] >= POWER_TARGET), None)


def seed_floor(p: np.ndarray, c: np.ndarray, n_seeds: int) -> dict:
    """MDE (deg C) from training randomness alone, large-G normal approximation:
    2.80 * sqrt(var_p/S + var_c/S), var = across-seed variance of RMSE (a deterministic
    model contributes 0). 95% interval from a chi-square with Satterthwaite df."""
    parts = [np.var(rmse_by_seed(m), ddof=1) / n_seeds for m in (p, c) if m.shape[0] > 1]
    dfs = [m.shape[0] - 1 for m in (p, c) if m.shape[0] > 1]
    v = float(sum(parts))
    df = v**2 / sum(pi**2 / d for pi, d in zip(parts, dfs))
    lo, hi = df * v / sp_stats.chi2.ppf(0.975, df), df * v / sp_stats.chi2.ppf(0.025, df)
    k = 1.96 + 0.84
    return {"floor": k * np.sqrt(v), "ci_low": k * np.sqrt(lo), "ci_high": k * np.sqrt(hi), "df": df}


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    results, floors = [], {}
    for template, child in TEMPLATES.items():
        for stratum, sizes in DESIGNS.items():
            p, c, clusters, base, corr = prepare(child, stratum)
            floors[f"{template} | {stratum}"] = {"5 seeds": seed_floor(p, c, 5), "10 seeds": seed_floor(p, c, 10),
                                                 "parent_rmse": base, "error_correlation": corr}
            for g in sizes:
                curve = power_curve(p, c, clusters, g, rng)
                m = mde(curve)
                results.append({"template": template, "stratum": stratum, "n_clusters": g, "parent_rmse": base,
                                "error_correlation": corr, "power": {f"{e:.3f}": v for e, v in curve.items()},
                                "false_positive_rate_one_sided": curve[0.0],
                                "mde_relative": m, "mde_deg_c": None if m is None else m * base})
                print(f"{template:38s} {stratum:8s} G={g:2d} corr={corr:.2f} FPR={curve[0.0]:.3f} "
                      f"MDE={'> 50%' if m is None else f'{m:.1%} ({m * base:.3f} C)'}", flush=True)

    sensitivity = []
    p, c, clusters, base, corr = prepare("damped_persistence", "all")
    for g in DESIGNS["all"]:
        curve = power_curve(p, c, clusters, g, rng, effect_model="cv1")
        m = mde(curve)
        sensitivity.append({"template": "less similar", "stratum": "all", "n_clusters": g, "effect_model": "eps*(1+z), CV=1",
                            "power": {f"{e:.3f}": v for e, v in curve.items()}, "mde_relative": m,
                            "max_power": max(curve.values())})
        print(f"sensitivity CV=1 all G={g:2d} MDE={'none' if m is None else f'{m:.1%}'} max power={max(curve.values()):.2f}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "mde.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump({"replicates": N_REP, "power_target": POWER_TARGET, "alpha": 0.05, "effects": EFFECTS,
                   "seed_noise_floor_deg_c": floors, "results": results, "sensitivity_cv1": sensitivity}, f, indent=2, default=float)

    lines = [
        "# Minimum detectable effect (paired cluster test, 80% power, alpha 0.05)", "",
        "Generated by `python -m evaluation.mde_simulation` from val predictions. Exact null + an UNEVEN planted",
        "relative RMSE improvement (log-normal across clusters, CV ~0.5). OPTIMISTIC lower bounds: pseudo-datasets reuse",
        "the few observed val clusters (9; 5 for extreme). Power has SE up to ~0.03 (300 replicates), so MDEs are",
        "accurate to about one grid step. Success = significant AND in the right direction, so the nominal",
        "false-positive rate is 0.025.", "",
        "| Template (error correlation) | Stratum | G | Parent RMSE | FPR (nominal 0.025) | MDE (relative) | MDE (deg C) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        rel = "> 50%" if r["mde_relative"] is None else f"{r['mde_relative']:.1%}"
        abs_c = "-" if r["mde_deg_c"] is None else f"{r['mde_deg_c']:.3f}"
        lines.append(f"| {r['template']} ({r['error_correlation']:.2f}) | {r['stratum']} | {r['n_clusters']} | "
                     f"{r['parent_rmse']:.3f} | {r['false_positive_rate_one_sided']:.3f} | {rel} | {abs_c} |")
    lines += ["", "Footnote: at G = 36 the less-similar template's false-positive rate was ~0.038 in a 1,500-replicate",
              "re-run (nominal 0.025), from the seed bootstrap; its 5% MDE is therefore slightly flattering.", "",
              "## Sensitivity: more uneven effect (eps * (1 + z), CV = 1; some clusters get worse)", "",
              "| Template | Stratum | G | MDE | Max power on grid |", "|---|---|---|---|---|"]
    for r in sensitivity:
        rel = "not reached" if r["mde_relative"] is None else f"{r['mde_relative']:.1%}"
        lines.append(f"| {r['template']} | {r['stratum']} | {r['n_clusters']} | {rel} | {r['max_power']:.2f} |")
    lines += ["", "With effects this uneven, power is capped by the number of clusters: on the val split alone (G = 9) an",
              "improvement may be undetectable at any size -- one more reason to use rolling-origin folds.", ""]
    lines += ["", "## Seed-noise floor", "",
              "MDE from training randomness alone (A1 has 5 seeds; damped persistence is deterministic). Adding clusters",
              "resampled from the same data does not reduce it; more seeds do (and more extreme instances per seed may",
              "reduce the per-seed RMSE variance itself). Interval: 95%, chi-square with Satterthwaite df.", "",
              "| Template / stratum | 5 seeds | 10 seeds |", "|---|---|---|"]
    for k, f in floors.items():
        a, b = f["5 seeds"], f["10 seeds"]
        lines.append(f"| {k} | {a['floor']:.3f} C [{a['ci_low']:.3f}, {a['ci_high']:.3f}] | "
                     f"{b['floor']:.3f} C [{b['ci_low']:.3f}, {b['ci_high']:.3f}] |")
    lines += ["", "Observed on val for comparison: RA-v1 vs A1 = +0.7% (all), -6.6% (extreme)."]
    (OUT_DIR / "mde.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
