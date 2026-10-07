"""
Retrieval information check (specified in context/decisions.md 2026-10-07, before it was run).

Question: do the retrieved analogues' own next 5 days tell us anything about the query's next 5
days BEYOND what the query's 14 input days already say? G3 found that retrieval does not help and
that random analogues do as well as retrieved ones. That has two possible causes:
  - the information is there but the network does not use it  -> tune how analogues are fed in;
  - the information is not there                               -> tune how analogues are matched.

Method (no training, no validation block used):
  * Per fold and family, queries = the fold's TRAINING windows with a full set of K analogues.
    Analogues come from retrieval.fold_retrieval.FoldRetriever, exactly as in the runs.
  * Signal = the mean standardised anomaly of the analogues' next 5 days, per lead (AnEn signal).
  * Baseline = ridge (alpha 1) on the query's own inputs: v1's 17 window features plus the
    target's last-day and 14-day-mean standardised anomaly. Augmented = baseline + signal.
  * Fit on training queries before the fold's last 2 training years; score on those 2 years,
    in deg C. Uncertainty: bootstrap over the 8 held-out years (fragile: few clusters).

Run from repo root:
    python -m evaluation.retrieval_information_check
Writes evaluation_v2/retrieval_information_check.{md,json}.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from retrieval.fold_retrieval import BUFFER_DAYS, K_DEFAULT, FoldRetriever, region_features, window_features
from training.folds import FOLDS, FORECAST_DAYS, INPUT_DAYS, fold_bounds

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "evaluation_v2"
FAMILIES = {"Tmax": ("t_max", "v2"), "WBGT (physical)": ("wbgt_lj_max", "wbgt")}
MODES = {"R0": ("sim", None), "R0-rand": ("rand", range(10)), "R1": ("time", None), "R1-rand": ("time_rand", range(10)),
         "Rg": ("region", None)}
PAIRS = (("R0", "R0-rand"), ("R1", "R1-rand"), ("Rg", "R0-rand"))  # Rg shares R0's pool (pre-registered 2026-10-07)
UPSTREAM_EXTRA = ("Rg",)  # descriptive: also scored against a baseline that is given the upstream readings
HELD_OUT_YEARS = 2
RIDGE_ALPHA = 1.0
N_BOOT = 2000


def clim_cols(target: str) -> tuple[str, str]:
    return ("clim_mean_t_max", "clim_std_t_max") if target == "t_max" else (f"clim_mean_{target}", f"clim_std_{target}")


def ridge_fit_predict(x_fit: np.ndarray, y_fit: np.ndarray, x_new: np.ndarray, alpha: float = RIDGE_ALPHA) -> np.ndarray:
    """Ridge on standardised columns (fit-set mean/SD), unpenalised intercept; y may be (n, L)."""
    mu, sd = x_fit.mean(axis=0), x_fit.std(axis=0)
    sd[sd == 0] = 1.0
    a, b = (x_fit - mu) / sd, (x_new - mu) / sd
    y_mu = y_fit.mean(axis=0)
    w = np.linalg.solve(a.T @ a + alpha * np.eye(a.shape[1]), a.T @ (y_fit - y_mu))
    return b @ w + y_mu


def analogue_signal(z: np.ndarray, idx: np.ndarray, cand_pos: np.ndarray) -> np.ndarray:
    """(N, L) mean standardised anomaly of each query's analogues' next L days.
    idx: (N, K) candidate positions (all >= 0); cand_pos: daily-table position of each candidate's first
    forecast day."""
    leads = np.arange(FORECAST_DAYS)
    return z[cand_pos[idx][..., None] + leads].mean(axis=1)


def fold_family(fold: str, target: str, labels: str) -> dict:
    """Held-out squared errors (deg C^2) for the baseline and for each mode's augmented model."""
    fr = FoldRetriever(fold, labels, target)
    d = fr.d
    mean_col, std_col = clim_cols(target)
    z = ((d[target] - d[mean_col]) / d[std_col]).to_numpy(dtype=np.float64)
    q = pd.DatetimeIndex(fr.train_w["query_date"])
    pos = d.index.get_indexer(q)
    cand_pos = d.index.get_indexer(fr.cand_dates)
    leads = np.arange(FORECAST_DAYS)
    y = z[pos[:, None] + leads]  # (N, L) truth, standardised
    own = np.column_stack([z[pos - 1], z[pos[:, None] + np.arange(-INPUT_DAYS, 0)].mean(axis=1)])
    base_x = np.column_stack([window_features(d, q), own])
    train_end, _, _ = fold_bounds(fold)
    cutoff = train_end - pd.DateOffset(years=HELD_OUT_YEARS)
    held = q > cutoff
    before = q <= cutoff - pd.Timedelta(days=BUFFER_DAYS)  # fit targets end before any held-out input starts

    retrieved = {}
    for name, (mode, seeds) in MODES.items():
        retrieved[name] = [fr.retrieve(q, mode, K_DEFAULT, seed=s).idx for s in (seeds or [None])]
    full = np.ones(len(q), dtype=bool)
    for runs in retrieved.values():
        for idx in runs:
            full &= (idx >= 0).all(axis=1)
    fit, ev = full & before, full & held

    scale = d[std_col].to_numpy()[pos[:, None] + leads][ev]
    def sq_err(pred_z: np.ndarray) -> np.ndarray:  # deg C^2 per (held query, lead)
        return ((pred_z - y[ev]) * scale) ** 2

    # descriptive: the same baseline also given the regional readings Rg matches on (the graph backbone sees them)
    up_x = np.column_stack([base_x, region_features(fr._region_state()[3], q)])
    out = {"base": sq_err(ridge_fit_predict(base_x[fit], y[fit], base_x[ev]))[None],
           "base+up": sq_err(ridge_fit_predict(up_x[fit], y[fit], up_x[ev]))[None]}

    def augmented(x: np.ndarray, sig: np.ndarray) -> np.ndarray:
        per_lead = []
        for lead in range(FORECAST_DAYS):  # the signal enters per lead (its own lead's analogue mean)
            xa = np.column_stack([x, sig[:, lead]])
            per_lead.append(ridge_fit_predict(xa[fit], y[fit, lead:lead + 1], xa[ev])[:, 0])
        return np.column_stack(per_lead)

    corr = {}
    for name, runs in retrieved.items():
        errs, errs_up, cs = [], [], []
        for idx in runs:
            sig = np.zeros((len(q), FORECAST_DAYS))
            sig[full] = analogue_signal(z, idx[full], cand_pos)
            errs.append(sq_err(augmented(base_x, sig)))
            if name in UPSTREAM_EXTRA:
                errs_up.append(sq_err(augmented(up_x, sig)))
            cs.append([np.corrcoef(sig[ev, lead], y[ev, lead])[0, 1] for lead in range(FORECAST_DAYS)])
        out[name] = np.stack(errs)  # (seeds, n_held, L)
        if errs_up:
            out[f"{name}+up"] = np.stack(errs_up)
        corr[name] = np.mean(cs, axis=0)
    stratum = d["stratum"].to_numpy()[pos[:, None] + leads][ev]
    return {"errors": out, "corr": corr, "year": np.repeat(q[ev].year.to_numpy()[:, None], FORECAST_DAYS, axis=1),
            "extreme": stratum == "extreme", "n_fit": int(fit.sum()), "n_held": int(ev.sum())}


def rmse_seedmean(se: np.ndarray, w: np.ndarray | None = None) -> float:
    """Mean over seeds of RMSE; se: (S, n). Optional per-row weights (bootstrap)."""
    w = np.ones(se.shape[1]) if w is None else w
    return float(np.mean(np.sqrt((se * w).sum(axis=1) / w.sum())))


def cluster_boot(diff_fn, years: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    """Point estimate and 95% percentile CI of diff_fn(weights) over a year-cluster bootstrap."""
    uniq = np.unique(years)
    inv = np.searchsorted(uniq, years)
    point = diff_fn(np.ones(len(years)))
    reps = []
    for _ in range(N_BOOT):
        counts = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        reps.append(diff_fn(counts[inv].astype(float)))
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return point, float(lo), float(hi)


def summarise(parts: list[dict]) -> dict:
    """Pool folds (and leads) and test the pre-specified contrasts."""
    rng = np.random.default_rng(20261007)
    res = {}
    for subset in ("all", "extreme"):
        rows = {}
        mask = np.concatenate([p["extreme"].ravel() if subset == "extreme" else np.ones(p["extreme"].size, bool) for p in parts])
        years = np.concatenate([p["year"].ravel() for p in parts])[mask]
        se = {k: np.concatenate([p["errors"][k].reshape(p["errors"][k].shape[0], -1) for p in parts], axis=1)[:, mask]
              for k in parts[0]["errors"]}
        rows["n_rows"], rows["n_years"] = int(mask.sum()), int(len(np.unique(years)))
        rows["rmse"] = {k: rmse_seedmean(v) for k, v in se.items()}
        for k in MODES:
            rows[f"{k} vs base"] = cluster_boot(lambda w, k=k: rmse_seedmean(se[k], w) - rmse_seedmean(se["base"], w), years, rng)
        for rung, rand in PAIRS:
            rows[f"{rung} vs {rand}"] = cluster_boot(lambda w, a=rung, b=rand: rmse_seedmean(se[a], w) - rmse_seedmean(se[b], w),
                                                    years, rng)
        for k in UPSTREAM_EXTRA:
            rows[f"{k}+up vs base+up"] = cluster_boot(lambda w, k=k: rmse_seedmean(se[f"{k}+up"], w) - rmse_seedmean(se["base+up"], w),
                                                      years, rng)
            rows["base+up vs base"] = cluster_boot(lambda w: rmse_seedmean(se["base+up"], w) - rmse_seedmean(se["base"], w),
                                                   years, rng)
        res[subset] = rows
    by_lead = {}
    for lead in range(FORECAST_DAYS):
        se = {k: np.concatenate([p["errors"][k][:, :, lead] for p in parts], axis=1) for k in parts[0]["errors"]}
        by_lead[lead + 1] = {k: rmse_seedmean(v) for k, v in se.items()}
    res["by_lead"] = by_lead
    res["corr_by_lead"] = {k: np.mean([p["corr"][k] for p in parts], axis=0).tolist() for k in MODES}
    verdict = {}
    for rung, rand in PAIRS:
        a, b = res["all"][f"{rung} vs base"], res["all"][f"{rung} vs {rand}"]
        verdict[rung] = bool(a[2] < 0 and b[2] < 0)
    res["information_present"] = verdict
    return res


def _ci(t) -> str:
    return f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"


def write_report(results: dict, meta: dict) -> None:
    L = ["# Retrieval information check (2026-10-07; Rg added the same day, pre-registered)", "",
         "Specified in `context/decisions.md` before it was run. No training; training years only (each fold's last "
         "2 training years held out; no validation block used).", "",
         "Question: do the analogues' own next 5 days add information beyond the query's own 14 input days? "
         "RMSE in °C, pooled over 4 folds x leads 1-5. Δ = augmented minus comparison; negative = analogues help. "
         "95% CI: bootstrap over the 8 held-out years (few clusters: fragile).", ""]
    for fam, r in results.items():
        L += [f"## {fam}", "",
              f"Held-out query-days: {r['all']['n_rows']} (extreme: {r['extreme']['n_rows']}); fit queries: {meta[fam]['n_fit']}.", "",
              "**Information present (pre-specified rule):** "
              + ", ".join(f"{k} {'yes' if v else 'no'}" for k, v in r["information_present"].items()), "",
              "| Comparison | All days Δ RMSE (95% CI) | Extreme days Δ RMSE (95% CI, descriptive) |", "|---|---|---|"]
        for k in [f"{m} vs base" for m in MODES] + [f"{a} vs {b}" for a, b in PAIRS]:
            L.append(f"| {k.replace('base', 'query-only baseline')} | {_ci(r['all'][k])} | {_ci(r['extreme'][k])} |")
        L += ["", "Descriptive (Rg, pre-registered 2026-10-07): the baseline is also given the upstream readings Rg "
              "matches on, as the graph backbone would be. Does Rg still add anything?", "",
              "| Comparison | All days Δ RMSE (95% CI) | Extreme days Δ RMSE (95% CI) |", "|---|---|---|"]
        for k in ["base+up vs base"] + [f"{m}+up vs base+up" for m in UPSTREAM_EXTRA]:
            label = {"base+up vs base": "query-only baseline + upstream vs query-only baseline"}.get(
                k, k.replace("+up vs base+up", " added to (query-only baseline + upstream)"))
            L.append(f"| {label} | "
                     f"{_ci(r['all'][k])} | {_ci(r['extreme'][k])} |")
        L += ["", f"Query-only baseline RMSE: all {r['all']['rmse']['base']:.3f} °C, extreme {r['extreme']['rmse']['base']:.3f} °C.", "",
              "By lead (RMSE °C, all days) and correlation of the analogue signal with the truth (held-out):", "",
              "| Lead | Baseline | " + " | ".join(MODES) + " | " + " | ".join(f"corr {m}" for m in MODES) + " |",
              "|---|---|" + "---|" * (2 * len(MODES))]
        for lead, v in r["by_lead"].items():
            L.append(f"| {lead} | {v['base']:.3f} | " + " | ".join(f"{v[m]:.3f}" for m in MODES) + " | "
                     + " | ".join(f"{r['corr_by_lead'][m][lead - 1]:+.3f}" for m in MODES) + " |")
        L.append("")
    L += ["Notes:",
          "- Random versions (R0-rand, R1-rand): 10 seeds; RMSE is the mean over seeds, as in the main comparison.",
          "- The signal enters a linear model. A network could in principle extract non-linear information this misses; "
          "a null here says the simple, AnEn-style information is absent.",
          "- The analogue features are v1's 17 Tmax-based features for both families (as in the runs)."]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "retrieval_information_check.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    results, meta = {}, {}
    for fam, (target, labels) in FAMILIES.items():
        parts = []
        for fold in FOLDS:
            parts.append(fold_family(fold, target, labels))
            print(f"{fam} {fold}: fit {parts[-1]['n_fit']}, held {parts[-1]['n_held']}", flush=True)
        results[fam] = summarise(parts)
        meta[fam] = {"n_fit": sum(p["n_fit"] for p in parts)}
    write_report(results, meta)
    (OUT_DIR / "retrieval_information_check.json").write_text(json.dumps(results, indent=2, default=float) + "\n", encoding="utf-8")
    for fam, r in results.items():
        print(f"{fam}: information present {r['information_present']}")
        for k in ["R0 vs base", "R1 vs base", "Rg vs base", "R0 vs R0-rand", "R1 vs R1-rand", "Rg vs R0-rand",
                  "base+up vs base", "Rg+up vs base+up"]:
            print(f"  {k:14s} all {_ci(r['all'][k])}")


if __name__ == "__main__":
    main()
