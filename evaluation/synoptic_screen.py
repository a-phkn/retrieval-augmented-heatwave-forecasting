"""
Screen of Rz, retrieval on the 500 hPa weather pattern (pre-registered 2026-10-17, context/decisions.md,
before any z500 data was downloaded). No training; training years only, with the method of
evaluation/retrieval_information_check.py (fit on each fold's training windows before its last 2
training years, score on those 2 years, deg C, bootstrap over the 8 held-out years).

The baseline is a linear stand-in for what C3-hop already sees: the query-only features (v1's 17 window
features + the target's last-day and 14-day-mean anomaly) + Rg's regional readings (upstream Tmax, and
dew point / RH for WBGT, last 3 input days) + C3-hop's own normalised upstream inputs on the last input
day (27 points x 12 features). Each retrieval mode's analogue signal (the mean standardised anomaly of
its analogues' next 5 days) is added per lead.

Pass rule per family (all days): (i) baseline + Rz vs baseline and (ii) baseline + Rz vs baseline +
R0-rand both have 95% CIs entirely below 0. Rg on the same baseline and extreme days: descriptive.
Rz is trained (C3hop_Rz) only in a family where it passes.

Run from repo root:  python -m evaluation.synoptic_screen   ->  evaluation_v2/synoptic_screen.{md,json}
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from evaluation.retrieval_information_check import (
    FAMILIES, HELD_OUT_YEARS, OUT_DIR, _ci, analogue_signal, clim_cols, cluster_boot, ridge_fit_predict, rmse_seedmean,
)
from retrieval.fold_retrieval import BUFFER_DAYS, DOY_WINDOW, K_DEFAULT, FoldRetriever, region_features, window_features
from training.folds import FOLDS, FORECAST_DAYS, INPUT_DAYS, fold_bounds
from training.graph_data import build_upstream, load_upstream

MODES = {"Rz": ("synoptic", None), "R0-rand": ("rand", range(10)), "Rg": ("region", None)}
CONTRASTS = (("Rz", "base"), ("Rz", "R0-rand"), ("Rg", "base"), ("R0-rand", "base"), ("Rz", "Rg"))


def fold_family(fold: str, target: str, labels: str, up_table: pd.DataFrame) -> dict:
    fr = FoldRetriever(fold, labels, target)
    d = fr.d
    mean_col, std_col = clim_cols(target)
    z = ((d[target] - d[mean_col]) / d[std_col]).to_numpy(dtype=np.float64)
    q = pd.DatetimeIndex(fr.train_w["query_date"])
    pos = d.index.get_indexer(q)
    cand_pos = d.index.get_indexer(fr.cand_dates)
    leads = np.arange(FORECAST_DAYS)
    y = z[pos[:, None] + leads]
    own = np.column_stack([z[pos - 1], z[pos[:, None] + np.arange(-INPUT_DAYS, 0)].mean(axis=1)])
    upd = build_upstream(fold, up_table)
    up_last = upd.x[upd.positions(q)[:, -1]].reshape(len(q), -1).astype(np.float64)
    x = np.column_stack([window_features(d, q), own, region_features(fr._region_state()[3], q), up_last])
    train_end, _, _ = fold_bounds(fold)
    cutoff = train_end - pd.DateOffset(years=HELD_OUT_YEARS)
    held, before = q > cutoff, q <= cutoff - pd.Timedelta(days=BUFFER_DAYS)

    retrieved = {name: [fr.retrieve(q, mode, K_DEFAULT, seed=s).idx for s in (seeds or [None])]
                 for name, (mode, seeds) in MODES.items()}
    full = np.ones(len(q), dtype=bool)
    for runs in retrieved.values():
        for idx in runs:
            full &= (idx >= 0).all(axis=1)
    fit, ev = full & before, full & held
    scale = d[std_col].to_numpy()[pos[:, None] + leads][ev]

    def sq_err(pred_z: np.ndarray) -> np.ndarray:
        return ((pred_z - y[ev]) * scale) ** 2

    out = {"base": sq_err(ridge_fit_predict(x[fit], y[fit], x[ev]))[None]}
    corr, same_season = {}, {}
    q_doy = (q - pd.Timedelta(days=1)).dayofyear.to_numpy()
    for name, runs in retrieved.items():
        errs, cs = [], []
        for idx in runs:
            sig = np.zeros((len(q), FORECAST_DAYS))
            sig[full] = analogue_signal(z, idx[full], cand_pos)
            per_lead = []
            for lead in range(FORECAST_DAYS):
                xa = np.column_stack([x, sig[:, lead]])
                per_lead.append(ridge_fit_predict(xa[fit], y[fit, lead:lead + 1], xa[ev])[:, 0])
            errs.append(sq_err(np.column_stack(per_lead)))
            cs.append([np.corrcoef(sig[ev, lead], y[ev, lead])[0, 1] for lead in range(FORECAST_DAYS)])
        out[name] = np.stack(errs)
        corr[name] = np.mean(cs, axis=0)
        diff = np.abs(fr.cand_doy[runs[0][ev]] - q_doy[ev][:, None])
        same_season[name] = float((np.minimum(diff, 365 - diff) <= DOY_WINDOW).mean())
    stratum = d["stratum"].to_numpy()[pos[:, None] + leads][ev]
    return {"errors": out, "corr": corr, "same_season": same_season, "n_fit": int(fit.sum()), "n_held": int(ev.sum()),
            "year": np.repeat(q[ev].year.to_numpy()[:, None], FORECAST_DAYS, axis=1), "extreme": stratum == "extreme"}


def passes(res: dict) -> bool:
    """Pre-registered rule: all days, Rz vs baseline and Rz vs R0-rand both with CI entirely below 0."""
    return bool(res["all"]["Rz vs base"][2] < 0 and res["all"]["Rz vs R0-rand"][2] < 0)


def summarise(parts: list[dict]) -> dict:
    rng = np.random.default_rng(20261017)
    res = {}
    for subset in ("all", "extreme"):
        mask = np.concatenate([p["extreme"].ravel() if subset == "extreme" else np.ones(p["extreme"].size, bool)
                               for p in parts])
        years = np.concatenate([p["year"].ravel() for p in parts])[mask]
        se = {k: np.concatenate([p["errors"][k].reshape(p["errors"][k].shape[0], -1) for p in parts], axis=1)[:, mask]
              for k in parts[0]["errors"]}
        rows = {"n_rows": int(mask.sum()), "rmse": {k: rmse_seedmean(v) for k, v in se.items()}}
        for a, b in CONTRASTS:
            rows[f"{a} vs {b}"] = cluster_boot(lambda w, a=a, b=b: rmse_seedmean(se[a], w) - rmse_seedmean(se[b], w),
                                               years, rng)
        res[subset] = rows
    res["corr_by_lead"] = {k: np.mean([p["corr"][k] for p in parts], axis=0).tolist() for k in MODES}
    res["same_season_share"] = {k: float(np.mean([p["same_season"][k] for p in parts])) for k in MODES}
    res["passes"] = passes(res)
    return res


def write_report(results: dict, meta: dict) -> None:
    L = ["# Rz screen: retrieval on the 500 hPa weather pattern (rule written 2026-10-17, before any data)", "",
         "No training; training years only (each fold's last 2 training years held out). RMSE in °C pooled over 4 folds "
         "x leads 1-5; Δ = first minus second (negative = the first is better); 95% CI by bootstrap over the 8 held-out "
         "years. Baseline = what C3-hop already sees, as a linear model (query-only features + Rg's regional readings + "
         "C3-hop's upstream inputs on the last input day).", ""]
    for fam, r in results.items():
        L += [f"## {fam}: **{'PASS: train C3hop_Rz' if r['passes'] else 'fail: not trained'}**", "",
              f"Held-out query-days: {r['all']['n_rows']} (extreme: {r['extreme']['n_rows']}); fit queries: "
              f"{meta[fam]['n_fit']}. Baseline RMSE: all {r['all']['rmse']['base']:.3f}, extreme "
              f"{r['extreme']['rmse']['base']:.3f} °C.", "",
              "| Comparison | All days Δ (95% CI) | Extreme days Δ (95% CI, descriptive) |", "|---|---|---|"]
        for a, b in CONTRASTS:
            k = f"{a} vs {b}"
            tag = " (rule)" if k in ("Rz vs base", "Rz vs R0-rand") else ""
            L.append(f"| {k.replace('base', 'baseline')}{tag} | {_ci(r['all'][k])} | {_ci(r['extreme'][k])} |")
        L += ["", "Correlation of the analogue signal with the truth, leads 1-5 (held out): " +
              "; ".join(f"{k} " + ", ".join(f"{c:+.2f}" for c in v) for k, v in r["corr_by_lead"].items()),
              "Analogues within ±30 days of the query's time of year: " +
              "; ".join(f"{k} {v:.0%}" for k, v in r["same_season_share"].items()), ""]
    L += ["Notes: the signal enters a linear model, so a null says the simple analogue-mean information is absent "
          "beyond the baseline. R0-rand: 10 seeds, RMSE averaged over seeds."]
    (OUT_DIR / "synoptic_screen.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    up_table = load_upstream()
    results, meta = {}, {}
    for fam, (target, labels) in FAMILIES.items():
        parts = []
        for fold in FOLDS:
            parts.append(fold_family(fold, target, labels, up_table))
            print(f"{fam} {fold}: fit {parts[-1]['n_fit']}, held {parts[-1]['n_held']}", flush=True)
        results[fam] = summarise(parts)
        meta[fam] = {"n_fit": sum(p["n_fit"] for p in parts)}
    write_report(results, meta)
    (OUT_DIR / "synoptic_screen.json").write_text(json.dumps(results, indent=2, default=float) + "\n", encoding="utf-8")
    print({fam: r["passes"] for fam, r in results.items()})


if __name__ == "__main__":
    main()
