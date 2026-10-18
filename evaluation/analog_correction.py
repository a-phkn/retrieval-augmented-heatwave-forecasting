"""
Option 2: analog error correction of the final C3-hop forecasts ("AnEn-corr"; pre-registered 2026-10-18,
context/decisions.md, before it was run; post-G-R* exploration). Analog post-processing after Delle Monache
et al. (2011, Monthly Weather Review 139: 3554): find past forecasts that looked like today's and remove the
error the model made on them.

Per family (final C3-hop), fold, seed and lead:
  pool       = honest past errors: the fold's own inner-block forecasts (its 2 early-stopping years, not
               fitted on) + earlier folds' validation forecasts on dates the inner block does not cover;
               only rows whose target date is before the fold's validation start;
  candidates = pool rows whose target day of year is within +-45 days (circular);
  predictors = forecast at this lead, mean of the window's 5 lead forecasts, last observed value (q-1),
               deg C, each standardised by its SD over the pool; Euclidean distance; K = 20 nearest;
  AnEn-corr  = forecast - mean error (forecast - actual) of the K.
Controls: raw; random (K random candidates from the same +-45-day set: season without similarity);
linear (OLS actual = a + b x forecast on the whole pool, per seed and lead).
Pass rule (all four): (1) vs raw, all days: CI not entirely above 0; (2) vs raw, extreme: CI entirely
below 0; (3) vs random, extreme: CI entirely below 0; (4) vs linear, extreme: CI entirely below 0.
Forecaster's-dilemma checks: all-days deltas, extreme bias, forecast-conditioned hot days.

Run from repo root:  python -m evaluation.analog_correction  ->  evaluation_v2/analog_correction.{md,json}
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from evaluation.compare_v2 import (
    PRED_DIR, _ci, _fmt_p, _tmax_clim_by_date, _wbgt_thresholds, compare, extreme_bias, forecast_conditioned,
    forecast_conditioned_wbgt, rmse_by,
)
from evaluation.retrieval_information_check import OUT_DIR
from training.folds import FOLDS, FORECAST_DAYS, fold_bounds, fold_daily

FAMILIES = {"Tmax": ("A1prime_hw5_C3hop", "t_max", "v2"), "WBGT (physical)": ("A2Lr_hw5_C3hop", "wbgt_lj_max", "wbgt")}
INNER_DIR = PRED_DIR / "inner"
K_ANALOGS, SEASON_DAYS = 20, 45
PREDICTORS = ["pred", "fc_mean", "last_obs"]
VARIANTS = ("raw", "anen", "random", "linear")


def doy_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Circular day-of-year distance (365-day year)."""
    d = np.abs(a - b)
    return np.minimum(d, 365 - d)


def add_predictors(df: pd.DataFrame, observed: pd.Series) -> pd.DataFrame:
    """Adds the window's mean forecast over its 5 leads, the last observed value (q-1) and the target doy."""
    out = df.copy()
    out["fc_mean"] = out.groupby(["seed", "query_date"])["pred"].transform("mean")
    out["last_obs"] = observed.reindex(pd.DatetimeIndex(out["query_date"]) - pd.Timedelta(days=1)).to_numpy()
    out["doy"] = pd.DatetimeIndex(out["target_date"]).dayofyear.to_numpy()
    if out[PREDICTORS].isna().any().any():
        raise ValueError("missing predictor values")
    return out


def build_pool(inner: pd.DataFrame, earlier_val: list[pd.DataFrame], val_start: pd.Timestamp) -> pd.DataFrame:
    """The fold's honest-error pool: its inner block, plus earlier folds' validation forecasts on query dates
    the inner block does not cover, restricted to target dates before the fold's validation start."""
    parts = [inner]
    if earlier_val:
        ev = pd.concat(earlier_val, ignore_index=True)
        parts.append(ev[~ev["query_date"].isin(set(inner["query_date"]))])
    pool = pd.concat(parts, ignore_index=True)
    return pool[pool["target_date"] < val_start].reset_index(drop=True)


def correct_block(val: pd.DataFrame, pool: pd.DataFrame, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Corrected forecasts for one seed and lead (val and pool: rows of that seed and lead only)."""
    xp, xv = pool[PREDICTORS].to_numpy(float), val[PREDICTORS].to_numpy(float)
    err, sd = pool["error"].to_numpy(float), xp.std(axis=0)
    sd[sd == 0] = 1.0
    season = doy_distance(val["doy"].to_numpy()[:, None], pool["doy"].to_numpy()[None, :]) <= SEASON_DAYS
    if season.sum(axis=1).min() < K_ANALOGS:
        raise ValueError("fewer than K in-season candidates")
    dist = ((((xv[:, None, :] - xp[None, :, :]) / sd) ** 2).sum(axis=-1))
    dist[~season] = np.inf
    nn = np.argpartition(dist, K_ANALOGS - 1, axis=1)[:, :K_ANALOGS]
    rand = np.array([err[rng.choice(np.flatnonzero(s), K_ANALOGS, replace=False)].mean() for s in season])
    b, a = np.polyfit(pool["pred"].to_numpy(float), pool["actual"].to_numpy(float), 1)
    pred = val["pred"].to_numpy(float)
    return {"raw": pred, "anen": pred - err[nn].mean(axis=1), "random": pred - rand, "linear": a + b * pred}


def family_frames(rid: str, target: str, labels: str) -> dict[str, pd.DataFrame]:
    """Pooled out-of-fold frames (long format, with 'fold') for every variant."""
    observed = fold_daily("f4", target, labels)[0][target]  # raw target, the same in every fold
    vals = {f: add_predictors(pd.read_parquet(PRED_DIR / rid / f"{f}.parquet").assign(fold=f), observed) for f in FOLDS}
    frames: dict[str, list[pd.DataFrame]] = {v: [] for v in VARIANTS}
    folds = list(FOLDS)
    for i, f in enumerate(folds):
        inner = add_predictors(pd.read_parquet(INNER_DIR / rid / f"{f}.parquet"), observed)
        pool = build_pool(inner, [vals[g] for g in folds[:i]], fold_bounds(f)[1])
        val = vals[f]
        for seed in sorted(val["seed"].unique()):
            for lead in range(1, FORECAST_DAYS + 1):
                v = val[(val["seed"] == seed) & (val["lead"] == lead)]
                p = pool[(pool["seed"] == seed) & (pool["lead"] == lead)]
                rng = np.random.default_rng([i, int(seed), lead, 20261018])
                for name, pred in correct_block(v, p, rng).items():
                    frames[name].append(v.assign(pred=pred, error=pred - v["actual"].to_numpy()))
    return {k: pd.concat(v, ignore_index=True).sort_values(["fold", "seed", "query_date", "lead"]).reset_index(drop=True)
            for k, v in frames.items()}


def gate(fr: dict[str, pd.DataFrame]) -> dict:
    a = fr["anen"]
    r = {"vs_raw_all": compare(fr["raw"], a, "all"), "vs_raw_ext": compare(fr["raw"], a, "extreme"),
         "vs_random_ext": compare(fr["random"], a, "extreme"), "vs_linear_ext": compare(fr["linear"], a, "extreme"),
         "vs_random_all": compare(fr["random"], a, "all"), "vs_linear_all": compare(fr["linear"], a, "all"),
         "linear_vs_raw_all": compare(fr["raw"], fr["linear"], "all"),
         "linear_vs_raw_ext": compare(fr["raw"], fr["linear"], "extreme"),
         "random_vs_raw_ext": compare(fr["raw"], fr["random"], "extreme")}
    r["conditions"] = (bool(r["vs_raw_all"]["ci_low"] <= 0), bool(r["vs_raw_ext"]["ci_high"] < 0),
                       bool(r["vs_random_ext"]["ci_high"] < 0), bool(r["vs_linear_ext"]["ci_high"] < 0))
    r["passes"] = all(r["conditions"])
    return r


def summary(fr: dict[str, pd.DataFrame], fam: str) -> dict:
    fc = (lambda df: forecast_conditioned(df, _tmax_clim_by_date())) if fam == "Tmax" else \
        (lambda df: forecast_conditioned_wbgt(df, _wbgt_thresholds()))
    return {k: {"rmse_all": rmse_by(df), "rmse_extreme": rmse_by(df[df["stratum"] == "extreme"]),
                "bias_all": float(df["error"].mean()), "bias_extreme": extreme_bias(df), "forecast_hot": fc(df)}
            for k, df in fr.items()}


def _t(r: dict) -> str:
    return f"{r['delta']:+.3f} {_ci(r)} (p {_fmt_p(r['p_value'])})"


def write_report(res: dict) -> None:
    L = ["# Option 2: analog error correction of C3-hop (rule written 2026-10-18, before it was run)", "",
         "Out of fold 2007-2018, 10 seeds x 4 folds, paired cluster-jackknife; Δ = AnEn-corr minus comparison "
         "(negative = AnEn-corr better). Honest past errors only (each fold's early-stopping years + earlier folds' "
         "validation years). K = 20 analogs within ±45 days of the time of year.", "",
         "## Decision", "",
         "| Family | (1) vs raw, all | (2) vs raw, extreme | (3) vs random, extreme | (4) vs linear, extreme | Result |",
         "|---|---|---|---|---|---|"]
    for fam, r in res.items():
        g = r["gate"]
        L.append(f"| {fam} | {_t(g['vs_raw_all'])} | {_t(g['vs_raw_ext'])} | {_t(g['vs_random_ext'])} | "
                 f"{_t(g['vs_linear_ext'])} | {'PASS' if g['passes'] else 'fail'} {g['conditions']} |")
    L += ["", "## Scores and forecaster's-dilemma checks", "",
          "| Family | Variant | RMSE all | RMSE extreme | Bias all | Bias extreme | Forecast hot days per seed | "
          "RMSE / bias on forecast hot days |", "|---|---|---|---|---|---|---|---|"]
    for fam, r in res.items():
        for k, s in r["summary"].items():
            h = s["forecast_hot"]
            L.append(f"| {fam} | {k} | {s['rmse_all']:.3f} | {s['rmse_extreme']:.3f} | {s['bias_all']:+.2f} | "
                     f"{s['bias_extreme']:+.2f} | {h['n_per_seed']:.0f} | {h['rmse']:.2f} / {h['bias']:+.2f} |")
    L += ["", "Also (all days): " + "; ".join(
        f"{fam}: AnEn vs random {_t(r['gate']['vs_random_all'])}, vs linear {_t(r['gate']['vs_linear_all'])}, "
        f"linear vs raw {_t(r['gate']['linear_vs_raw_all'])}" for fam, r in res.items()),
          "Extreme days, controls vs raw: " + "; ".join(
        f"{fam}: linear {_t(r['gate']['linear_vs_raw_ext'])}, random {_t(r['gate']['random_vs_raw_ext'])}"
        for fam, r in res.items())]
    (OUT_DIR / "analog_correction.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    res = {}
    for fam, (rid, target, labels) in FAMILIES.items():
        fr = family_frames(rid, target, labels)
        res[fam] = {"gate": gate(fr), "summary": summary(fr, fam)}
        print(fam, res[fam]["gate"]["conditions"], flush=True)
    write_report(res)
    (OUT_DIR / "analog_correction.json").write_text(json.dumps(res, indent=2, default=float) + "\n", encoding="utf-8")
    print({fam: r["gate"]["passes"] for fam, r in res.items()})


if __name__ == "__main__":
    main()
