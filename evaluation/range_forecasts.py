"""
Range (probabilistic) forecasts (option 2; specified 2026-10-18 in context/decisions.md before it was
computed; descriptive). Instead of one number per day, a forecast distribution:

  centre    the 10-seed ensemble mean, recalibrated on the fold's inner block (as calibration_check);
  spread    (a) Gaussian with the inner-block residual SD per lead, or (b) the empirical inner-block
            residuals per lead added to the centre (keeps their skew).
Scores out of fold 2007-2018: CRPS (continuous ranked probability score, deg C; lower is better), its skill
vs a climatology forecast (fold-train day-of-year mean and SD, Gaussian), and how often the truth falls in
the 80% (10th-90th percentile) and 50% intervals. C3-hop vs the Delhi-only control is tested on sqrt(CRPS)
with the paired cluster-jackknife (a monotone transform: tests equal mean CRPS).
Deliverable: predictions_v2/range/<run>.parquet with the 10th, 50th and 90th percentiles per forecast day.

Run from repo root:  python -m evaluation.range_forecasts   ->  evaluation_v2/range_forecasts.{md,json}
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import norm

from evaluation.calibration_check import apply_recal, fit_recal
from evaluation.compare_v2 import PRED_DIR, _ci, _fmt_p, compare, ensemble
from evaluation.retrieval_information_check import OUT_DIR
from training.folds import FOLDS, fold_daily

FAMILIES = {"Tmax": ("t_max", "v2", "A1prime_hw5"), "WBGT (physical)": ("wbgt_lj_max", "wbgt", "A2Lr_hw5")}
INNER_DIR = PRED_DIR / "inner"
RANGE_DIR = PRED_DIR / "range"
Z80, Z50 = norm.ppf(0.9), norm.ppf(0.75)


def crps_gaussian(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Closed-form CRPS of N(mu, sigma^2) at y (Gneiting et al. 2005)."""
    z = (y - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def crps_empirical(y: np.ndarray, centre: np.ndarray, resid: np.ndarray) -> np.ndarray:
    """CRPS of the sample distribution {centre - r : r in resid} (resid = forecast - actual), per row:
    E|X - y| - 0.5 E|X - X'|."""
    x = centre[:, None] - resid[None, :]
    term1 = np.abs(x - y[:, None]).mean(axis=1)
    r = np.sort(resid)
    m = len(r)
    term2 = 2.0 * np.sum((2 * np.arange(1, m + 1) - m - 1) * r) / (m * m)  # E|X - X'| from sorted samples
    return term1 - 0.5 * term2


def fold_rows(rid: str, target: str, labels: str, fold: str) -> pd.DataFrame:
    val = pd.read_parquet(PRED_DIR / rid / f"{fold}.parquet")
    inner = pd.read_parquet(INNER_DIR / rid / f"{fold}.parquet")
    coef = fit_recal(inner)
    ve, ie = ensemble(apply_recal(val, coef)), ensemble(apply_recal(inner, coef))
    d, _ = fold_daily(fold, target, labels)
    mcol, scol = ("clim_mean_t_max", "clim_std_t_max") if target == "t_max" else (f"clim_mean_{target}", f"clim_std_{target}")
    t = pd.DatetimeIndex(ve["target_date"])
    clim_mu, clim_sd = d[mcol].reindex(t).to_numpy(), d[scol].reindex(t).to_numpy()
    y, mu = ve["actual"].to_numpy(float), ve["pred"].to_numpy(float)
    out = []
    for lead, g in ve.groupby("lead"):
        idx = g.index.to_numpy()
        r = ie.loc[ie["lead"] == lead, "error"].to_numpy(float)
        sd = r.std(ddof=1)
        q10, q50, q90 = (np.quantile(mu[idx, None] - r[None, :], q, axis=1) for q in (0.1, 0.5, 0.9))
        q25, q75 = (np.quantile(mu[idx, None] - r[None, :], q, axis=1) for q in (0.25, 0.75))
        out.append(pd.DataFrame({
            "query_date": ve.loc[idx, "query_date"].to_numpy(), "lead": lead, "target_date": t[idx],
            "actual": y[idx], "centre": mu[idx], "stratum": ve.loc[idx, "stratum"].to_numpy(),
            "cluster_id": ve.loc[idx, "cluster_id"].to_numpy(), "fold": fold,
            "crps_gauss": crps_gaussian(y[idx], mu[idx], np.full(len(idx), sd)),
            "crps_emp": crps_empirical(y[idx], mu[idx], r),
            "crps_clim": crps_gaussian(y[idx], clim_mu[idx], clim_sd[idx]),
            "in80_gauss": np.abs(y[idx] - mu[idx]) <= Z80 * sd, "in50_gauss": np.abs(y[idx] - mu[idx]) <= Z50 * sd,
            "in80_emp": (y[idx] >= q10) & (y[idx] <= q90), "in50_emp": (y[idx] >= q25) & (y[idx] <= q75),
            "q10": q10, "q50": q50, "q90": q90}))
    return pd.concat(out, ignore_index=True)


def family_rows(rid: str, target: str, labels: str) -> pd.DataFrame:
    return pd.concat([fold_rows(rid, target, labels, f) for f in FOLDS], ignore_index=True).sort_values(
        ["query_date", "lead"]).reset_index(drop=True)


def scores(df: pd.DataFrame) -> dict:
    out = {}
    for s, sub in (("all", df), ("extreme", df[df["stratum"] == "extreme"])):
        out[s] = {k: float(sub[k].mean()) for k in ("crps_gauss", "crps_emp", "crps_clim", "in80_gauss", "in50_gauss",
                                                    "in80_emp", "in50_emp")}
        out[s]["crpss_gauss"] = 1 - out[s]["crps_gauss"] / out[s]["crps_clim"]
        out[s]["crpss_emp"] = 1 - out[s]["crps_emp"] / out[s]["crps_clim"]
    return out


def as_test_frame(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Long frame whose 'error' is sqrt(CRPS), so the RMSE test compares mean CRPS."""
    return df[["query_date", "lead", "stratum", "cluster_id"]].assign(seed=0, error=np.sqrt(df[col].to_numpy()))


def main() -> None:
    res = {}
    RANGE_DIR.mkdir(parents=True, exist_ok=True)
    for fam, (target, labels, ctrl) in FAMILIES.items():
        rows = {m: family_rows(rid, target, labels) for m, rid in (("C3-hop", f"{ctrl}_C3hop"), ("control", ctrl))}
        rows["C3-hop"][["query_date", "lead", "target_date", "fold", "q10", "q50", "q90", "centre", "actual"]].to_parquet(
            RANGE_DIR / f"{ctrl}_C3hop.parquet", index=False)
        r = {m: scores(df) for m, df in rows.items()}
        r["tests"] = {f"C3-hop vs control, {col} ({s})": compare(as_test_frame(rows["control"], col),
                                                                 as_test_frame(rows["C3-hop"], col), s)
                      for col in ("crps_gauss", "crps_emp") for s in ("all", "extreme")}
        res[fam] = r
    L = ["# Range forecasts (specified 2026-10-18, before they were computed)", "",
         "Out of fold 2007-2018. Centre = recalibrated 10-seed ensemble mean; spread from the fold's inner-block "
         "residuals (Gaussian, or the empirical residuals). CRPS in °C (lower is better); CRPSS = skill vs a climatology "
         "forecast. Coverage: share of days whose truth falls inside the 80% / 50% interval (ideal 0.80 / 0.50).", ""]
    for fam, r in res.items():
        L += [f"## {fam}", "", "| Model | Days | CRPS Gaussian | CRPS empirical | CRPS climatology | CRPSS (Gauss / emp) | "
              "80% coverage (Gauss / emp) | 50% coverage (Gauss / emp) |", "|---|---|---|---|---|---|---|---|"]
        for m in ("C3-hop", "control"):
            for s in ("all", "extreme"):
                v = r[m][s]
                L.append(f"| {m} | {s} | {v['crps_gauss']:.3f} | {v['crps_emp']:.3f} | {v['crps_clim']:.3f} | "
                         f"{v['crpss_gauss']:+.3f} / {v['crpss_emp']:+.3f} | {v['in80_gauss']:.2f} / {v['in80_emp']:.2f} | "
                         f"{v['in50_gauss']:.2f} / {v['in50_emp']:.2f} |")
        L += ["", "| Test (sqrt-CRPS, paired cluster-jackknife) | Δ (95% CI) | p |", "|---|---|---|"]
        L += [f"| {k} | {t['delta']:+.4f} {_ci(t)} | {_fmt_p(t['p_value'])} |" for k, t in r["tests"].items()]
        L.append("")
    (OUT_DIR / "range_forecasts.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT_DIR / "range_forecasts.json").write_text(json.dumps(res, indent=2, default=float) + "\n", encoding="utf-8")
    print({fam: {m: round(r[m]["all"]["crpss_gauss"], 3) for m in ("C3-hop", "control")} for fam, r in res.items()})


if __name__ == "__main__":
    main()
