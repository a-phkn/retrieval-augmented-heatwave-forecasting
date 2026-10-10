"""
Ridge baselines for every results table (decisions.md 2026-10-08, reporting addition 1, taken on
the independent ML review). Evaluation only; no gate uses them.

  ridge      the G-D0 ridge exactly (evaluation/gd0_upstream_signal.py "upstream_all"): per lead,
             Delhi's last standardised anomaly + the 27 points' standardised Tmax anomalies on
             the last 3 input days; ridge strength from {0.1, 1, 10, 100, 1000} on the fold's
             last 2 training years (plain MSE), then refitted on ALL training years.
  ridge_hw5  the same inputs, fitted by weighted least squares with the neural models' loss
             weights 1 + (5 - 1) x hot (the family's hot labels per target day, hot_weight 5);
             the ridge strength is chosen by the same weighted error on the inner block.

Note (disclosed in the report): the ridges refit on all training years, the neural models do
not (they never train on the last 2 training years).

Writes predictions_v2/{control}_ridge{,_hw5}/{fold}.parquet (one pseudo-seed 0, the trained
models' windows, actuals, strata and clusters) and evaluation_v2/ridge_baselines.{md,json}.
Run from repo root:  python -m evaluation.ridge_baselines
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from evaluation.compare_v2 import PRED_DIR, _ci, _fmt_p, compare, extreme_bias, load_run, rmse_by
from evaluation.gd0_upstream_signal import (
    ALPHAS, INNER_YEARS, NODE_IDS, OUT_DIR, delhi_z, design, fit_with_inner_alpha, load_upstream_tmax, targets,
    upstream_z,
)
from evaluation.predict_v1 import long_frame
from training.folds import FOLDS, FORECAST_DAYS, build_fold, fold_bounds, fold_daily, fold_windows

HOT_WEIGHT = 5.0
FAMILIES = {"Tmax": ("A1prime_hw5", "t_max", "v2"), "WBGT (physical)": ("A2Lr_hw5", "wbgt_lj_max", "wbgt")}
COMPARE_WITH = ("control", "C2", "U1", "C3",  # trained runs: {control}, {control}_C2, ...; missing ones skipped
                "U1_tuned", "C3_tuned", "C3pool_tuned", "C3hop")


def weighted_ridge(X: np.ndarray, y: np.ndarray, alpha: float, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weighted ridge with an unpenalised intercept: minimises sum w (y - Xb - c)^2 + alpha |b|^2.
    With w = 1 it equals gd0_upstream_signal.ridge."""
    w = w / w.mean()
    xm, ym = (w @ X) / w.sum(), (w @ y) / w.sum()
    Xc, yc = X - xm, y - ym
    Xw = Xc * w[:, None]
    coef = np.linalg.solve(Xc.T @ Xw + alpha * np.eye(X.shape[1]), Xw.T @ yc)
    return coef, ym - xm @ coef


def fit_weighted_inner_alpha(X: np.ndarray, Y: np.ndarray, W: np.ndarray, q: pd.DatetimeIndex,
                             train_end: pd.Timestamp) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """fit_with_inner_alpha with per-(window, lead) weights W (N, 5): alpha chosen by the weighted
    error on the last INNER_YEARS training years, then refitted on all training rows."""
    stop_start = train_end - pd.DateOffset(years=INNER_YEARS) + pd.Timedelta(days=1)
    fit = (q + pd.Timedelta(days=FORECAST_DAYS - 1)) < stop_start
    stop = q >= stop_start
    coef, icpt, chosen = np.empty((X.shape[1], FORECAST_DAYS)), np.empty(FORECAST_DAYS), np.empty(FORECAST_DAYS)
    for lead in range(FORECAST_DAYS):
        errs = []
        for a in ALPHAS:
            c, b = weighted_ridge(X[fit], Y[fit][:, [lead]], a, W[fit, lead])
            errs.append(np.average((X[stop] @ c + b - Y[stop][:, [lead]])[:, 0] ** 2, weights=W[stop, lead]))
        chosen[lead] = ALPHAS[int(np.argmin(errs))]
        c, b = weighted_ridge(X, Y[:, [lead]], chosen[lead], W[:, lead])
        coef[:, lead], icpt[lead] = c[:, 0], b[0]
    return coef, icpt, chosen


def fold_frames(fold: str, target: str, labels: str, up: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Validation forecasts (deg C) of both ridges, long format, on the trained models' windows."""
    train_end, _, _ = fold_bounds(fold)
    d, _ = fold_daily(fold, target, labels)
    data = build_fold(fold, target, labels)
    train_w, val_w = fold_windows(fold)
    qt, qv = pd.DatetimeIndex(train_w["query_date"]), pd.DatetimeIndex(val_w["query_date"])
    if not (qv.equals(pd.DatetimeIndex(data.val.query_dates)) and qt.equals(pd.DatetimeIndex(data.train.query_dates))):
        raise ValueError("windows disagree with the trained models'")
    dz = delhi_z(d, target)
    uz = upstream_z(up.loc[d.index], train_end)
    Xt, Xv, Yt = design(qt, dz, uz, NODE_IDS), design(qv, dz, uz, NODE_IDS), targets(qt, dz)
    mean_col, std_col = (("clim_mean_t_max", "clim_std_t_max") if target == "t_max"
                         else (f"clim_mean_{target}", f"clim_std_{target}"))
    pos = d.index.get_indexer(qv)[:, None] + np.arange(FORECAST_DAYS)[None, :]
    to_c = lambda z: d[mean_col].to_numpy()[pos] + d[std_col].to_numpy()[pos] * z  # noqa: E731
    W = 1.0 + (HOT_WEIGHT - 1.0) * data.train.hot.astype(np.float64)
    out = {}
    for name, (coef, icpt, _) in (("ridge", fit_with_inner_alpha(Xt, Yt, qt, train_end)),
                                  ("ridge_hw5", fit_weighted_inner_alpha(Xt, Yt, W, qt, train_end))):
        out[name] = long_frame(name, 0, data.val.query_dates, to_c(Xv @ coef + icpt), data.val.y_raw, data.val.stratum)
    return out


def write_runs(control: str, target: str, labels: str, up: pd.DataFrame) -> None:
    for fold in FOLDS:
        for name, df in fold_frames(fold, target, labels, up).items():
            path = PRED_DIR / f"{control}_{name}"
            path.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path / f"{fold}.parquet", index=False)


def _lead_rmse(df: pd.DataFrame) -> list[float]:
    return [rmse_by(df[df["lead"] == lead]) for lead in range(1, FORECAST_DAYS + 1)]


def evaluate(control: str) -> dict:
    ridges = {n: load_run(f"{control}_{n}") for n in ("ridge", "ridge_hw5")}
    trained = {m: load_run(control if m == "control" else f"{control}_{m}") for m in COMPARE_WITH}
    models = ridges | {m: df for m, df in trained.items() if df is not None}
    res = {"rmse": {}, "tests": {}}
    for m, df in models.items():
        ext = df[df["stratum"] == "extreme"]
        res["rmse"][m] = {"all": rmse_by(df), "extreme": rmse_by(ext), "bias_all": float(df["error"].mean()),
                          "bias_extreme": extreme_bias(df), "by_lead": _lead_rmse(df)}
    for r in ridges:
        for m in trained:
            if trained[m] is None:
                continue
            for s in ("all", "extreme"):
                res["tests"][f"{m} vs {r} ({s})"] = compare(ridges[r], trained[m], s)
    return res


def write_report(results: dict) -> None:
    L = ["# Ridge baselines (decisions.md 2026-10-08, reporting addition 1)", "",
         "Out of fold 2007-2018; every ridge fitted per fold on training years only. `ridge` = the G-D0 ridge "
         "(Delhi's last anomaly + 27 upstream points' Tmax anomalies, 3 lags); `ridge_hw5` = the same, weighted like "
         "the neural loss (1 + 4 x hot). The ridges refit on ALL training years; the neural models never train on "
         "the last 2 (no refit). Δ = trained model minus ridge (negative = the trained model is better). "
         "Trained models: mean of per-seed RMSE, 10 seeds.", ""]
    for fam, r in results.items():
        L += [f"## {fam}", "", "| Model | All | Extreme | Bias all | Bias extreme | Lead 1 | 2 | 3 | 4 | 5 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for m, v in r["rmse"].items():
            L.append(f"| {m} | {v['all']:.3f} | {v['extreme']:.3f} | {v['bias_all']:+.2f} | {v['bias_extreme']:+.2f} | "
                     + " | ".join(f"{x:.3f}" for x in v["by_lead"]) + " |")
        L += ["", "| Comparison | Δ RMSE (95% CI) | p |", "|---|---|---|"]
        L += [f"| {k} | {t['delta']:+.3f} {_ci(t)} | {_fmt_p(t['p_value'])} |" for k, t in r["tests"].items()]
        L.append("")
    (OUT_DIR / "ridge_baselines.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    up = load_upstream_tmax()
    results = {}
    for fam, (control, target, labels) in FAMILIES.items():
        write_runs(control, target, labels, up)
        results[fam] = evaluate(control)
    write_report(results)
    (OUT_DIR / "ridge_baselines.json").write_text(json.dumps(results, indent=2, default=float) + "\n", encoding="utf-8")
    print({f: {m: round(v["all"], 3) for m, v in r["rmse"].items()} for f, r in results.items()})


if __name__ == "__main__":
    main()
