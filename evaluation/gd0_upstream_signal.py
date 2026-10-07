"""
Gate G-D0: does upstream heat help Delhi's forecast? (plan v5, Week 3; rule pre-registered
2026-10-06 in context/decisions.md, before any upstream number was computed.)

A cheap LINEAR check run before any graph network is trained: if yesterday's heat at the
27 upstream points adds nothing to what Delhi's own recent weather already says about the
next 5 days, a graph backbone has little to work with.

  reference  Delhi damped persistence (per-lead factor on Delhi's last standardised anomaly;
             the same forecast compare_v2 uses)
  augmented  per lead, ridge regression of Delhi's standardised anomaly on Delhi's last
             anomaly + the 27 points' standardised Tmax anomalies on the last 3 input days
             (81 extra inputs); ridge strength from {0.1, 1, 10, 100, 1000} chosen on the
             fold's last 2 training years, then refitted on all training years
  per point  the same with Delhi + ONE point's 3 lags (which points carry the signal?)

Everything is fitted per fold on that fold's training years (climatologies, anomalies,
coefficients); validation blocks 2007-2018 pooled; nothing from 2019 on is read.

PASS: augmented RMSE significantly lower than damped persistence on ALL days over leads 1-3
for Delhi Tmax (paired cluster-jackknife test, 95% CI entirely below 0).
Reported, not pass/fail: each lead, extreme days, physical WBGT, per-point gains on a map,
ridge weights by lag, north-west/west vs south-east/east, and best lag vs distance.

Outputs: evaluation_v2/gd0_upstream.md, .json, figures/gd0_upstream_map.png
Run from repo root:  python -m evaluation.gd0_upstream_signal
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.compare_v2 import _ci, _fmt_p, compare, rmse_by
from evaluation.predict_v1 import long_frame
from pipeline.build_upstream_daily import OUT as UPSTREAM_PATH
from pipeline.climatology import apply_climatology, doy_climatology
from pipeline.download_era5_upstream import NODES, node_id
from pipeline.graph import bearing_deg, distance_km, node_coords
from training.folds import FOLDS, FORECAST_DAYS, TEST_START, build_fold, fold_bounds, fold_daily, fold_windows

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "evaluation_v2"
LAGS = 3
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
PASS_LEADS = (1, 2, 3)
INNER_YEARS = 2
NODE_IDS = [node_id(a, o) for a, o in NODES]
FAMILIES = {"Tmax": ("t_max", "v2"), "WBGT (physical)": ("wbgt_lj_max", "wbgt")}
UPWIND = (247.5, 337.5)  # compass bearing from Delhi to the point: W to NW
DOWNWIND = (67.5, 157.5)  # E to SE


def load_upstream_tmax() -> pd.DataFrame:
    """(dates, 27) upstream daily Tmax, pre-2019 rows only (filtered at read time)."""
    cols = [f"temperature_2m_max__{n}" for n in NODE_IDS]
    up = pd.read_parquet(UPSTREAM_PATH, columns=cols, filters=[("date", "<", TEST_START)])
    up.columns = NODE_IDS
    return up[up.index < TEST_START]


def upstream_z(up: pd.DataFrame, train_end: pd.Timestamp) -> pd.DataFrame:
    """Standardised anomalies of every point, climatology from training years only."""
    train = up.index <= train_end
    out = {}
    for n in up.columns:
        m, s = apply_climatology(up.index, doy_climatology(up[n], train))
        out[n] = (up[n].to_numpy() - m) / s
    return pd.DataFrame(out, index=up.index)


def delhi_z(d: pd.DataFrame, target: str) -> pd.Series:
    mean_col, std_col = (("clim_mean_t_max", "clim_std_t_max") if target == "t_max"
                         else (f"clim_mean_{target}", f"clim_std_{target}"))
    return (d[target] - d[mean_col]) / d[std_col]


def design(query_dates, dz: pd.Series, uz: pd.DataFrame, nodes: list[str]) -> np.ndarray:
    """(N, 1 + LAGS*len(nodes)): Delhi z on the last input day, then each node's z on the
    last LAGS input days (q-1, q-2, q-3), node-major."""
    q = dz.index.get_indexer(pd.DatetimeIndex(query_dates))
    if (q < LAGS).any():
        raise ValueError("query dates too early for the lags")
    cols = [dz.to_numpy()[q - 1][:, None]]
    u = uz[nodes].to_numpy()
    for j in range(len(nodes)):
        cols.append(np.stack([u[q - 1 - k, j] for k in range(LAGS)], axis=1))
    return np.concatenate(cols, axis=1)


def targets(query_dates, dz: pd.Series) -> np.ndarray:
    q = dz.index.get_indexer(pd.DatetimeIndex(query_dates))
    return dz.to_numpy()[q[:, None] + np.arange(FORECAST_DAYS)[None, :]]


def ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Ridge with an unpenalised intercept: (coef (p, 5), intercept (5,))."""
    xm, ym = X.mean(axis=0), y.mean(axis=0)
    Xc, yc = X - xm, y - ym
    coef = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(X.shape[1]), Xc.T @ yc)
    return coef, ym - xm @ coef


def fit_with_inner_alpha(X: np.ndarray, Y: np.ndarray, q: pd.DatetimeIndex, train_end: pd.Timestamp):
    """Per lead: choose alpha on the last INNER_YEARS training years, refit on all training rows.
    Returns (coef (p, 5), intercept (5,), alphas (5,))."""
    stop_start = train_end - pd.DateOffset(years=INNER_YEARS) + pd.Timedelta(days=1)
    fit = (q + pd.Timedelta(days=FORECAST_DAYS - 1)) < stop_start
    stop = q >= stop_start
    coef = np.empty((X.shape[1], FORECAST_DAYS))
    icpt, chosen = np.empty(FORECAST_DAYS), np.empty(FORECAST_DAYS)
    for lead in range(FORECAST_DAYS):
        errs = []
        for a in ALPHAS:
            c, b = ridge(X[fit], Y[fit][:, [lead]], a)
            errs.append(np.mean((X[stop] @ c + b - Y[stop][:, [lead]]) ** 2))
        chosen[lead] = ALPHAS[int(np.argmin(errs))]
        c, b = ridge(X, Y[:, [lead]], chosen[lead])
        coef[:, lead], icpt[lead] = c[:, 0], b[0]
    return coef, icpt, chosen


def fold_results(fold: str, target: str, labels: str, up: pd.DataFrame) -> dict:
    """Validation forecasts (deg C) of damped persistence, the full upstream ridge and each
    single-point ridge, in long format; plus coefficients for the weight map."""
    train_end, _, _ = fold_bounds(fold)
    d, _ = fold_daily(fold, target, labels)
    val = build_fold(fold, target, labels).val
    train_w, val_w = fold_windows(fold)
    if not pd.DatetimeIndex(val_w["query_date"]).equals(pd.DatetimeIndex(val.query_dates)):
        raise ValueError("validation windows disagree")
    dz = delhi_z(d, target)
    uz = upstream_z(up.loc[d.index], train_end)
    qt, qv = pd.DatetimeIndex(train_w["query_date"]), pd.DatetimeIndex(val_w["query_date"])
    Yt = targets(qt, dz)
    mean_col, std_col = (("clim_mean_t_max", "clim_std_t_max") if target == "t_max"
                         else (f"clim_mean_{target}", f"clim_std_{target}"))
    pos = d.index.get_indexer(qv)[:, None] + np.arange(FORECAST_DAYS)[None, :]
    to_c = lambda z: d[mean_col].to_numpy()[pos] + d[std_col].to_numpy()[pos] * z  # noqa: E731

    def frame(name, pred):
        return long_frame(name, 0, val.query_dates, pred, val.y_raw, val.stratum).assign(fold=fold)

    out = {"frames": {"damped_persistence": frame("damped_persistence", val.damped)}, "coef": {}, "alpha": {}}
    for name, nodes in [("upstream_all", NODE_IDS)] + [(f"node_{n}", [n]) for n in NODE_IDS]:
        coef, icpt, alphas = fit_with_inner_alpha(design(qt, dz, uz, nodes), Yt, qt, train_end)
        z_hat = design(qv, dz, uz, nodes) @ coef + icpt
        out["frames"][name] = frame(name, to_c(z_hat))
        out["coef"][name], out["alpha"][name] = coef, alphas
    return out


def _leads(df: pd.DataFrame, leads) -> pd.DataFrame:
    return df[df["lead"].isin(leads)]


def evaluate(family: str, up: pd.DataFrame, folds=tuple(FOLDS)) -> dict:
    target, labels = FAMILIES[family]
    per_fold = [fold_results(f, target, labels, up) for f in folds]
    pooled = {name: pd.concat([r["frames"][name] for r in per_fold], ignore_index=True)
              for name in per_fold[0]["frames"]}
    dp, aug = pooled["damped_persistence"], pooled["upstream_all"]
    res = {"family": family, "target": target, "labels": labels}
    head = compare(_leads(dp, PASS_LEADS), _leads(aug, PASS_LEADS), "all")
    res["primary"] = {"delta": head["delta"], "ci_low": head["ci_low"], "ci_high": head["ci_high"],
                      "p_value": head["p_value"], "n": head["n_instances"],
                      "rmse_dp": rmse_by(_leads(dp, PASS_LEADS)), "rmse_aug": rmse_by(_leads(aug, PASS_LEADS))}
    res["passes"] = bool(head["ci_high"] < 0)
    res["by_lead"] = []
    for lead in range(1, FORECAST_DAYS + 1):
        a, e = compare(_leads(dp, [lead]), _leads(aug, [lead]), "all"), compare(_leads(dp, [lead]), _leads(aug, [lead]), "extreme")
        res["by_lead"].append({"lead": lead, "rmse_dp": rmse_by(_leads(dp, [lead])), "rmse_aug": rmse_by(_leads(aug, [lead])),
                               "all_delta": a["delta"], "all_ci": _ci(a), "all_p": a["p_value"],
                               "ext_delta": e["delta"], "ext_ci": _ci(e), "ext_p": e["p_value"]})
    ext = compare(_leads(dp, PASS_LEADS), _leads(aug, PASS_LEADS), "extreme")
    res["extreme_leads_1_3"] = {"delta": ext["delta"], "ci": _ci(ext), "p": ext["p_value"]}

    coords = node_coords()
    bearing, dist = bearing_deg(coords)[0, 1:], distance_km(coords)[0, 1:]
    nodes = []
    for j, n in enumerate(NODE_IDS):
        r = compare(_leads(dp, PASS_LEADS), _leads(pooled[f"node_{n}"], PASS_LEADS), "all")
        w = np.mean([pf["coef"][f"node_{n}"][1:, :3] for pf in per_fold], axis=0)  # (LAGS, leads 1-3)
        best_lag = int(np.argmax(np.abs(w).mean(axis=1))) + 1
        nodes.append({"node": n, "lat": NODES[j][0], "lon": NODES[j][1], "bearing_from_delhi": float(bearing[j]),
                      "distance_km": float(dist[j]), "delta_leads_1_3": r["delta"], "ci": _ci(r),
                      "ci_high": r["ci_high"], "p": r["p_value"], "best_lag_days": best_lag,
                      "weight_by_lag": np.abs(w).mean(axis=1).tolist()})
    res["nodes"] = nodes
    in_sector = lambda b, s: s[0] <= b < s[1]  # noqa: E731
    up_g = [x["delta_leads_1_3"] for x in nodes if in_sector(x["bearing_from_delhi"], UPWIND)]
    dn_g = [x["delta_leads_1_3"] for x in nodes if in_sector(x["bearing_from_delhi"], DOWNWIND)]
    res["direction"] = {"upwind_W_NW": {"n": len(up_g), "mean_delta": float(np.mean(up_g)) if up_g else float("nan")},
                        "downwind_E_SE": {"n": len(dn_g), "mean_delta": float(np.mean(dn_g)) if dn_g else float("nan")}}
    from scipy.stats import spearmanr

    lags = [x["best_lag_days"] for x in nodes]
    if len(set(lags)) == 1:  # correlation undefined: say so instead of reporting NaN
        res["lag_vs_distance"] = {"spearman_rho": float("nan"), "p": float("nan"),
                                  "note": f"every point's best lag is {lags[0]} day(s)"}
    else:
        lag_dist = spearmanr([x["distance_km"] for x in nodes], lags)
        res["lag_vs_distance"] = {"spearman_rho": float(lag_dist.statistic), "p": float(lag_dist.pvalue),
                                  "note": f"{sum(v == 1 for v in lags)} of {len(lags)} points have best lag 1 day"}
    w_all = np.mean([pf["coef"]["upstream_all"][1:, :3] for pf in per_fold], axis=0).reshape(len(NODE_IDS), LAGS, 3)
    res["full_model_weight_by_node"] = {n: np.abs(w_all[j]).mean(axis=1).tolist() for j, n in enumerate(NODE_IDS)}
    res["alphas_full_model"] = {pf_f: per_fold[i]["alpha"]["upstream_all"].tolist() for i, pf_f in enumerate(folds)}
    return res


def _map(results: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fams = list(results)
    fig, axes = plt.subplots(1, len(fams), figsize=(6.5 * len(fams), 5.5), squeeze=False)
    for ax, fam in zip(axes[0], fams):
        nd = results[fam]["nodes"]
        gain = np.array([-x["delta_leads_1_3"] for x in nd])  # positive = point helps
        lim = max(np.abs(gain).max(), 1e-3)
        sc = ax.scatter([x["lon"] for x in nd], [x["lat"] for x in nd], c=gain, cmap="RdBu", vmin=-lim, vmax=lim,
                        s=260, edgecolors="k")
        for x in nd:
            if x["ci_high"] < 0:
                ax.scatter(x["lon"], x["lat"], s=420, facecolors="none", edgecolors="k", linewidths=2)
            ax.annotate(str(x["best_lag_days"]), (x["lon"], x["lat"]), ha="center", va="center", fontsize=8)
        ax.plot(77.21, 28.61, marker="*", color="gold", markersize=20, markeredgecolor="k")
        ax.set_title(f"{fam}: RMSE gain over damped persistence\n(leads 1-3; ring = significant; number = best lag, days)")
        ax.set_xlabel("longitude (E)")
        ax.set_ylabel("latitude (N)")
        fig.colorbar(sc, ax=ax, label="deg C (positive = point helps)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    up = load_upstream_tmax()
    results = {fam: evaluate(fam, up) for fam in FAMILIES}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "gd0_upstream.json").write_text(json.dumps(results, indent=2, default=float) + "\n", encoding="utf-8")
    _map(results, OUT_DIR / "figures" / "gd0_upstream_map.png")

    t = results["Tmax"]
    p = t["primary"]
    L = ["# Gate G-D0: does upstream heat help Delhi's forecast?", "",
         "Generated by `python -m evaluation.gd0_upstream_signal`. Rule pre-registered 2026-10-06 "
         "(context/decisions.md) before any upstream number was computed. Validation 2007-2018 (folds f1-f4 pooled); "
         "everything fitted per fold on training years only; no 2019+ data read. RMSE in deg C; Δ = augmented minus "
         "damped persistence (negative = upstream helps).", "",
         f"## Verdict: **{'PASS' if t['passes'] else 'FAIL'}**", "",
         f"Delhi Tmax, all days, leads 1-3: RMSE {p['rmse_dp']:.3f} (damped persistence) → {p['rmse_aug']:.3f} "
         f"(+ upstream); Δ {p['delta']:+.3f} [{p['ci_low']:+.3f}, {p['ci_high']:+.3f}], p {_fmt_p(p['p_value'])}.", ""]
    for fam, r in results.items():
        L += [f"## {fam}: by lead", "",
              "| Lead | RMSE damped persistence | RMSE + upstream | Δ all days (95% CI) | p | Δ extreme days (95% CI) |",
              "|---|---|---|---|---|---|"]
        for b in r["by_lead"]:
            L.append(f"| {b['lead']} | {b['rmse_dp']:.3f} | {b['rmse_aug']:.3f} | {b['all_delta']:+.3f} {b['all_ci']} | "
                     f"{_fmt_p(b['all_p'])} | {b['ext_delta']:+.3f} {b['ext_ci']} |")
        e = r["extreme_leads_1_3"]
        L += ["", f"Extreme days, leads 1-3: Δ {e['delta']:+.3f} {e['ci']} (p {_fmt_p(e['p'])}).",
              f"Ridge strength chosen per fold (leads 1-5): {r['alphas_full_model']}.", "",
              f"### {fam}: which points help (Delhi + one point, leads 1-3, all days)", "",
              f"Direction check: mean Δ of points to the W/NW of Delhi {r['direction']['upwind_W_NW']['mean_delta']:+.4f} "
              f"(n={r['direction']['upwind_W_NW']['n']}) vs E/SE {r['direction']['downwind_E_SE']['mean_delta']:+.4f} "
              f"(n={r['direction']['downwind_E_SE']['n']}; the download area lies almost entirely west of Delhi, so this "
              f"contrast is weak by design). Best lag vs distance: Spearman ρ "
              f"{r['lag_vs_distance']['spearman_rho']:+.2f} (p {_fmt_p(r['lag_vs_distance']['p'])}); "
              f"{r['lag_vs_distance']['note']}.", "",
              "| Point | Bearing from Delhi | Distance km | Δ leads 1-3 (95% CI) | p | Best lag (days) |",
              "|---|---|---|---|---|---|"]
        for x in sorted(r["nodes"], key=lambda x: x["delta_leads_1_3"]):
            L.append(f"| {x['node']} | {x['bearing_from_delhi']:.0f}° | {x['distance_km']:.0f} | {x['delta_leads_1_3']:+.4f} {x['ci']} | "
                     f"{_fmt_p(x['p'])} | {x['best_lag_days']} |")
        L.append("")
    L += ["Map: `evaluation_v2/figures/gd0_upstream_map.png`.", "",
          "Notes:",
          "- Linear check only: a graph network could extract more (non-linear, wind-gated). A FAIL here means the "
          "simplest version of the upstream signal is absent; a PASS means there is something for the graph to use.",
          "- Per-point tests are descriptive (27 tests, no multiple-comparison correction); only the pre-registered "
          "primary test decides the gate.",
          "- Best lag is 1 day almost everywhere: in this linear check the signal looks like the large-scale heat "
          "pattern seen a day earlier, not a travel time that grows with distance. Whether a graph can use "
          "travel-time structure is tested later (G-D3, against an LSTM given the same upstream data).",
          "- Bearing = compass direction from Delhi to the point (270° = due west)."]
    (OUT_DIR / "gd0_upstream.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    for fam, r in results.items():
        q = r["primary"]
        print(f"G-D0 [{fam}] leads 1-3 all days: delta {q['delta']:+.4f} [{q['ci_low']:+.4f}, {q['ci_high']:+.4f}] "
              f"-> {'PASS' if r['passes'] else 'FAIL'}" + (" (decides the gate)" if fam == "Tmax" else " (descriptive)"))
    print(f"-> {(OUT_DIR / 'gd0_upstream.md').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
