"""
Screen S2: can new information, delivered directly or by retrieval, improve WBGT heatwave days?
(pre-registered 2026-10-18 in context/decisions.md, before the humid-air data was downloaded).

No training; training years only, as evaluation/synoptic_screen.py (fit on each fold's training windows
before its last 2 training years, score on those 2 years, deg C, bootstrap over the 8 held-out years), but the
ridge is fitted with the neural loss weights 1 + 4 x hot. Baseline = the C3-hop-like linear model (query-only
features + Rg's regional readings + C3-hop's upstream inputs on the last input day). Arms added to it:
  A  humid-air direct: 12 points' standardised dew point / RH / Tmax / surface pressure, last 3 input days
  B  Rh retrieval on A's features (K = 5, Rg's pool and rules); random control R0-rand
  C  heatwave-onset matching: mean of the top-5 cosine similarities to training windows whose 5 target days
     contain a WBGT-label episode onset (eligible by the chronological rule, not the query's own episode);
     control: the same with an equal number of random non-onset windows
  D  upper-air direct: the Rz state's 20 principal-component scores (500 hPa, last 3 input days)
Pass per arm (WBGT): (1) extreme days vs baseline CI entirely below 0; (2) all days CI not entirely above 0;
(3) for B and C, extreme days vs their random control CI entirely below 0. Tmax: descriptive.

Run from repo root:  python -m evaluation.heatwave_info_screen [--arms D]
Writes evaluation_v2/heatwave_info_screen.{md,json}.
"""
from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd

from evaluation.retrieval_information_check import (
    FAMILIES, HELD_OUT_YEARS, OUT_DIR, _ci, analogue_signal, clim_cols, cluster_boot, rmse_seedmean,
)
from evaluation.ridge_baselines import weighted_ridge
from retrieval.fold_retrieval import (
    BUFFER_DAYS, HUMID_LAGS, K_DEFAULT, FoldRetriever, _unit, humid_daily, lagged_features, region_features,
    synoptic_features, window_episode, window_features,
)
from training.folds import FOLDS, FORECAST_DAYS, INPUT_DAYS, fold_bounds
from training.graph_data import build_upstream, load_upstream

HOT_WEIGHT, ALPHA, N_RAND, TOP = 5.0, 1.0, 10, 5
ARMS = ("A", "B", "C", "D")
CONTROLS = {"B": "B-rand", "C": "C-rand"}


def wridge_fit_predict(x_fit: np.ndarray, y_fit: np.ndarray, w_fit: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    """Weighted ridge (alpha 1) on columns standardised with the fit set's mean / SD."""
    mu, sd = x_fit.mean(axis=0), x_fit.std(axis=0)
    sd[sd == 0] = 1.0
    coef, icpt = weighted_ridge((x_fit - mu) / sd, y_fit, ALPHA, w_fit)
    return ((x_new - mu) / sd) @ coef + icpt


def onset_windows(d: pd.DataFrame, cand_dates: pd.DatetimeIndex) -> np.ndarray:
    """True for candidate windows whose 5 target days contain the first day of a labelled episode."""
    ep = d["episode_id"].to_numpy()
    onset = (ep > 0) & (ep != np.r_[0, ep[:-1]])
    pos = d.index.get_indexer(cand_dates)
    return onset[pos[:, None] + np.arange(FORECAST_DAYS)].any(axis=1)


def match_score(qv: np.ndarray, cv: np.ndarray, members: np.ndarray, q_dates: pd.DatetimeIndex,
                cand_dates: pd.DatetimeIndex, q_ep: np.ndarray, cand_ep: np.ndarray) -> np.ndarray:
    """Per query: mean of the TOP highest cosine similarities to eligible member windows (0 if none)."""
    m_idx = np.flatnonzero(members)
    sims = qv @ cv[m_idx].T
    ok = cand_dates.values[m_idx][None, :] <= (q_dates - pd.Timedelta(days=BUFFER_DAYS)).values[:, None]
    ok &= ~((q_ep[:, None] > 0) & (cand_ep[m_idx][None, :] == q_ep[:, None]))
    sims = np.where(ok, sims, -np.inf)
    top = -np.sort(-sims, axis=1)[:, :TOP]
    top = np.where(np.isfinite(top), top, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # queries with no eligible member: score 0 below
        out = np.nanmean(top, axis=1)
    return np.nan_to_num(out, nan=0.0)


def fold_family(fold: str, target: str, labels: str, up_table: pd.DataFrame, arms: tuple[str, ...]) -> dict:
    fr = FoldRetriever(fold, labels, target)
    d = fr.d
    mean_col, std_col = clim_cols(target)
    z = ((d[target] - d[mean_col]) / d[std_col]).to_numpy(dtype=np.float64)
    q = pd.DatetimeIndex(fr.train_w["query_date"])
    pos = d.index.get_indexer(q)
    cand_pos = d.index.get_indexer(fr.cand_dates)
    leads = np.arange(FORECAST_DAYS)
    y = z[pos[:, None] + leads]
    w_all = 1.0 + (HOT_WEIGHT - 1.0) * d["hot"].to_numpy(dtype=np.float64)[pos[:, None] + leads]
    own = np.column_stack([z[pos - 1], z[pos[:, None] + np.arange(-INPUT_DAYS, 0)].mean(axis=1)])
    upd = build_upstream(fold, up_table)
    up_last = upd.x[upd.positions(q)[:, -1]].reshape(len(q), -1).astype(np.float64)
    reg = region_features(fr._region_state()[3], q)
    x = np.column_stack([window_features(d, q), own, reg, up_last])
    train_end, _, _ = fold_bounds(fold)
    cutoff = train_end - pd.DateOffset(years=HELD_OUT_YEARS)
    held, before = q > cutoff, q <= cutoff - pd.Timedelta(days=BUFFER_DAYS)

    static: dict[str, list[np.ndarray]] = {}  # arm -> list over random draws of (N, F) extra columns
    per_lead_sig: dict[str, list[np.ndarray]] = {}  # arm -> list over draws of (N, L) analogue signals
    full = np.ones(len(q), dtype=bool)
    if "A" in arms or "B" in arms or "C" in arms:
        hum = lagged_features(humid_daily(fold, d), q, HUMID_LAGS)
    if "A" in arms:
        static["A"] = [hum]
    if "D" in arms:
        _, mean, comps, za, w = fr._synoptic_state()
        static["D"] = [(synoptic_features(za, w, q) - mean) @ comps.T]
    if "B" in arms:
        runs = {"B": [fr.retrieve(q, "humid", K_DEFAULT).idx], "B-rand": [fr.retrieve(q, "rand", K_DEFAULT, seed=s).idx
                                                                       for s in range(N_RAND)]}
        for idxs in runs.values():
            for idx in idxs:
                full &= (idx >= 0).all(axis=1)
        for name, idxs in runs.items():
            sigs = []
            for idx in idxs:
                s = np.zeros((len(q), FORECAST_DAYS))
                s[full] = analogue_signal(z, idx[full], cand_pos)
                sigs.append(s)
            per_lead_sig[name] = sigs
    if "C" in arms:
        def zs(a):
            sd = a.std(axis=0)
            sd[sd == 0] = 1.0
            return (a - a.mean(axis=0)) / sd
        vec = _unit(np.column_stack([zs(reg), zs(hum)]))  # queries are the training windows = candidates
        q_ep = window_episode(d, q)
        members = onset_windows(d, fr.cand_dates)
        static["C"] = [match_score(vec, vec, members, q, fr.cand_dates, q_ep, fr.cand_episode)[:, None]]
        rng = np.random.default_rng([list(FOLDS).index(fold), 20261018])
        non = np.flatnonzero(~members)
        rand_sets = []
        for _ in range(N_RAND):
            m = np.zeros(len(members), dtype=bool)
            m[rng.choice(non, members.sum(), replace=False)] = True
            rand_sets.append(match_score(vec, vec, m, q, fr.cand_dates, q_ep, fr.cand_episode)[:, None])
        static["C-rand"] = rand_sets

    fit, ev = full & before, full & held
    scale = d[std_col].to_numpy()[pos[:, None] + leads][ev]

    def predict(extra_static: np.ndarray | None, sig: np.ndarray | None) -> np.ndarray:
        cols = []
        for lead in range(FORECAST_DAYS):
            parts = [x] + ([extra_static] if extra_static is not None else []) + ([sig[:, lead:lead + 1]] if sig is not None else [])
            xa = np.column_stack(parts)
            cols.append(wridge_fit_predict(xa[fit], y[fit, lead], w_all[fit, lead], xa[ev]))
        return np.column_stack(cols)

    def sq_err(pred: np.ndarray) -> np.ndarray:
        return ((pred - y[ev]) * scale) ** 2

    out = {"base": sq_err(predict(None, None))[None]}
    for name, draws in static.items():
        out[name] = np.stack([sq_err(predict(e, None)) for e in draws])
    for name, draws in per_lead_sig.items():
        out[name] = np.stack([sq_err(predict(None, s)) for s in draws])
    stratum = d["stratum"].to_numpy()[pos[:, None] + leads][ev]
    return {"errors": out, "n_fit": int(fit.sum()), "n_held": int(ev.sum()), "n_onset_windows": int(onset_windows(
        d, fr.cand_dates).sum()) if "C" in arms else None,
            "year": np.repeat(q[ev].year.to_numpy()[:, None], FORECAST_DAYS, axis=1), "extreme": stratum == "extreme"}


def contrasts(arms: tuple[str, ...]) -> list[tuple[str, str]]:
    out = [(a, "base") for a in arms]
    return out + [(a, CONTROLS[a]) for a in arms if a in CONTROLS]


def passes(res: dict, arm: str) -> tuple[bool, ...]:
    """Pre-registered conditions for one arm (WBGT family)."""
    c = [bool(res["extreme"][f"{arm} vs base"][2] < 0), bool(res["all"][f"{arm} vs base"][1] <= 0)]
    if arm in CONTROLS:
        c.append(bool(res["extreme"][f"{arm} vs {CONTROLS[arm]}"][2] < 0))
    return tuple(c)


def summarise(parts: list[dict], arms: tuple[str, ...]) -> dict:
    rng = np.random.default_rng(20261018)
    res = {}
    for subset in ("all", "extreme"):
        mask = np.concatenate([p["extreme"].ravel() if subset == "extreme" else np.ones(p["extreme"].size, bool)
                               for p in parts])
        years = np.concatenate([p["year"].ravel() for p in parts])[mask]
        se = {k: np.concatenate([p["errors"][k].reshape(p["errors"][k].shape[0], -1) for p in parts], axis=1)[:, mask]
              for k in parts[0]["errors"]}
        rows = {"n_rows": int(mask.sum()), "rmse": {k: rmse_seedmean(v) for k, v in se.items()}}
        for a, b in contrasts(arms):
            rows[f"{a} vs {b}"] = cluster_boot(lambda w, a=a, b=b: rmse_seedmean(se[a], w) - rmse_seedmean(se[b], w),
                                               years, rng)
        res[subset] = rows
    res["conditions"] = {a: passes(res, a) for a in arms}
    res["passes"] = {a: all(c) for a, c in res["conditions"].items()}
    return res


def write_report(results: dict, meta: dict, arms: tuple[str, ...]) -> None:
    names = {"A": "humid-air direct", "B": "Rh retrieval (humid-air analogues)", "C": "heatwave-onset matching",
             "D": "upper-air (500 hPa) direct"}
    L = ["# Screen S2: new information for WBGT heatwave days (rule written 2026-10-18, before the new data)", "",
         "No training; training years only (each fold's last 2 training years held out). Ridge weighted like the "
         "neural loss (1 + 4 x hot). RMSE in °C pooled over 4 folds x leads 1-5; Δ = first minus second (negative = "
         "the arm helps); 95% CI by bootstrap over the 8 held-out years. Baseline = what C3-hop already sees, as a "
         f"linear model. Arms run: {', '.join(f'{a} ({names[a]})' for a in arms)}.", ""]
    for fam, r in results.items():
        L += [f"## {fam}" + (" (decides)" if fam.startswith("WBGT") else " (descriptive)"), "",
              f"Held-out query-days: {r['all']['n_rows']} (extreme: {r['extreme']['n_rows']}); fit queries: "
              f"{meta[fam]['n_fit']}. Baseline RMSE: all {r['all']['rmse']['base']:.3f}, extreme "
              f"{r['extreme']['rmse']['base']:.3f} °C.", "",
              "| Comparison | Extreme days Δ (95% CI) | All days Δ (95% CI) |", "|---|---|---|"]
        for a, b in contrasts(arms):
            L.append(f"| {a} ({names[a]}) vs {b.replace('base', 'baseline')} | {_ci(r['extreme'][f'{a} vs {b}'])} | "
                     f"{_ci(r['all'][f'{a} vs {b}'])} |")
        if fam.startswith("WBGT"):
            L += ["", "**Decision:** " + "; ".join(f"{a} {'PASS' if r['passes'][a] else 'fail'} {r['conditions'][a]}"
                                                   for a in arms)]
        L.append("")
    (OUT_DIR / "heatwave_info_screen.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    arms = tuple(sorted(ap.parse_args(argv).arms))
    up_table = load_upstream()
    results, meta = {}, {}
    for fam, (target, labels) in FAMILIES.items():
        parts = []
        for fold in FOLDS:
            parts.append(fold_family(fold, target, labels, up_table, arms))
            print(f"{fam} {fold}: fit {parts[-1]['n_fit']}, held {parts[-1]['n_held']}", flush=True)
        results[fam] = summarise(parts, arms)
        meta[fam] = {"n_fit": sum(p["n_fit"] for p in parts)}
    write_report(results, meta, arms)
    (OUT_DIR / "heatwave_info_screen.json").write_text(json.dumps({"arms": arms, "results": results}, indent=2,
                                                                  default=float) + "\n", encoding="utf-8")
    print({fam: r["passes"] for fam, r in results.items()})


if __name__ == "__main__":
    main()
