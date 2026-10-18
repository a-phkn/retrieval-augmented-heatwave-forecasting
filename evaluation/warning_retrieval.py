"""
W: retrieval for heatwave warnings (pre-registered 2026-10-18, context/decisions.md, before it was run).

Does "how many of the 20 most similar past situations became hot days" make C3-hop's hot-day probability
more reliable? Temperature forecasts are not changed. Brier score, in season only.

  p_model   calibration_check's "recal" probability for C3-hop (ensemble mean, inner-block recalibration,
            Gaussian with the inner residual SD per lead, event = the hot-label rule)
  p_analog  share of K = 20 analogues (fold training windows, Rg's pool and rules) whose own outcome day at
            that lead was a hot day; arms Rg ("region") and Rz ("synoptic")
  blend     logit P = a + b logit(p_model) + c p_analog, fitted on the fold's inner block (in season, leads pooled)
Controls: model-only logistic (logit P = a + b logit(p_model), same rows) and the random-analogue blend
(R0-rand, 10 seeds). Pass per arm and family: blend vs model-only and vs random blend, sqrt-Brier CI entirely
below 0 (paired cluster-jackknife as in calibration_check).

Run from repo root:  python -m evaluation.warning_retrieval   ->  evaluation_v2/warning_retrieval.{md,json}
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

from evaluation.calibration_check import apply_recal, brier, fit_recal, probability_frames
from evaluation.compare_v2 import PRED_DIR, _ci, _fmt_p, compare, ensemble
from evaluation.retrieval_information_check import OUT_DIR
from retrieval.fold_retrieval import FoldRetriever
from training.folds import FOLDS, FORECAST_DAYS

FAMILIES = {"Tmax": ("A1prime_hw5_C3hop", "t_max", "v2"), "WBGT (physical)": ("A2Lr_hw5_C3hop", "wbgt_lj_max", "wbgt")}
INNER_DIR = PRED_DIR / "inner"
K_ANALOGS, N_RAND, EPS = 20, 10, 1e-4
ARMS = {"Rg": "region", "Rz": "synoptic"}


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def analog_prob(fr: FoldRetriever, query_dates: pd.Series, leads: np.ndarray, mode: str, seed: int | None) -> np.ndarray:
    """Per row (query_date, lead): share of the K analogues whose outcome day at that lead was hot (NaN if none)."""
    q = pd.DatetimeIndex(query_dates)
    uq = q.unique()
    idx = fr.retrieve(uq, mode, K_ANALOGS, seed=seed).idx  # (U, K)
    hot = fr.d["hot"].to_numpy(dtype=float)
    cand_pos = fr.d.index.get_indexer(fr.cand_dates)
    row = uq.get_indexer(q)
    ii = idx[row]  # (N, K)
    valid = ii >= 0
    h = hot[cand_pos[np.maximum(ii, 0)] + (leads[:, None] - 1)]
    with np.errstate(invalid="ignore"):
        return np.where(valid.any(axis=1), (h * valid).sum(axis=1) / valid.sum(axis=1), np.nan)


def fit_logistic(o: np.ndarray, *cols: np.ndarray) -> np.ndarray:
    x = sm.add_constant(np.column_stack(cols), has_constant="add")
    return sm.GLM(o, x, family=sm.families.Binomial()).fit().params


def predict_logistic(params: np.ndarray, *cols: np.ndarray) -> np.ndarray:
    x = sm.add_constant(np.column_stack(cols), has_constant="add")
    return 1.0 / (1.0 + np.exp(-(x @ params)))


def fold_frames(rid: str, target: str, labels: str, family: str, fold: str) -> dict[str, list[pd.DataFrame]]:
    """In-season validation rows of one fold with p for every variant (random blends: one frame per seed)."""
    val = pd.read_parquet(PRED_DIR / rid / f"{fold}.parquet")
    inner = pd.read_parquet(INNER_DIR / rid / f"{fold}.parquet")
    coef = fit_recal(inner)
    ie, ve = ensemble(apply_recal(inner, coef)), ensemble(apply_recal(val, coef))
    pi, pv = probability_frames(ie, ie, fold, family), probability_frames(ve, ie, fold, family)
    fr = FoldRetriever(fold, labels, target)
    li, lv = pi["lead"].to_numpy(), pv["lead"].to_numpy()

    out: dict[str, list[pd.DataFrame]] = {}
    m_par = fit_logistic(pi["o"].to_numpy(), logit(pi["p"].to_numpy()))
    p_mod = predict_logistic(m_par, logit(pv["p"].to_numpy()))
    out["model-only"] = [pv.assign(p=p_mod, error=p_mod - pv["o"].to_numpy())]

    def blend(mode: str, seed: int | None) -> pd.DataFrame:
        ai, av = analog_prob(fr, pi["query_date"], li, mode, seed), analog_prob(fr, pv["query_date"], lv, mode, seed)
        ok = np.isfinite(ai)
        par = fit_logistic(pi["o"].to_numpy()[ok], logit(pi["p"].to_numpy()[ok]), ai[ok])
        av = np.where(np.isfinite(av), av, np.nanmean(ai))  # a window without analogues gets the inner mean
        p = predict_logistic(par, logit(pv["p"].to_numpy()), av)
        return pv.assign(p=p, error=p - pv["o"].to_numpy(), seed=0 if seed is None else seed)

    for arm, mode in ARMS.items():
        out[arm] = [blend(mode, None)]
    out["random"] = [blend("rand", s) for s in range(N_RAND)]
    return out


def family_result(rid: str, target: str, labels: str, family: str) -> dict:
    parts: dict[str, list[pd.DataFrame]] = {}
    for fold in FOLDS:
        for k, frames in fold_frames(rid, target, labels, family, fold).items():
            parts.setdefault(k, []).extend(f.assign(fold=fold) for f in frames)
    fr = {k: pd.concat(v, ignore_index=True) for k, v in parts.items()}
    drop = ["fold", "p", "o", "p_clim"]
    t = {k: v.drop(columns=drop) for k, v in fr.items()}
    res = {"scores": {}, "tests": {}, "conditions": {}, "passes": {}}
    for k, v in fr.items():
        sc = brier(v.groupby("seed").get_group(v["seed"].min())) if k != "random" else brier(v)
        warn = v["p"] >= 0.5
        n_seasons = v["target_date"].dt.year.nunique() * (N_RAND if k == "random" else 1)
        res["scores"][k] = {**{x: sc[x] for x in ("brier", "brier_clim", "bss", "n_events")},
                            "hits_per_season": float((warn & (v["o"] == 1)).sum() / n_seasons),
                            "false_alarms_per_season": float((warn & (v["o"] == 0)).sum() / n_seasons),
                            "reliability": sc["reliability"]}
    for arm in ARMS:
        a, b = compare(t["model-only"], t[arm], "all"), compare(t["random"], t[arm], "all")
        res["tests"][arm] = {"vs_model_only": a, "vs_random": b}
        res["conditions"][arm] = (bool(a["ci_high"] < 0), bool(b["ci_high"] < 0))
        res["passes"][arm] = all(res["conditions"][arm])
    res["tests"]["random_vs_model_only"] = compare(t["model-only"], t["random"], "all")
    return res


def write_report(res: dict) -> None:
    L = ["# W: retrieval for heatwave warnings (rule written 2026-10-18, before it was run)", "",
         "Hot-day probability of C3-hop, in season only, out of fold 2007-2018. Brier score (lower is better); BSS = "
         "skill vs the training-years base rate. Blends fitted on each fold's inner block. Tests: paired cluster-"
         "jackknife on sqrt-Brier (Δ in sqrt-probability units; negative = the blend is better).", ""]
    for fam, r in res.items():
        L += [f"## {fam}", "", "| Variant | Brier | BSS | Hot days | Hits / season (P >= 0.5) | False alarms / season |",
              "|---|---|---|---|---|---|"]
        for k, s in r["scores"].items():
            L.append(f"| {k} | {s['brier']:.4f} | {s['bss']:+.3f} | {s['n_events']} | {s['hits_per_season']:.1f} | "
                     f"{s['false_alarms_per_season']:.1f} |")
        L += ["", "| Arm | vs model-only | vs random-analogue blend | Result |", "|---|---|---|---|"]
        for arm in ARMS:
            a, b = r["tests"][arm]["vs_model_only"], r["tests"][arm]["vs_random"]
            L.append(f"| {arm} | {a['delta']:+.4f} {_ci(a)} (p {_fmt_p(a['p_value'])}) | {b['delta']:+.4f} {_ci(b)} "
                     f"(p {_fmt_p(b['p_value'])}) | {'PASS' if r['passes'][arm] else 'fail'} {r['conditions'][arm]} |")
        rm = r["tests"]["random_vs_model_only"]
        L += ["", f"Random-analogue blend vs model-only: {rm['delta']:+.4f} {_ci(rm)} (p {_fmt_p(rm['p_value'])}).", ""]
    (OUT_DIR / "warning_retrieval.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    res = {fam: family_result(rid, target, labels, fam) for fam, (rid, target, labels) in FAMILIES.items()}
    write_report(res)
    (OUT_DIR / "warning_retrieval.json").write_text(json.dumps(res, indent=2, default=float) + "\n", encoding="utf-8")
    print({fam: r["passes"] for fam, r in res.items()})


if __name__ == "__main__":
    main()
