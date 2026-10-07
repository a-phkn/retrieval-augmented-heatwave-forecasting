"""
Screen of retrieval rungs R2-R4 built on Rg's regional matching (specified in
context/decisions.md 2026-10-07, before any R2-R4 number was computed). No training; training
years only, exactly as evaluation/retrieval_information_check.py.

  R2  Rg + MMR diversity: greedy maximal marginal relevance over the 200 most similar eligible
      candidates, score = 0.7 sim(query, c) - 0.3 max_chosen sim(c, s), dedup rules applied.
  R3  Rg + one drift gate: ADF (autolag AIC) and KPSS ("c", automatic lags) on the target's
      standardised anomaly over the 365 days before the query's first input day; drift if
      ADF p > 0.05 or KPSS p < 0.05; then Rg restricted to the last 15 years. Power check:
      firing rate outside 5-95% -> inert, not screened.
  R4  Rg + climate-shift adjustment: each analogue outcome + beta (query year - analogue
      year), beta = OLS slope of the heat-season (Mar-Sep) mean standardised anomaly on year,
      fold training years.
A rung is trained only if (i) its gain vs the query-only baseline and vs its random control
both have 95% CIs entirely below 0 and (ii) it adds to Rg (CI of the difference entirely
below 0).

Run from repo root:  python -m evaluation.retrieval_ladder_screen
Writes evaluation_v2/retrieval_ladder_screen.{md,json}.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

from evaluation.retrieval_information_check import (
    FAMILIES, HELD_OUT_YEARS, OUT_DIR, _ci, analogue_signal, clim_cols, cluster_boot, ridge_fit_predict, rmse_seedmean,
)
from retrieval.fold_retrieval import (
    BUFFER_DAYS, K_DEFAULT, MAX_PER_EPISODE, MIN_DAYS_APART, FoldRetriever, window_episode, window_features,
)
from training.folds import FOLDS, FORECAST_DAYS, INPUT_DAYS, fold_bounds

MMR_LAMBDA, MMR_POOL = 0.7, 200
DRIFT_DAYS, RECENT_YEARS, ALPHA = 365, 15, 0.05
GATE_INERT = (0.05, 0.95)
SEEDS = range(10)
HEAT_SEASON = range(3, 10)  # Mar-Sep


def drift_flags(z: np.ndarray, first_input_pos: np.ndarray) -> np.ndarray:
    """Per query: ADF p > 0.05 or KPSS p < 0.05 on the DRIFT_DAYS before its first input day.
    Queries without that much history get False (no drift, plain Rg)."""
    from statsmodels.tsa.stattools import adfuller, kpss

    out = np.zeros(len(first_input_pos), dtype=bool)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # KPSS p-values outside its table are clipped (InterpolationWarning)
        for i, p in enumerate(first_input_pos):
            if p < DRIFT_DAYS:
                continue
            x = z[p - DRIFT_DAYS:p]
            out[i] = adfuller(x, autolag="AIC")[1] > ALPHA or kpss(x, regression="c", nlags="auto")[1] < ALPHA
    return out


def trend_beta(d: pd.DataFrame, z: np.ndarray, train_end: pd.Timestamp) -> float:
    """OLS slope (standardised anomaly per year) of the heat-season mean anomaly, training years."""
    s = pd.Series(z, index=d.index)
    s = s[(s.index <= train_end) & s.index.month.isin(HEAT_SEASON)]
    yearly = s.groupby(s.index.year).mean()
    return float(np.polyfit(yearly.index.to_numpy(float), yearly.to_numpy(), 1)[0])


class Selector:
    """Per-query eligibility exactly as FoldRetriever.retrieve (modes sim / rand / region), then a
    selection policy. Eligibility parity with retrieve() is checked by the tests."""

    def __init__(self, fr: FoldRetriever, q: pd.DatetimeIndex):
        self.fr, self.q = fr, q
        self.vectors = fr._region_state()[0]
        self.qv = fr.region_query_vectors(q)
        self.q_ep = window_episode(fr.d, q)
        self.n_cand = np.searchsorted(fr.cand_dates.values, (q - pd.Timedelta(days=BUFFER_DAYS)).values, side="right")
        self.cand_day = fr.cand_dates.values.astype("datetime64[D]").astype(np.int64)

    def eligible(self, i: int, recent: bool = False) -> np.ndarray:
        elig = np.zeros(len(self.fr.cand_dates), dtype=bool)
        elig[: self.n_cand[i]] = True
        if self.q_ep[i] > 0:
            elig &= self.fr.cand_episode != self.q_ep[i]
        if recent:
            elig &= self.fr.cand_dates.values >= (self.q[i] - pd.DateOffset(years=RECENT_YEARS)).to_datetime64()
        return elig

    def _ok(self, c: int, days: list, counts: dict) -> bool:
        if any(abs(self.cand_day[c] - x) < MIN_DAYS_APART for x in days):
            return False
        ep = self.fr.cand_episode[c]
        return not (ep > 0 and counts.get(ep, 0) >= MAX_PER_EPISODE)

    def _take(self, c: int, chosen: list, days: list, counts: dict) -> None:
        chosen.append(int(c))
        days.append(self.cand_day[c])
        ep = self.fr.cand_episode[c]
        if ep > 0:
            counts[ep] = counts.get(ep, 0) + 1

    def mmr(self, i: int, k: int = K_DEFAULT) -> list[int]:
        elig = self.eligible(i)
        cands = np.flatnonzero(elig)
        if len(cands) == 0:
            return []
        rel = self.vectors[cands] @ self.qv[i]
        order = np.argsort(-rel, kind="stable")
        for pool in (cands[order[:MMR_POOL]], cands[order]):  # widen only if dedup leaves fewer than k
            chosen, days, counts = [], [], {}
            rel_p = self.vectors[pool] @ self.qv[i]
            redund = np.full(len(pool), -np.inf)
            free = np.ones(len(pool), dtype=bool)
            while len(chosen) < k and free.any():
                score = np.where(free, MMR_LAMBDA * rel_p - (1 - MMR_LAMBDA) * np.where(np.isfinite(redund), redund, 0.0),
                                 -np.inf)
                j = int(np.argmax(score))
                free[j] = False
                c = pool[j]
                if self._ok(c, days, counts):
                    self._take(c, chosen, days, counts)
                    redund = np.maximum(redund, self.vectors[pool] @ self.vectors[c])
            if len(chosen) == k or len(pool) == len(cands):
                return chosen
        return chosen

    def top(self, i: int, recent: bool, k: int = K_DEFAULT) -> list[int]:
        cands = np.flatnonzero(self.eligible(i, recent))
        order = cands[np.argsort(-(self.vectors[cands] @ self.qv[i]), kind="stable")]
        return self.fr._dedup(order, k)

    def random(self, i: int, recent: bool, rng: np.random.Generator, k: int = K_DEFAULT) -> list[int]:
        return self.fr._dedup(rng.permutation(np.flatnonzero(self.eligible(i, recent))), k)


def _as_idx(rows: list[list[int]], k: int = K_DEFAULT) -> np.ndarray:
    idx = np.full((len(rows), k), -1, dtype=np.int64)
    for i, r in enumerate(rows):
        idx[i, : len(r)] = r
    return idx


def fold_family(fold: str, target: str, labels: str) -> dict:
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
    base_x = np.column_stack([window_features(d, q), own])
    train_end, _, _ = fold_bounds(fold)
    cutoff = train_end - pd.DateOffset(years=HELD_OUT_YEARS)
    held, before = q > cutoff, q <= cutoff - pd.Timedelta(days=BUFFER_DAYS)

    sel = Selector(fr, q)
    drift = drift_flags(z, pos - INPUT_DAYS)
    fold_seed = list(FOLDS).index(fold)
    idx = {"Rg": [fr.retrieve(q, "region").idx],
           "R0-rand": [fr.retrieve(q, "rand", seed=s).idx for s in SEEDS],
           "R2": [_as_idx([sel.mmr(i) for i in range(len(q))])],
           "R3": [_as_idx([sel.top(i, recent=bool(drift[i])) for i in range(len(q))])]}
    idx["R3-rand"] = []
    for s in SEEDS:
        rng = np.random.default_rng([fold_seed, s, 20261007])
        idx["R3-rand"].append(_as_idx([sel.random(i, bool(drift[i]), rng) for i in range(len(q))]))
    full = np.ones(len(q), dtype=bool)
    for runs in idx.values():
        for a in runs:
            full &= (a >= 0).all(axis=1)
    fit, ev = full & before, full & held

    beta = trend_beta(d, z, train_end)
    q_year = q.year.to_numpy()
    cand_year = fr.cand_dates.year.to_numpy()

    def signal(a: np.ndarray, adjust: bool) -> np.ndarray:
        sig = np.zeros((len(q), FORECAST_DAYS))
        sig[full] = analogue_signal(z, a[full], cand_pos)
        if adjust:  # mean over analogues of beta x (query year - analogue year)
            sig[full] += beta * (q_year[full, None] - cand_year[a[full]]).mean(axis=1)[:, None]
        return sig

    scale = d[std_col].to_numpy()[pos[:, None] + leads][ev]

    def sq_err(pred_z: np.ndarray) -> np.ndarray:
        return ((pred_z - y[ev]) * scale) ** 2

    def augmented(sig: np.ndarray) -> np.ndarray:
        cols = []
        for lead in range(FORECAST_DAYS):
            xa = np.column_stack([base_x, sig[:, lead]])
            cols.append(ridge_fit_predict(xa[fit], y[fit, lead:lead + 1], xa[ev])[:, 0])
        return np.column_stack(cols)

    errors = {"base": sq_err(ridge_fit_predict(base_x[fit], y[fit], base_x[ev]))[None]}
    for name, runs in idx.items():
        errors[name] = np.stack([sq_err(augmented(signal(a, adjust=False))) for a in runs])
    errors["R4"] = np.stack([sq_err(augmented(signal(a, adjust=True))) for a in idx["Rg"]])
    errors["R4-rand"] = np.stack([sq_err(augmented(signal(a, adjust=True))) for a in idx["R0-rand"]])
    return {"errors": errors, "year": np.repeat(q[ev].year.to_numpy()[:, None], FORECAST_DAYS, axis=1),
            "firing_rate": float(drift[full].mean()), "beta_per_decade": 10 * beta,
            "redundancy": {k: float(_mean_pair_sim(sel.vectors, idx[k][0][ev])) for k in ("Rg", "R2")}}


def _mean_pair_sim(vectors: np.ndarray, idx: np.ndarray) -> float:
    """Mean pairwise regional similarity among each query's K analogues (redundancy)."""
    v = vectors[idx]  # (N, K, D)
    g = np.einsum("nkd,njd->nkj", v, v)
    k = idx.shape[1]
    return float(((g.sum(axis=(1, 2)) - k) / (k * (k - 1))).mean())


RUNGS = {"R2": "R0-rand", "R3": "R3-rand", "R4": "R4-rand"}


def summarise(parts: list[dict]) -> dict:
    rng = np.random.default_rng(20261007)
    years = np.concatenate([p["year"].ravel() for p in parts])
    se = {k: np.concatenate([p["errors"][k].reshape(p["errors"][k].shape[0], -1) for p in parts], axis=1)
          for k in parts[0]["errors"]}
    firing = float(np.mean([p["firing_rate"] for p in parts]))
    res = {"firing_rate": firing, "beta_per_decade": [p["beta_per_decade"] for p in parts],
           "redundancy": {k: float(np.mean([p["redundancy"][k] for p in parts])) for k in ("Rg", "R2")},
           "rmse": {k: rmse_seedmean(v) for k, v in se.items()}, "tests": {}, "train": {}}

    def diff(a: str, b: str):
        return cluster_boot(lambda w: rmse_seedmean(se[a], w) - rmse_seedmean(se[b], w), years, rng)

    for rung, rand in RUNGS.items():
        t = {"vs base": diff(rung, "base"), f"vs {rand}": diff(rung, rand), "vs Rg": diff(rung, "Rg")}
        res["tests"][rung] = t
        inert = rung == "R3" and not (GATE_INERT[0] <= firing <= GATE_INERT[1])
        res["train"][rung] = "inert (gate firing rate outside 5-95%)" if inert else (
            "yes" if all(v[2] < 0 for v in t.values()) else "no")
    res["tests"]["Rg"] = {"vs base": diff("Rg", "base"), "vs R0-rand": diff("Rg", "R0-rand")}
    return res


def write_report(results: dict) -> None:
    L = ["# Retrieval ladder screen: R2-R4 on Rg (2026-10-07)", "",
         "Specified in `context/decisions.md` before it was run. No training; training years only (each fold's last 2 "
         "training years held out). Δ = RMSE (°C) of the query-only linear forecast with the rung's analogue signal "
         "minus the comparison; negative = the rung helps. 95% CI: bootstrap over the 8 held-out years (fragile).", "",
         "Trained only if the gain vs the baseline, vs its random control AND vs Rg are all significant.", ""]
    for fam, r in results.items():
        L += [f"## {fam}", "",
              f"Drift-gate firing rate (R3 power check): {r['firing_rate']:.1%}. Trend beta (R4): "
              + ", ".join(f"{b:+.3f}" for b in r["beta_per_decade"]) + " SD/decade per fold. Mean pairwise similarity "
              f"of the 5 analogues (redundancy): Rg {r['redundancy']['Rg']:.3f}, R2 {r['redundancy']['R2']:.3f}.", "",
              "| Rung | vs query-only baseline | vs its random control | vs Rg | Train? |", "|---|---|---|---|---|"]
        for rung, rand in RUNGS.items():
            t = r["tests"][rung]
            L.append(f"| {rung} | {_ci(t['vs base'])} | {_ci(t[f'vs {rand}'])} ({rand}) | {_ci(t['vs Rg'])} | {r['train'][rung]} |")
        t = r["tests"]["Rg"]
        L += [f"| Rg (reference) | {_ci(t['vs base'])} | {_ci(t['vs R0-rand'])} (R0-rand) | | |", ""]
    L += ["Notes:", "- R4-rand = R0-rand with the same climate-shift adjustment; R3-rand = random draws from R3's gated pool.",
          "- A linear check: a null says the simple analogue-mean information is absent, not that a network could not use it."]
    (OUT_DIR / "retrieval_ladder_screen.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    results = {}
    for fam, (target, labels) in FAMILIES.items():
        parts = []
        for fold in FOLDS:
            parts.append(fold_family(fold, target, labels))
            print(f"{fam} {fold}: drift firing {parts[-1]['firing_rate']:.1%}", flush=True)
        results[fam] = summarise(parts)
    write_report(results)
    (OUT_DIR / "retrieval_ladder_screen.json").write_text(json.dumps(results, indent=2, default=float) + "\n",
                                                          encoding="utf-8")
    for fam, r in results.items():
        print(fam, r["train"], f"firing {r['firing_rate']:.1%}")


if __name__ == "__main__":
    main()
