"""
Retrieval mechanism metrics (plan v5, Week 4: "mechanism metrics: redundancy, age, season
mismatch"). Descriptive: they explain why a rung helps or not; they decide nothing.

For each family (Tmax / physical WBGT) and each trained retrieval run (R0, R0-rand, R1, R1-rand,
Rg), over the validation queries of all 4 folds (out-of-fold, 2007-2018; the analogue files the
trainer saved):
  age              query year - analogue year (mean; share older than 20 years)
  season mismatch  circular day-of-year distance between query and analogue window ends
                   (mean days; share within +-30 days)
  redundancy       mean pairwise similarity among a query's 5 analogues, in v1's Delhi feature
                   space AND in Rg's regional space (same spaces for every run, so comparable)
  outcome signal   correlation, per lead, of the analogues' mean standardised outcome with the
                   query's truth (the AnEn signal; "does what followed the analogues track what
                   followed the query?"), all days and observed-extreme days
  attention        how unevenly the network weights its 5 analogues: 1 - entropy / log 5
                   (0 = uniform), and the within-query Spearman correlation of attention with
                   similarity (does it trust the most similar analogue more?)
Random runs: averaged over their 10 seeds. Nothing from 2019 on is read.

Run from repo root:  python -m evaluation.retrieval_mechanisms
Writes evaluation_v2/retrieval_mechanisms.{md,json}.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from evaluation.retrieval_information_check import FAMILIES, OUT_DIR, clim_cols
from retrieval.fold_retrieval import FoldRetriever, region_features, window_features
from training.folds import FOLDS, FORECAST_DAYS

PRED_DIR = OUT_DIR.parent / "predictions_v2"
RUNS = {"Tmax": "A1prime_hw5", "WBGT (physical)": "A2Lr_hw5"}
RUNGS = {"R0": "_R0", "R0-rand": "_R0rand", "R1": "_R1", "R1-rand": "_R1rand", "Rg": "_Rg"}
K = 5


def doy_distance(a: pd.DatetimeIndex, b: pd.DatetimeIndex) -> np.ndarray:
    """Circular day-of-year distance (0-182) between window-end days (query_date - 1)."""
    da = (a - pd.Timedelta(days=1)).dayofyear.to_numpy()
    db = (b - pd.Timedelta(days=1)).dayofyear.to_numpy()
    d = np.abs(da - db)
    return np.minimum(d, 365 - d)


def mean_pairwise(vectors: np.ndarray) -> np.ndarray:
    """(N, K, D) unit vectors -> (N,) mean pairwise cosine similarity among the K."""
    g = np.einsum("nkd,njd->nkj", vectors, vectors)
    k = vectors.shape[1]
    return (g.sum(axis=(1, 2)) - np.trace(g, axis1=1, axis2=2)) / (k * (k - 1))


def attention_unevenness(att: np.ndarray) -> np.ndarray:
    """(N, K) attention weights -> 1 - entropy / log K (0 = uniform, 1 = all on one)."""
    p = np.clip(att / att.sum(axis=1, keepdims=True), 1e-12, 1.0)
    return 1.0 - (-(p * np.log(p)).sum(axis=1)) / np.log(att.shape[1])


def rowwise_spearman(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N, K) x (N, K) -> (N,) Spearman correlation per row (NaN if a row is constant)."""
    ra = a.argsort(axis=1).argsort(axis=1).astype(float)
    rb = b.argsort(axis=1).argsort(axis=1).astype(float)
    ra -= ra.mean(axis=1, keepdims=True)
    rb -= rb.mean(axis=1, keepdims=True)
    den = np.sqrt((ra**2).sum(axis=1) * (rb**2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (ra * rb).sum(axis=1) / den, np.nan)


def _unit(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(n == 0, 1.0, n)


def fold_metrics(fold: str, target: str, labels: str, run_ids: dict[str, str]) -> dict[str, dict]:
    fr = FoldRetriever(fold, labels, target)
    d = fr.d
    mean_col, std_col = clim_cols(target)
    z = ((d[target] - d[mean_col]) / d[std_col]).to_numpy(np.float64)
    # common similarity spaces, normalised on the fold's candidate (training) windows
    delhi_raw = window_features(d, fr.cand_dates)
    region_raw = region_features(fr._region_state()[3], fr.cand_dates)
    spaces = {}
    for name, raw in (("delhi", delhi_raw), ("region", region_raw)):
        mu, sd = raw.mean(axis=0), raw.std(axis=0)
        sd[sd == 0] = 1.0
        spaces[name] = (_unit((raw - mu) / sd))
    cand_index = pd.Series(np.arange(len(fr.cand_dates)), index=fr.cand_dates)
    out = {}
    for rung, rid in run_ids.items():
        a = pd.read_parquet(PRED_DIR / rid / f"{fold}_analogues.parquet")
        p = pd.read_parquet(PRED_DIR / rid / f"{fold}.parquet", columns=["seed", "query_date", "lead", "stratum"])
        per_seed = []
        for seed, g in a.groupby("seed"):
            g = g.sort_values(["query_date", "rank"])
            q = pd.DatetimeIndex(g["query_date"].to_numpy()[::K])
            an = pd.DatetimeIndex(g["analogue_query_date"]).to_numpy().reshape(-1, K)
            if pd.isna(an).any():
                raise ValueError(f"{rid} {fold}: validation query without a full analogue set")
            an_idx = cand_index.reindex(pd.DatetimeIndex(an.ravel())).to_numpy().reshape(-1, K)
            att = g["attention"].to_numpy().reshape(-1, K)
            sim = g["similarity"].to_numpy().reshape(-1, K)
            age = q.year.to_numpy()[:, None] - pd.DatetimeIndex(an.ravel()).year.to_numpy().reshape(-1, K)
            season = doy_distance(pd.DatetimeIndex(np.repeat(q.values, K)), pd.DatetimeIndex(an.ravel())).reshape(-1, K)
            apos = d.index.get_indexer(pd.DatetimeIndex(an.ravel())).reshape(-1, K)
            zbar = z[apos[..., None] + np.arange(FORECAST_DAYS)].mean(axis=1)  # (N, L)
            qpos = d.index.get_indexer(q)
            truth = z[qpos[:, None] + np.arange(FORECAST_DAYS)]
            st = (p[p["seed"] == seed].pivot(index="query_date", columns="lead", values="stratum")
                  .reindex(q).to_numpy())
            per_seed.append({
                "age": age.ravel(), "season": season.ravel(),
                "red_delhi": mean_pairwise(spaces["delhi"][an_idx]), "red_region": mean_pairwise(spaces["region"][an_idx]),
                "zbar": zbar, "truth": truth, "extreme": st == "extreme",
                "uneven": attention_unevenness(att), "att_sim": rowwise_spearman(att, sim),
            })
        out[rung] = per_seed
    return out


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 2 else float("nan")


def summarise(parts: list[dict[str, list[dict]]]) -> dict[str, dict]:
    res = {}
    for rung in parts[0]:
        seeds = [[p[rung][s] for p in parts] for s in range(len(parts[0][rung]))]
        rows = []
        for folds in seeds:
            cat = {k: np.concatenate([f[k] for f in folds]) for k in folds[0]}
            ext = cat["extreme"]
            rows.append({
                "age_mean": cat["age"].mean(), "age_over_20": (cat["age"] > 20).mean(),
                "season_mean_days": cat["season"].mean(), "season_within_30": (cat["season"] <= 30).mean(),
                "redundancy_delhi": cat["red_delhi"].mean(), "redundancy_region": cat["red_region"].mean(),
                "signal_corr_by_lead": [_corr(cat["zbar"][:, l], cat["truth"][:, l]) for l in range(FORECAST_DAYS)],
                "signal_corr_extreme_by_lead": [_corr(cat["zbar"][ext[:, l], l], cat["truth"][ext[:, l], l])
                                                for l in range(FORECAST_DAYS)],
                "attention_unevenness": float(np.nanmean(cat["uneven"])),
                "attention_vs_similarity_spearman": float(np.nanmean(cat["att_sim"])),
            })
        res[rung] = {k: (np.mean([r[k] for r in rows], axis=0).tolist() if isinstance(rows[0][k], list)
                         else float(np.mean([r[k] for r in rows]))) for k in rows[0]}
    return res


def write_report(results: dict) -> None:
    f3 = lambda v: f"{v:.3f}"  # noqa: E731
    L = ["# Retrieval mechanism metrics (2026-10-07)", "",
         "Descriptive (plan v5 Week 4). Validation queries of all 4 folds (out of fold, 2007-2018), from the analogue "
         "files each run saved. Random runs: mean over 10 seeds. Redundancy is measured in the same two spaces for every "
         "run (Delhi's 17 features; Rg's regional pattern), so rungs are comparable.", ""]
    for fam, r in results.items():
        L += [f"## {fam}", "",
              "| Rung | Age (years) | Older than 20 y | Season gap (days) | Within ±30 d | Redundancy, Delhi space | "
              "Redundancy, regional space | Attention unevenness | Attention vs similarity (Spearman) |",
              "|---|---|---|---|---|---|---|---|---|"]
        for rung, m in r.items():
            L.append(f"| {rung} | {m['age_mean']:.1f} | {m['age_over_20']:.0%} | {m['season_mean_days']:.0f} | "
                     f"{m['season_within_30']:.0%} | {f3(m['redundancy_delhi'])} | {f3(m['redundancy_region'])} | "
                     f"{f3(m['attention_unevenness'])} | {m['attention_vs_similarity_spearman']:+.2f} |")
        L += ["", "Outcome signal: correlation of the analogues' mean standardised outcome with the query's truth, by lead "
              "(all days / observed-extreme days):", "",
              "| Rung | " + " | ".join(f"Lead {l + 1}" for l in range(FORECAST_DAYS)) + " |", "|---|" + "---|" * FORECAST_DAYS]
        for rung, m in r.items():
            L.append(f"| {rung} | " + " | ".join(f"{a:+.2f} / {b:+.2f}" for a, b in
                                                zip(m["signal_corr_by_lead"], m["signal_corr_extreme_by_lead"])) + " |")
        L.append("")
    L += ["Notes:", "- Attention unevenness 0 = the network weights its 5 analogues equally; a network that trusts some "
          "analogues more than others scores higher.",
          "- The outcome-signal correlation is not 'skill beyond the query's own inputs'; that is "
          "`evaluation_v2/retrieval_information_check.md`."]
    (OUT_DIR / "retrieval_mechanisms.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    results = {}
    for fam, (target, labels) in FAMILIES.items():
        run_ids = {rung: RUNS[fam] + suffix for rung, suffix in RUNGS.items()}
        parts = [fold_metrics(fold, target, labels, run_ids) for fold in FOLDS]
        results[fam] = summarise(parts)
        print(fam, "done", flush=True)
    write_report(results)
    (OUT_DIR / "retrieval_mechanisms.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
