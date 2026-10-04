"""
Week-1 premise checks (plan v5): two questions that decide how the retrieval /
non-stationarity hypotheses can be framed before any v2 model is built.

1. Trend check -- is the heatwave-season climate drifting, and is it drifting in
   humid heat (WBGT, Heat Index) rather than in air temperature (Tmax)? Seasonal
   (Mar 15-Jul 31) means per year of the domain daily maxima from hourly data
   (pipeline/hourly_features.py); Sen slope with 95% CI (scipy theilslopes) and
   Mann-Kendall p (Kendall tau of value vs year). TRAIN YEARS ONLY (1980-2015) as
   primary, train+val (1980-2018) as sensitivity; test years (2019+) are never used,
   so no decision here is informed by test-period targets.

2. Future-similarity retention (SARAF, Zhou et al. 2026, arXiv 2606.04135) -- for each
   val query, does a higher retrieval similarity mean the analogue's FUTURE is closer
   to the truth? Per query: Spearman rho between the 20 analogues' similarity and the
   negative RMSE of their standardised 5-day anomaly outcomes vs the query's own.
   Summarised as the mean rho, per year x season cluster, plus how the outcome error of
   ranks 1-5 compares with ranks 16-20.

Writes evaluation_v2/premise_checks.json and evaluation_v2/premise_report.md.
Run from repo root:  python -m evaluation.premise_checks
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from evaluation.stats import make_cluster_ids
from pipeline.hourly_features import domain_daily
from training.data import _load_daily, _load_windows

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "evaluation_v2"
ANALOGUES_PATH = REPO_ROOT / "retrieval" / "analogues_top20.parquet"
PERIODS = {"train 1980-2015 (primary)": (1980, 2015), "train+val 1980-2018": (1980, 2018)}
SEASONS = {
    "heat season Mar15-Jul31": (315, 731),
    "pre-monsoon Mar15-Jun15": (315, 615),
    "monsoon onset Jun16-Jul31": (616, 731),
}
VARIABLES = {
    "t_max": "Tmax",
    "wbgt_bom_max": "BoM WBGT approx.",
    "tw_max": "Wet-bulb Tw (Stull)",
    "hi_max": "Heat Index",
    "e_at_tmax": "Vapour pressure at Tmax (hPa)",
    "rh_at_tmax": "RH at Tmax (%)",
}


def trend_table(domain: pd.DataFrame) -> list[dict]:
    md = domain.index.month * 100 + domain.index.day
    rows = []
    for period, (y0, y1) in PERIODS.items():
        for season, (s0, s1) in SEASONS.items():
            sub = domain[(md >= s0) & (md <= s1) & (domain.index.year >= y0) & (domain.index.year <= y1)]
            annual = sub.groupby(sub.index.year).mean()
            years = annual.index.to_numpy(dtype=float)
            for var, label in VARIABLES.items():
                slope, _, lo, hi = sp_stats.theilslopes(annual[var].to_numpy(), years, alpha=0.95)
                tau, p = sp_stats.kendalltau(years, annual[var].to_numpy())
                resid = annual[var].to_numpy() - (slope * years + np.median(annual[var].to_numpy() - slope * years))
                lag1 = float(np.corrcoef(resid[:-1], resid[1:])[0, 1])
                rows.append({
                    "period": period, "season": season, "variable": label,
                    "sen_slope_per_decade": 10 * slope, "ci_low": 10 * lo, "ci_high": 10 * hi,
                    "mann_kendall_tau": tau, "mann_kendall_p": p, "n_years": len(annual),
                    "mean": float(annual[var].mean()),
                    "lag1_autocorr_detrended": lag1,
                })
    return rows


def retention(split: str = "val") -> dict:
    daily = _load_daily()
    windows = _load_windows()
    queries = windows.loc[windows["split"] == split, "query_date"]
    z = (daily["t_max_anomaly"] / daily["clim_std_t_max"]).to_numpy()
    pos_of = lambda dates: daily.index.get_indexer(pd.DatetimeIndex(dates))  # noqa: E731
    leads = np.arange(5)

    an = pd.read_parquet(ANALOGUES_PATH)
    an = an[an["query_date"].isin(queries)].sort_values(["query_date", "rank"])
    sims = an.pivot(index="query_date", columns="rank", values="similarity").reindex(queries)
    dates = an.pivot(index="query_date", columns="rank", values="analogue_query_date").reindex(queries)
    if sims.isna().any().any():
        raise ValueError("retention check expects 20 analogues for every query")

    truth = z[pos_of(queries)[:, None] + leads]  # (n, 5)
    outcome_rmse = np.stack(
        [np.sqrt(np.mean((z[pos_of(dates[r])[:, None] + leads] - truth) ** 2, axis=1)) for r in sims.columns], axis=1
    )  # (n, 20)
    rho = np.array([sp_stats.spearmanr(s, -e).statistic for s, e in zip(sims.to_numpy(), outcome_rmse)])

    clusters = make_cluster_ids(queries)
    per_cluster = pd.Series(rho).groupby(clusters).mean()
    # Cluster-level uncertainty of the mean rho: delete-one-cluster jackknife, t(G-1).
    rho_s, cl = pd.Series(rho), pd.Series(clusters)
    jack = np.array([rho_s[cl != g].mean() for g in per_cluster.index])
    g_n = len(jack)
    se = float(np.sqrt((g_n - 1) / g_n * np.sum((jack - jack.mean()) ** 2)))
    crit = sp_stats.t.ppf(0.975, g_n - 1)
    sim = sims.to_numpy()
    top, bottom = outcome_rmse[:, :5].mean(axis=1), outcome_rmse[:, 15:].mean(axis=1)
    return {
        "split": split,
        "n_queries": int(len(queries)),
        "mean_spearman_rho": float(np.nanmean(rho)),
        "mean_rho_ci95_cluster_jackknife": [float(np.nanmean(rho) - crit * se), float(np.nanmean(rho) + crit * se)],
        "similarity_rank1_mean": float(sim[:, 0].mean()),
        "similarity_rank20_mean": float(sim[:, -1].mean()),
        "share_queries_rho_positive": float(np.nanmean(rho > 0)),
        "per_cluster_mean_rho": {str(k): float(v) for k, v in per_cluster.items()},
        "clusters_with_positive_mean_rho": int((per_cluster > 0).sum()),
        "n_clusters": int(per_cluster.size),
        "outcome_rmse_ranks_1_5": float(top.mean()),
        "outcome_rmse_ranks_16_20": float(bottom.mean()),
        # Reference: same metric for a single 'analogue' that is pure climatology (z = 0).
        "outcome_rmse_climatology_reference": float(np.sqrt(np.mean(truth**2, axis=1)).mean()),
        "share_queries_top5_better_than_ranks_16_20": float(np.mean(top < bottom)),
        "note": "rho > 0: higher input similarity goes with analogue futures closer to the truth "
                "(standardised Tmax anomaly, 5-day outcome).",
    }


def anen_lines() -> list[str]:
    """Key AnEn rows from the G0 comparison (run evaluation.compare_v1 first)."""
    path = OUT_DIR / "g0_val_comparison.json"
    if not path.exists():
        return ["- (run `python -m evaluation.compare_v1` first for the AnEn numbers)"]
    rows = {(r["parent"], r["child"], r["stratum"]): r for r in json.loads(path.read_text(encoding="utf-8"))["results"]}
    out = []
    for (pa, ch), text in {
        ("climatology", "anen"): "Retrieved-analogue average vs climatology (the honest control)",
        ("damped_persistence", "anen"): "Analogue average vs damped persistence",
        ("anen_random", "anen"): "Retrieved vs 5 random past windows (mostly the noise penalty of random draws)",
    }.items():
        r = rows.get((pa, ch, "all"))
        if r:
            out.append(f"- {text}: delta {r['delta']:+.3f} C [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}], "
                       f"p={r['p_value']:.3f}, child better in {r['clusters_child_better']}/{r['n_clusters']} clusters (all days).")
    return out


def report(trends: list[dict], ret: dict) -> str:
    lines = ["# Week-1 premise checks", "", "Generated by `python -m evaluation.premise_checks`. Test years (2019+) are not used.", ""]
    lines += ["## 1. Seasonal trends (Sen slope per decade, 95% CI; Mann-Kendall p)", ""]
    for period in PERIODS:
        lines += [f"### {period}", "", "| Season | Variable | Mean | Slope / decade | 95% CI | MK p | Lag-1 r |", "|---|---|---|---|---|---|---|"]
        for r in (r for r in trends if r["period"] == period):
            sig = " **" if r["ci_low"] > 0 or r["ci_high"] < 0 else ""
            lines.append(
                f"| {r['season']} | {r['variable']} | {r['mean']:.2f} | {r['sen_slope_per_decade']:+.3f}{sig} | "
                f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | {r['mann_kendall_p']:.3f} | {r['lag1_autocorr_detrended']:+.2f} |"
            )
        lines.append("")
    lines += [
        "`**` = 95% CI of the Sen slope excludes zero.", "",
        "Reading the trends carefully:",
        "- Tmax: **no detectable** trend (the CI includes zero but also allows up to roughly +0.5 C/decade) -- "
        "not proof of no change.",
        "- The BoM WBGT approximation's trend comes mostly (~80%) from its vapour-pressure term, so check it "
        "against wet-bulb temperature and vapour pressure, which are physically interpretable.",
        "- Serial correlation: the Lag-1 r column is the lag-1 autocorrelation of each detrended annual series; "
        "values near zero mean the plain Mann-Kendall p-values are not inflated by persistence.",
        "- Caveat: rising humidity in ERA5 could partly reflect reanalysis inhomogeneity (vapour pressure at "
        "Tmax shows a step around 2000-01). Cross-check with station humidity (IMD or HadISDH) before claiming "
        "a physical trend in the paper. The 9 cells are neighbouring ERA5 points, not independent evidence.",
        "",
    ]
    lines += [
        "## 2. Future-similarity retention (val queries, 20 analogues each)", "",
        f"- Mean Spearman rho (similarity vs analogue-future accuracy): **{ret['mean_spearman_rho']:+.3f}** "
        f"(95% CI over clusters [{ret['mean_rho_ci95_cluster_jackknife'][0]:+.3f}, {ret['mean_rho_ci95_cluster_jackknife'][1]:+.3f}]); "
        f"rho > 0 for {ret['share_queries_rho_positive']:.0%} of queries; positive mean rho in "
        f"{ret['clusters_with_positive_mean_rho']}/{ret['n_clusters']} year x season clusters.",
        f"- Outcome RMSE (standardised anomaly) of ranks 1-5: {ret['outcome_rmse_ranks_1_5']:.3f} vs ranks 16-20: "
        f"{ret['outcome_rmse_ranks_16_20']:.3f}; top-5 better for {ret['share_queries_top5_better_than_ranks_16_20']:.0%} of queries. "
        f"Reference (climatology, z = 0): {ret['outcome_rmse_climatology_reference']:.3f}.",
        f"- Interpretation (limited by range): similarity only spans {ret['similarity_rank1_mean']:.2f} (rank 1) to "
        f"{ret['similarity_rank20_mean']:.2f} (rank 20) within the top 20, so this shows similarity does not RANK "
        "analogues within the top 20 -- not that similarity is uninformative overall. A single analogue's future is "
        "noisier than climatology; only averages approach it.",
        "",
        "## 3. Analogue-ensemble checks (from `evaluation_v2/g0_val_comparison.md`)", "",
        *anen_lines(),
        "- Caveat: strata are defined on the OBSERVED outcome, which favours warm-biased forecasters",
        "  (forecaster's dilemma, Lerch et al. 2017, Statistical Science 32(1)).",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    _, domain = domain_daily()
    trends = trend_table(domain)
    ret = retention("val")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "premise_checks.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump({"trends": trends, "retention": ret}, f, indent=2, default=float)
    text = report(trends, ret)
    (OUT_DIR / "premise_report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
