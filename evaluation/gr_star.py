"""
G-R*: does retrieval (Rg) help the C3-hop backbone? Judged by the rule written before any run
(context/decisions.md 2026-10-13), the same three G3 conditions as every retrieval rung:
  (1) C3hop_Rg vs C3hop, all days: CI not entirely above 0;
  (2) C3hop_Rg vs C3hop, extreme days: CI entirely below 0;
  (3) C3hop_Rg vs C3hop_rand (random analogues, identical model), extreme days: CI entirely below 0.
Consequence: a family's final test model is C3-hop + Rg only if all three hold, else C3-hop alone.

Exploratory diagnostics (no claims; "which spatio-temporal part does retrieval help"):
  (a) per-lead deltas C3hop_Rg - C3hop (temporal);
  (b) windows with upstream heat arriving: mean of the 27 upstream points' standardised Tmax anomalies
      (Rg's own, fold-train climatology) on the last input day in the top quarter of the fold's
      training windows; delta inside vs outside (spatial);
  (c) attention concentration: mean max weight and normalised entropy (1 = uniform), next to LSTM Rg;
  (d) 2x2 overlap: (C3hop_Rg - C3hop) - (LSTM_Rg - LSTM), plain cluster jackknife over year x season
      clusters (seed-mean RMSE; seed variance not included, so read as indicative).
Retrieval acts only at C3-hop's Delhi readout (models/dstgnn_retrieval.py).

Run from repo root:  python -m evaluation.gr_star   ->  evaluation_v2/gr_star.{md,json}
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from evaluation.compare_v2 import (
    PRED_DIR, _ci, _fmt_p, _labels, _seed_errors, compare, extreme_bias, g3_conditions, load_run, mde80, rmse_by,
)
from evaluation.retrieval_information_check import OUT_DIR
from training.folds import FOLDS, FORECAST_DAYS

FAMILIES = {"Tmax": "A1prime_hw5", "WBGT (physical)": "A2Lr_hw5"}
UPSTREAM_QUANTILE = 0.75


def _need(run_id: str) -> pd.DataFrame:
    df = load_run(run_id)
    if df is None:
        raise FileNotFoundError(f"{run_id}: not all folds are trained yet")
    return df


def gate_rows() -> dict:
    """The pre-registered G-R* decision per family."""
    out = {}
    for fam, base in FAMILIES.items():
        ctrl, rg, rand = _need(f"{base}_C3hop"), _need(f"{base}_C3hop_Rg"), _need(f"{base}_C3hop_rand")
        a, e, re = compare(ctrl, rg, "all"), compare(ctrl, rg, "extreme"), compare(rand, rg, "extreme")
        row = {"rmse_all": rmse_by(rg), "rmse_extreme": rmse_by(rg[rg["stratum"] == "extreme"]),
               "ctrl_rmse_all": rmse_by(ctrl), "ctrl_rmse_extreme": rmse_by(ctrl[ctrl["stratum"] == "extreme"]),
               "ctrl_all": a, "ctrl_ext": e, "rand_all": compare(rand, rg, "all"), "rand_ext": re,
               "rand_vs_ctrl_all": compare(ctrl, rand, "all"),
               "ctrl_all_ci_low": a["ci_low"], "ctrl_ext_ci_high": e["ci_high"], "rand_ext_ci_high": re["ci_high"],
               "ext_mde": mde80(e), "ext_bias": extreme_bias(rg), "ctrl_ext_bias": extreme_bias(ctrl)}
        row["g3"] = g3_conditions(row)
        row["passes"] = all(row["g3"])
        row["final_model"] = f"{base}_C3hop_Rg" if row["passes"] else f"{base}_C3hop"
        out[fam] = row
    return out


# ------------------------------------------------------------------ diagnostics


def per_lead(base: str) -> list[dict]:
    ctrl, rg = _need(f"{base}_C3hop"), _need(f"{base}_C3hop_Rg")
    return [compare(ctrl[ctrl["lead"] == L], rg[rg["lead"] == L], "all") for L in range(1, FORECAST_DAYS + 1)]


def top_quarter(train_scores: np.ndarray, scores: np.ndarray, q: float = UPSTREAM_QUANTILE) -> np.ndarray:
    """True where a score is at or above the q-quantile of the training windows' scores."""
    return scores >= np.quantile(train_scores, q)


def upstream_heat_dates() -> pd.DatetimeIndex:
    """Validation query dates (all folds) whose last input day had upstream heat in the top quarter."""
    from retrieval.fold_retrieval import FoldRetriever, regional_daily

    hot = []
    for f in FOLDS:
        r = FoldRetriever(f, "v2", "t_max")
        rz = regional_daily(f, r.d, "t_max")
        up = rz[[c for c in rz.columns if c.startswith("temperature_2m_max__")]].mean(axis=1)
        def score(q):
            return up.reindex(pd.DatetimeIndex(q) - pd.Timedelta(days=1)).to_numpy()
        val_q = pd.DatetimeIndex(r.val_w["query_date"])
        hot.append(val_q[top_quarter(score(r.cand_dates), score(val_q))])
    return pd.DatetimeIndex(np.concatenate([h.values for h in hot]))


def upstream_split(base: str, hot_dates: pd.DatetimeIndex) -> dict:
    ctrl, rg = _need(f"{base}_C3hop"), _need(f"{base}_C3hop_Rg")
    out = {}
    for name, keep in (("upstream heat", True), ("other days", False)):
        sel = lambda df: df[df["query_date"].isin(hot_dates) == keep]
        out[name] = {**compare(sel(ctrl), sel(rg), "all"), "n_windows": int(sel(ctrl)["query_date"].nunique())}
    return out


def attention_stats(an: pd.DataFrame) -> dict:
    """Mean over (seed, query) of the largest weight and of the entropy / log(number of analogues)."""
    an = an[an["attention"] > 0]
    g = an.groupby(["seed", "query_date"])["attention"]
    k = g.size()
    ent = g.apply(lambda w: float(-(w * np.log(w)).sum()))
    norm = (ent / np.log(k.where(k > 1))).dropna()
    return {"mean_max_weight": float(g.max().mean()), "uniform_max_weight": float((1 / k).mean()),
            "mean_norm_entropy": float(norm.mean())}


def load_analogues(run_id: str) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(PRED_DIR / run_id / f"{f}_analogues.parquet") for f in FOLDS], ignore_index=True)


def did_jackknife(errors: dict[str, list[np.ndarray]], clusters: np.ndarray, ci_level: float = 0.95) -> dict:
    """(RMSE(c3_rg) - RMSE(c3)) - (RMSE(lstm_rg) - RMSE(lstm)), seed-mean RMSE per run, with a
    leave-one-cluster-out jackknife SE and t(G-1) interval. errors: run -> per-seed aligned errors."""
    codes, _ = pd.factorize(np.asarray(clusters, dtype=object))
    g = int(codes.max()) + 1
    loss = {k: np.stack([np.bincount(codes, weights=e ** 2, minlength=g) for e in v]) for k, v in errors.items()}
    count = np.bincount(codes, minlength=g).astype(float)

    def did(drop: int | None) -> float:
        keep = np.ones(g, bool) if drop is None else np.arange(g) != drop
        r = {k: float(np.sqrt(v[:, keep].sum(axis=1) / count[keep].sum()).mean()) for k, v in loss.items()}
        return (r["c3_rg"] - r["c3"]) - (r["lstm_rg"] - r["lstm"])

    full = did(None)
    loo = np.array([did(i) for i in range(g)])
    se = float(np.sqrt((g - 1) / g * ((loo - loo.mean()) ** 2).sum()))
    t = stats.t.ppf(0.5 + ci_level / 2, g - 1)
    p = float(2 * stats.t.sf(abs(full) / se, g - 1)) if se > 0 else float("nan")
    return {"delta": full, "ci_low": full - t * se, "ci_high": full + t * se, "p_value": p, "n_clusters": g}


def overlap_2x2(base: str, stratum: str) -> dict:
    runs = {"lstm": base, "lstm_rg": f"{base}_Rg", "c3": f"{base}_C3hop", "c3_rg": f"{base}_C3hop_Rg"}
    dfs = {k: _need(v) for k, v in runs.items()}
    ref = _labels(dfs["c3"])
    for k, df in dfs.items():
        if not _labels(df).equals(ref):
            raise ValueError(f"{runs[k]} disagrees with C3-hop on windows, strata or clusters")
    if stratum != "all":
        ref = ref[ref["stratum"] == stratum].reset_index(drop=True)
    res = did_jackknife({k: _seed_errors(df, ref) for k, df in dfs.items()}, ref["cluster_id"].to_numpy())
    res["lstm_gain"] = compare(dfs["lstm"], dfs["lstm_rg"], stratum)["delta"]
    res["c3_gain"] = compare(dfs["c3"], dfs["c3_rg"], stratum)["delta"]
    return res


def diagnostics() -> dict:
    hot = upstream_heat_dates()
    out = {}
    for fam, base in FAMILIES.items():
        out[fam] = {"per_lead": per_lead(base), "upstream": upstream_split(base, hot),
                    "attention_c3_rg": attention_stats(load_analogues(f"{base}_C3hop_Rg")),
                    "attention_lstm_rg": attention_stats(load_analogues(f"{base}_Rg")),
                    "overlap_all": overlap_2x2(base, "all"), "overlap_extreme": overlap_2x2(base, "extreme")}
    return out


# ------------------------------------------------------------------ report


def _t(r: dict) -> str:
    return f"{r['delta']:+.3f} {_ci(r)} (p {_fmt_p(r['p_value'])})"


def write_report(gate: dict, diag: dict) -> None:
    L = ["# G-R*: retrieval (Rg) on the C3-hop backbone (rule written 2026-10-13, before any run)", "",
         "Out of fold 2007-2018, 10 seeds x 4 folds, paired cluster-jackknife; Δ = first-named minus second "
         "(negative = C3hop_Rg better). Retrieval acts only at C3-hop's Delhi readout.", "",
         "## Decision (G3's three conditions)", "",
         "| Family | RMSE all: Rg / C3-hop | Extreme: Rg / C3-hop | (1) vs C3-hop, all | (2) vs C3-hop, extreme | "
         "(3) vs random, extreme | Result | Final test model |", "|---|---|---|---|---|---|---|---|"]
    for fam, r in gate.items():
        L.append(f"| {fam} | {r['rmse_all']:.3f} / {r['ctrl_rmse_all']:.3f} | {r['rmse_extreme']:.3f} / "
                 f"{r['ctrl_rmse_extreme']:.3f} | {_t(r['ctrl_all'])} | {_t(r['ctrl_ext'])} | {_t(r['rand_ext'])} | "
                 f"{'PASS' if r['passes'] else 'fail'} {r['g3']} | `{r['final_model']}` |")
    L += ["", "Also: C3hop_Rg vs random, all days: " + "; ".join(f"{f} {_t(r['rand_all'])}" for f, r in gate.items()),
          "Random analogues vs C3-hop, all days: " + "; ".join(f"{f} {_t(r['rand_vs_ctrl_all'])}" for f, r in gate.items()),
          "Extreme-day detectable effect (80% power): " + "; ".join(f"{f} {r['ext_mde']:.2f} °C" for f, r in gate.items()),
          "Extreme-day bias Rg / C3-hop: " + "; ".join(f"{f} {r['ext_bias']:+.2f} / {r['ctrl_ext_bias']:+.2f}"
                                                        for f, r in gate.items()),
          "", "## Diagnostics (exploratory, no claims)", ""]
    for fam, d in diag.items():
        L += [f"### {fam}", "", "**When (per lead, all days, Rg - C3-hop):** " +
              "; ".join(f"day {i} {_t(r)}" for i, r in enumerate(d["per_lead"], 1)), ""]
        u = d["upstream"]
        L += ["**Where (upstream heat arriving = top quarter of upstream Tmax anomaly on the last input day):** " +
              "; ".join(f"{k} ({v['n_windows']} windows) {_t(v)}" for k, v in u.items()), ""]
        a, b = d["attention_c3_rg"], d["attention_lstm_rg"]
        L += [f"**How (attention):** C3hop_Rg max weight {a['mean_max_weight']:.3f}, normalised entropy "
              f"{a['mean_norm_entropy']:.3f}; LSTM Rg {b['mean_max_weight']:.3f} / {b['mean_norm_entropy']:.3f} "
              f"(uniform: max {a['uniform_max_weight']:.3f}, entropy 1).", ""]
        for s in ("all", "extreme"):
            o = d[f"overlap_{s}"]
            L.append(f"**Overlap, {s} days:** Rg's gain on the LSTM {o['lstm_gain']:+.3f}, on C3-hop {o['c3_gain']:+.3f}; "
                     f"difference {_t(o)} (positive = smaller gain on the graph model).")
        L.append("")
    (OUT_DIR / "gr_star.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    gate = gate_rows()
    diag = diagnostics()
    write_report(gate, diag)
    (OUT_DIR / "gr_star.json").write_text(json.dumps({"gate": gate, "diagnostics": diag}, indent=2, default=float) + "\n",
                                          encoding="utf-8")
    print({f: (r["passes"], r["final_model"]) for f, r in gate.items()})


if __name__ == "__main__":
    main()
