"""
Control comparison (plan v5, gate G2): every trained v2 run against the simple baselines,
pooled out-of-fold over the four rolling folds (validation blocks 2007-2018).

Runs are read from configs/*.json (all except A1_repro) with their predictions in
predictions_v2/<run_id>/<fold>.parquet; a run with a missing fold is listed and skipped.

Baselines are rebuilt PER FOLD, PER TARGET and PER LABEL SET from that fold's training years
only (training/folds.py: train-only climatology, labels, damped-persistence factors):
    persistence         every lead = target on the last input day
    climatology         target's train-only day-of-year mean
    damped_persistence  clim_mean + clim_std * phi_L * z_last, phi_L fitted on TRAIN windows
written to predictions_v2/baselines_<target>_<labels>/<fold>.parquet (predict_v1 format).

Comparisons only pair forecasts of the SAME target with the SAME labels (RMSE of different
variables is not comparable; strata must match):
  * every run vs damped persistence and vs climatology (all days and every stratum);
  * every run vs its config parent, when the parent is in the same family;
  * runs on the WBGT label are ALSO scored on the Tmax label (decision 2026-10-05);
  * the 10-seed ensemble mean is reported as an extra (decision 2026-10-04: not primary).
Statistics: paired cluster-jackknife t-test (evaluation/stats.py) over year x season
clusters, folds pooled; per-seed errors are concatenated across folds, so seed variance
enters the standard error.

Forecast-conditioned check (forecaster's dilemma): days a model FORECASTS a hot day by its
own label's rule (Tmax label: in season, Tmax >= 40 C and anomaly >= 3 C, or >= 45 C; WBGT
label: in season and forecast >= the fold's WBGT threshold). Descriptive, not paired.

Control choice (pre-registered 2026-10-05, context/decisions.md), per family:
  among runs NOT significantly worse than damped persistence on all days, the lowest
  all-days RMSE; within 0.02 C prefer fewer changes (non-raw target form, hot_weight != 20).

Writes evaluation_v2/week3_controls.md and .json. The test period is never read.
Run from repo root:  python -m evaluation.compare_v2
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.predict_v1 import LEADS, long_frame
from evaluation.stats import paired_cluster_test
from pipeline.labels_v2 import hot_days_tmax, in_wbgt_season, wbgt_threshold
from training.folds import FOLDS, build_fold, fold_bounds, fold_daily, fold_windows, wbgt_label_setting
from training.folds import fit_phi as _fit_phi

REPO_ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = REPO_ROOT / "predictions_v2"
CONFIG_DIR = REPO_ROOT / "configs"
OUT_DIR = REPO_ROOT / "evaluation_v2"
STRATA = ("all", "normal", "unusual", "extreme")
KEY = ["query_date", "lead"]
BASELINES = ("persistence", "climatology", "damped_persistence")
DEFAULT_HOT_WEIGHT = 20
TIE_MARGIN_C = 0.02
FAMILIES = {"Tmax": ("t_max", "v2"), "WBGT (physical)": ("wbgt_lj_max", "wbgt")}


# ------------------------------------------------------------------ runs and baselines


def load_runs(config_dir: Path = CONFIG_DIR) -> dict[str, dict]:
    """run_id -> config, for every v2 run config (A1_repro and label configs excluded)."""
    from training.train_unified import load_config

    runs = {}
    for path in sorted(config_dir.glob("*.json")):
        if path.stem in ("A1_repro", "wbgt_label"):
            continue
        cfg = load_config(path)
        runs[cfg["run_id"]] = cfg
    return runs


def fit_phi(fold: str, target: str, labels: str = "v2") -> np.ndarray:
    """Damped-persistence factors of one fold and target (training windows only)."""
    d, _ = fold_daily(fold, target, labels)
    return _fit_phi(d, target, fold_windows(fold)[0])


def fold_baselines(fold: str, target: str, labels: str = "v2") -> dict[str, pd.DataFrame]:
    """Validation-block forecasts of the three baselines, in predict_v1 long format, on
    exactly the windows, actuals and strata the trained models are scored on."""
    val = build_fold(fold, target, labels).val
    preds = {
        "persistence": np.repeat(val.persist[:, None], LEADS, axis=1),
        "climatology": val.clim_target,
        "damped_persistence": val.damped,
    }
    return {name: long_frame(name, 0, val.query_dates, p, val.y_raw, val.stratum) for name, p in preds.items()}


def write_baselines(target: str, labels: str, folds=tuple(FOLDS), out_dir: Path = PRED_DIR) -> dict[str, pd.DataFrame]:
    """Build, save and pool (over folds) the baselines for one target and label set."""
    pooled: dict[str, list[pd.DataFrame]] = {b: [] for b in BASELINES}
    path = out_dir / f"baselines_{target}_{labels}"
    path.mkdir(parents=True, exist_ok=True)
    for fold in folds:
        frames = fold_baselines(fold, target, labels)
        pd.concat(frames.values(), ignore_index=True).assign(fold=fold).to_parquet(path / f"{fold}.parquet", index=False)
        for b in BASELINES:
            pooled[b].append(frames[b].assign(fold=fold))
    return {b: pd.concat(v, ignore_index=True) for b, v in pooled.items()}


def load_run(run_id: str, folds=tuple(FOLDS), pred_dir: Path = PRED_DIR) -> pd.DataFrame | None:
    """Pooled out-of-fold predictions of one trained run (None if any fold is missing)."""
    frames = []
    for fold in folds:
        path = pred_dir / run_id / f"{fold}.parquet"
        if not path.exists():
            return None
        frames.append(pd.read_parquet(path).assign(fold=fold))
    df = pd.concat(frames, ignore_index=True)
    seeds = df.groupby("fold")["seed"].apply(lambda s: tuple(sorted(s.unique())))
    if seeds.nunique() != 1:
        raise ValueError(f"{run_id}: folds have different seed sets: {seeds.to_dict()}")
    return df


def tmax_strata_by_date(folds=tuple(FOLDS)) -> pd.Series:
    """Tmax-label (v2) stratum of every validation date, from ITS fold's training years."""
    parts = []
    for fold in folds:
        d, _ = fold_daily(fold, "t_max", "v2")
        _, a, b = fold_bounds(fold)
        parts.append(d.loc[a:b, "stratum"])
    return pd.concat(parts).sort_index()


def relabel(df: pd.DataFrame, strata_by_date: pd.Series) -> pd.DataFrame:
    """Same forecasts, scored on another label set (stratum looked up by target date)."""
    return df.assign(stratum=strata_by_date.reindex(pd.DatetimeIndex(df["target_date"])).to_numpy())


def ensemble(df: pd.DataFrame) -> pd.DataFrame:
    """Seed-mean forecast as a single pseudo-seed (reported as an extra, not primary)."""
    g = df.groupby(KEY, sort=True)
    out = g.agg(target_date=("target_date", "first"), pred=("pred", "mean"), actual=("actual", "first"),
                stratum=("stratum", "first"), cluster_id=("cluster_id", "first")).reset_index()
    return out.assign(seed=0, error=out["pred"] - out["actual"])


# ------------------------------------------------------------------ statistics


def _labels(df: pd.DataFrame) -> pd.DataFrame:
    """Stratum and cluster per (query_date, lead) from the lowest seed, sorted by key."""
    first = df[df["seed"] == df["seed"].min()].sort_values(KEY).reset_index(drop=True)
    return first[KEY + ["stratum", "cluster_id"]]


def _seed_errors(df: pd.DataFrame, ref: pd.DataFrame) -> list[np.ndarray]:
    """Per-seed 1-D error arrays aligned to ref's (query_date, lead) order."""
    idx = pd.MultiIndex.from_frame(ref[KEY])
    return [g.set_index(KEY).loc[idx, "error"].to_numpy() for _, g in df.groupby("seed", sort=True)]


def compare(parent: pd.DataFrame, child: pd.DataFrame, stratum: str) -> dict:
    """Paired cluster-jackknife test of RMSE(child) - RMSE(parent) on one stratum."""
    ref, child_ref = _labels(parent), _labels(child)
    if not ref.equals(child_ref):
        raise ValueError("parent and child disagree on windows, strata or clusters")
    if stratum != "all":
        ref = ref[ref["stratum"] == stratum].reset_index(drop=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = paired_cluster_test(parent=_seed_errors(parent, ref), child=_seed_errors(child, ref),
                                cluster_ids=ref["cluster_id"].to_numpy())
    row = r.to_dict()
    row["n_instances"] = len(ref)
    row["warnings"] = [str(w.message) for w in caught]
    return row


def rmse_by(df: pd.DataFrame, by: str | list[str] | None = None) -> pd.Series | float:
    """Mean over seeds of the per-seed RMSE (optionally per group)."""
    se = df.assign(se=df["error"] ** 2)
    if by is None:
        return float(se.groupby("seed")["se"].mean().pipe(np.sqrt).mean())
    keys = [by] if isinstance(by, str) else list(by)
    return se.groupby(["seed", *keys])["se"].mean().pipe(np.sqrt).groupby(keys).mean()


def _summarise_forecast_hot(df: pd.DataFrame, hot_fc: np.ndarray) -> dict:
    sub = df[hot_fc]
    if sub.empty:
        return {"n_per_seed": 0.0, "rmse": float("nan"), "bias": float("nan")}
    per_seed = sub.groupby("seed")["error"]
    return {"n_per_seed": float(per_seed.size().mean()),
            "rmse": float(per_seed.apply(lambda e: np.sqrt(np.mean(e.to_numpy() ** 2))).mean()),
            "bias": float(per_seed.mean().mean())}


def forecast_conditioned(df: pd.DataFrame, clim_by_date: pd.Series) -> dict:
    """Tmax forecasts: on days the model FORECASTS a v2 (Tmax) hot day, the number of such
    forecast-days, the RMSE and the mean error (bias), averaged over seeds."""
    clim = clim_by_date.reindex(pd.DatetimeIndex(df["target_date"])).to_numpy()
    return _summarise_forecast_hot(df, hot_days_tmax(df["target_date"], df["pred"], df["pred"].to_numpy() - clim))


def forecast_conditioned_wbgt(df: pd.DataFrame, threshold_by_fold: dict[str, float]) -> dict:
    """WBGT-label forecasts: days forecast in the WBGT season at or above the fold's threshold."""
    thr = df["fold"].map(threshold_by_fold).to_numpy()
    return _summarise_forecast_hot(df, in_wbgt_season(df["target_date"]) & (df["pred"].to_numpy() >= thr))


def _tmax_clim_by_date(folds=tuple(FOLDS)) -> pd.Series:
    """Each validation date's Tmax climatology, from ITS fold's training years."""
    parts = []
    for fold in folds:
        d, _ = fold_daily(fold, "t_max", "v2")
        _, a, b = fold_bounds(fold)
        parts.append(d.loc[a:b, "clim_mean_t_max"])
    return pd.concat(parts).sort_index()


def _wbgt_thresholds(folds=tuple(FOLDS)) -> dict[str, float]:
    var, pct = wbgt_label_setting()
    out = {}
    for fold in folds:
        d, _ = fold_daily(fold, var, "wbgt")
        train_end, _, _ = fold_bounds(fold)
        out[fold] = wbgt_threshold(d.index, d[var], d.index <= train_end, pct)
    return out


def complexity(cfg: dict) -> int:
    """Changes from the family's base recipe (for the pre-registered tie rule)."""
    return int(cfg.get("target_form", "raw") != "raw") + int(cfg["hot_weight"] != DEFAULT_HOT_WEIGHT)


def choose_control(rows: list[dict]) -> dict:
    """Pre-registered G2 rule over one family's summary rows (see module docstring)."""
    passing = [r for r in rows if r["floor_pass"]]
    if not passing:
        best = min(rows, key=lambda r: r["dp_all_delta"])
        return {"choice": best["run_id"], "passes_floor": False,
                "reason": "no run passes the floor; closest to damped persistence"}
    best_rmse = min(r["rmse_all"] for r in passing)
    tied = [r for r in passing if r["rmse_all"] - best_rmse <= TIE_MARGIN_C]
    pick = min(tied, key=lambda r: (r["complexity"], r["rmse_all"]))
    return {"choice": pick["run_id"], "passes_floor": True,
            "reason": f"lowest all-days RMSE among {len(passing)} passing runs (ties within {TIE_MARGIN_C} C -> simpler)"}


# ------------------------------------------------------------------ report


def _fmt_p(p: float) -> str:
    if np.isnan(p):
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _ci(r: dict) -> str:
    return "n/a" if np.isnan(r["ci_low"]) else f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"


def main() -> None:
    runs = load_runs()
    preds: dict[str, pd.DataFrame] = {}
    missing = []
    for run_id in runs:
        df = load_run(run_id)
        if df is None:
            missing.append(run_id)
        else:
            preds[run_id] = df
    families_used = sorted({(c["target"], c["labels"]) for r, c in runs.items() if r in preds})
    base = {key: write_baselines(*key) for key in families_used}
    tmax_strata = tmax_strata_by_date()
    tmax_clim = _tmax_clim_by_date()
    wbgt_thr = _wbgt_thresholds() if any(k[1] == "wbgt" for k in families_used) else {}

    summary, tests = [], []
    for run_id, df in preds.items():
        cfg = runs[run_id]
        key = (cfg["target"], cfg["labels"])
        dp, clim = base[key]["damped_persistence"], base[key]["climatology"]
        res = {s: compare(dp, df, s) for s in STRATA}
        res_clim = {s: compare(clim, df, s) for s in ("all", "extreme")}
        ens = ensemble(df)
        ens_dp = compare(dp, ens, "all")
        row = {"run_id": run_id, "parent": cfg["parent"], "target": cfg["target"], "labels": cfg["labels"],
               "target_form": cfg.get("target_form", "raw"), "hot_weight": cfg["hot_weight"], "complexity": complexity(cfg),
               "rmse_all": rmse_by(df), "rmse_extreme": rmse_by(df[df["stratum"] == "extreme"]),
               "dp_all_delta": res["all"]["delta"], "dp_all_ci": _ci(res["all"]), "dp_all_p": res["all"]["p_value"],
               "dp_ext_delta": res["extreme"]["delta"], "dp_ext_ci": _ci(res["extreme"]),
               "clim_all_delta": res_clim["all"]["delta"], "clim_ext_delta": res_clim["extreme"]["delta"],
               "ens_rmse_all": rmse_by(ens), "ens_dp_delta": ens_dp["delta"], "ens_dp_p": ens_dp["p_value"],
               # floor (decision 2026-10-04): NOT significantly worse than damped persistence
               "floor_pass": not (res["all"]["delta"] > 0 and res["all"]["ci_low"] > 0)}
        if cfg["labels"] == "wbgt":
            on_t = relabel(df, tmax_strata)
            dp_t = relabel(dp, tmax_strata)
            r_t = compare(dp_t, on_t, "extreme")
            row.update(tmaxlabel_ext_rmse=rmse_by(on_t[on_t["stratum"] == "extreme"]), tmaxlabel_dp_ext_delta=r_t["delta"],
                       tmaxlabel_dp_ext_ci=_ci(r_t))
            row["fc"] = forecast_conditioned_wbgt(df, wbgt_thr)
        elif cfg["target"] == "t_max":
            row["fc"] = forecast_conditioned(df, tmax_clim)
        summary.append(row)
        for s in STRATA:
            tests.append({**res[s], "parent": "damped_persistence", "child": run_id, "target": cfg["target"],
                          "labels": cfg["labels"], "stratum": s})
        par = cfg["parent"]
        if par in preds and (runs[par]["target"], runs[par]["labels"]) == key:
            for s in ("all", "extreme"):
                tests.append({**compare(preds[par], df, s), "parent": par, "child": run_id, "target": cfg["target"],
                              "labels": cfg["labels"], "stratum": s})

    baseline_fc = {}
    for key, b in base.items():
        for name in ("damped_persistence", "persistence"):
            if key == ("t_max", "v2"):
                baseline_fc[f"{name} (t_max)"] = forecast_conditioned(b[name], tmax_clim)
            elif key[1] == "wbgt":
                baseline_fc[f"{name} ({key[0]})"] = forecast_conditioned_wbgt(b[name], wbgt_thr)

    choices = {}
    for fam, key in FAMILIES.items():
        rows = [r for r in summary if (r["target"], r["labels"]) == key]
        if rows:
            choices[fam] = choose_control(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "week3_controls.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"pooled_folds": list(FOLDS), "metric": "rmse", "missing_runs": missing, "summary": summary,
                   "tests": tests, "baseline_forecast_conditioned": baseline_fc, "control_choice": choices,
                   "wbgt_thresholds": wbgt_thr}, fh, indent=2, default=float)

    L = ["# Week 3: control models vs simple baselines (gate G2)", "",
         "Generated by `python -m evaluation.compare_v2`. Out-of-fold validation 2007-2018 (folds f1-f4 pooled), "
         "RMSE in deg C, 10 seeds per run (primary score = mean of per-seed RMSEs). Runs are only compared within "
         "the same target and label set.", ""]
    if missing:
        L += [f"**Runs without complete predictions (skipped):** {', '.join(missing)}", ""]
    L += ["## Control choice (pre-registered rule, context/decisions.md 2026-10-05)", ""]
    for fam, c in choices.items():
        L.append(f"- **{fam}:** `{c['choice']}` (passes floor: {'yes' if c['passes_floor'] else 'NO'}; {c['reason']})")
    for fam, key in FAMILIES.items():
        rows = [r for r in summary if (r["target"], r["labels"]) == key]
        others = [r for r in summary if (r["target"], r["labels"]) not in FAMILIES.values()]
        for title, rr in ((f"{fam} family ({key[0]}, {key[1]} labels)", rows),):
            if not rr:
                continue
            L += ["", f"## {title}", "",
                  "| Run | Form | hot_weight | All RMSE | Δ vs climatology | Δ vs damped persistence (95% CI) | p | Floor | "
                  "Extreme RMSE | Δ extreme vs damped | 10-seed ensemble Δ vs damped (p) |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
            for r in sorted(rr, key=lambda x: x["rmse_all"]):
                L.append(f"| {r['run_id']} | {r['target_form']} | {r['hot_weight']} | {r['rmse_all']:.3f} | {r['clim_all_delta']:+.3f} | "
                         f"{r['dp_all_delta']:+.3f} {r['dp_all_ci']} | {_fmt_p(r['dp_all_p'])} | {'✅' if r['floor_pass'] else '❌'} | "
                         f"{r['rmse_extreme']:.3f} | {r['dp_ext_delta']:+.3f} {r['dp_ext_ci']} | {r['ens_dp_delta']:+.3f} ({_fmt_p(r['ens_dp_p'])}) |")
    other_rows = [r for r in summary if (r["target"], r["labels"]) not in FAMILIES.values()]
    if other_rows:
        L += ["", "## Other runs (historical BoM-index runs and chain steps)", "",
              "| Run | Target | Labels | Form | hot_weight | All RMSE | Δ vs damped persistence (95% CI) | Floor | Extreme RMSE | Δ extreme vs damped |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for r in other_rows:
            L.append(f"| {r['run_id']} | {r['target']} | {r['labels']} | {r['target_form']} | {r['hot_weight']} | {r['rmse_all']:.3f} | "
                     f"{r['dp_all_delta']:+.3f} {r['dp_all_ci']} | {'✅' if r['floor_pass'] else '❌'} | {r['rmse_extreme']:.3f} | "
                     f"{r['dp_ext_delta']:+.3f} {r['dp_ext_ci']} |")
    wl = [r for r in summary if r["labels"] == "wbgt"]
    if wl:
        L += ["", "## WBGT-label runs scored on the Tmax label (decision 2026-10-05)", "",
              "| Run | Extreme RMSE (Tmax-label days) | Δ vs damped persistence (95% CI) |", "|---|---|---|"]
        for r in wl:
            L.append(f"| {r['run_id']} | {r['tmaxlabel_ext_rmse']:.3f} | {r['tmaxlabel_dp_ext_delta']:+.3f} {r['tmaxlabel_dp_ext_ci']} |")
    L += ["", "## Forecast-conditioned: days each model FORECASTS a hot day by its own label rule", "",
          "Descriptive. A warm-biased model forecasts many hot days with a positive bias (forecaster's dilemma).", "",
          "| Model | Forecast hot days (per seed) | RMSE on them | Mean error (bias) |", "|---|---|---|---|"]
    for r in summary:
        if "fc" in r:
            v = r["fc"]
            L.append(f"| {r['run_id']} | {v['n_per_seed']:.0f} | {v['rmse']:.3f} | {v['bias']:+.3f} |")
    for name, v in baseline_fc.items():
        L.append(f"| {name} | {v['n_per_seed']:.0f} | {v['rmse']:.3f} | {v['bias']:+.3f} |")
    L += ["", "## All paired tests (delta = child - parent; negative = child better)", "",
          "| Target | Labels | Parent -> child | Stratum | n | Delta | 95% CI | p | G | Clusters child better |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for t in tests:
        frag = "*" if t["few_clusters"] else ""
        L.append(f"| {t['target']} | {t['labels']} | {t['parent']} -> {t['child']} | {t['stratum']} | {t['n_instances']} | "
                 f"{t['delta']:+.3f} | {_ci(t)} | {_fmt_p(t['p_value'])} | {t['n_clusters']}{frag} | "
                 f"{t['clusters_child_better']}/{t['n_clusters']} |")
    L += ["", "Notes:",
          "- Floor (decision 2026-10-04): not significantly worse than damped persistence on ALL days (95% CI of the "
          "difference not entirely above 0). Gains on observed-extreme days alone are not enough.",
          "- 95% CI and p: cluster-jackknife t-test, year x season clusters, G-1 df, seed variance included. "
          "`*` = fewer than 10 clusters (fragile).",
          "- 10-seed ensemble = mean of the 10 seeds' forecasts; reported as an extra, not the primary score.",
          "- Neural models early-stop on the last 2 TRAINING years of each fold; validation blocks never choose a checkpoint.",
          "- These are development (out-of-fold) results. The test period (2019+) is locked until Week 7."]
    (OUT_DIR / "week3_controls.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    # ASCII-only console summary (the Windows console may not encode the report's symbols)
    for r in sorted(summary, key=lambda x: (x["target"], x["labels"], x["rmse_all"])):
        print(f"{r['run_id']:14s} {r['target']:13s} {r['labels']:5s} RMSE {r['rmse_all']:.3f}  "
              f"vs damped {r['dp_all_delta']:+.3f} {r['dp_all_ci']}  floor {'PASS' if r['floor_pass'] else 'FAIL'}")
    for fam, c in choices.items():
        print(f"control [{fam}]: {c['choice']} (passes floor: {c['passes_floor']})")
    print(f"-> {(OUT_DIR / 'week3_controls.md').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
