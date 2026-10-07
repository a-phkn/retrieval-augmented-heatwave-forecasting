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
Retrieval ladder (pre-registered 2026-10-06): runs with a "retrieval" config key are kept
  out of the control choice and reported in their own section against their control
  (A1prime_hw5 / A2Lr_hw5), against R0-rand, and next to the analogue-ensemble (AnEn)
  baseline; choose_retrieval_rung applies the G3 rule.
Adopted control (decision 2026-10-06, a disclosed deviation): the rule above picks
  hot_weight = 1, whose WBGT model never forecasts a hot day. hot_weight is therefore fixed
  at 5 and the same rule chooses among the hot_weight = 5 runs. Both choices are reported.

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
from scipy import stats as sp_stats

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
ADOPTED_HOT_WEIGHT = 5  # decision 2026-10-06 (deviation from the pre-registered rule, disclosed)
# Retrieval ladder (pre-registered 2026-10-06): controls and rung order for the tie rule.
RETRIEVAL_CONTROLS = {"Tmax": "A1prime_hw5", "WBGT (physical)": "A2Lr_hw5"}
RUNG_NAMES = {"sim": "R0", "rand": "R0-rand", "time": "R1", "time_rand": "R1-rand", "region": "Rg"}
RUNG_SIMPLICITY = {"sim": 0, "time": 1, "region": 2}
# Each rung's random control (amendment 2026-10-06: R1 is judged against calendar-matched R1-rand;
# Rg, pre-registered 2026-10-07, shares R0's pool, so R0-rand is its random control).
RANDOM_CONTROL = {"sim": "rand", "time": "time_rand", "region": "rand"}
# G3 is decided once these have run. Rg is optional: it is trained only for a family whose
# training-years information screen passed (decision 2026-10-07).
G3_REQUIRED = {"sim", "rand", "time", "time_rand"}


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


def anen_run(target: str, labels: str, folds=tuple(FOLDS)) -> pd.DataFrame:
    """Analogue-ensemble baseline (non-neural) on the validation blocks: the same top-5 R0
    analogues (retrieval.fold_retrieval, "sim"), averaged as standardised anomalies."""
    from retrieval.fold_retrieval import FoldRetriever, anen_forecast

    frames = []
    for fold in folds:
        fr = FoldRetriever(fold, labels, target)
        val = build_fold(fold, target, labels).val
        r = fr.retrieve(val.query_dates, "sim")
        dates = np.where(r.idx >= 0, fr.cand_dates.values[np.maximum(r.idx, 0)], np.datetime64("NaT"))
        pred = anen_forecast(fr.d, target, val.query_dates, dates)
        frames.append(long_frame("anen", 0, val.query_dates, pred, val.y_raw, val.stratum).assign(fold=fold))
    return pd.concat(frames, ignore_index=True)


def choose_retrieval_rung(rows: list[dict]) -> dict:
    """Pre-registered G3 rule (context/decisions.md 2026-10-06, as amended before results),
    over one family's rungs (the random controls are not candidates). A rung helps only if
    (1) it is not significantly worse than the control on all days, (2) it is significantly
    better on extreme days, and (3) it is significantly better than ITS random control on
    extreme days (R0 vs R0-rand, R1 vs R1-rand). Lowest extreme RMSE wins; within 0.02 C the
    simpler rung (R0 before R1). The forecast-conditioned bias is descriptive only."""
    helps = [r for r in rows if all(g3_conditions(r))]
    if not helps:
        return {"choice": None, "helps": [], "reason": "no rung meets all three conditions: retrieval does not help yet"}
    best = min(r["rmse_extreme"] for r in helps)
    tied = [r for r in helps if r["rmse_extreme"] - best <= TIE_MARGIN_C]
    pick = min(tied, key=lambda r: (RUNG_SIMPLICITY[r["mode"]], r["rmse_extreme"]))
    return {"choice": pick["run_id"], "helps": [r["run_id"] for r in helps],
            "reason": f"lowest extreme-day RMSE among {len(helps)} helping rung(s) (ties within {TIE_MARGIN_C} C -> simpler)"}


def extreme_bias(df: pd.DataFrame) -> float:
    """Mean error (forecast - actual) on observed-extreme days, averaged over seeds (descriptive)."""
    e = df[df["stratum"] == "extreme"]
    return float((e["pred"] - e["actual"]).groupby(e["seed"]).mean().mean()) if len(e) else float("nan")


def g3_conditions(r: dict) -> tuple[bool, bool, bool]:
    """The three G3 conditions for one rung row. A missing (NaN) CI bound fails its condition."""
    return (bool(r["ctrl_all_ci_low"] <= 0),  # (1) all days: CI not entirely above 0
            bool(r["ctrl_ext_ci_high"] < 0),  # (2) extreme days vs control: CI entirely below 0
            bool(r.get("rand_ext_ci_high", np.nan) < 0))  # (3) extreme days vs own random control


def mde80(res: dict) -> float:
    """Smallest true difference the paired test detects with 80% power (two-sided 5%),
    from the CI half-width and the cluster t distribution (G-1 df)."""
    df = res["n_clusters"] - 1
    if df < 1 or np.isnan(res["ci_low"]):
        return float("nan")
    t975 = sp_stats.t.ppf(0.975, df)
    return float((t975 + sp_stats.t.ppf(0.80, df)) * (res["ci_high"] - res["ci_low"]) / (2 * t975))


def retrieval_rows(fam: str, runs: dict, preds: dict, summary: list[dict], anen: pd.DataFrame | None) -> list[dict]:
    """One row per retrieval run of a family (and the AnEn baseline): deltas vs the control,
    vs the rung's own random control on every stratum, and the forecast-conditioned check."""
    control = RETRIEVAL_CONTROLS[fam]
    if control not in preds:
        return []
    ctrl = preds[control]
    by_id = {r["run_id"]: r for r in summary}
    rungs: dict[str, str] = {}
    family_key = (runs[control]["target"], runs[control]["labels"])
    for rid, c in runs.items():
        if not (c.get("retrieval") and rid in preds and (c["target"], c["labels"]) == family_key):
            continue
        if c["parent"] not in (control, f"{control}_R0"):  # never drop a run silently (it would block G3)
            raise ValueError(f"{fam}: retrieval run {rid} has parent {c['parent']!r}; "
                             f"expected {control!r} or {control + '_R0'!r}")
        mode = c["retrieval"]["mode"]
        if mode in rungs:  # never let a second run (e.g. a k ablation) silently replace a rung
            raise ValueError(f"{fam}: runs {rungs[mode]} and {rid} both claim retrieval mode {mode!r}")
        rungs[mode] = rid
    rows = []
    entries = [(m, rid, preds[rid]) for m, rid in rungs.items()] + ([("anen", "anen", anen)] if anen is not None else [])
    for mode, rid, df in entries:
        a, e = compare(ctrl, df, "all"), compare(ctrl, df, "extreme")
        row = {"run_id": rid, "mode": mode, "rung": RUNG_NAMES.get(mode, "AnEn (no network)"),
               "rmse_all": rmse_by(df), "rmse_extreme": rmse_by(df[df["stratum"] == "extreme"]),
               "ctrl_all_delta": a["delta"], "ctrl_all_ci_low": a["ci_low"], "ctrl_all_ci": _ci(a), "ctrl_all_p": a["p_value"],
               "ctrl_ext_delta": e["delta"], "ctrl_ext_ci_high": e["ci_high"], "ctrl_ext_ci": _ci(e),
               "ctrl_ext_p": e["p_value"], "ctrl_ext_mde": mde80(e), "fc": by_id.get(rid, {}).get("fc"),
               "ext_bias": extreme_bias(df), "ctrl_ext_bias": extreme_bias(ctrl)}
        rand_id = rungs.get(RANDOM_CONTROL.get(mode, ""))
        if rand_id is not None:
            rand = preds[rand_id]
            row["rand_run"] = rand_id
            for s in STRATA:
                if s != "all" and not (df["stratum"] == s).any():  # stratum absent: report n/a
                    row[f"rand_{s}_delta"], row[f"rand_{s}_ci"], row[f"rand_{s}_p"] = float("nan"), "n/a", float("nan")
                    continue
                v = compare(rand, df, s)
                row[f"rand_{s}_delta"], row[f"rand_{s}_ci"], row[f"rand_{s}_p"] = v["delta"], _ci(v), v["p_value"]
                if s == "extreme":
                    row["rand_ext_ci_high"], row["rand_ext_mde"] = v["ci_high"], mde80(v)
        rows.append(row)
    return rows


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


def adopted_control(rows: list[dict]) -> dict | None:
    """Decision 2026-10-06: hot_weight fixed at ADOPTED_HOT_WEIGHT, then the pre-registered rule."""
    fixed = [r for r in rows if r["hot_weight"] == ADOPTED_HOT_WEIGHT]
    if not fixed:
        return None
    c = choose_control(fixed)
    return {**c, "reason": f"hot_weight fixed at {ADOPTED_HOT_WEIGHT} (decision 2026-10-06); then {c['reason']}"}


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
               "retrieval": cfg.get("retrieval", {}).get("mode", ""),
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

    choices, rule_choices = {}, {}
    for fam, key in FAMILIES.items():
        rows = [r for r in summary if (r["target"], r["labels"]) == key and not r["retrieval"]]
        if rows:
            rule_choices[fam] = choose_control(rows)
            adopted = adopted_control(rows)
            if adopted is not None:
                choices[fam] = adopted

    ret_rows, ret_choice = {}, {}
    for fam, key in FAMILIES.items():
        has_rungs = any(c.get("retrieval") and (c["target"], c["labels"]) == key and r in preds for r, c in runs.items())
        if not has_rungs:
            continue
        anen = anen_run(*key)
        anen_fc = forecast_conditioned(anen, tmax_clim) if key[1] == "v2" else forecast_conditioned_wbgt(anen, wbgt_thr)
        rows = retrieval_rows(fam, runs, preds, summary, anen)
        for r in rows:
            if r["mode"] == "anen":
                r["fc"] = anen_fc
        ret_rows[fam] = rows
        complete = {r["mode"] for r in rows} >= G3_REQUIRED
        rungs = [r for r in rows if r["mode"] in RUNG_SIMPLICITY]
        ret_choice[fam] = (choose_retrieval_rung(rungs) if complete
                           else {"choice": None, "helps": [], "reason": "not all rungs have run yet"})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "week3_controls.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"pooled_folds": list(FOLDS), "metric": "rmse", "missing_runs": missing, "summary": summary,
                   "tests": tests, "baseline_forecast_conditioned": baseline_fc, "control_choice": choices,
                   "control_choice_preregistered_rule": rule_choices, "wbgt_thresholds": wbgt_thr,
                   "retrieval": ret_rows, "retrieval_choice": ret_choice}, fh, indent=2, default=float)

    L = ["# Week 3: control models vs simple baselines (gate G2)", "",
         "Generated by `python -m evaluation.compare_v2`. Out-of-fold validation 2007-2018 (folds f1-f4 pooled), "
         "RMSE in deg C, 10 seeds per run (primary score = mean of per-seed RMSEs). Runs are only compared within "
         "the same target and label set.", ""]
    if missing:
        L += [f"**Runs without complete predictions (skipped):** {', '.join(missing)}", ""]
    L += ["## Control choice", "", "**Adopted controls (decision 2026-10-06):**", ""]
    for fam, c in choices.items():
        L.append(f"- **{fam}:** `{c['choice']}` (passes floor: {'yes' if c['passes_floor'] else 'NO'}; {c['reason']})")
    L += ["", "**Pre-registered rule as written (2026-10-05), reported for transparency:**", ""]
    for fam, c in rule_choices.items():
        L.append(f"- **{fam}:** `{c['choice']}` (passes floor: {'yes' if c['passes_floor'] else 'NO'}; {c['reason']})")
    L += ["", "Why they differ: the rule scores all days only, and a lower hot_weight always helps there. It picks "
          "hot_weight = 1, which gives no extreme-day gain over damped persistence for Tmax and never forecasts a "
          "WBGT hot day. The deviation fixes hot_weight at 5 and keeps the rule for everything else; it was decided "
          "on development folds only, with the test period still locked. See context/decisions.md 2026-10-06."]
    for fam, key in FAMILIES.items():
        rows = [r for r in summary if (r["target"], r["labels"]) == key and not r["retrieval"]]
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
    for fam, rows in ret_rows.items():
        c = ret_choice[fam]
        L += ["", f"## Retrieval ladder, {fam} family (gate G3, rule pre-registered 2026-10-06, amended before results)", "",
              f"Control: `{RETRIEVAL_CONTROLS[fam]}`. Δ = rung minus control; negative = rung better. "
              "Each rung also has a random control fed the same model random eligible past windows (R0 vs R0-rand; "
              "R1 vs R1-rand, random within ±30 days of the same time of year): beating it shows the retrieved "
              "*information* is used, not just the extra machinery or the season.", ""]
        if any(r["mode"] == "region" for r in rows):
            L += ["**Rg was added after G3 had been seen** (G3 = none for R0/R1). It was pre-registered on 2026-10-07 and "
                  "screened on training years before any Rg training (`evaluation_v2/retrieval_information_check.md`). "
                  "The idea also followed G-D0, which was scored on these validation blocks; Rg reuses G-D0's "
                  "pre-registered layout (27 points, 3 lags) untuned. Treat Rg as post-G3 exploration.", ""]
        L += [
              f"**G3 choice:** {('`' + c['choice'] + '`') if c['choice'] else 'none'} ({c['reason']})", "",
              "| Rung | Run | All RMSE | Δ all vs control (95% CI) | p | Extreme RMSE | Δ extreme vs control (95% CI) | p | "
              "G3 conditions 1 / 2 / 3 | Extreme-day bias, rung / control (descriptive) | Forecast hot days / bias (descriptive) |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            fc = r.get("fc") or {}
            fc_txt = f"{fc['n_per_seed']:.0f} / {fc['bias']:+.2f}" if fc else "n/a"
            if r["mode"] in RUNG_SIMPLICITY:
                g3 = " / ".join("✅" if ok else "❌" for ok in g3_conditions(r))
            else:
                g3 = "(control)" if r["mode"] in RANDOM_CONTROL.values() else "(baseline)"
            L.append(f"| {r['rung']} | {r['run_id']} | {r['rmse_all']:.3f} | {r['ctrl_all_delta']:+.3f} {r['ctrl_all_ci']} | "
                     f"{_fmt_p(r['ctrl_all_p'])} | {r['rmse_extreme']:.3f} | {r['ctrl_ext_delta']:+.3f} {r['ctrl_ext_ci']} | "
                     f"{_fmt_p(r['ctrl_ext_p'])} | {g3} | {r['ext_bias']:+.2f} / {r['ctrl_ext_bias']:+.2f} | {fc_txt} |")
        if any(r["mode"] in RUNG_SIMPLICITY for r in rows):
            L += ["", "Conditions: (1) not significantly worse than the control on all days; (2) significantly better than the "
                  "control on extreme days; (3) significantly better than its own random control on extreme days. "
                  "p < 0.05 with a positive Δ on all days fails (1), even when the rounded CI shows +0.000."]
        vs = [r for r in rows if "rand_run" in r]
        if vs:
            L += ["", "Rung vs its random control, every stratum (Δ = rung minus random control, 95% CI, p). "
                  "Last column: smallest extreme-day difference this test detects with 80% power.", "",
                  "| Rung | Random control | " + " | ".join(STRATA) + " | Detectable (extreme, 80% power) |",
                  "|---|---|" + "---|" * len(STRATA) + "---|"]
            for r in vs:
                L.append(f"| {r['rung']} | {r['rand_run']} | "
                         + " | ".join(f"{r[f'rand_{s}_delta']:+.3f} {r[f'rand_{s}_ci']} (p {_fmt_p(r[f'rand_{s}_p'])})" for s in STRATA)
                         + f" | {r.get('rand_ext_mde', float('nan')):.2f} °C |")
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
    if ret_rows:
        L += ["- Retrieval caveats (independent review 2026-10-06):",
              "  - R0/R1 similarity uses v1's 17 features, which are Tmax-based, for the WBGT family too (as pre-registered); "
              "Rg matches on the regional pattern instead (upstream Tmax anomalies, plus dew point for WBGT, last 3 input days);",
              "  - the network receives analogue outcomes in its own target units (for A2Lr: WBGT anomaly in deg C, scaled), "
              "while AnEn averages standardised anomalies;",
              "  - the 'not from the query's own episode' rule looks at the query's target days, as in v1. That is future "
              "label information, but it can only remove candidates. It affects a handful of training queries and no "
              "validation query, because validation queries only see training windows;",
              "  - the forecast-conditioned count and bias are descriptive, not a pass/fail condition (amendment 2026-10-06);",
              "  - power: a null result means no gain of about the 'Detectable' size; smaller benefits are not ruled out. "
              "Extreme days come from few year x season clusters (about 11-14), close to the fragile region;",
              f"  - multiple comparisons: 2 families x {len({r['mode'] for rr in ret_rows.values() for r in rr if r['mode'] in RUNG_SIMPLICITY})} "
              "rungs x 3 conditions, no correction, and more rungs follow. A future pass should be read with that in mind; "
              "it cannot turn a fail into a pass;",
              "  - extreme-day bias column: observed-extreme days are by construction ones the models under-forecast, so a "
              "gain there can come from a smaller cold bias rather than a better day-to-day forecast (forecaster's dilemma);",
              "  - provenance: the retrieval runs were trained from uncommitted code (registry git_commit is the previous "
              "commit, dirty = True). The registry code_sha256 identifies the exact code: 49129b3d... = commit 312ec5f "
              "(R0, R0-rand, R1); a99ec74b... = commit 5a3cb99 (R1-rand); f1dd5eae... = commit 'Add regional-pattern "
              "retrieval rung Rg' (Rg). Rg also reads datasets_v2/upstream_daily.parquet (sha256 ab7b3685...), which the "
              "registry data_sha256 of those runs does not include;",
              "  - AnEn being far worse than the control says the raw analogue outcomes carry little skill on their own; "
              "it does not test how the network uses them."]
    (OUT_DIR / "week3_controls.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    # ASCII-only console summary (the Windows console may not encode the report's symbols)
    for r in sorted(summary, key=lambda x: (x["target"], x["labels"], x["rmse_all"])):
        print(f"{r['run_id']:14s} {r['target']:13s} {r['labels']:5s} RMSE {r['rmse_all']:.3f}  "
              f"vs damped {r['dp_all_delta']:+.3f} {r['dp_all_ci']}  floor {'PASS' if r['floor_pass'] else 'FAIL'}")
    for fam, c in choices.items():
        print(f"control [{fam}]: {c['choice']} (passes floor: {c['passes_floor']}; "
              f"pre-registered rule alone: {rule_choices[fam]['choice']})")
    for fam, rows in ret_rows.items():
        for r in rows:
            vs_rand = (f"  ext vs {r['rand_run']} {r['rand_extreme_delta']:+.3f} {r['rand_extreme_ci']}"
                       if "rand_run" in r else "")
            print(f"retrieval [{fam}] {r['rung']:8s} all vs control {r['ctrl_all_delta']:+.3f} {r['ctrl_all_ci']}  "
                  f"ext vs control {r['ctrl_ext_delta']:+.3f} {r['ctrl_ext_ci']}{vs_rand}")
        print(f"G3 [{fam}]: {ret_choice[fam]['choice']} ({ret_choice[fam]['reason']})")
    print(f"-> {(OUT_DIR / 'week3_controls.md').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
