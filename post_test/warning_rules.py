"""
Post-test exploration 3 (pre-registered in context/decisions.md 2026-10-27 before any number was computed;
descriptive only, it changes no model and no confirmatory result): better yes/no heat warnings from the same forecasts.

Rules, on the 10-seed ensemble forecast, one row per (query date, lead), in-season rows only:
  R-point  warn if the forecast >= the heatwave threshold (the current rule);
  R-prob   warn if the hot-day probability >= p* (probability as in the test report; C3-hop and the LSTM only);
  R-shift  warn if the forecast >= the threshold - delta (any point-forecast model);
  persistence  warn on every lead if the day before the query date was a heatwave day.
p* and delta are chosen on the four development folds pooled (2007-2018) by the highest equitable threat score
(ETS), then applied unchanged to 2019-2026. CIs: cluster bootstrap over year x season clusters.

Reading the 2019-2026 thresholds needs the test lock (valid while the frozen code is unchanged).

Run from repo root:  python -m post_test.warning_rules  ->  evaluation_post_test/warning_rules.{md,json}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from post_test.test_period_extras import event_scores

REPO_ROOT = Path(__file__).resolve().parents[1]
PRED_TEST = REPO_ROOT / "predictions_test"
OUT_DIR = REPO_ROOT / "evaluation_post_test"
FAMILIES = {"WBGT (physical)": ("A2Lr_hw5", "wbgt_lj_max", "wbgt"), "Tmax": ("A1prime_hw5", "t_max", "v2")}
PROB_MODELS = {"C3-hop": "_C3hop", "Delhi-only LSTM": ""}
SHIFT_MODELS = {"C3-hop": "_C3hop", "Delhi-only LSTM": "", "C2": "_C2", "U1_tuned": "_U1_tuned",
                "C3pool_tuned": "_C3pool_tuned", "ridge": "_ridge", "ridge_hw5": "_ridge_hw5"}
P_GRID = np.round(np.arange(0.01, 1.00, 0.01), 2)
DELTA_GRID = np.round(np.arange(0.0, 6.0001, 0.05), 2)
TRADEOFF_P = (0.1, 0.2, 0.3, 0.5)
N_BOOT, BOOT_SEED = 2000, 20261027
KEY = ["query_date", "lead"]


# ------------------------------------------------------------------ scoring and choice

def scores(o: np.ndarray, w: np.ndarray) -> dict:
    """Event scores plus frequency bias (warnings / events)."""
    s = event_scores(o, w)
    events = s["hits"] + s["misses"]
    s["frequency_bias"] = (s["hits"] + s["false_alarms"]) / events if events else float("nan")
    return s


def _ets_counts(h, m, f, n):
    """ETS from counts (arrays allowed); 0/0 -> nan."""
    chance = (h + m) * (h + f) / n
    denom = h + m + f - chance
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom != 0, (h - chance) / denom, np.nan)


def choose(o: np.ndarray, score: np.ndarray, grid: np.ndarray, kind: str) -> tuple[float, float]:
    """The grid value with the highest ETS. kind 'prob': warn if score >= t, ties -> the largest t;
    kind 'shift': score is forecast - threshold, warn if score >= -t, ties -> the smallest t."""
    o = np.asarray(o, bool)
    best_t, best = None, -np.inf
    order = grid[::-1] if kind == "prob" else grid  # visit the value closest to R-point first
    for t in order:
        w = score >= (t if kind == "prob" else -t)
        e = float(_ets_counts((o & w).sum(), (o & ~w).sum(), (~o & w).sum(), o.size))
        if np.isfinite(e) and e > best + 1e-12:
            best_t, best = float(t), e
    if best_t is None:
        raise ValueError("no grid value gives a finite ETS")
    return best_t, best


def cluster_counts(clusters: np.ndarray, o: np.ndarray, w: np.ndarray) -> pd.DataFrame:
    """Per cluster: hits, misses, false alarms, rows."""
    o, w = np.asarray(o, bool), np.asarray(w, bool)
    df = pd.DataFrame({"c": clusters, "h": o & w, "m": o & ~w, "f": ~o & w, "n": 1})
    return df.groupby("c")[["h", "m", "f", "n"]].sum()


def bootstrap(clusters: np.ndarray, o: np.ndarray, rules: dict[str, np.ndarray], pairs: list[tuple[str, str]],
              n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """95% cluster-bootstrap CIs for each rule's ETS and hit rate, and for paired ETS / hit-rate differences
    (first minus second) on the same rows."""
    counts = {k: cluster_counts(clusters, o, w) for k, w in rules.items()}
    ids = next(iter(counts.values())).index
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(ids), size=(n_boot, len(ids)))
    stats = {}
    for k, c in counts.items():
        a = c.loc[ids].to_numpy(float)  # (G, 4)
        tot = a[draws].sum(axis=1)  # (n_boot, 4)
        h, m, f, n = tot.T
        with np.errstate(invalid="ignore", divide="ignore"):
            stats[k] = {"ets": _ets_counts(h, m, f, n), "hit_rate": np.where(h + m > 0, h / (h + m), np.nan)}
    ci = lambda x: [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]  # noqa: E731
    out = {"rules": {k: {"ets_ci": ci(v["ets"]), "hit_rate_ci": ci(v["hit_rate"])} for k, v in stats.items()},
           "pairs": {}}
    for a, b in pairs:
        d_ets, d_hit = stats[a]["ets"] - stats[b]["ets"], stats[a]["hit_rate"] - stats[b]["hit_rate"]
        out["pairs"][f"{a} minus {b}"] = {"ets_ci": ci(d_ets), "hit_rate_ci": ci(d_hit),
                                          "share_ets_above_0": float(np.nanmean(d_ets > 0))}
    return out


# ------------------------------------------------------------------ data

def _with_events(ens: pd.DataFrame, fold: str, family: str) -> pd.DataFrame:
    """In-season ensemble rows with the threshold and the observed event (checked against the hot labels)."""
    from evaluation.calibration_check import event_thresholds

    thr = event_thresholds(fold, family, ens["target_date"])
    e = ens[np.isfinite(thr)].assign(thr=thr[np.isfinite(thr)]).reset_index(drop=True)
    e["o"] = e["actual"].to_numpy() >= e["thr"].to_numpy()
    if not np.array_equal(e["o"].to_numpy(), e["stratum"].isin(["extreme", "unusual"]).to_numpy()):
        raise ValueError(f"{family} {fold}: the threshold rule does not reproduce the hot labels")
    return e


def point_rows(rid: str, family: str, period: str) -> pd.DataFrame:
    """Ensemble point forecasts with events; period 'dev' (folds f1-f4 pooled) or 'test'."""
    from evaluation.compare_v2 import ensemble, load_run
    from training.folds import FOLDS

    if period == "test":
        return _with_events(ensemble(pd.read_parquet(PRED_TEST / rid / "test.parquet")), "test", family).assign(fold="test")
    val = load_run(rid)
    if val is None:
        raise FileNotFoundError(f"{rid}: development forecasts missing")
    return pd.concat([_with_events(ensemble(val[val["fold"] == f].drop(columns="fold")), f, family).assign(fold=f)
                      for f in FOLDS], ignore_index=True)


def prob_rows(rid: str, family: str, period: str) -> pd.DataFrame:
    """Hot-day probabilities exactly as in the test report (recalibration + Gaussian spread from the inner block in
    development, from the 2017-2018 stop block on the test)."""
    from evaluation.calibration_check import apply_recal, ensemble, fit_recal, inner_forecasts, probability_frames
    from evaluation.compare_v2 import load_run
    from training.folds import FOLDS

    if period == "test":
        val, stop = (pd.read_parquet(PRED_TEST / rid / f"test{s}.parquet") for s in ("", "_stop"))
        coef = fit_recal(stop)
        return probability_frames(ensemble(apply_recal(val, coef)), ensemble(apply_recal(stop, coef)), "test",
                                  family).assign(fold="test")
    val, parts = load_run(rid), []
    for f in FOLDS:
        inner, v = inner_forecasts(rid, f), val[val["fold"] == f].drop(columns="fold")
        coef = fit_recal(inner)
        parts.append(probability_frames(ensemble(apply_recal(v, coef)), ensemble(apply_recal(inner, coef)), f,
                                        family).assign(fold=f))
    return pd.concat(parts, ignore_index=True)


def persistence_warning(rows: pd.DataFrame, target: str, labels: str) -> np.ndarray:
    """Warn on every lead if the day before the query date was a heatwave day (its own fold's labels)."""
    from training.folds import fold_daily

    out = np.zeros(len(rows), bool)
    for fold, idx in rows.groupby("fold").groups.items():
        hot = fold_daily(fold, target, labels)[0]["hot"]
        prev = pd.DatetimeIndex(rows.loc[idx, "query_date"]) - pd.Timedelta(days=1)
        out[np.asarray(idx)] = hot.reindex(prev).fillna(False).to_numpy(bool)
    return out


# ------------------------------------------------------------------ the analysis

def family_analysis(family: str) -> dict:
    ctrl, target, labels = FAMILIES[family]
    res = {"choices": {}, "test": {}, "bootstrap": None, "tradeoff": {}}
    test_rows, rules = None, {}
    for name, suffix in SHIFT_MODELS.items():
        dev, test = point_rows(ctrl + suffix, family, "dev"), point_rows(ctrl + suffix, family, "test")
        delta, dev_ets = choose(dev["o"].to_numpy(), (dev["pred"] - dev["thr"]).to_numpy(), DELTA_GRID, "shift")
        res["choices"][f"{name} R-shift"] = {"delta_C": delta, "dev_ets": dev_ets,
                                             "dev_ets_r_point": scores(dev["o"], dev["pred"] >= dev["thr"])["ets"]}
        if test_rows is None:
            test_rows = test[KEY + ["cluster_id", "o", "fold", "target_date"]]
        elif not test[KEY].equals(test_rows[KEY]) or not np.array_equal(test["o"], test_rows["o"]):
            raise ValueError(f"{name}: test rows differ from the other models'")
        rules[f"{name} R-point"] = (test["pred"] >= test["thr"]).to_numpy()
        rules[f"{name} R-shift"] = (test["pred"] >= test["thr"] - delta).to_numpy()
    for name, suffix in PROB_MODELS.items():
        dev, test = prob_rows(ctrl + suffix, family, "dev"), prob_rows(ctrl + suffix, family, "test")
        p_star, dev_ets = choose(dev["o"].to_numpy(), dev["p"].to_numpy(), P_GRID, "prob")
        res["choices"][f"{name} R-prob"] = {"p_star": p_star, "dev_ets": dev_ets}
        t = test_rows[KEY].merge(test[KEY + ["p", "o"]], on=KEY, how="left", validate="one_to_one")
        if t["p"].isna().any() or not np.array_equal(t["o"].to_numpy(bool), test_rows["o"].to_numpy(bool)):
            raise ValueError(f"{name}: probability rows do not match the point-forecast rows")
        rules[f"{name} R-prob"] = (t["p"] >= p_star).to_numpy()
        if name == "C3-hop":
            res["tradeoff"] = {str(p): scores(test_rows["o"], t["p"] >= p) for p in TRADEOFF_P}
    rules["persistence"] = persistence_warning(test_rows, target, labels)
    o = test_rows["o"].to_numpy(bool)
    res["test"] = {k: scores(o, w) for k, w in rules.items()}
    res["n_rows"], res["n_events"] = int(len(o)), int(o.sum())
    pairs = [("C3-hop R-prob", "C3-hop R-point"), ("C3-hop R-shift", "C3-hop R-point"),
             ("C3-hop R-prob", "Delhi-only LSTM R-prob"), ("C3-hop R-prob", "persistence")]
    pairs += [("C3-hop R-shift", f"{m} R-shift") for m in SHIFT_MODELS if m != "C3-hop"]
    res["bootstrap"] = bootstrap(test_rows["cluster_id"].to_numpy(), o, rules, pairs)
    return res


def _pct(x: float) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:.0f}%"


def write_report(res: dict) -> None:
    L = ["# Better yes/no heat warnings from the same forecasts (post-test, descriptive)", "",
         "Pre-registered in `context/decisions.md` (2026-10-27) before any number was computed. Warning levels (p*, "
         "delta) chosen on 2007-2018 by the highest equitable threat score (ETS), then applied unchanged to "
         "2019-2026. Rows = 10-seed ensemble forecasts, one per (query date, lead), in season. Hit rate = share of "
         "heatwave days warned; false-alarm ratio = share of warnings that were wrong; ETS: 0 = no better than "
         "chance, 1 = perfect; frequency bias = warnings / heatwave days. 95% CIs: cluster bootstrap over "
         "year x season blocks.", "",
         "Event definition (as pre-registered): every day that meets the heatwave label rule, including isolated "
         "hot days. The RMSE tables and `test_period_extras.md` count only hot days inside a heatwave run, so the "
         "current rule's hit rate is a little different there (WBGT 9% vs 6% here, Tmax 27% vs 24%).", ""]
    for fam, r in res.items():
        b = r["bootstrap"]
        L += [f"## {fam} ({r['n_events']} heatwave rows of {r['n_rows']})", "",
              "| Model and rule | Chosen on 2007-2018 | Hit rate (95% CI) | False-alarm ratio | ETS (95% CI) | Frequency bias |",
              "|---|---|---|---|---|---|"]
        order = sorted(r["test"], key=lambda k: -r["test"][k]["ets"] if np.isfinite(r["test"][k]["ets"]) else 1)
        for k in order:
            s, c = r["test"][k], b["rules"][k]
            ch = r["choices"].get(k, {})
            chosen = (f"p* = {ch['p_star']:.2f}" if "p_star" in ch else f"delta = {ch['delta_C']:.2f} C" if "delta_C" in ch
                      else "")
            L.append(f"| {k} | {chosen} | {_pct(s['hit_rate'])} ({_pct(c['hit_rate_ci'][0])} to {_pct(c['hit_rate_ci'][1])}) "
                     f"| {_pct(s['false_alarm_ratio'])} | {s['ets']:.2f} ({c['ets_ci'][0]:.2f} to {c['ets_ci'][1]:.2f}) "
                     f"| {s['frequency_bias']:.2f} |")
        L += ["", "**Paired differences (first minus second, same days):**", "",
              "| Comparison | ETS difference (95% CI) | Hit-rate difference (95% CI) |", "|---|---|---|"]
        for k, v in b["pairs"].items():
            L.append(f"| {k} | {v['ets_ci'][0]:+.2f} to {v['ets_ci'][1]:+.2f} | "
                     f"{100 * v['hit_rate_ci'][0]:+.0f} to {100 * v['hit_rate_ci'][1]:+.0f} points |")
        L += ["", "**C3-hop, probability rule at other levels (descriptive trade-off; not chosen on the test):**", "",
              "| Warn if probability >= | Hit rate | False-alarm ratio | ETS |", "|---|---|---|---|"]
        for p, s in r["tradeoff"].items():
            L.append(f"| {p} | {_pct(s['hit_rate'])} | {_pct(s['false_alarm_ratio'])} | {s['ets']:.2f} |")
        L.append("")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "warning_rules.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    from training import test_lock

    test_lock.require()
    res = {fam: family_analysis(fam) for fam in FAMILIES}
    write_report(res)
    (OUT_DIR / "warning_rules.json").write_text(json.dumps(res, indent=2, default=float) + "\n", encoding="utf-8")
    print((OUT_DIR / "warning_rules.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
