"""
Bias split, recalibration control, ensemble and hot-day probability scores (decisions.md
2026-10-08, reporting addition 2, taken on the independent ML review). Evaluation only.

Why: on observed-extreme days most of every model's error is a cold bias (WBGT control: bias
-3.0 °C of RMSE 3.3), so "better on extremes" can just mean "forecasts warmer" (the forecaster's
dilemma, Lerch et al. 2017, Statistical Science 32(1)). These checks separate skill from bias:

1. Bias split: all-days and extreme-day RMSE, mean error (bias) and error SD per model.
2. Recalibration control: per fold, seed and lead, actual = a + b x forecast fitted on the
   fold's INNER block (last 2 training years; forecasts re-made from the saved checkpoints),
   applied to the validation forecasts. If extreme-day gaps between models shrink a lot after
   it, they were calibration, not skill.
3. The 10-seed ensemble mean as the forecast (raw and recalibrated).
4. Hot-day probability: P(hot) = 1 - Phi((threshold - mu) / sigma), mu = the (recalibrated)
   ensemble mean, sigma = SD of the inner-block residuals of that ensemble mean, per fold and
   lead. The threshold is the label rule itself (Tmax: in season and Tmax >= min(45, max(40,
   climatology + 3)); WBGT: in season and >= the fold's training-years percentile), checked to
   reproduce the hot labels exactly. Brier score, Brier skill vs the training-years in-season
   base rate, reliability bins. Brier differences are tested on sqrt(Brier) with the same
   paired cluster-jackknife (p - o as the "error").

Inner-block forecasts are cached in predictions_v2/inner/<run>/<fold>.parquet. The trained
models' validation forecasts are re-made for seed 0 and must match the saved ones (checks that
each checkpoint is rebuilt correctly). Nothing from 2019 on is read.

Run from repo root:  python -m evaluation.calibration_check [--models control C2 U1 C3 ...]
Writes evaluation_v2/calibration_check.{md,json}.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

import training.train_unified as tu
from evaluation.compare_v2 import KEY, PRED_DIR, _ci, _fmt_p, _wbgt_thresholds, compare, ensemble, load_run, rmse_by
from evaluation.predict_v1 import long_frame
from evaluation.retrieval_information_check import OUT_DIR
from models.lstm import LSTMForecaster
from pipeline.labels_v2 import (ANOMALY_HOT_C, TMAX_ABSOLUTE_C, TMAX_FLOOR_C, in_season, in_wbgt_season)
from training.folds import FOLDS, FORECAST_DAYS, build_fold, fold_bounds, fold_daily
from training.graph_data import build_upstream, load_upstream

FAMILIES = {"Tmax": "A1prime_hw5", "WBGT (physical)": "A2Lr_hw5"}
DEFAULT_MODELS = ("control", "C2", "U1", "C3", "C4", "C4a")
INNER_DIR = PRED_DIR / "inner"
BINS = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0001)


def run_id(control: str, model: str) -> str:
    return control if model == "control" else f"{control}_{model}"


# ------------------------------------------------------------------ inner-block forecasts


def _predict(cfg: dict, model: torch.nn.Module, X: np.ndarray, pos: np.ndarray | None, upd) -> np.ndarray:
    if cfg.get("backbone", "lstm") == "dstgnn":
        return tu.predict_graph(model, X, pos, torch.from_numpy(upd.x), torch.from_numpy(upd.adj))
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X)).numpy()


def inner_forecasts(rid: str, fold: str, up_table=None) -> pd.DataFrame:
    """All seeds' inner-block forecasts of one run and fold (deg C, long format), from the
    checkpoints; seed 0's validation forecasts are re-made and checked against the saved ones."""
    cache = INNER_DIR / rid / f"{fold}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    cfg = tu.load_config(tu.REPO_ROOT / "configs" / f"{rid}.json")
    if cfg.get("retrieval") or cfg.get("head"):
        raise ValueError(f"{rid}: retrieval / head runs are not handled here")
    backbone, hp = cfg.get("backbone", "lstm"), cfg.get("hparams")
    data = build_fold(fold, cfg["target"], cfg["labels"], target_form=cfg["target_form"])
    _, stop = tu.fit_and_stop_sets(data, cfg["early_stop"])
    val, upd, pos_stop, pos_val = data.val, None, None, None
    if backbone != "lstm":
        upd = build_upstream(fold, up_table if up_table is not None else load_upstream())
        if backbone == "lstm_upstream":
            stop, val = tu.with_upstream(stop, upd), tu.with_upstream(val, upd)
        else:
            pos_stop, pos_val = upd.positions(stop.query_dates), upd.positions(val.query_dates)
    saved = pd.read_parquet(PRED_DIR / rid / f"{fold}.parquet")
    frames = []
    for seed in sorted(saved["seed"].unique()):
        if backbone == "dstgnn":
            model = tu.make_dstgnn(cfg["graph"], stop.X.shape[2], upd, hp)
        elif hp is None:
            model = LSTMForecaster(n_features=stop.X.shape[2])
        else:
            model = LSTMForecaster(n_features=stop.X.shape[2], hidden_size=hp["hidden"], dropout=float(hp["dropout"]))
        ckpt = tu.MODEL_DIR / rid / fold / f"seed_{seed}" / "checkpoint.pt"
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        if seed == saved["seed"].min():
            v = data.to_raw(_predict(cfg, model, val.X, pos_val, upd), "val")
            s = saved[saved["seed"] == seed].sort_values(KEY)
            if not np.allclose(v.reshape(-1), s["pred"].to_numpy(), atol=1e-4):
                raise ValueError(f"{rid} {fold}: rebuilt checkpoint does not reproduce the saved forecasts")
        p = tu._sub_to_raw(data, _predict(cfg, model, stop.X, pos_stop, upd), stop)
        frames.append(long_frame(rid, int(seed), stop.query_dates, p, stop.y_raw, stop.stratum))
    df = pd.concat(frames, ignore_index=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


# ------------------------------------------------------------------ recalibration


def fit_recal(inner: pd.DataFrame) -> pd.DataFrame:
    """Per (seed, lead): OLS actual = a + b x pred on the inner block -> columns seed, lead, a, b."""
    rows = []
    for (seed, lead), g in inner.groupby(["seed", "lead"]):
        b, a = np.polyfit(g["pred"].to_numpy(), g["actual"].to_numpy(), 1)
        rows.append({"seed": seed, "lead": lead, "a": a, "b": b})
    return pd.DataFrame(rows)


def apply_recal(val: pd.DataFrame, coef: pd.DataFrame) -> pd.DataFrame:
    m = val.merge(coef, on=["seed", "lead"], how="left", validate="many_to_one")
    if m[["a", "b"]].isna().any().any():
        raise ValueError("missing recalibration coefficients")
    pred = m["a"] + m["b"] * m["pred"]
    return val.assign(pred=pred.to_numpy(), error=(pred - m["actual"]).to_numpy())


# ------------------------------------------------------------------ hot-day probability


def event_thresholds(fold: str, family: str, target_dates: pd.Series) -> np.ndarray:
    """The hot-label threshold (deg C) per target date; +inf outside the label season."""
    t = pd.DatetimeIndex(target_dates)
    if family == "Tmax":
        d, _ = fold_daily(fold, "t_max", "v2")
        clim = d["clim_mean_t_max"].reindex(t).to_numpy()
        thr = np.minimum(TMAX_ABSOLUTE_C, np.maximum(TMAX_FLOOR_C, clim + ANOMALY_HOT_C))
        return np.where(in_season(t), thr, np.inf)
    return np.where(in_wbgt_season(t), _wbgt_thresholds((fold,))[fold], np.inf)


def base_rate(fold: str, family: str) -> float:
    """Hot-day frequency among in-season training days (the climatological probability)."""
    target, labels, season = (("t_max", "v2", in_season) if family == "Tmax" else ("wbgt_lj_max", "wbgt", in_wbgt_season))
    d, _ = fold_daily(fold, target, labels)
    train = (d.index <= fold_bounds(fold)[0]) & season(d.index)
    return float(d.loc[train, "hot"].mean())


def probability_frames(val_ens: pd.DataFrame, inner_ens: pd.DataFrame, fold: str, family: str) -> pd.DataFrame:
    """val_ens / inner_ens: one-pseudo-seed ensemble forecasts of one fold. Returns the val rows
    with p (forecast probability), o (observed hot), p_clim; error = p - o for the paired test."""
    sigma = inner_ens.groupby("lead")["error"].std(ddof=1)
    thr = event_thresholds(fold, family, val_ens["target_date"])
    o = (val_ens["actual"].to_numpy() >= thr).astype(float)
    hot_label = val_ens["stratum"].isin(["extreme", "unusual"]).to_numpy()  # v2 / WBGT: hot <=> not 'normal'
    if not np.array_equal(o.astype(bool), hot_label):
        raise ValueError(f"{family} {fold}: the threshold rule does not reproduce the hot labels")
    s = sigma.reindex(val_ens["lead"]).to_numpy()
    p = norm.sf((thr - val_ens["pred"].to_numpy()) / s)
    p_clim = np.where(np.isfinite(thr), base_rate(fold, family), 0.0)
    return val_ens.assign(p=p, o=o, p_clim=p_clim, error=p - o)


def brier(df: pd.DataFrame) -> dict:
    bs, bs_clim = float(np.mean((df["p"] - df["o"]) ** 2)), float(np.mean((df["p_clim"] - df["o"]) ** 2))
    rel = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        sel = (df["p"] >= lo) & (df["p"] < hi)
        if sel.any():
            rel.append({"bin": f"{lo:.1f}-{min(hi, 1.0):.1f}", "n": int(sel.sum()), "mean_p": float(df.loc[sel, "p"].mean()),
                        "observed": float(df.loc[sel, "o"].mean())})
    return {"brier": bs, "brier_clim": bs_clim, "bss": 1.0 - bs / bs_clim, "n_events": int(df["o"].sum()),
            "reliability": rel}


# ------------------------------------------------------------------ per family


def _split(df: pd.DataFrame) -> dict:
    out = {}
    for s, sub in (("all", df), ("extreme", df[df["stratum"] == "extreme"])):
        per_seed = sub.groupby("seed")["error"]
        out[s] = {"rmse": rmse_by(sub), "bias": float(per_seed.mean().mean()), "sd": float(per_seed.std(ddof=0).mean())}
    return out


def evaluate(family: str, models: tuple[str, ...]) -> dict:
    control = FAMILIES[family]
    up = load_upstream()
    res = {"split": {}, "split_recal": {}, "ensemble": {}, "prob": {}, "tests": {}}
    recal, prob = {}, {}
    for m in models:
        rid = run_id(control, m)
        val = load_run(rid)
        if val is None:
            continue
        vr, pr = [], []
        for fold in FOLDS:
            inner = inner_forecasts(rid, fold, up)
            v = val[val["fold"] == fold].drop(columns="fold")
            vr.append(apply_recal(v, fit_recal(inner)).assign(fold=fold))
            for tag, vv, ii in (("raw", v, inner), ("recal", vr[-1].drop(columns="fold"), apply_recal(inner, fit_recal(inner)))):
                pr.append(probability_frames(ensemble(vv), ensemble(ii), fold, family).assign(fold=fold, kind=tag))
        recal[m] = pd.concat(vr, ignore_index=True)
        prob[m] = pd.concat(pr, ignore_index=True)
        res["split"][m], res["split_recal"][m] = _split(val), _split(recal[m])
        res["ensemble"][m] = {"raw": rmse_by(ensemble(val)), "recal": rmse_by(ensemble(recal[m])),
                              "raw_extreme": rmse_by(ensemble(val)[lambda d: d["stratum"] == "extreme"]),
                              "recal_extreme": rmse_by(ensemble(recal[m])[lambda d: d["stratum"] == "extreme"])}
        res["prob"][m] = {k: brier(prob[m][prob[m]["kind"] == k]) for k in ("raw", "recal")}
    for m in recal:
        if m == "control":
            continue
        for s in ("all", "extreme"):
            res["tests"][f"{m} vs control, recalibrated ({s})"] = compare(recal["control"], recal[m], s)
        for k in ("raw", "recal"):
            pc, pm = (prob[x][prob[x]["kind"] == k].drop(columns=["kind", "fold", "p", "o", "p_clim"]) for x in ("control", m))
            res["tests"][f"{m} vs control, sqrt-Brier hot day ({k})"] = compare(pc, pm, "all")
    return res


def write_report(results: dict) -> None:
    L = ["# Bias split, recalibration and hot-day probability (decisions.md 2026-10-08, addition 2)", "",
         "Out of fold 2007-2018. Recalibration: actual = a + b x forecast per fold, seed and lead, fitted on the "
         "fold's inner block (last 2 training years) and applied to the validation block. Probabilities: Gaussian "
         "around the 10-seed ensemble mean, SD from inner-block residuals; event = the hot label itself. Inner-block "
         "residuals come from the early-stopping block, so sigma is slightly optimistic (disclosed).", ""]
    for fam, r in results.items():
        L += [f"## {fam}", "", "| Model | All RMSE / bias / SD | Extreme RMSE / bias / SD | Recalibrated all | "
              "Recalibrated extreme / bias | Ensemble all (recal) | Ensemble extreme (recal) |", "|---|---|---|---|---|---|---|"]
        for m, v in r["split"].items():
            c, e = r["split_recal"][m], r["ensemble"][m]
            L.append(f"| {m} | {v['all']['rmse']:.3f} / {v['all']['bias']:+.2f} / {v['all']['sd']:.2f} | "
                     f"{v['extreme']['rmse']:.3f} / {v['extreme']['bias']:+.2f} / {v['extreme']['sd']:.2f} | "
                     f"{c['all']['rmse']:.3f} | {c['extreme']['rmse']:.3f} / {c['extreme']['bias']:+.2f} | "
                     f"{e['raw']:.3f} ({e['recal']:.3f}) | {e['raw_extreme']:.3f} ({e['recal_extreme']:.3f}) |")
        L += ["", "| Model | Brier raw | BSS raw | Brier recal | BSS recal | Hot days |", "|---|---|---|---|---|---|"]
        for m, v in r["prob"].items():
            L.append(f"| {m} | {v['raw']['brier']:.4f} | {v['raw']['bss']:+.3f} | {v['recal']['brier']:.4f} | "
                     f"{v['recal']['bss']:+.3f} | {v['raw']['n_events']} |")
        L += ["", "| Comparison | Δ (95% CI) | p |", "|---|---|---|"]
        L += [f"| {k} | {t['delta']:+.4f} {_ci(t)} | {_fmt_p(t['p_value'])} |" for k, t in r["tests"].items()]
        rel = r["prob"].get("control", {}).get("recal", {}).get("reliability", [])
        if rel:
            L += ["", "Reliability, control (recalibrated): " + "; ".join(
                f"{b['bin']}: n {b['n']}, forecast {b['mean_p']:.2f}, observed {b['observed']:.2f}" for b in rel)]
        L.append("")
    (OUT_DIR / "calibration_check.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    args = ap.parse_args(argv)
    torch.set_num_threads(2)  # light inference next to the training queue
    results = {fam: evaluate(fam, tuple(args.models)) for fam in FAMILIES}
    write_report(results)
    (OUT_DIR / "calibration_check.json").write_text(json.dumps(results, indent=2, default=float) + "\n", encoding="utf-8")
    print({f: {m: round(v["recal"]["bss"], 3) for m, v in r["prob"].items()} for f, r in results.items()})


if __name__ == "__main__":
    main()
