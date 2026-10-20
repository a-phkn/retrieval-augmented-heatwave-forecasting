"""
Post-test exploration 1 (specified 2026-10-20 in context/decisions.md before any model was run): pretrained
time-series foundation models, zero-shot on Delhi's own history, vs our models on the 2019-2026 test windows.

For every C3-hop test window (both families) the model gets the target's daily series for the CONTEXT_DAYS days
before the query date and forecasts 5 days; the point forecast is its median. The daily series is rebuilt from
pre-2019 data (training.folds, development folds) plus the ERA5 truth already stored in the committed test outputs
(predictions_test/), so the test lock is not used. Model weights are cached inside the project (.hf_cache/).

Run from repo root:  python -m post_test.foundation_baselines [--models chronos2 timesfm3]
Writes predictions_post_test/<model>_<family>.parquet and evaluation_post_test/foundation_baselines.{md,json}.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".hf_cache"))  # weights stay inside the project, never ~/.cache
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from evaluation.compare_v2 import _ci, _fmt_p, compare, extreme_bias, rmse_by  # noqa: E402
from training.folds import FORECAST_DAYS, fold_daily  # noqa: E402

PRED_TEST = REPO_ROOT / "predictions_test"
OUT_PRED = REPO_ROOT / "predictions_post_test"
OUT_DIR = REPO_ROOT / "evaluation_post_test"
FAMILIES = {"WBGT (physical)": ("A2Lr_hw5", "wbgt_lj_max", "wbgt"), "Tmax": ("A1prime_hw5", "t_max", "v2")}
CONTEXT_DAYS = 1024
MODELS = {"chronos2": ("Chronos-2 (Amazon)", "amazon/chronos-2"), "timesfm3": ("TimesFM 3.0 (Google)", "google/timesfm-3.0-pytorch")}
BATCH = 128


def daily_series(control: str, target: str, labels: str) -> pd.Series:
    """The target's daily ERA5 values 1980-01-01 .. the end of the test period, gap-free."""
    pre = fold_daily("f4", target, labels)[0][target]  # development fold: pre-2019 rows only
    test = pd.read_parquet(PRED_TEST / f"{control}_C3hop" / "test.parquet", columns=["target_date", "actual"])
    per_day = test.groupby("target_date")["actual"]
    if (per_day.max() - per_day.min()).max() > 1e-4:
        raise ValueError("inconsistent truth for the same date in the test outputs")
    s = pd.concat([pre, per_day.first()]).sort_index()
    s = s[~s.index.duplicated()]
    full = pd.date_range(s.index.min(), s.index.max(), freq="D")
    if not s.index.equals(full) or s.isna().any():
        raise ValueError("the rebuilt daily series has gaps")
    return s.astype(np.float64)


def contexts(series: pd.Series, query_dates: pd.DatetimeIndex, n: int = CONTEXT_DAYS) -> list[np.ndarray]:
    """For each query date q: the n daily values ending on q - 1 (nothing from q onwards)."""
    pos = series.index.get_indexer(query_dates)
    if (pos < n).any() or (pos < 0).any():
        raise ValueError("not enough history before some query dates")
    v = series.to_numpy()
    return [v[p - n:p] for p in pos]


def median_from_quantiles(q, levels: list[float]) -> np.ndarray:
    """(horizon,) median from one series' quantile output of any shape (..., horizon, n_levels)."""
    a = np.asarray(q, dtype=np.float64).reshape(-1, len(levels))
    return a[:, levels.index(0.5)]


def forecast(model: str, ctx: list[np.ndarray]) -> np.ndarray:
    """(N, 5) zero-shot median forecasts."""
    out = []
    if model == "chronos2":
        import torch
        from chronos import BaseChronosPipeline

        pipe = BaseChronosPipeline.from_pretrained(MODELS[model][1], device_map="cpu", torch_dtype=torch.float32)
        for s in range(0, len(ctx), BATCH):
            q, _ = pipe.predict_quantiles([torch.tensor(c, dtype=torch.float32) for c in ctx[s:s + BATCH]],
                                          prediction_length=FORECAST_DAYS, quantile_levels=[0.5])
            out += [median_from_quantiles(qi, [0.5])[:FORECAST_DAYS] for qi in q]
    elif model == "timesfm3":
        import timesfm

        fc = timesfm.TimesFM3Forecaster.from_pretrained(MODELS[model][1], device="cpu")
        for s in range(0, len(ctx), BATCH):
            out += [np.asarray(o.forecast, dtype=np.float64).reshape(-1)[:FORECAST_DAYS]
                    for o in fc.predict_batch(ctx[s:s + BATCH], horizon=FORECAST_DAYS)]
    else:
        raise ValueError(model)
    return np.stack(out)


def as_frame(model: str, ref: pd.DataFrame, query_dates: pd.DatetimeIndex, pred: np.ndarray) -> pd.DataFrame:
    """Long frame on exactly C3-hop's (query_date, lead) rows, strata and clusters; one pseudo-seed."""
    p = pd.DataFrame({"query_date": np.repeat(query_dates, FORECAST_DAYS),
                      "lead": np.tile(np.arange(1, FORECAST_DAYS + 1), len(query_dates)), "pred": pred.reshape(-1)})
    base = ref[ref["seed"] == ref["seed"].min()][["query_date", "lead", "target_date", "actual", "stratum", "cluster_id"]]
    df = base.merge(p, on=["query_date", "lead"], how="left", validate="one_to_one")
    if df["pred"].isna().any():
        raise ValueError("missing forecasts")
    return df.assign(model=model, seed=0, error=df["pred"] - df["actual"])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    models = ap.parse_args(argv).models
    reg = pd.read_csv(REPO_ROOT / "registry" / "test_runs.csv")
    res = {}
    OUT_PRED.mkdir(parents=True, exist_ok=True)
    for fam, (ctrl, target, labels) in FAMILIES.items():
        c3 = pd.read_parquet(PRED_TEST / f"{ctrl}_C3hop" / "test.parquet")
        lstm = pd.read_parquet(PRED_TEST / ctrl / "test.parquet")
        series = daily_series(ctrl, target, labels)
        q = pd.DatetimeIndex(sorted(c3["query_date"].unique()))
        ctx = contexts(series, q)
        row = reg[reg["run_id"] == f"{ctrl}_C3hop"].iloc[0]
        r = {"C3-hop": {"rmse_all": rmse_by(c3), "rmse_extreme": rmse_by(c3[c3["stratum"] == "extreme"]),
                        "bias_extreme": extreme_bias(c3)},
             "Delhi-only LSTM": {"rmse_all": rmse_by(lstm), "rmse_extreme": rmse_by(lstm[lstm["stratum"] == "extreme"]),
                                 "bias_extreme": extreme_bias(lstm)},
             "climatology": {"rmse_all": float(row["clim_rmse_all"]), "rmse_extreme": float(row["clim_rmse_extreme"])},
             "persistence": {"rmse_all": float(row["persist_rmse_all"]), "rmse_extreme": float(row["persist_rmse_extreme"])}}
        for m in models:
            path = OUT_PRED / f"{m}_{ctrl}.parquet"
            if path.exists():
                fm = pd.read_parquet(path)
            else:
                fm = as_frame(m, c3, q, forecast(m, ctx))
                fm.to_parquet(path, index=False)
            r[MODELS[m][0]] = {"rmse_all": rmse_by(fm), "rmse_extreme": rmse_by(fm[fm["stratum"] == "extreme"]),
                               "bias_extreme": extreme_bias(fm),
                               "C3-hop vs it (all)": compare(fm, c3, "all"), "C3-hop vs it (extreme)": compare(fm, c3, "extreme"),
                               "LSTM vs it (all)": compare(fm, lstm, "all"), "LSTM vs it (extreme)": compare(fm, lstm, "extreme")}
            print(f"{fam} {m}: done", flush=True)
        res[fam] = r
    write_report(res)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "foundation_baselines.json").write_text(json.dumps(res, indent=2, default=float) + "\n", encoding="utf-8")


def write_report(res: dict) -> None:
    L = ["# Pretrained time-series foundation models vs our models, test years 2019-2026 (post-test exploration)", "",
         "Specified 2026-10-20 before running (context/decisions.md). Zero-shot: each foundation model sees only the "
         f"target's own last {CONTEXT_DAYS} days and forecasts 5 days (median). RMSE in °C (lower is better): ours = mean "
         "of 10 seeds' RMSE; foundation models = one deterministic run. Δ = first-named minus the foundation model "
         "(negative = ours better); paired cluster-jackknife, year x season. Caveat: their pretraining data may "
         "include weather series from these years.", ""]
    for fam, r in res.items():
        L += [f"## {fam}", "", "| Model | RMSE all days | RMSE heatwave days | Bias heatwave days |", "|---|---|---|---|"]
        for name, v in r.items():
            b = v.get("bias_extreme")
            L.append(f"| {name} | {v['rmse_all']:.3f} | {v['rmse_extreme']:.3f} | {'' if b is None else f'{b:+.2f}'} |")
        L += ["", "| Comparison | Δ RMSE (95% CI) | p |", "|---|---|---|"]
        for name, v in r.items():
            for k in ("C3-hop vs it (all)", "C3-hop vs it (extreme)", "LSTM vs it (all)", "LSTM vs it (extreme)"):
                if k in v:
                    t = v[k]
                    L.append(f"| {k.replace('it', name)} | {t['delta']:+.3f} {_ci(t)} | {_fmt_p(t['p_value'])} |")
        L.append("")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "foundation_baselines.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
