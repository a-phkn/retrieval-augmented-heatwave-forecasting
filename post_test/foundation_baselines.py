"""
Post-test exploration 1 (specified 2026-10-20 in context/decisions.md before any model was run): pretrained
time-series foundation models, zero-shot on Delhi's own history, vs our models on the 2019-2026 test windows.

For every C3-hop test window (both families) the model gets the target's daily series for the CONTEXT_DAYS days
before the query date and forecasts 5 days; the point forecast is its median. The daily series is rebuilt from
pre-2019 data (training.folds, development folds) plus the ERA5 truth already stored in the committed test outputs
(predictions_test/), so the test lock is not used. Model weights (~1.8 GB) are cached inside the project
(.hf_cache/) and pinned to the exact Hugging Face commits used on 2026-10-21 (REVISION).

Install first (on top of the core environment):  .venv\\Scripts\\python.exe -m pip install -r requirements-post-test.lock.txt
Run from repo root:  python -m post_test.foundation_baselines [--models chronos2 timesfm3]
Writes predictions_post_test/<model>_<family>.parquet (+ .meta.json: the settings it was made with) and
evaluation_post_test/foundation_baselines.{md,json}.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.compare_v2 import _ci, _fmt_p, compare, extreme_bias, rmse_by
from training.folds import FORECAST_DAYS, fold_daily

REPO_ROOT = Path(__file__).resolve().parents[1]
HF_HOME = REPO_ROOT / ".hf_cache"  # weights stay inside the project, never ~/.cache, whatever the global setting
PRED_TEST = REPO_ROOT / "predictions_test"
OUT_PRED = REPO_ROOT / "predictions_post_test"
OUT_DIR = REPO_ROOT / "evaluation_post_test"
FAMILIES = {"WBGT (physical)": ("A2Lr_hw5", "wbgt_lj_max", "wbgt"), "Tmax": ("A1prime_hw5", "t_max", "v2")}
CONTEXT_DAYS = 1024
MODELS = {"chronos2": ("Chronos-2 (Amazon)", "amazon/chronos-2"), "timesfm3": ("TimesFM 3.0 (Google)", "google/timesfm-3.0-pytorch")}
# Hugging Face commit of each model repo used for the reported numbers (both unchanged since before 2026-10-21)
REVISION = {"chronos2": "29ec3766d36d6f73f0696f85560a422f50e8498c", "timesfm3": "43046b85ec22d584a13f8098c2ed39c889e129c2"}
BATCH = 128


def use_project_cache() -> None:
    """Point the Hugging Face cache at the project folder (overriding any global HF_HOME) and turn off telemetry.
    Called at the start of main(), before any Hugging Face library is imported, so importing this module changes
    nothing."""
    os.environ["HF_HOME"] = str(HF_HOME)
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def five_days(a, what: str) -> np.ndarray:
    """One forecast as a (FORECAST_DAYS,) float array; anything else is an error, never silently trimmed."""
    v = np.asarray(a, dtype=np.float64).reshape(-1)
    if v.shape != (FORECAST_DAYS,):
        raise ValueError(f"{what}: expected {FORECAST_DAYS} forecast days, got shape {np.shape(a)}")
    return v


def run_settings(model: str) -> dict:
    """What a cached forecast file depends on; a cached file is reused only if its settings match exactly."""
    return {"model": MODELS[model][1], "revision": REVISION[model], "context_days": CONTEXT_DAYS,
            "forecast_days": FORECAST_DAYS, "point_forecast": "median"}


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

        pipe = BaseChronosPipeline.from_pretrained(MODELS[model][1], revision=REVISION[model], device_map="cpu",
                                                   torch_dtype=torch.float32)
        for s in range(0, len(ctx), BATCH):
            q, _ = pipe.predict_quantiles([torch.tensor(c, dtype=torch.float32) for c in ctx[s:s + BATCH]],
                                          prediction_length=FORECAST_DAYS, quantile_levels=[0.5])
            out += [five_days(median_from_quantiles(qi, [0.5]), model) for qi in q]
    elif model == "timesfm3":
        import timesfm

        fc = timesfm.TimesFM3Forecaster.from_pretrained(MODELS[model][1], device="cpu", revision=REVISION[model])
        for s in range(0, len(ctx), BATCH):
            out += [five_days(o.forecast, model) for o in fc.predict_batch(ctx[s:s + BATCH], horizon=FORECAST_DAYS)]
    else:
        raise ValueError(model)
    if len(out) != len(ctx):
        raise ValueError(f"{model}: {len(out)} forecasts for {len(ctx)} contexts")
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


def cached_forecasts(model: str, path: Path) -> pd.DataFrame | None:
    """The cached forecast frame if it was made with exactly run_settings(model); None if there is no cache.
    A cache made with other settings is an error (delete it to recompute), never silently reused."""
    if not path.exists():
        return None
    meta = path.with_suffix(".meta.json")
    found = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else None
    if found != run_settings(model):
        raise ValueError(f"{path.name} was made with settings {found}, not {run_settings(model)}; delete it to recompute")
    return pd.read_parquet(path)


def main(argv: list[str] | None = None) -> None:
    """Forecast every C3-hop test window with each foundation model (or reuse matching cached forecasts) and write
    the comparison report."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    models = ap.parse_args(argv).models
    use_project_cache()
    reg = pd.read_csv(REPO_ROOT / "registry" / "test_runs.csv")
    res = {}
    OUT_PRED.mkdir(parents=True, exist_ok=True)
    for fam, (ctrl, target, labels) in FAMILIES.items():
        c3 = pd.read_parquet(PRED_TEST / f"{ctrl}_C3hop" / "test.parquet")
        lstm = pd.read_parquet(PRED_TEST / ctrl / "test.parquet")
        series = daily_series(ctrl, target, labels)
        q = pd.DatetimeIndex(sorted(c3["query_date"].unique()))
        ctx = contexts(series, q)
        rows = reg[reg["run_id"] == f"{ctrl}_C3hop"]
        if len(rows) != 1:
            raise ValueError(f"expected exactly one registry row for {ctrl}_C3hop, found {len(rows)}")
        row = rows.iloc[0]
        r = {"C3-hop": {"rmse_all": rmse_by(c3), "rmse_extreme": rmse_by(c3[c3["stratum"] == "extreme"]),
                        "bias_extreme": extreme_bias(c3)},
             "Delhi-only LSTM": {"rmse_all": rmse_by(lstm), "rmse_extreme": rmse_by(lstm[lstm["stratum"] == "extreme"]),
                                 "bias_extreme": extreme_bias(lstm)},
             "climatology": {"rmse_all": float(row["clim_rmse_all"]), "rmse_extreme": float(row["clim_rmse_extreme"])},
             "persistence": {"rmse_all": float(row["persist_rmse_all"]), "rmse_extreme": float(row["persist_rmse_extreme"])}}
        for m in models:
            path = OUT_PRED / f"{m}_{ctrl}.parquet"
            fm = cached_forecasts(m, path)
            if fm is None:
                fm = as_frame(m, c3, q, forecast(m, ctx))
                fm.to_parquet(path, index=False)
                path.with_suffix(".meta.json").write_text(json.dumps(run_settings(m), indent=2) + "\n", encoding="utf-8")
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
    """Write evaluation_post_test/foundation_baselines.md from the results dict built in main()."""
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
