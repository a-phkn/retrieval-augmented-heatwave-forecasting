"""
Evaluation for the baseline LSTM, trimmed to exactly the 4 metrics that
matter for this project (see context/ml_notes.md's "which metric matters"
discussion for the reasoning). Deliberately DROPPED from earlier versions:
R^2, MAPE, per-horizon breakdown, plain accuracy, F1 (replaced with F2,
since recall matters more than precision for this use case), and the raw
confusion-matrix panel. None of those were wrong, they just weren't the
metrics this project's conclusions should be staked on.

The 4 metrics kept:
  1. Global MAE/RMSE           -- credibility floor, is the baseline sane at all
  2. Stratified RMSE           -- normal/unusual/extreme; THE metric RAG has
                                   to beat later (Execution_Pipeline.md Step 6)
  3. Detection precision/recall/F2 -- supporting diagnostic, not the target
  4. Bias diagnostic           -- explains WHY detection looks the way it does
                                   (tail-shrinkage, not a bug)

Uses the val split only -- test is reserved for Step 6 per project convention.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from models.lstm import LSTMForecaster
from training.baselines import mae as mae_np
from training.baselines import rmse as rmse_np
from training.baselines import persistence_predict, climatology_predict
from training.data import (
    FEATURE_COLUMNS,
    build_split_target_climatology,
    build_split_target_stratum,
    denormalize_y,
    load_all_splits,
    _load_daily,
    _load_windows,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models" / "baseline_lstm"
EVAL_DIR = REPO_ROOT / "evaluation" / "baseline_lstm"
SEEDS = [0, 1, 2, 3, 4]
HOT_DAY_SIGMA = 1.5  # must match prepare_datasets.py's own definition
STRATA = ["normal", "unusual", "extreme"]

def _sanitize_nan(obj):
    """Recursively replace float('nan') with None so json.dump produces
    valid JSON (bare NaN tokens are a Python-ism, not valid JSON, and will
    break strict parsers even though Python's own json module reads them
    back fine)."""
    if isinstance(obj, float) and obj != obj:  # NaN != NaN is always True
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_nan(v) for v in obj]
    return obj

def load_seed_model(seed: int) -> LSTMForecaster:
    model = LSTMForecaster()
    state = torch.load(MODELS_DIR / f"seed_{seed}" / "checkpoint.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def stratified_rmse(pred: np.ndarray, target: np.ndarray, stratum: np.ndarray) -> dict:
    """pred/target/stratum: (N,5) each."""
    pred_f, target_f, stratum_f = pred.ravel(), target.ravel(), stratum.ravel()
    out = {}
    for s in STRATA:
        mask = stratum_f == s
        out[s] = {"rmse": rmse_np(pred_f[mask], target_f[mask]) if mask.sum() else float("nan"), "n": int(mask.sum())}
    return out


def detection_metrics(pred_hot: np.ndarray, actual_hot: np.ndarray) -> dict:
    tp = int(np.sum(pred_hot & actual_hot))
    fp = int(np.sum(pred_hot & ~actual_hot))
    fn = int(np.sum(~pred_hot & actual_hot))
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    beta = 2  # F2: weights recall 2x precision -- matches this project's stated priority
    f2 = ((1 + beta**2) * precision * recall / (beta**2 * precision + recall)
          if (precision + recall) > 0 and not np.isnan(precision) and not np.isnan(recall)
          else float("nan"))
    return {"precision": precision, "recall": recall, "f2": f2, "n_actual_hot": int(actual_hot.sum()), "n_total": int(actual_hot.size)}


def bias_diagnostic(pred: np.ndarray, target: np.ndarray, actual_hot: np.ndarray) -> dict:
    return {
        "mean_bias_on_hot_days": float((pred[actual_hot] - target[actual_hot]).mean()),
        "mean_bias_on_normal_days": float((pred[~actual_hot] - target[~actual_hot]).mean()),
        "n_hot": int(actual_hot.sum()),
        "n_normal": int((~actual_hot).sum()),
    }


def main():
    print("Loading data (refit_stats=False -- reusing the exact stats saved during training)...")
    data = load_all_splits(refit_stats=False)
    stats = data["stats"]
    split = "val"

    daily = _load_daily()
    windows = _load_windows()
    clim_mean, clim_std, actual_hot = build_split_target_climatology(split, daily, windows)
    threshold = clim_mean + HOT_DAY_SIGMA * clim_std
    stratum = build_split_target_stratum(split, daily, windows)

    # Persistence/climatology predictions, for the same 4 metrics.
    split_windows = windows.loc[windows["split"] == split].reset_index(drop=True)
    feature_block = daily[FEATURE_COLUMNS]
    X_raw = np.empty((len(split_windows), 14, len(FEATURE_COLUMNS)), dtype=np.float32)
    for i, row in enumerate(split_windows.itertuples(index=False)):
        X_raw[i] = feature_block.loc[row.input_start : row.input_end].to_numpy(dtype=np.float32)
    persistence_pred = persistence_predict(X_raw)
    climatology_pred = climatology_predict(daily, windows, split)
    y_raw = data[split]["y_raw"]

    # LSTM: 5 seeds.
    per_seed_global, per_seed_stratified = [], []
    all_pred_hot, all_pred_raw = [], []

    for seed in SEEDS:
        model = load_seed_model(seed)
        with torch.no_grad():
            pred_norm = model(torch.from_numpy(data[split]["X_norm"])).numpy()
        pred_raw = denormalize_y(pred_norm, stats)

        per_seed_global.append({"mae": mae_np(pred_raw, y_raw), "rmse": rmse_np(pred_raw, y_raw)})
        per_seed_stratified.append(stratified_rmse(pred_raw, y_raw, stratum))
        all_pred_hot.append(pred_raw > threshold)
        all_pred_raw.append(pred_raw)

    def agg(vals):
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    lstm_global = {
        "mae": agg([s["mae"] for s in per_seed_global]),
        "rmse": agg([s["rmse"] for s in per_seed_global]),
    }
    lstm_stratified = {
        s: {"rmse_mean": float(np.mean([p[s]["rmse"] for p in per_seed_stratified])),
            "rmse_std": float(np.std([p[s]["rmse"] for p in per_seed_stratified])),
            "n": per_seed_stratified[0][s]["n"]}
        for s in STRATA
    }

    pooled_pred_hot = np.concatenate(all_pred_hot, axis=0)
    pooled_actual_hot = np.concatenate([actual_hot] * len(SEEDS), axis=0)
    lstm_detection = detection_metrics(pooled_pred_hot, pooled_actual_hot)

    pooled_pred_raw = np.concatenate(all_pred_raw, axis=0)
    pooled_y_raw = np.concatenate([y_raw] * len(SEEDS), axis=0)
    lstm_bias = bias_diagnostic(pooled_pred_raw, pooled_y_raw, pooled_actual_hot)

    # Persistence/climatology on the same 4 metrics (no seeds, deterministic).
    baselines = {}
    for name, pred in [("persistence", persistence_pred), ("climatology", climatology_pred)]:
        baselines[name] = {
            "global": {"mae": mae_np(pred, y_raw), "rmse": rmse_np(pred, y_raw)},
            "stratified_rmse": stratified_rmse(pred, y_raw, stratum),
            "detection": detection_metrics(pred > threshold, actual_hot),
        }

    result = {
        "split": split,
        "note": "test split intentionally not evaluated here -- reserved for Step 6",
        "1_global_accuracy": {
            "lstm": lstm_global,
            "persistence": baselines["persistence"]["global"],
            "climatology": baselines["climatology"]["global"],
        },
        "2_stratified_rmse": {
            "lstm": lstm_stratified,
            "persistence": baselines["persistence"]["stratified_rmse"],
            "climatology": baselines["climatology"]["stratified_rmse"],
        },
        "3_detection": {
            "lstm": lstm_detection,
            "persistence": baselines["persistence"]["detection"],
            "climatology": {**baselines["climatology"]["detection"],
                             "note": "Recall is mechanically 0: climatology always predicts its own mean, which can never exceed mean+1.5sigma by construction."},
        },
        "4_bias_diagnostic": lstm_bias,
    }

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVAL_DIR / "val_extended_metrics.json"
    with open(out_path, "w") as f:
        json.dump(_sanitize_nan(result), f, indent=2)

    print(f"\n=== 1. Global accuracy (val, 5-seed mean +/- std) ===")
    print(f"  Persistence: MAE={baselines['persistence']['global']['mae']:.3f}  RMSE={baselines['persistence']['global']['rmse']:.3f}")
    print(f"  Climatology: MAE={baselines['climatology']['global']['mae']:.3f}  RMSE={baselines['climatology']['global']['rmse']:.3f}")
    print(f"  LSTM:        MAE={lstm_global['mae']['mean']:.3f}+/-{lstm_global['mae']['std']:.3f}  RMSE={lstm_global['rmse']['mean']:.3f}+/-{lstm_global['rmse']['std']:.3f}")

    print(f"\n=== 2. Stratified RMSE (the metric that matters most) ===")
    for name, strat in [("Persistence", baselines["persistence"]["stratified_rmse"]),
                         ("Climatology", baselines["climatology"]["stratified_rmse"]),
                         ("LSTM", None)]:
        if name == "LSTM":
            print(f"  {name:12s} normal={lstm_stratified['normal']['rmse_mean']:.3f} "
                  f"unusual={lstm_stratified['unusual']['rmse_mean']:.3f} "
                  f"extreme={lstm_stratified['extreme']['rmse_mean']:.3f} (n_extreme={lstm_stratified['extreme']['n']})")
        else:
            print(f"  {name:12s} normal={strat['normal']['rmse']:.3f} unusual={strat['unusual']['rmse']:.3f} "
                  f"extreme={strat['extreme']['rmse']:.3f} (n_extreme={strat['extreme']['n']})")

    print(f"\n=== 3. Detection (precision / recall / F2) ===")
    for name, d in [("Persistence", baselines["persistence"]["detection"]),
                    ("Climatology", baselines["climatology"]["detection"]),
                    ("LSTM", lstm_detection)]:
        print(f"  {name:12s} precision={d['precision']:.3f} recall={d['recall']:.3f} f2={d['f2']:.3f}")

    print(f"\n=== 4. Bias diagnostic (LSTM, pooled across seeds) ===")
    print(f"  Hot days:    {lstm_bias['mean_bias_on_hot_days']:+.3f} degC (n={lstm_bias['n_hot']})")
    print(f"  Normal days: {lstm_bias['mean_bias_on_normal_days']:+.3f} degC (n={lstm_bias['n_normal']})")

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()