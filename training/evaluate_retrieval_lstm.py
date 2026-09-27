"""
Evaluates the retrieval-augmented model on the SAME 4 metrics + stratified
breakdown as training/evaluate_lstm.py, across all 5 trained seeds (not
just the one whose predictions got saved during training) -- this is the
actual apples-to-apples comparison against the canonical baseline that
Execution_Pipeline.md Step 6 is about.

Run from repo root:
    python -m training.evaluate_retrieval_lstm
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from models.retrieval_lstm import RetrievalAugmentedLSTM
from training.baselines import mae as mae_np, rmse as rmse_np
from training.data import (
    build_split_target_climatology,
    build_split_target_stratum,
    denormalize_y,
    load_normalization_stats,
    normalize_y,
    _load_daily,
    _load_windows,
)
from training.evaluate_lstm import detection_metrics
from training.retrieval_data import build_split_arrays_with_analogues
from training.train_retrieval_lstm import K, MODELS_DIR, SEEDS

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO_ROOT / "evaluation" / "retrieval_augmented"
HOT_DAY_SIGMA = 1.5


def stratified_rmse(pred, target, stratum):
    pred_f, target_f, stratum_f = pred.ravel(), target.ravel(), stratum.ravel()
    out = {}
    for s in ["normal", "unusual", "extreme"]:
        mask = stratum_f == s
        out[s] = {"rmse": rmse_np(pred_f[mask], target_f[mask]) if mask.sum() else float("nan"), "n": int(mask.sum())}
    return out


def main():
    stats = load_normalization_stats()
    daily = _load_daily()
    windows = _load_windows()
    clim_mean, clim_std, actual_hot = build_split_target_climatology("val", daily, windows)
    threshold = clim_mean + HOT_DAY_SIGMA * clim_std
    stratum = build_split_target_stratum("val", daily, windows)

    d = build_split_arrays_with_analogues("val", k=K)
    X_norm = torch.from_numpy(((d["X"] - stats.feature_mean) / stats.feature_std).astype(np.float32))
    Xa_norm = torch.from_numpy(((d["X_analogues"] - stats.feature_mean) / stats.feature_std).astype(np.float32))
    ya_norm = torch.from_numpy(normalize_y(d["y_analogues"], stats).astype(np.float32))
    mask = torch.from_numpy(d["analogue_mask"])
    y_raw = d["y"]

    per_seed_global, per_seed_stratified = [], []
    all_pred_hot, all_pred_raw = [], []

    for seed in SEEDS:
        model = RetrievalAugmentedLSTM()
        state = torch.load(MODELS_DIR / f"seed_{seed}" / "checkpoint.pt", map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            pred_norm, _ = model(X_norm, Xa_norm, ya_norm, mask)
        pred_raw = denormalize_y(pred_norm.numpy(), stats)

        per_seed_global.append({"mae": mae_np(pred_raw, y_raw), "rmse": rmse_np(pred_raw, y_raw)})
        per_seed_stratified.append(stratified_rmse(pred_raw, y_raw, stratum))
        all_pred_hot.append(pred_raw > threshold)
        all_pred_raw.append(pred_raw)

    lstm_global = {
        "mae": {"mean": float(np.mean([s["mae"] for s in per_seed_global])), "std": float(np.std([s["mae"] for s in per_seed_global]))},
        "rmse": {"mean": float(np.mean([s["rmse"] for s in per_seed_global])), "std": float(np.std([s["rmse"] for s in per_seed_global]))},
    }
    lstm_stratified = {
        s: {"rmse_mean": float(np.mean([p[s]["rmse"] for p in per_seed_stratified])),
            "rmse_std": float(np.std([p[s]["rmse"] for p in per_seed_stratified])),
            "n": per_seed_stratified[0][s]["n"]}
        for s in ["normal", "unusual", "extreme"]
    }
    pooled_pred_hot = np.concatenate(all_pred_hot, axis=0)
    pooled_actual_hot = np.concatenate([actual_hot] * len(SEEDS), axis=0)
    detection = detection_metrics(pooled_pred_hot, pooled_actual_hot)

    result = {
        "1_global_accuracy": lstm_global,
        "2_stratified_rmse": lstm_stratified,
        "3_detection": detection,
    }
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(EVAL_DIR / "val_extended_metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    print("=== Retrieval-augmented model, val split, 5 seeds ===")
    print(f"Global MAE:  {lstm_global['mae']['mean']:.3f} +/- {lstm_global['mae']['std']:.3f}")
    print(f"Global RMSE: {lstm_global['rmse']['mean']:.3f} +/- {lstm_global['rmse']['std']:.3f}")
    print()
    for s in ["normal", "unusual", "extreme"]:
        r = lstm_stratified[s]
        print(f"{s:10s} RMSE: {r['rmse_mean']:.3f} +/- {r['rmse_std']:.3f} (n={r['n']})")
    print()
    print(f"Detection: precision={detection['precision']:.3f} recall={detection['recall']:.3f} f2={detection['f2']:.3f}")
    print(f"\nSaved to {EVAL_DIR / 'val_extended_metrics.json'}")


if __name__ == "__main__":
    main()
