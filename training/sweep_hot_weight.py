"""
Sweeps HOT_WEIGHT in {5, 10, 15, 20, 25} -- the loss-weighting hyperparameter
that was flagged as untuned in context/ml_notes.md. Trains 5 seeds per
value (same architecture, optimizer, data, early stopping as
training/train_lstm.py), evaluates each on the SAME 4 metrics as
training/evaluate_lstm.py, and reports which value actually performs best
on the metric that matters (extreme-stratum RMSE), not just detection
recall -- see context/decisions.md for why recall alone is not the right
thing to optimize.

Does NOT touch models/baseline_lstm/ (the canonical, hot_weight=10
checkpoints) -- each swept value trains into its own
models/hw_sweep/hw_{value}/ directory. Canonical is only replaced if this
sweep finds a clearly better value AND that's confirmed, not automatically.

Run from repo root:
    python -m training.sweep_hot_weight
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.lstm import LSTMForecaster
from training.baselines import mae as mae_np, rmse as rmse_np
from training.data import (
    build_split_target_climatology,
    build_split_target_stratum,
    denormalize_y,
    load_all_splits,
    _load_daily,
    _load_windows,
)
from training.evaluate_lstm import detection_metrics, stratified_rmse

REPO_ROOT = Path(__file__).resolve().parents[1]
SWEEP_DIR = REPO_ROOT / "models" / "hw_sweep"
EVAL_DIR = REPO_ROOT / "evaluation" / "hw_sweep"

SEEDS = [0, 1, 2, 3, 4]
BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE = 10
LR = 1e-3
HOT_DAY_SIGMA = 1.5
SWEEP_VALUES = [5.0, 10.0, 15.0, 20.0, 25.0]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def weighted_mse(pred, target, hot_mask, hot_weight):
    weight = 1.0 + (hot_weight - 1.0) * hot_mask.float()
    return torch.mean(weight * (pred - target) ** 2)


def make_loader(X, y, hot, batch_size, shuffle):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(hot.astype(np.float32)))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_one_seed(seed, hot_weight, data, train_hot, val_hot, save_dir):
    set_seed(seed)
    train_loader = make_loader(data["train"]["X_norm"], data["train"]["y_norm"], train_hot, BATCH_SIZE, True)
    val_X = torch.from_numpy(data["val"]["X_norm"])
    val_y = torch.from_numpy(data["val"]["y_norm"])
    val_hot_t = torch.from_numpy(val_hot.astype(np.float32))

    model = LSTMForecaster()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_val_loss, best_state, epochs_without_improvement = float("inf"), None, 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb, hb in train_loader:
            optimizer.zero_grad()
            loss = weighted_mse(model(xb), yb, hb, hot_weight)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = weighted_mse(model(val_X), val_y, val_hot_t, hot_weight).item()

        if val_loss < best_val_loss - 1e-6:
            best_val_loss, best_state, epochs_without_improvement = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                break

    model.load_state_dict(best_state)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / "checkpoint.pt")
    return model, epoch


def evaluate(model, data, stats, stratum, threshold, actual_hot):
    model.eval()
    with torch.no_grad():
        pred_norm = model(torch.from_numpy(data["val"]["X_norm"])).numpy()
    pred_raw = denormalize_y(pred_norm, stats)
    y_raw = data["val"]["y_raw"]
    return {
        "mae": mae_np(pred_raw, y_raw),
        "rmse": rmse_np(pred_raw, y_raw),
        "stratified": stratified_rmse(pred_raw, y_raw, stratum),
        "detection": detection_metrics(pred_raw > threshold, actual_hot),
        "pred_raw": pred_raw,
    }


def main(values=None):
    values = values if values is not None else SWEEP_VALUES
    print("Loading data...")
    data = load_all_splits(refit_stats=False)
    stats = data["stats"]
    daily = _load_daily()
    windows = _load_windows()
    _, _, train_hot = build_split_target_climatology("train", daily, windows)
    clim_mean, clim_std, val_hot = build_split_target_climatology("val", daily, windows)
    threshold = clim_mean + HOT_DAY_SIGMA * clim_std
    stratum = build_split_target_stratum("val", daily, windows)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    results_path = EVAL_DIR / "sweep_results.json"
    all_results = {}
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)

    t0 = time.time()
    for hw in values:
        print(f"\n=== hot_weight={hw} ===")
        seed_metrics = []
        all_pred_hot, all_pred_raw = [], []
        for seed in SEEDS:
            save_dir = SWEEP_DIR / f"hw_{hw:g}" / f"seed_{seed}"
            model, epochs = train_one_seed(seed, hw, data, train_hot, val_hot, save_dir)
            m = evaluate(model, data, stats, stratum, threshold, val_hot)
            print(f"  seed {seed}: {epochs} epochs, MAE={m['mae']:.3f} RMSE={m['rmse']:.3f} "
                  f"extreme_rmse={m['stratified']['extreme']['rmse']:.3f} recall={m['detection']['recall']:.3f}")
            seed_metrics.append(m)
            all_pred_hot.append(m["pred_raw"] > threshold)
            all_pred_raw.append(m["pred_raw"])

        pooled_pred_hot = np.concatenate(all_pred_hot, axis=0)
        pooled_actual_hot = np.concatenate([val_hot] * len(SEEDS), axis=0)
        pooled_detection = detection_metrics(pooled_pred_hot, pooled_actual_hot)

        agg = {
            "mae_mean": float(np.mean([m["mae"] for m in seed_metrics])),
            "mae_std": float(np.std([m["mae"] for m in seed_metrics])),
            "rmse_mean": float(np.mean([m["rmse"] for m in seed_metrics])),
            "rmse_std": float(np.std([m["rmse"] for m in seed_metrics])),
            "stratified": {
                s: {"rmse_mean": float(np.mean([m["stratified"][s]["rmse"] for m in seed_metrics])),
                    "rmse_std": float(np.std([m["stratified"][s]["rmse"] for m in seed_metrics])),
                    "n": seed_metrics[0]["stratified"][s]["n"]}
                for s in ["normal", "unusual", "extreme"]
            },
            "detection_pooled": pooled_detection,
        }
        all_results[f"hw_{hw:g}"] = agg
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)

    total_elapsed = time.time() - t0
    print(f"\nThis batch took {total_elapsed:.1f}s. Saved to {results_path}")
    return all_results


def print_table(all_results):
    print(f"\n{'hot_weight':<12}{'Global MAE':>12}{'Global RMSE':>13}{'Extreme RMSE':>14}{'Recall':>9}{'Precision':>11}{'F2':>7}")
    print("-" * 78)
    for hw in SWEEP_VALUES:
        key = f"hw_{hw:g}"
        if key not in all_results:
            continue
        r = all_results[key]
        d = r["detection_pooled"]
        print(f"{hw:<12g}{r['mae_mean']:>12.3f}{r['rmse_mean']:>13.3f}"
              f"{r['stratified']['extreme']['rmse_mean']:>14.3f}{d['recall']:>9.3f}{d['precision']:>11.3f}{d['f2']:>7.3f}")


if __name__ == "__main__":
    import sys
    values = [float(v) for v in sys.argv[1:]] if len(sys.argv) > 1 else None
    results = main(values)
    print_table(results)

