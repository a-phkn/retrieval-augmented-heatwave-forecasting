"""
Trains the retrieval-augmented LSTM (models.retrieval_lstm.RetrievalAugmentedLSTM),
5 seeds, weighted MSE (hot_weight=20 -- MUST match training/train_lstm.py's
canonical baseline for the Step 5 "same training procedure, retrieval
on/off only" comparison to be valid; see context/decisions.md).

COMPUTE NOTE: each training step encodes the query window AND all K
analogue windows through the shared LSTM encoder -- roughly a (K+1)x
forward-pass cost per batch vs. the plain baseline. Per context/decisions.md's
Colab-first decision (user's laptop is ~5 years old), this is written to
run identically on CPU or GPU (just moves tensors to whatever `device` is)
so it works unmodified in a Colab notebook -- see context/RUN_COMMANDS.md
for the exact Colab cells. A local run is fine for a quick smoke-test
(reduce MAX_EPOCHS / use 1 seed) but full 5-seed training is intended for
Colab.

Saves:
  models/retrieval_augmented/seed_{n}/checkpoint.pt
  evaluation/retrieval_augmented/val_metrics.json
  predictions/retrieval_augmented/val_predictions.parquet
    (query_date, predicted t_max x5, actual t_max x5, attention weights x K)
    -- per Execution_Pipeline.md Step 5's output spec.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.retrieval_lstm import RetrievalAugmentedLSTM
from training.baselines import mae as mae_np
from training.baselines import rmse as rmse_np
from training.data import (
    build_split_target_climatology,
    denormalize_y,
    load_normalization_stats,
    normalize_y,
    _load_daily,
    _load_windows,
)
from training.retrieval_data import build_split_arrays_with_analogues

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models" / "retrieval_augmented"
EVAL_DIR = REPO_ROOT / "evaluation" / "retrieval_augmented"
PRED_DIR = REPO_ROOT / "predictions" / "retrieval_augmented"

SEEDS = [0, 1, 2, 3, 4]
BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE = 10
LR = 1e-3
HOT_WEIGHT = 20.0  # MUST match training/train_lstm.py's canonical value
K = 5              # roadmap's core K; the {1,3,5,10,20} sweep is Step 7, not this script

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def weighted_mse(pred, target, hot_mask, hot_weight):
    weight = 1.0 + (hot_weight - 1.0) * hot_mask.float()
    return torch.mean(weight * (pred - target) ** 2)


def prepare_split_tensors(split, stats, hot):
    """Builds and normalizes all tensors for one split, once (reused across
    seeds -- retrieval/data don't depend on the model)."""
    d = build_split_arrays_with_analogues(split, k=K)
    X_norm = (d["X"] - stats.feature_mean) / stats.feature_std
    y_norm = normalize_y(d["y"], stats)
    y_analogues_norm = normalize_y(d["y_analogues"], stats)  # same t_max stats apply
    return {
        "X": torch.from_numpy(X_norm.astype(np.float32)),
        "y": torch.from_numpy(y_norm.astype(np.float32)),
        "X_analogues": torch.from_numpy((d["X_analogues"] - stats.feature_mean) / stats.feature_std).float(),
        "y_analogues": torch.from_numpy(y_analogues_norm.astype(np.float32)),
        "analogue_mask": torch.from_numpy(d["analogue_mask"]),
        "hot": torch.from_numpy(hot.astype(np.float32)),
        "y_raw": d["y"],
        "query_dates": d["query_dates"],
    }


def make_loader(t, batch_size, shuffle):
    ds = TensorDataset(t["X"], t["y"], t["X_analogues"], t["y_analogues"], t["analogue_mask"], t["hot"])
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_one_seed(seed, train_t, val_t):
    set_seed(seed)
    train_loader = make_loader(train_t, BATCH_SIZE, shuffle=True)

    model = RetrievalAugmentedLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    val_X = val_t["X"].to(device)
    val_y = val_t["y"].to(device)
    val_Xa = val_t["X_analogues"].to(device)
    val_ya = val_t["y_analogues"].to(device)
    val_mask = val_t["analogue_mask"].to(device)
    val_hot = val_t["hot"].to(device)

    best_val_loss, best_state, epochs_without_improvement = float("inf"), None, 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb, xab, yab, mb, hb in train_loader:
            xb, yb, xab, yab, mb, hb = (t.to(device) for t in (xb, yb, xab, yab, mb, hb))
            optimizer.zero_grad()
            pred, _ = model(xb, xab, yab, mb)
            loss = weighted_mse(pred, yb, hb, HOT_WEIGHT)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred, _ = model(val_X, val_Xa, val_ya, val_mask)
            val_loss = weighted_mse(val_pred, val_y, val_hot, HOT_WEIGHT).item()

        if val_loss < best_val_loss - 1e-6:
            best_val_loss, epochs_without_improvement = val_loss, 0
            best_state = {k_: v.clone() for k_, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                break

    model.load_state_dict(best_state)
    seed_dir = MODELS_DIR / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), seed_dir / "checkpoint.pt")
    return model, epoch


def evaluate_and_collect(model, val_t, stats):
    model.eval()
    with torch.no_grad():
        pred_norm, attn = model(
            val_t["X"].to(device), val_t["X_analogues"].to(device),
            val_t["y_analogues"].to(device), val_t["analogue_mask"].to(device),
        )
    pred_raw = denormalize_y(pred_norm.cpu().numpy(), stats)
    y_raw = val_t["y_raw"]
    return {
        "mae": mae_np(pred_raw, y_raw), "rmse": rmse_np(pred_raw, y_raw),
        "pred_raw": pred_raw, "attn": attn.cpu().numpy(),
    }


def main(seeds=None):
    seeds = seeds if seeds is not None else SEEDS
    print(f"Device: {device}")
    stats = load_normalization_stats()
    daily = _load_daily()
    windows = _load_windows()
    _, _, train_hot = build_split_target_climatology("train", daily, windows)
    _, _, val_hot = build_split_target_climatology("val", daily, windows)

    print("Building train tensors (query + analogues)...")
    train_t = prepare_split_tensors("train", stats, train_hot)
    print("Building val tensors...")
    val_t = prepare_split_tensors("val", stats, val_hot)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    results_path = EVAL_DIR / "val_metrics.json"
    seed_results = []
    if results_path.exists():
        with open(results_path) as f:
            existing = json.load(f)
            seed_results = existing.get("per_seed", [])

    t0 = time.time()
    last_attn = None
    for seed in seeds:
        print(f"\n--- Training seed {seed} ---")
        seed_t0 = time.time()
        model, epochs = train_one_seed(seed, train_t, val_t)
        elapsed = time.time() - seed_t0
        m = evaluate_and_collect(model, val_t, stats)
        print(f"seed {seed}: {epochs} epochs, val MAE={m['mae']:.3f} RMSE={m['rmse']:.3f}, time={elapsed:.1f}s")
        seed_results = [r for r in seed_results if r["seed"] != seed]
        seed_results.append({"seed": seed, "epochs": epochs, "val_mae": m["mae"], "val_rmse": m["rmse"], "train_time_seconds": elapsed})
        last_attn = m

        maes = [r["val_mae"] for r in seed_results]
        rmses = [r["val_rmse"] for r in seed_results]
        summary = {
            "hot_weight": HOT_WEIGHT, "k": K, "device": str(device),
            "val_mae_mean": float(np.mean(maes)), "val_mae_std": float(np.std(maes)),
            "val_rmse_mean": float(np.mean(rmses)), "val_rmse_std": float(np.std(rmses)),
            "n_seeds_so_far": len(seed_results),
            "per_seed": sorted(seed_results, key=lambda r: r["seed"]),
        }
        with open(results_path, "w") as f:
            json.dump(summary, f, indent=2)

    total_elapsed = time.time() - t0

    if last_attn is not None:
        PRED_DIR.mkdir(parents=True, exist_ok=True)
        pred_df = pd.DataFrame({
            "query_date": val_t["query_dates"].values,
            **{f"pred_day{i+1}": last_attn["pred_raw"][:, i] for i in range(5)},
            **{f"actual_day{i+1}": val_t["y_raw"][:, i] for i in range(5)},
            **{f"attn_rank{j+1}": last_attn["attn"][:, j] for j in range(K)},
        })
        pred_df.to_parquet(PRED_DIR / "val_predictions.parquet", index=False)

    print(f"\n=== Retrieval-augmented model, val split, {len(seed_results)}/{len(SEEDS)} seeds done, device={device} ===")
    print(f"MAE:  {summary['val_mae_mean']:.3f} +/- {summary['val_mae_std']:.3f}")
    print(f"RMSE: {summary['val_rmse_mean']:.3f} +/- {summary['val_rmse_std']:.3f}")
    print(f"This batch took: {total_elapsed:.1f}s")
    print(f"Saved to {results_path}")


if __name__ == "__main__":
    import sys
    seeds = [int(s) for s in sys.argv[1:]] if len(sys.argv) > 1 else None
    main(seeds)
