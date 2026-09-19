"""
Trains the baseline LSTM forecaster with 5 seeds, early stopping on val loss.

Loss: WEIGHTED MSE on z-normalized targets (see WEIGHTED-LOSS RATIONALE
below) -- this superseded plain MSE after evaluation showed plain MSE
produces a model with excellent overall fit but near-total failure to
detect actual heatwave days (4.8% recall vs. persistence's 22.7%; see
context/decisions.md and context/ml_notes.md for the full investigation).
Everything else matches the original roadmap spec: Adam lr=1e-3, batch
size 64, up to 100 epochs, patience 10, 5 seeds.

WEIGHTED-LOSS RATIONALE:
loss = mean( weight * (pred - target)^2 ),  weight = 1 + (HOT_WEIGHT-1) * hot_mask
where hot_mask flags forecast-day-instances that are actually "hot" per the
project's own train-only climatology + 1.5sigma definition (same definition
used everywhere else in the pipeline, e.g. event_catalogue.parquet). Plain
MSE is dominated by the ~95% of normal days, so the MSE-optimal function
shrinks extreme predictions toward the mean. Upweighting hot-day targets
forces the optimizer to treat underpredicting them as costly.
HOT_WEIGHT=10 is a deliberate, moderate first choice (true inverse-frequency
weighting would be ~21.7x) -- NOT swept/tuned. A sweep (5/10/15/20) is a
legitimate next step if further gains are wanted; see context/ml_notes.md's
open questions.

IMPORTANT: this is now the single canonical baseline used everywhere,
including as Step 5's future "retrieval off" reference. If/when the
retrieval-augmented model is built, it must use this SAME weighted loss
(hot_weight=10) for the "same training procedure, retrieval on/off only"
comparison to remain valid -- see context/decisions.md.
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
from training.baselines import mae as mae_np
from training.baselines import rmse as rmse_np
from training.data import build_split_target_climatology, denormalize_y, load_all_splits, _load_daily, _load_windows

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models" / "baseline_lstm"
EVAL_DIR = REPO_ROOT / "evaluation" / "baseline_lstm"

SEEDS = [0, 1, 2, 3, 4]
BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE = 10
LR = 1e-3
HOT_WEIGHT = 10.0


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def weighted_mse(pred: torch.Tensor, target: torch.Tensor, hot_mask: torch.Tensor, hot_weight: float) -> torch.Tensor:
    weight = 1.0 + (hot_weight - 1.0) * hot_mask.float()
    return torch.mean(weight * (pred - target) ** 2)


def make_loader(X: np.ndarray, y: np.ndarray, hot: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(hot.astype(np.float32)))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_one_seed(seed: int, data: dict, train_hot: np.ndarray, val_hot: np.ndarray) -> dict:
    set_seed(seed)

    train_loader = make_loader(data["train"]["X_norm"], data["train"]["y_norm"], train_hot, BATCH_SIZE, shuffle=True)
    val_X_norm = torch.from_numpy(data["val"]["X_norm"])
    val_y_norm = torch.from_numpy(data["val"]["y_norm"])
    val_hot_t = torch.from_numpy(val_hot.astype(np.float32))

    model = LSTMForecaster()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_losses = []
        for xb, yb, hb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = weighted_mse(pred, yb, hb, HOT_WEIGHT)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_pred = model(val_X_norm)
            # Early stopping tracks the SAME weighted loss used to train, so
            # "improvement" is measured on the objective actually optimized.
            val_loss = weighted_mse(val_pred, val_y_norm, val_hot_t, HOT_WEIGHT).item()

        train_loss = float(np.mean(train_losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                break

    model.load_state_dict(best_state)

    seed_dir = MODELS_DIR / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), seed_dir / "checkpoint.pt")

    return {
        "seed": seed,
        "best_val_loss_weighted": best_val_loss,
        "epochs_trained": len(history),
        "history": history,
        "model_state": best_state,
    }


def evaluate_model_on_split(model: LSTMForecaster, data: dict, split: str, stats) -> dict:
    model.eval()
    with torch.no_grad():
        X_norm = torch.from_numpy(data[split]["X_norm"])
        pred_norm = model(X_norm).numpy()
    pred_raw = denormalize_y(pred_norm, stats)
    y_raw = data[split]["y_raw"]
    return {"mae": mae_np(pred_raw, y_raw), "rmse": rmse_np(pred_raw, y_raw)}


def main():
    print(f"Loading data and fitting normalization stats (train-only)... [hot_weight={HOT_WEIGHT}]")
    data = load_all_splits(refit_stats=True)
    stats = data["stats"]

    daily = _load_daily()
    windows = _load_windows()
    _, _, train_hot = build_split_target_climatology("train", daily, windows)
    _, _, val_hot = build_split_target_climatology("val", daily, windows)

    seed_results = []
    seed_val_metrics = []
    t0 = time.time()
    for seed in SEEDS:
        print(f"\n--- Training seed {seed} ---")
        seed_t0 = time.time()
        result = train_one_seed(seed, data, train_hot, val_hot)
        elapsed = time.time() - seed_t0

        model = LSTMForecaster()
        model.load_state_dict(result["model_state"])
        val_metrics = evaluate_model_on_split(model, data, "val", stats)

        print(f"seed {seed}: {result['epochs_trained']} epochs, "
              f"best val loss (weighted, norm)={result['best_val_loss_weighted']:.4f}, "
              f"val MAE={val_metrics['mae']:.3f}, val RMSE={val_metrics['rmse']:.3f}, "
              f"time={elapsed:.1f}s")

        seed_results.append({
            "seed": seed,
            "epochs_trained": result["epochs_trained"],
            "best_val_loss_weighted_norm": result["best_val_loss_weighted"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "train_time_seconds": elapsed,
        })
        seed_val_metrics.append(val_metrics)

    total_elapsed = time.time() - t0

    val_maes = [m["mae"] for m in seed_val_metrics]
    val_rmses = [m["rmse"] for m in seed_val_metrics]
    summary = {
        "hot_weight": HOT_WEIGHT,
        "loss": "weighted_mse",
        "val_mae_mean": float(np.mean(val_maes)),
        "val_mae_std": float(np.std(val_maes)),
        "val_rmse_mean": float(np.mean(val_rmses)),
        "val_rmse_std": float(np.std(val_rmses)),
        "n_seeds": len(SEEDS),
        "total_train_time_seconds": total_elapsed,
        "per_seed": seed_results,
    }

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(EVAL_DIR / "val_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== LSTM baseline (weighted MSE, hot_weight={HOT_WEIGHT}), val split, {len(SEEDS)} seeds ===")
    print(f"MAE:  {summary['val_mae_mean']:.3f} +/- {summary['val_mae_std']:.3f}")
    print(f"RMSE: {summary['val_rmse_mean']:.3f} +/- {summary['val_rmse_std']:.3f}")
    print(f"Total training time: {total_elapsed:.1f}s")
    print(f"Saved to {EVAL_DIR / 'val_metrics.json'}")


if __name__ == "__main__":
    main()
