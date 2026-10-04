"""
Per-seed, per-window predictions of the frozen v1 models (plan v5, Week 1).

Writes one long-format parquet per (split, model) to predictions_v1/<split>/<model>.parquet,
one row per (seed, window, lead day):
    model, seed, query_date, lead, target_date, pred, actual, error,
    stratum (v1 normal/unusual/extreme), cluster_id (year x season of target date)
plus, for the retrieval-augmented model, analogue provenance for ranks 1..K:
    analogue{k}_date, analogue{k}_similarity, analogue{k}_attention, analogue{k}_age_years

Why a separate script: training/train_retrieval_lstm.py saves predictions of only the
LAST seed it trains (the archived val_predictions.parquet is seed 4 only), and no script
saved per-window LSTM predictions. Paired, seed-aware statistics (evaluation/stats.py)
need both. Prediction is separated from training so it can be regenerated from the
frozen checkpoints at any time without retraining; no training code is changed.

Models:
    persistence, climatology -- deterministic baselines (training/baselines.py), seed 0
    lstm_a1                  -- models/frozen/lstm_tmax_v1/seed_*/checkpoint.pt
    ra_v1                    -- models/frozen/ra_lstm_v1/seed_*/checkpoint.pt (if present)

The test split is refused: it stays locked until the pre-registered test run.

Run from repo root:
    python -m evaluation.predict_v1                 # val split
    python -m evaluation.predict_v1 --splits train val
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluation.stats import make_cluster_ids
from models.lstm import LSTMForecaster
from models.retrieval_lstm import RetrievalAugmentedLSTM
from training.baselines import climatology_predict, persistence_predict
from training.data import (
    _load_daily,
    _load_windows,
    build_split_arrays,
    build_split_target_stratum,
    denormalize_y,
    load_normalization_stats,
    normalize_X,
    normalize_y,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "predictions_v1"
LSTM_DIR = REPO_ROOT / "models" / "frozen" / "lstm_tmax_v1"
RA_DIR = REPO_ROOT / "models" / "frozen" / "ra_lstm_v1"
ANALOGUES_PATH = REPO_ROOT / "retrieval" / "analogues_top20.parquet"
ALLOWED_SPLITS = ("train", "val")
K = 5  # must match training/train_retrieval_lstm.py
LEADS = 5


def seed_dirs(model_dir: Path) -> list[tuple[int, Path]]:
    """[(seed, checkpoint_path)] sorted by seed; empty if the model isn't frozen yet."""
    out = []
    for d in sorted(model_dir.glob("seed_*")):
        ckpt = d / "checkpoint.pt"
        if ckpt.exists():
            out.append((int(d.name.split("_")[1]), ckpt))
    return out


def long_frame(model: str, seed: int, query_dates: pd.Series, pred: np.ndarray, actual: np.ndarray,
               stratum: np.ndarray) -> pd.DataFrame:
    """(n_windows, LEADS) arrays -> one row per (window, lead)."""
    n = len(query_dates)
    qd = np.repeat(pd.DatetimeIndex(query_dates).to_numpy(), LEADS)
    lead = np.tile(np.arange(1, LEADS + 1), n)
    target = qd + (lead - 1).astype("timedelta64[D]")  # lead 1 = query_date (first forecast day)
    df = pd.DataFrame({
        "model": model,
        "seed": seed,
        "query_date": qd,
        "lead": lead,
        "target_date": target,
        "pred": pred.reshape(-1).astype(np.float64),
        "actual": actual.reshape(-1).astype(np.float64),
        "stratum": stratum.reshape(-1),
    })
    df["error"] = df["pred"] - df["actual"]
    df["cluster_id"] = make_cluster_ids(df["target_date"])
    return df


@torch.no_grad()
def predict_lstm(ckpt: Path, X_norm: np.ndarray, stats) -> np.ndarray:
    model = LSTMForecaster()
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval()
    return denormalize_y(model(torch.from_numpy(X_norm)).numpy(), stats)


def analogue_inputs(split_windows: pd.DataFrame, X_raw: np.ndarray):
    """Analogue tensors + provenance, built exactly as training.retrieval_data does
    (slot j = rank j+1 of retrieval/analogues_top20.parquet, padded slots masked)."""
    from training.retrieval_data import build_split_arrays_with_analogues

    split = split_windows["split"].iloc[0]
    d = build_split_arrays_with_analogues(split, k=K)
    if not np.array_equal(d["X"], X_raw):
        raise RuntimeError("analogue builder and window builder disagree on query inputs")
    analogues = pd.read_parquet(ANALOGUES_PATH)
    analogues = analogues[(analogues["rank"] <= K) & analogues["query_date"].isin(split_windows["query_date"])]
    wide = analogues.pivot(index="query_date", columns="rank", values=["analogue_query_date", "similarity"])
    wide = wide.reindex(split_windows["query_date"])
    return d, wide


@torch.no_grad()
def predict_ra(ckpt: Path, d: dict, stats) -> tuple[np.ndarray, np.ndarray]:
    model = RetrievalAugmentedLSTM()
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval()
    pred_norm, attn = model(
        torch.from_numpy(normalize_X(d["X"], stats).astype(np.float32)),
        torch.from_numpy(((d["X_analogues"] - stats.feature_mean) / stats.feature_std).astype(np.float32)),
        torch.from_numpy(normalize_y(d["y_analogues"], stats).astype(np.float32)),
        torch.from_numpy(d["analogue_mask"]),
    )
    return denormalize_y(pred_norm.numpy(), stats), attn.numpy()


def add_provenance(df: pd.DataFrame, wide: pd.DataFrame, attn: np.ndarray, query_dates: pd.Series) -> pd.DataFrame:
    """Adds analogue{k}_* columns (repeated across the 5 lead rows of each window).
    Slots with no eligible analogue have NaT date / NaN similarity and age. Known v1
    model limitation: for a query with ZERO analogues, models/retrieval_lstm.py spreads
    attention uniformly (0.2) over the padded slots, so attention can be non-zero where
    provenance is NaN (163/13,131 train windows; none in val)."""
    qd = pd.DatetimeIndex(query_dates)
    for k in range(1, K + 1):
        a_date = pd.DatetimeIndex(wide[("analogue_query_date", k)]) if ("analogue_query_date", k) in wide else pd.DatetimeIndex([pd.NaT] * len(qd))
        sim = wide[("similarity", k)].to_numpy(dtype=np.float64) if ("similarity", k) in wide else np.full(len(qd), np.nan)
        age = (qd - a_date).days.to_numpy(dtype=np.float64) / 365.25
        df[f"analogue{k}_date"] = np.repeat(a_date.to_numpy(), LEADS)
        df[f"analogue{k}_similarity"] = np.repeat(sim, LEADS)
        df[f"analogue{k}_attention"] = np.repeat(attn[:, k - 1].astype(np.float64), LEADS)
        df[f"analogue{k}_age_years"] = np.repeat(age, LEADS)
    return df


def run_split(split: str) -> dict[str, Path]:
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"split {split!r} refused: only {ALLOWED_SPLITS} (test stays locked until the test run)")
    stats = load_normalization_stats()
    daily, windows = _load_daily(), _load_windows()
    split_windows = windows.loc[windows["split"] == split].reset_index(drop=True)
    X_raw, y_raw, query_dates = build_split_arrays(split, daily, windows)
    stratum = build_split_target_stratum(split, daily, windows)
    out_dir = OUT_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}

    frames = {
        "persistence": [long_frame("persistence", 0, query_dates, persistence_predict(X_raw), y_raw, stratum)],
        "climatology": [long_frame("climatology", 0, query_dates, climatology_predict(daily, windows, split), y_raw, stratum)],
    }
    X_norm = normalize_X(X_raw, stats).astype(np.float32)
    lstm_seeds = seed_dirs(LSTM_DIR)
    if not lstm_seeds:
        raise FileNotFoundError(f"no frozen A1 checkpoints in {LSTM_DIR}")
    frames["lstm_a1"] = [long_frame("lstm_a1", s, query_dates, predict_lstm(c, X_norm, stats), y_raw, stratum)
                         for s, c in lstm_seeds]

    ra_seeds = seed_dirs(RA_DIR)
    if ra_seeds:
        d, wide = analogue_inputs(split_windows, X_raw)
        frames["ra_v1"] = []
        for s, c in ra_seeds:
            pred, attn = predict_ra(c, d, stats)
            df = long_frame("ra_v1", s, query_dates, pred, y_raw, stratum)
            frames["ra_v1"].append(add_provenance(df, wide, attn, query_dates))
    else:
        print(f"  (no frozen RA checkpoints in {RA_DIR}; skipping ra_v1)")

    for name, parts in frames.items():
        path = out_dir / f"{name}.parquet"
        pd.concat(parts, ignore_index=True).to_parquet(path, index=False)
        written[name] = path
        seeds = sorted({int(p['seed'].iloc[0]) for p in parts})
        rmse = np.mean([np.sqrt(np.mean(p["error"] ** 2)) for p in parts])
        shown = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
        print(f"  [{split}] {name:12s} seeds={seeds} rows={sum(len(p) for p in parts):7d} mean RMSE={rmse:.4f} -> {shown}")
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Per-seed predictions of the frozen v1 models.")
    parser.add_argument("--splits", nargs="+", default=["val"], choices=ALLOWED_SPLITS)
    args = parser.parse_args(argv)
    torch.set_num_threads(max(1, (torch.get_num_threads() or 2) // 2))
    for split in args.splits:
        run_split(split)


if __name__ == "__main__":
    main()
