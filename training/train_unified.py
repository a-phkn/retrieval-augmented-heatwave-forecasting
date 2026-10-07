"""
Config-driven training of LSTM variants on rolling-origin folds (plan v5, Week 2).

One JSON config = one run (e.g. A1', A2, A2r); each run changes one factor from its
parent. For every (fold, seed) the script trains models.lstm.LSTMForecaster with EXACTLY
the recipe of training/train_lstm.py (v1, unchanged): Adam lr 1e-3, batch 64, up to 100
epochs, early stopping with patience 10 on the weighted loss of the early-stopping set
(see below), weighted MSE on
normalised targets (weight = 1 + (hot_weight - 1) * hot mask), and the same order of
random-number use, so a seed gives the same model as v1 for the same data.

Early stopping ("early_stop" key):
    "inner_2y"   v2 default: fit on the fold's training years minus the last 2, early-stop
                 on those last 2 training years; the validation block is only reported.
                 No refit on the stop years afterwards (models never train on them). The
                 stop block holds only ~15-26 hot days, so expect seed spread.
    "val_block"  v1 behaviour (early-stop on the reported block). Optimistically biased;
                 used only by A1_repro to reproduce the frozen A1 bit-for-bit.

Config keys (configs/*.json):
    run_id, parent, description, target ("t_max" | "wbgt_bom_max"), labels ("v1" | "v2"),
    anomaly_target (bool), hot_weight, early_stop, seeds (list), folds (list of "f1".."f4"),
    target_form (optional: "raw" | "anomaly" | "dp_residual"; default from anomaly_target),
    threads (optional int; results are bit-reproducible only at the same thread count),
    retrieval (optional: {"mode": "sim" | "rand" | "time" | "time_rand" | "region", "k": 5}). With it, the model is
        models.retrieval_lstm_v2 fed K analogues per window from retrieval.fold_retrieval
        (this fold's training windows only; "rand"/"time_rand" draws are fixed per fold and seed). The
        analogues' outcomes are given in the model's own target units (same target form and
        scaling). Without it, the plain LSTM path below is used unchanged.

Registry RMSE values of different targets (Tmax vs WBGT) are NOT comparable. Compare runs
through the skill columns, 1 - RMSE / RMSE_ref, where climatology and persistence are
computed on the same windows and target (damped persistence: evaluation scripts, Week 3).

Outputs:
    predictions_v2/<run_id>/<fold>.parquet  val predictions, predict_v1 long format
                                            (seed, query_date, lead, target_date, pred,
                                            actual, error, stratum, cluster_id)
    predictions_v2/<run_id>/<fold>_analogues.parquet  retrieval runs only: per seed and
                                            val query, the K analogues (rank, date,
                                            similarity) and the model's attention weights
    models/v2/<run_id>/<fold>/seed_<n>/checkpoint.pt   (gitignored; regenerable)
    registry/runs.csv                       one row per (run, fold): config, data and code
                                            hashes, git commit + dirty flag, seeds, threads,
                                            torch version, val RMSE and skill (all / extreme)
The test split is never built (training/folds.py drops 2019+ before anything else).

Run from repo root:
    python -m training.train_unified --config configs/A1prime.json
    python -m training.train_unified --config configs/A2.json --folds f4 --seeds 0 1
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from evaluation.predict_v1 import long_frame
from models.lstm import LSTMForecaster
from models.retrieval_lstm_v2 import RetrievalAugmentedLSTMv2
from retrieval.fold_retrieval import MODES as RETRIEVAL_MODES, RANDOM_MODES, UPSTREAM_PATH
from retrieval.fold_retrieval import FoldRetriever
from training.folds import (
    DAILY_V2_PATH, FOLDS, LABEL_VERSIONS, LILJEGREN_DAILY_PATH, TARGET_FORMS, TARGETS, WBGT_LABEL_CONFIG,
    FoldData, SplitArrays, build_fold, inner_split,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = REPO_ROOT / "predictions_v2"
MODEL_DIR = REPO_ROOT / "models" / "v2"
REGISTRY = REPO_ROOT / "registry" / "runs.csv"
WINDOW_INDEX = REPO_ROOT / "splits" / "window_index_v1.parquet"
# Code that determines a run's data, labels and training; hashed into every registry row.
CODE_FILES = [
    "training/folds.py", "training/train_unified.py", "pipeline/labels_v2.py",
    "pipeline/climatology.py", "models/lstm.py", "training/data.py", "evaluation/predict_v1.py",
    "evaluation/stats.py", "retrieval/fold_retrieval.py", "retrieval/features.py", "models/retrieval_lstm_v2.py",
    "pipeline/download_era5_upstream.py",  # its NODES list sets the order of Rg's regional columns
]
EARLY_STOP = ("inner_2y", "val_block")

# Recipe of training/train_lstm.py (v1) -- keep identical.
BATCH_SIZE, MAX_EPOCHS, PATIENCE, LR = 64, 100, 10, 1e-3
REQUIRED_KEYS = {"run_id", "parent", "target", "labels", "anomaly_target", "hot_weight", "early_stop", "seeds", "folds"}
REGISTRY_FIELDS = [
    "timestamp_utc", "run_id", "parent", "fold", "target", "labels", "anomaly_target", "target_form", "retrieval",
    "hot_weight",
    "early_stop", "seeds", "config_sha256", "data_sha256", "code_sha256", "git_commit", "git_dirty",
    "device", "torch_threads", "torch_version", "n_fit_windows", "n_stop_windows", "n_val_windows",
    "val_rmse_all", "val_rmse_extreme", "clim_rmse_all", "clim_rmse_extreme",
    "persist_rmse_all", "persist_rmse_extreme", "skill_clim_all", "skill_clim_extreme",
    "skill_persist_all", "skill_persist_extreme", "wall_seconds",
]


def load_config(path: Path) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = REQUIRED_KEYS - set(cfg)
    if missing:
        raise ValueError(f"config {path} missing keys: {sorted(missing)}")
    if cfg["target"] not in TARGETS or cfg["labels"] not in LABEL_VERSIONS:
        raise ValueError("invalid target or labels in config")
    if not set(cfg["folds"]) <= set(FOLDS):
        raise ValueError(f"folds must be among {list(FOLDS)}")
    if cfg["early_stop"] not in EARLY_STOP:
        raise ValueError(f"early_stop must be one of {EARLY_STOP}")
    if "threads" in cfg and not (isinstance(cfg["threads"], int) and cfg["threads"] > 0):
        raise ValueError("threads must be a positive integer")
    form = cfg.get("target_form", "anomaly" if cfg["anomaly_target"] else "raw")
    if form not in TARGET_FORMS or (cfg["anomaly_target"] and form != "anomaly"):
        raise ValueError(f"target_form must be one of {TARGET_FORMS} and agree with anomaly_target")
    cfg["target_form"] = form
    if "retrieval" in cfg:
        r = cfg["retrieval"]
        if not (isinstance(r, dict) and r.get("mode") in RETRIEVAL_MODES
                and isinstance(r.get("k"), int) and r["k"] > 0 and set(r) <= {"mode", "k"}):
            raise ValueError(f'retrieval must be {{"mode": one of {RETRIEVAL_MODES}, "k": positive int}}')
    return cfg


def set_seed(seed: int) -> None:  # as training/train_lstm.py
    torch.manual_seed(seed)
    np.random.seed(seed)


def weighted_mse(pred, target, hot_mask, hot_weight):  # as training/train_lstm.py
    weight = 1.0 + (hot_weight - 1.0) * hot_mask.float()
    return torch.mean(weight * (pred - target) ** 2)


def train_one_seed(seed: int, tr: SplitArrays, va: SplitArrays, hot_weight: float) -> LSTMForecaster:
    """Mirrors training/train_lstm.py:train_one_seed step for step. The model is fitted on
    `tr`; `va` only chooses the early-stopping checkpoint."""
    set_seed(seed)
    ds = TensorDataset(torch.from_numpy(tr.X), torch.from_numpy(tr.y), torch.from_numpy(tr.hot.astype(np.float32)))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    val_X, val_y = torch.from_numpy(va.X), torch.from_numpy(va.y)
    val_hot = torch.from_numpy(va.hot.astype(np.float32))

    model = LSTMForecaster(n_features=tr.X.shape[2])
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_val, best_state, stale = float("inf"), None, 0
    for _ in range(MAX_EPOCHS):
        model.train()
        for xb, yb, hb in loader:
            optimizer.zero_grad()
            loss = weighted_mse(model(xb), yb, hb, hot_weight)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = weighted_mse(model(val_X), val_y, val_hot, hot_weight).item()
        if val_loss < best_val - 1e-6:
            best_val, stale = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model


def _analogue_batch(pool_X: torch.Tensor, pool_y: torch.Tensor, idx: torch.Tensor):
    """(x_analogues, y_analogues, mask) for a (B, K) block of pool positions (-1 = none)."""
    mask = idx >= 0
    safe = idx.clamp(min=0)
    return pool_X[safe], pool_y[safe], mask


def predict_ra(model: RetrievalAugmentedLSTMv2, X: np.ndarray, idx: np.ndarray, pool_X: torch.Tensor,
               pool_y: torch.Tensor, batch: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Normalised predictions (N, 5) and attention weights (N, K), in eval mode."""
    model.eval()
    preds, atts = [], []
    with torch.no_grad():
        for s in range(0, len(X), batch):
            xa, ya, m = _analogue_batch(pool_X, pool_y, torch.from_numpy(idx[s:s + batch]))
            p, a = model(torch.from_numpy(X[s:s + batch]), xa, ya, m)
            preds.append(p.numpy())
            atts.append(a.numpy())
    return np.concatenate(preds), np.concatenate(atts)


def train_one_seed_ra(seed: int, tr: SplitArrays, va: SplitArrays, hot_weight: float, idx_tr: np.ndarray,
                      idx_va: np.ndarray, pool_X: torch.Tensor, pool_y: torch.Tensor) -> RetrievalAugmentedLSTMv2:
    """train_one_seed with analogues: same recipe (seed handling, Adam, batch, epochs,
    patience, weighted MSE); each window carries the pool positions of its analogues."""
    set_seed(seed)
    ds = TensorDataset(torch.from_numpy(tr.X), torch.from_numpy(tr.y), torch.from_numpy(tr.hot.astype(np.float32)),
                       torch.from_numpy(idx_tr))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    val_y, val_hot = torch.from_numpy(va.y), torch.from_numpy(va.hot.astype(np.float32))

    model = RetrievalAugmentedLSTMv2(n_features=tr.X.shape[2])
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_val, best_state, stale = float("inf"), None, 0
    for _ in range(MAX_EPOCHS):
        model.train()
        for xb, yb, hb, ib in loader:
            optimizer.zero_grad()
            pred, _ = model(xb, *_analogue_batch(pool_X, pool_y, ib))
            loss = weighted_mse(pred, yb, hb, hot_weight)
            loss.backward()
            optimizer.step()
        val_pred, _ = predict_ra(model, va.X, idx_va, pool_X, pool_y)
        val_loss = weighted_mse(torch.from_numpy(val_pred), val_y, val_hot, hot_weight).item()
        if val_loss < best_val - 1e-6:
            best_val, stale = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model


def _positions(sub: SplitArrays, full: SplitArrays) -> np.ndarray:
    """Row positions of sub's windows within full (sub is a subset of full)."""
    pos = pd.DatetimeIndex(full.query_dates).get_indexer(pd.DatetimeIndex(sub.query_dates))
    if (pos < 0).any():
        raise ValueError("subset windows not found in the full split")
    return pos


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _code_sha256() -> str:
    """Hash of CODE_FILES (line endings normalised, so a checkout's autocrlf doesn't matter)."""
    h = hashlib.sha256()
    for rel in CODE_FILES:
        h.update(rel.encode() + b"\0" + (REPO_ROOT / rel).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def _git_dirty() -> str:
    """"True" if any hashed code file differs from HEAD or is untracked, "False" if clean,
    "unknown" if git can't be run (never report clean without checking)."""
    if not _git("rev-parse", "HEAD"):
        return "unknown"
    return str(bool(_git("status", "--porcelain", "--", *CODE_FILES)))


def _rmse_by_seed(df: pd.DataFrame) -> float:
    """Mean over seeds of the per-seed RMSE (NaN if df is empty)."""
    if df.empty:
        return float("nan")
    return float(df.groupby("seed")["error"].apply(lambda e: np.sqrt(np.mean(e.to_numpy() ** 2))).mean())


def _reference_rmse(split: SplitArrays) -> dict[str, float]:
    """Climatology and persistence RMSE on the same windows and target (all / extreme)."""
    ext = split.stratum == "extreme"
    y = split.y_raw.astype(np.float64)
    out = {}
    for name, pred in (("clim", split.clim_target), ("persist", np.repeat(split.persist[:, None], y.shape[1], axis=1))):
        err = pred - y
        out[f"{name}_rmse_all"] = float(np.sqrt(np.mean(err**2)))
        out[f"{name}_rmse_extreme"] = float(np.sqrt(np.mean(err[ext] ** 2))) if ext.any() else float("nan")
    return out


def fit_and_stop_sets(data: FoldData, early_stop: str) -> tuple[SplitArrays, SplitArrays]:
    """(arrays to fit on, arrays that choose the early-stopping checkpoint)."""
    if early_stop == "val_block":
        return data.train, data.val
    return inner_split(data.train, data.train_end, years=2)


def _append_registry(row: dict) -> None:
    """Append one row. If the file was written with an older column set, it is rewritten
    with the union of columns first (old rows get blanks), so columns never misalign."""
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    if REGISTRY.exists():
        with open(REGISTRY, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), [])
        if header != REGISTRY_FIELDS:
            old = pd.read_csv(REGISTRY, dtype=str, keep_default_na=False)
            extra = [c for c in old.columns if c not in REGISTRY_FIELDS]
            if extra:
                raise ValueError(f"registry has unknown columns {extra}; refusing to drop them")
            old.reindex(columns=REGISTRY_FIELDS, fill_value="").to_csv(REGISTRY, index=False)
    new = not REGISTRY.exists()
    with open(REGISTRY, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=REGISTRY_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def run(cfg: dict, folds: list[str] | None = None, seeds: list[int] | None = None,
        out_dir: Path = PRED_DIR, model_dir: Path = MODEL_DIR, write_registry: bool = True,
        threads: int | None = None) -> dict[str, pd.DataFrame]:
    """Train every (fold, seed); write predictions, checkpoints and registry rows. Torch
    threads: `threads`, else the config's "threads", else unchanged; restored afterwards."""
    folds = folds or cfg["folds"]
    seeds = seeds if seeds is not None else cfg["seeds"]
    n_threads = threads or cfg.get("threads")
    prev_threads = torch.get_num_threads()
    if n_threads:
        torch.set_num_threads(n_threads)
    try:
        return _run(cfg, folds, seeds, out_dir, model_dir, write_registry)
    finally:
        torch.set_num_threads(prev_threads)


def _data_files(cfg: dict) -> list[Path]:
    """Data files a run reads; their hashes form the registry data_sha256."""
    files = [DAILY_V2_PATH, WINDOW_INDEX] + ([LILJEGREN_DAILY_PATH] if LILJEGREN_DAILY_PATH.exists() else [])
    if cfg["labels"] == "wbgt":
        files.append(WBGT_LABEL_CONFIG)
    if (cfg.get("retrieval") or {}).get("mode") == "region":
        files.append(UPSTREAM_PATH)  # Rg reads the upstream dataset
    return files


def _run(cfg: dict, folds: list[str], seeds: list[int], out_dir: Path, model_dir: Path,
         write_registry: bool) -> dict[str, pd.DataFrame]:
    cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    data_hash = hashlib.sha256("".join(_sha256_file(p) for p in _data_files(cfg)).encode()).hexdigest()
    code_hash, commit, dirty = _code_sha256(), _git("rev-parse", "HEAD"), _git_dirty()
    results = {}
    for fold in folds:
        t0 = time.time()
        data = build_fold(fold, cfg["target"], cfg["labels"], target_form=cfg["target_form"])
        fit, stop = fit_and_stop_sets(data, cfg["early_stop"])
        ret = cfg.get("retrieval")
        if ret:
            retriever = FoldRetriever(fold, cfg["labels"], cfg["target"])
            if not pd.DatetimeIndex(retriever.train_w["query_date"]).equals(pd.DatetimeIndex(data.train.query_dates)):
                raise ValueError("retrieval pool and training windows disagree")
            pool_X, pool_y = torch.from_numpy(data.train.X), torch.from_numpy(data.train.y)
            pos_fit, pos_stop = _positions(fit, data.train), _positions(stop, data.train)
            fixed = None  # sim / time retrieval does not depend on the seed: compute once
        frames, an_frames = [], []
        for seed in seeds:
            ckpt = model_dir / cfg["run_id"] / fold / f"seed_{seed}" / "checkpoint.pt"
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            if ret:
                if ret["mode"] in RANDOM_MODES or fixed is None:
                    s = seed if ret["mode"] in RANDOM_MODES else None
                    fixed = (retriever.retrieve(data.train.query_dates, ret["mode"], ret["k"], seed=s),
                             retriever.retrieve(data.val.query_dates, ret["mode"], ret["k"], seed=s))
                r_tr, r_va = fixed
                model = train_one_seed_ra(seed, fit, stop, float(cfg["hot_weight"]), r_tr.idx[pos_fit],
                                          r_tr.idx[pos_stop], pool_X, pool_y)
                torch.save(model.state_dict(), ckpt)
                pred_norm, att = predict_ra(model, data.val.X, r_va.idx, pool_X, pool_y)
                pred = data.to_raw(pred_norm, "val")
                k = r_va.idx.shape[1]
                an_frames.append(pd.DataFrame({
                    "seed": seed, "query_date": np.repeat(pd.DatetimeIndex(data.val.query_dates), k),
                    "rank": np.tile(np.arange(1, k + 1), len(r_va.idx)),
                    "analogue_query_date": np.where(r_va.idx >= 0, retriever.cand_dates.values[np.maximum(r_va.idx, 0)],
                                                    np.datetime64("NaT")).ravel(),
                    "similarity": r_va.sim.ravel(), "attention": att.ravel()}))
            else:
                model = train_one_seed(seed, fit, stop, float(cfg["hot_weight"]))
                torch.save(model.state_dict(), ckpt)
                model.eval()
                with torch.no_grad():
                    pred = data.to_raw(model(torch.from_numpy(data.val.X)).numpy(), "val")  # float32, as v1
            frames.append(long_frame(cfg["run_id"], seed, data.val.query_dates, pred, data.val.y_raw, data.val.stratum))
            print(f"  [{cfg['run_id']} {fold}] seed {seed}: val RMSE {np.sqrt(np.mean(frames[-1]['error'] ** 2)):.4f}", flush=True)
        df = pd.concat(frames, ignore_index=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / cfg["run_id"]).mkdir(parents=True, exist_ok=True)
        if an_frames:  # written first, so a complete {fold}.parquet implies its analogues exist
            pd.concat(an_frames, ignore_index=True).to_parquet(out_dir / cfg["run_id"] / f"{fold}_analogues.parquet",
                                                               index=False)
        df.to_parquet(out_dir / cfg["run_id"] / f"{fold}.parquet", index=False)
        results[fold] = df

        by_seed_all = _rmse_by_seed(df)
        by_seed_ext = _rmse_by_seed(df[df["stratum"] == "extreme"])
        ref = _reference_rmse(data.val)
        skill = {
            f"skill_{name}_{part}": 1.0 - model_rmse / ref[f"{name}_rmse_{part}"]
            for name in ("clim", "persist")
            for part, model_rmse in (("all", by_seed_all), ("extreme", by_seed_ext))
        }
        if write_registry:
            _append_registry({
                "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "run_id": cfg["run_id"], "parent": cfg["parent"], "fold": fold, "target": cfg["target"],
                "labels": cfg["labels"], "anomaly_target": cfg["anomaly_target"], "target_form": cfg["target_form"],
                "retrieval": f"{ret['mode']} k={ret['k']}" if ret else "",
                "hot_weight": cfg["hot_weight"],
                "early_stop": cfg["early_stop"], "seeds": " ".join(map(str, seeds)), "config_sha256": cfg_hash,
                "data_sha256": data_hash, "code_sha256": code_hash, "git_commit": commit, "git_dirty": dirty,
                "device": "cpu", "torch_threads": torch.get_num_threads(), "torch_version": torch.__version__,
                "n_fit_windows": len(fit.query_dates), "n_stop_windows": len(stop.query_dates),
                "n_val_windows": len(data.val.query_dates),
                "val_rmse_all": round(by_seed_all, 6), "val_rmse_extreme": round(by_seed_ext, 6),
                **{k: round(v, 6) for k, v in {**ref, **skill}.items()},
                "wall_seconds": round(time.time() - t0, 1),
            })
        print(f"[{cfg['run_id']} {fold}] {len(seeds)} seeds: mean val RMSE {by_seed_all:.4f} "
              f"(extreme {by_seed_ext:.4f}); skill vs clim {skill['skill_clim_all']:+.3f}, "
              f"vs persistence {skill['skill_persist_all']:+.3f}; {time.time() - t0:.0f}s", flush=True)
    return results


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Train an LSTM variant on rolling-origin folds.")
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--folds", nargs="+", choices=list(FOLDS))
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--threads", type=int, default=None,
                    help="torch threads (default: the config's 'threads', else half the cores)")
    args = ap.parse_args(argv)
    import os
    cfg = load_config(args.config)
    threads = args.threads or cfg.get("threads") or max(1, (os.cpu_count() or 2) // 2)
    run(cfg, args.folds, args.seeds, threads=threads)


if __name__ == "__main__":
    main()
