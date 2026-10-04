# Reproducibility log

One entry per re-run of a frozen v1 artefact: what was re-run, where, what changed.
Newest first.

---

## 2026-10-04: raw data restored, A1 baseline LSTM retrained

**Environment:** Windows 11, local CPU (16 cores; capped at 8 threads), Python 3.13.2,
torch 2.14.1 (CPU), numpy 2.5.3, pandas 3.0.6 (exact set in `requirements.lock.txt`).

### 1. Raw ERA5 data (`data/raw/era5_monthly/`)
- **Source:** the original download, supplied by the user (it was not in git).
- **Contents:** 9 cells × 561 monthly JSON files (1980-01 … 2026-09), 5,049 files.
  Fingerprint in `data/MANIFEST.json` → `raw_inputs`.
- **Check:** `process_weather.py` was re-run on it from a scratch copy, writing outside the
  repo, and compared with the frozen `data/processed/weather_daily.parquet`.
  - Result: 17,051 rows, same columns, same date range (1980-01-01 … 2026-09-06).
  - **All weather columns are identical (max |diff| = 0).** `doy_sin`/`doy_cos` differ
    by ≤ 1.1e-16 (floating-point rounding).
- **Conclusion:** this is exactly the data v1 was built from. No re-download is needed.
- **Also kept:** `data/raw/legacy_full_range/cell_1..6.json` are older single-request
  downloads (hourly, 1980-01-01 … 2026-09-11, cells 1–6 only). They aren't used by any
  code and are kept for reference only.

### 2. Run A1: canonical weighted-MSE LSTM (`training/train_lstm.py`, unchanged)
- **Why:** the original checkpoints were gitignored and nobody on the team still had them.
- **Command:** `python -m training.train_lstm`, then `python -m training.evaluate_lstm`.
  Recipe: hot_weight=20, seeds 0–4, Adam 1e-3, bs 64, patience 10.
  Wall time 121 s.
- **Comparison with `archive_v1/evaluation/baseline_lstm/`:**

| File | Numbers compared | Exact | Max abs diff |
|---|---|---|---|
| `val_extended_metrics.json` (computed from the checkpoints) | 46 | 34 | **1.2e-7** |
| `val_metrics.json` (training log) | 31 | 19 | 1.2e-3 (seed 1 only) |
| `normalization_stats.json` | byte hash | identical | 0 |

- **Interpretation:**
  - The headline evaluation reproduces to float32 noise: global RMSE 2.40123 vs 2.40123,
    and identical stratified RMSE and detection metrics.
  - In the training log, seeds 0, 2, 3 and 4 are bit-identical, with the same epochs and
    losses.
  - Seed 1 differs only in the *archived training log* (val RMSE 2.31974 vs 2.32095). The
    archived evaluation file matches the new run, not that log. So the archived log came
    from a different run than the checkpoints that were actually evaluated.
  - The new `val_metrics.json` is consistent with the evaluation, which the old pair was not.
- **Frozen as A1:** `models/frozen/lstm_tmax_v1/seed_0..4/checkpoint.pt` (1.1 MB,
  committed, fingerprinted in `data/MANIFEST.json`).

### 3. Run RA-v1: retrieval-augmented LSTM (`training/train_retrieval_lstm.py`, unchanged)
- **Why:** the original checkpoints, trained on a Colab GPU, weren't recoverable, and the
  archived `val_predictions.parquet` holds seed 4 only.
- **Steps:**
  1. Rebuilt the gitignored retrieval artefacts with `python -m retrieval.build_index`
     (`feature_normalization_stats.json` came out byte-identical) and
     `python -m retrieval.precompute_analogues` (301,461 query–analogue pairs; its
     built-in check against `retrieval.query` passed).
  2. Ran `python -m training.train_retrieval_lstm` on the **local CPU** (8 threads):
     5 seeds in 352 s.
- **Not bit-identical, as expected:** GPU and CPU floating-point arithmetic differ, so early
  stopping lands on different epochs (e.g. seed 0: 17 → 15 epochs). Every aggregate stayed
  within one seed-SD:

| Metric (val) | Archived (Colab GPU) | Retrained (CPU) |
|---|---|---|
| Global RMSE | 2.442 ± 0.067 | 2.420 ± 0.037 |
| Global MAE | 1.830 ± 0.048 | 1.809 ± 0.030 |
| Extreme RMSE | 1.231 ± 0.116 | 1.201 ± 0.095 |
| Unusual RMSE | 1.677 ± 0.094 | 1.713 ± 0.084 |
| Normal RMSE | 2.571 ± 0.086 | 2.543 ± 0.051 |
| Recall / precision / F2 | 0.404 / 0.245 / 0.358 | 0.401 / 0.252 / 0.358 |

- **Frozen as RA-v1:** `models/frozen/ra_lstm_v1/seed_0..4/checkpoint.pt` (1.3 MB,
  committed, fingerprinted).
- **Overwritten tracked files** (originals kept in `archive_v1/`):
  `evaluation/retrieval_augmented/*.json` and
  `predictions/retrieval_augmented/val_predictions.parquet`. The latter is again a single
  seed (the script's last-seed limitation), so use `predictions_v1/val/ra_v1.parquet`
  for all 5 seeds.

### 4. Per-seed predictions (`evaluation/predict_v1.py`, new; no training code changed)
- Generated from the frozen checkpoints into `predictions_v1/val/`.
- They reproduce `evaluate_lstm` / `evaluate_retrieval_lstm` / the baselines to ≤1e-6:
  - A1 RMSE 2.401233, normal/unusual/extreme 2.523807/1.681848/1.285354;
  - RA 2.419582;
  - persistence 2.61508, climatology 2.344342.
- **Lead alignment:** lead 1 equals `forecast_start` and lead 5 equals `forecast_end` for
  every val window.
