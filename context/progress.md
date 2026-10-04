# Progress Tracker

Short, frequently-updated. See `build_plan.md` for full milestone detail.

## Done
- P1 data pipeline fully verified.
- P2 (baseline forecaster) closed — canonical weighted-MSE LSTM beats
  persistence on all 3 severity strata; stratified RMSE established as the
  headline metric.
- **P3 (retrieval system) COMPLETE:**
  - `retrieval/features.py` — 17-dim statistical feature vector per window
  - `retrieval/build_index.py` — FAISS index over all 17,025 windows
  - `retrieval/query.py` — `query_analogues(query_date, k)`, 3-rule
    eligibility filter + 2-rule dedup (episode cap + temporal spacing)
  - `tests/test_retrieval_eligibility.py` — 6/6 passing
  - Manual inspection: plausible, season-matched, diverse analogues for
    real heatwave query dates
  - Two findings flagged (not blocking): dedup gap found+fixed;
    "heatwave" episodes skew winter, not summer — see `decisions.md`

## Next (plan v5 — see `docs/PLAN_REVIEW_v5.md`)
- Step 5 (retrieval-augmented LSTM) HAS been run: 5 seeds on Colab, val split only,
  hot_weight=20 (same as canonical baseline). Results in
  `evaluation/retrieval_augmented/`.
- Week 1 (in progress, 2026-10-04): venv + pinned requirements (done); v1 backup in
  `archive_v1/`; frozen window index `splits/window_index_v1.parquet`; data
  checksum manifest; `evaluation/stats.py` (cluster-jackknife t-test + Diebold-Mariano).

## Blocked / waiting on teammates (asked 2026-10-04)
These gitignored v1 artifacts are missing locally. Restoring them beats retraining
(retraining in the new venv won't reproduce the metrics bit-for-bit):
1. `data/raw/era5_monthly/`: raw hourly JSON, 9 cells. Avoids a 2–3 day re-download.
2. `models/baseline_lstm/seed_0..4/checkpoint.pt`: canonical hw=20 LSTM (run A1).
3. `models/retrieval_augmented/seed_0..4/checkpoint.pt`: RA-v1 from Colab.

If restored: run `training.evaluate_lstm` and compare to `archive_v1/evaluation/`.
If they match (~4 decimals), freeze to `models/frozen/lstm_tmax_v1/`. If unavailable,
retrain A1 and record the differences.

## Last verified state (2026-10-04)
- `pytest` → 66 passed (leakage 3, retrieval eligibility 6, manifest 11, stats 46).
- `retrieval/candidates.parquet` + `retrieval/faiss_index.bin` rebuilt in the
  new venv; `feature_normalization_stats.json` reproduced byte-identically.
- v1 frozen: `data/MANIFEST.json` (25 files), window index
  `splits/window_index_v1.parquet` (all training code reads it via
  `training.data._load_windows`), `prepare_datasets.py` refuses to overwrite
  frozen data without `--force`.
- `evaluation/stats.py`: paired cluster-jackknife t-test (primary) + Diebold-Mariano
  (secondary), calibrated on simulated overlapping-window data (3.5-6.4% false
  positives at nominal 5%). Replaces the window-level bootstrap.

## Known limitation found 2026-10-04
- `archive_v1/predictions/retrieval_augmented/val_predictions.parquet` holds ONLY
  seed 4 (the last seed `train_retrieval_lstm.py` trained), not a seed average:
  RA RMSE ranges 2.339-2.549 across seeds. Seed-aware comparisons need per-seed
  predictions; fix when provenance logging is added (plan v5, Week 1).
- Val (2016-18) has only 9 year x season clusters; extreme-stratum val
  comparisons have ~5. Headline inference must use rolling-origin folds.
- `process_weather.py` has no freeze guard (only `prepare_datasets.py` does): a
  re-run would overwrite `data/processed/weather_daily.parquet`. The manifest test
  would catch it afterwards; it also needs the missing raw data to run at all.

## 2026-09-22 update
- Canonical LSTM changed again: hot_weight 15 -> 20, after user correctly
  flagged the 15->20 marginal trade-off was better than 10->15 (see
  decisions.md for the full marginal analysis). Current canonical numbers:
  extreme RMSE=1.285, MAE=1.775, recall=0.339.
- Bootstrap CI on the 15-vs-20 extreme-RMSE gap was later run: CI includes zero
  (`evaluation/hw_sweep/bootstrap_ci_15_vs_20.json`, see decisions.md). Note it
  used a window-level bootstrap, now known to give CIs that are too narrow.
