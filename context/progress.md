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

## Missing v1 artefacts: resolved 2026-10-04 (details in `docs/REPRODUCIBILITY_LOG.md`)
1. `data/raw/era5_monthly/`: RESTORED (original download, 9 cells × 561 months).
   Re-processing reproduces `weather_daily.parquet` exactly.
2. `models/baseline_lstm/` (A1): no teammate had it, so it was RETRAINED locally with the
   unchanged script. It reproduces the archived evaluation to 1e-7. Frozen at
   `models/frozen/lstm_tmax_v1/` (committed).
3. `models/retrieval_augmented/` (RA-v1): the originals weren't recoverable, so it was
   RETRAINED locally on CPU with the unchanged script. Aggregates are within one seed-SD
   of the archived Colab run. Frozen at `models/frozen/ra_lstm_v1/`. Per-seed predictions
   + analogue provenance come from the new `evaluation/predict_v1.py` (no training code
   changed).

## Last verified state (2026-10-04)
- `pytest` → 145 passed (leakage 3, retrieval eligibility 6, manifest 12, stats 46, predict_v1 8, hourly features 16, downloader v2 13, labels v2 13, folds 19, trainer 9).
- `retrieval/candidates.parquet` + `retrieval/faiss_index.bin` rebuilt in the
  new venv; `feature_normalization_stats.json` reproduced byte-identically.
- v1 frozen: `data/MANIFEST.json` (30 files incl. A1 checkpoints + raw-data fingerprint), window index
  `splits/window_index_v1.parquet` (all training code reads it via
  `training.data._load_windows`), `prepare_datasets.py` refuses to overwrite
  frozen data without `--force`.
- `evaluation/stats.py`: paired cluster-jackknife t-test (primary) + Diebold-Mariano
  (secondary), calibrated on simulated overlapping-window data (3.5-6.4% false
  positives at nominal 5%). Replaces the window-level bootstrap.

## CI added (2026-10-05): `.github/workflows/ci.yml`
- Windows job: pinned lockfile env, rebuild retrieval index, pytest, manifest check.
  Verified on a fresh local clone: 144 passed, 1 skipped (raw-data test), Manifest OK.
- Security job: gitleaks over full history (local run: 15 commits, no leaks) and
  pip-audit on both requirement files (no known vulnerabilities).
- The workflow passes actionlint. Actions are SHA-pinned; gitleaks is checksum-verified.
- A rebuilt analogue table differs from the local one only in float rounding of the
  similarity scores (≤ 2.4e-7) and in 1 analogue at rank 20 (a float tie), so it has no
  effect on the top-5 retrieval used by RA-v1.

## Week 2 status (2026-10-04): v2 data, labels, folds and trainer built
- `datasets_v2/` (daily + per-cell; no climatology/labels stored, as they are fold-specific).
- `pipeline/labels_v2.py`, with an acceptance test on pre-2019 events only: 1998, 2002,
  2010 and 2015 (to be confirmed against IMD records).
- `pipeline/climatology.py` reproduces v1 climatology exactly; the shared episode code
  reproduces v1's 102-episode catalogue exactly.
- `training/folds.py` builds 4 rolling folds. On the primary fold with v1 settings it
  reproduces v1's training arrays bit-for-bit.
- `training/train_unified.py` with `configs/` is in place. Configured as A1, it reproduces
  the frozen A1 weights bit-for-bit (5/5 seeds, 8 threads; a test checks seed 0 exactly).
- Review fixes (critical reviewer, 2026-10-04):
  - **v2 runs early-stop on the last 2 training years** (`early_stop: inner_2y`), not on
    the validation block they are scored on. v1/A1_repro keep `val_block` only to
    reproduce v1.
  - Registry rows carry skill vs climatology and persistence on the same target. Raw RMSE
    of Tmax and WBGT runs is not comparable.
  - Threads are pinned in configs (8). Rows record the code hash and a git-dirty flag.
  - No test reads 2019+ rows.
  - A full perturbation-leakage test covers all folds.
- Sanity run (1 seed, f4, not a result): A1′ RMSE 2.18 (skill vs climatology +0.07); A2r
  WBGT RMSE 1.52 (+0.26).
- The v2 download is running on the user's own terminal (not needed until Week 3–4).
- Next: run A1′, A2, A2r (10 seeds × 4 folds), compare against damped persistence, then the
  hot_weight re-sweep and the WBGT percentile label.

## Week 1 results (2026-10-04): what the premise checks found
Reports: `evaluation_v2/g0_val_comparison.md`, `premise_report.md`, `mde.md`.
- **Damped anomaly persistence is the strongest model on val.** It uses 5 coefficients fitted
  on train and scores RMSE 2.134, against climatology 2.344, A1 LSTM 2.401 and RA-v1 2.420.
  - Both neural models lose to it overall (+0.27 / +0.29 °C, p ≤ 0.003) and are worse than
    climatology from lead 3 onwards.
  - They win only on observed unusual and extreme days (the hot_weight=20 warm bias; the
    forecaster's dilemma).
  - => Credibility floor proposal: no worse than damped persistence overall; A2r
    (anomaly target) is essential.
- **Analogue ensemble (no training):** ≈ climatology overall (p = 0.76) and better on extreme
  days; its skill sits in leads 1–2. Retrieved analogues beat random past windows, but that
  mostly measures the noise of random draws; the honest control is climatology.
- **Trends (train years only):** no detectable Tmax trend. Humid heat rises: BoM index
  +0.38, wet-bulb Tw +0.35, Heat Index +0.47 °C/decade (all CIs exclude 0).
  Caveat: ERA5 humidity homogeneity is unverified (a step around 2000–01); check against
  station data.
- **BoM "WBGT" is a T–humidity index:** ~6 °C above a shade WBGT in July. It is renamed
  `wbgt_bom_*` and gets no absolute thresholds; validate against Liljegren once the v2
  download (radiation) exists.
- **Retention:** similarity barely ranks analogues within the top 20 (mean ρ +0.034, CI
  [+0.004, +0.064]); the similarity range there is narrow (0.94 → 0.84).
- **MDE (80% power):** all days 5–6% (≈0.12–0.14 °C) with 24–36 clusters; extreme days
  25–35% (0.32–0.45 °C) with 10–20 clusters. The seed-noise floor is 0.22–0.26 °C on extreme
  days with 5 seeds => use 10 seeds for decision runs.
- **Downloader v2** (`pipeline/download_era5_v2.py`) is ready, verified against the live API
  and tested offline. The full run hasn't started yet (about 1.1 days of free API quota).

## G0 result (2026-10-04): `evaluation_v2/g0_val_comparison.md`
- **Retrieval vs no retrieval (RA-v1 vs A1), val, 5 seeds each:**
  - all days +0.018 °C [−0.11, +0.15], p=0.76;
  - extreme days −0.085 °C [−0.58, +0.41], p=0.66, child better in 4/5 clusters.
  - **Inconclusive.** This is the expected starting point (plan v5): don't tune RA-v1 further;
    the retrieval ladder (R1+) starts from here.
- **A1 vs climatology:** worse on normal days (+0.58 °C, 0/9 clusters), better on unusual
  and extreme days, no difference overall (p=0.58). This motivates the anomaly-target
  control A2r.
- RA attention is nearly uniform (mean max weight ≈ 0.21): RA-v1 effectively averages
  its 5 analogues' outcomes.

## Known limitation found 2026-10-04 (RA-v1 model)
- `models/retrieval_lstm.py`: for a query with ZERO eligible analogues, the model spreads
  attention uniformly over zero-padded slots, i.e. fake normalised "0" outcomes. This
  affects 163/13,131 train windows and no val windows.
  - RA-v1 is kept as originally built (frozen reference).
  - The fix (zero context vector for empty retrieval) goes into the new v2 retrieval
    model, not into v1 code.

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
