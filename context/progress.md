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
- `pytest` → 323 passed (2026-10-08, after the review additions). 315 earlier that day. 272 passed (2026-10-07). Earlier: 196 passed (2026-10-06; leakage 3, retrieval eligibility 6, manifest 12, stats 46, predict_v1 8, hourly features 16, downloader v2 13, upstream downloader 11, compare v2 13, wbgt liljegren 11, wbgt label 5, labels v2 13, folds 27, trainer 12).
- `retrieval/candidates.parquet` + `retrieval/faiss_index.bin` rebuilt in the
  new venv; `feature_normalization_stats.json` reproduced byte-identically.
- v1 frozen: `data/MANIFEST.json` (30 files incl. A1 checkpoints + raw-data fingerprint), window index
  `splits/window_index_v1.parquet` (all training code reads it via
  `training.data._load_windows`), `prepare_datasets.py` refuses to overwrite
  frozen data without `--force`.
- `evaluation/stats.py`: paired cluster-jackknife t-test (primary) + Diebold-Mariano
  (secondary), calibrated on simulated overlapping-window data (3.5-6.4% false
  positives at nominal 5%). Replaces the window-level bootstrap.

## Week 3 results (2026-10-05)
- **G2** (`evaluation_v2/week3_controls.md`): out-of-fold 2007-2018, 10 seeds. Delta vs damped persistence on all days:
  A1′ +0.107 (fails the floor), A2 −0.043 (n.s.), A2r −0.056 (p=0.024). All beat climatology and are much better on extremes.
  The 10-seed ensemble gives A2 −0.074, A2r −0.077 (both significant) and A1′ +0.070.
  A1′ over-forecasts heatwaves (~1,200 forecast hot days/seed vs ~730 observed; +1.66 °C bias on them).
  The control choice is held until the improvement candidates have run: ensemble, hot_weight sweep, Tmax anomaly target, damped-persistence residual.
- **Improvement runs done (2026-10-06)**: all 15 pre-registered runs, 10 seeds × 4 folds, 72 registry rows (18 runs × 4 folds).
  - hot_weight is the only lever: lower = better all days, worse extremes. The anomaly target (Tmax) and the dp-residual did not help.
  - **Controls (decision 2026-10-06, disclosed deviation): `A1prime_hw5` (Tmax), `A2Lr_hw5` (WBGT physical).**
    - The pre-registered rule alone picks hw1 (`A1prime_hw1`, `A2L_hw1`). The WBGT hw1 model forecasts 0 hot days.
    - hw5: ties damped persistence on all days (−0.009 / +0.004), beats it on extremes (−0.93 / −1.20 °C).
  - Queue note: the 2 h background cap stopped A2L_hw5 mid-fold. Its partial outputs and 3 registry rows were removed and the run was redone in full.
- **Part D (2026-10-06)** verified against saved documents: IMD "Hot & Humid Weather" (FAQ p.2, qualitative);
  current IMD warning lead time = 7 days since July 2023 (PIB 26 Jul 2023), not 5 as Cowork's result file says
  (its file is stale vs its own JSON: 51 vs 58 rows); D1 none found (limitation stands). Fixes sent back to Cowork.
- **Retrieval rungs (2026-10-06)**: `retrieval/fold_retrieval.py` (per-fold pool, features, normalisation) reproduces
  v1's analogues exactly on f4 / v1 labels (1,092/1,092 val queries). It also found that v1's precompute
  truncated 881 training queries (1980-88): 500-candidate screen, no fallback.
  - `models/retrieval_lstm_v2.py`: zero context when there is no analogue.
  - Trainer `retrieval` config key; 6 runs queued, ~37 min each; log in scratchpad `week3_retrieval.log`.
  - G3 rule pre-registered in decisions.md before the runs.
- **G3 result (2026-10-07): none.** All 8 retrieval runs done (R1-rand = mode `time_rand`, added after the
  first 6; their 192 analogue-list fingerprints re-checked unchanged). Commits 312ec5f, 5a3cb99 (code hashes
  match the registry). No rung beats its random control on extreme days; Tmax rungs are slightly worse than the
  control, WBGT rungs (and the random ones) slightly better but not significantly. Detectable effect about 0.24 °C
  (Tmax) and 0.08-0.12 °C (WBGT). Independently reviewed and signed off with reporting fixes (applied). Details in
  decisions.md. Next: Week 4 R2-R4 + mechanism metrics.
- **Retrieval information check (2026-10-07)**: `evaluation/retrieval_information_check.py`. On held-out training
  years, adding the analogues' outcomes to a query-only linear forecast changes RMSE by ~0.000 °C (both families,
  both rungs): the analogues are redundant with the query's own inputs. Tuning the fusion is not worth it.
- **Rg (2026-10-07, post-G3, pre-registered + training-years screen)**: regional-pattern matching. WBGT: all days
  -0.044 °C (p = 0.003), beats R0-rand on all days (p < 0.001); extreme vs R0-rand p = 0.114 -> G3 still none.
  Tmax: no detectable change. Reviewed, signed off with changes (applied). G-R* will test Rg on the graph backbone.
- **Week 4 (2026-10-07)**: physics head decided (exact Liljegren, per-cell at each cell's own peak hour; D3).
  Graph trainer built (`training/graph_data.py`, `backbone` key in the trainer); G-D4 PASS (3.2 min per C4
  seed-fold); graph queue of 10 runs (C2, C3, C4, C4a, U1 x 2 families) running fold by fold, code b5e778d6...
  R2-R4 screen (`evaluation/retrieval_ladder_screen.py`): R2 passes, R3/R4 no; R2 trains after the graph queue.
- **G-D3 (2026-10-08): U1** (`evaluation_v2/graph_gates.md`, commit 8749dc6). All graph configs pass G-D2 and (a);
  C3 fails (b) for WBGT (+0.076 [+0.042, +0.110]). Edges hurt (C3/C4/C4a vs C2 +0.08 to +0.10); edgeless C2 is best.
  User told first, as agreed. Tuning round running (`training/tune_backbone.py`, 48 configs in `configs/tuning/`,
  selection on inner blocks only, registry `registry/tuning_runs.csv`) with the post-result arm C3-pool
  (decisions.md 2026-10-08), then the 6 tuned 10-seed runs, then PH_lstm + R2 (patches applied 2026-10-08).
  Queue log: scratchpad `week5.log`. The first tuning combo reproduces original C3 (seed 0 f1 2.1386 both).
  Afterwards: `python -m evaluation.graph_gates --tuned` (amended rule: provisional pick, user approves).
- **ML review (2026-10-08)**, read-only reviewer; decisions in decisions.md. Added: `evaluation/ridge_baselines.py`
  (G-D0 ridge reproduced 2.104 / 2.217; hot-weighted ridge), `evaluation/calibration_check.py` (bias split,
  inner-block recalibration from checkpoints, ensemble, hot-day Brier; every checkpoint reproduces its saved
  forecasts; inner caches in predictions_v2/inner, regenerable), selection-free-years tests and the amended rule
  in `evaluation/graph_gates.py`, `docs/CONFIRMATORY_PROTOCOL.md` (draft). PH_lstm + R2 held (queue file in
  scratchpad `week4_ph_r2_queue.held.txt`) until the physics NaN guard / smooth clamps are applied after the
  tuned runs.
- **C3-hop (2026-10-08)**: `models/dstgnn_multihop.py` + 7 tests (not yet in CODE_FILES, so the running queue's hash
  is unchanged). Trainer hook, physics fixes and C3-hop configs are prepared patches in scratchpad `post_queue/`
  (RUNBOOK.md), dry-run tested on a scratch copy of the repo; applied after the tuned runs.
- **DSTGNN skeleton (2026-10-06)**: `models/dstgnn.py` (graph GRU, modes none/static/dynamic + adaptive),
  `pipeline/graph.py` (Delhi + 27 nodes, geographic and wind-gated advective edges). G-D1: 13 tests pass.
  The plan said `data/graph.py`; it is `pipeline/graph.py` because `data/` holds raw data.
- **Upstream + G-D0 (2026-10-07)**: raw download 1269/1269 verified (all days, no NaN, grid/timezone/elevation
  OK; 29 tiny negative soil-moisture values clipped). `pipeline/build_upstream_daily.py` ->
  `datasets_v2/upstream_daily.parquet` (17,051 x 324, raw fingerprint in `.meta.json`).
  G-D0 (`evaluation/gd0_upstream_signal.py`, rule pre-registered first): **PASS**, Tmax leads 1-3 Δ -0.111
  [-0.144, -0.077]; west and south-west points strongest; best lag 1 day almost everywhere (no
  travel-time pattern in the linear check).
- **Physics-head design study (2026-10-07)**: `evaluation/physics_head_standin.py`.
  - The stand-in reproduces the exact Liljegren formula with RMSE 0.12 overall but 0.40 at hot-day peak hours,
    so the evidence favours exact.
  - Averaging the 9 cells' ingredients first loses 1.08 °C (bias -0.88) on hot days even with the exact formula,
    which argues for a per-cell head or a learned correction.
  - Decision deferred to Week 4 (user).
- **CI (2026-10-06)**: commits ba6659b and 7523bd6 failed 2 tests on GitHub only, from runner CPU float differences:
  - the index rebuild rewrote the frozen stats file in the last digits;
  - A1 bit-for-bit retraining.
  Fix: CI checks the rebuilt stats to 1e-9 and then restores the committed file; the bit-for-bit test runs locally only.
- **Liljegren WBGT** (`pipeline/wbgt_liljegren.py`, `pipeline/build_wbgt_liljegren.py`):
  - Liljegren/Argonne C ported (option B); Kong & Huber (2022) methodology; checked against PyWBGT
    (max diff 0.007 °C on 3,000 real hours) and against the original iteration (within 0.02 K).
  - BoM is +2–3 °C in Jun–Sep; corr 0.74 Apr–Sep.
  - Heat-season trend: physical WBGT +0.32 °C/decade [0.16, 0.48], BoM +0.375, Tmax +0.14 (n.s.).
- **WBGT label:** season Mar 15–Sep 30, 97.5th percentile of BoM (≈37.7), 28 pooled episodes
  (`configs/wbgt_label.json`). Physical-WBGT sensitivity: 95th pct, 29 episodes.
  Tmax heatwaves fall in Apr–Jun and BoM-WBGT heatwaves in Jun–Aug; only 27 days overlap.
- **Decided 2026-10-05:** the WBGT models forecast the physical WBGT with the WBGT label
  (95th pct, 29 episodes); see `decisions.md`.
- **Pre-registered improvement queue** (`configs/week3_queue.txt`): 15 runs started 23:49.
  The queue is resumable (it skips runs with all 4 folds done); the background job cap is 2 h,
  so relaunch the same loop if it is cut off.
- **Cowork Part D requested** (brief): humid-heat events before 2019, IMD humid-heat criteria,
  IMD lead time, report housekeeping.
- **Open:** advisory tier mapping (user, later).

## Week 3 data (2026-10-05)
- **v2 download verified:**
  - 5,049/5,049 files, correct coordinates, IST, 409,224 hours per cell, no duplicates;
  - physical checks pass (no night radiation, dew point ≤ T, valid ranges, monsoon dew-point peak);
  - wind speed identical to v1 on all 3.68M hours, so there were no ERA5 revisions.
- **`data/raw/legacy_full_range/`** was made on 2026-10-04 during the raw-data reorganisation. It holds old single-request cell_1..6 files from the user's upload. It is unused and gitignored; whether to keep it is the user's call.
- **Upstream downloader** `pipeline/download_era5_upstream.py`:
  - 27 daily nodes, 10 variables (names probed against the live API);
  - 11 offline tests;
  - one real node-year (n26e072, 2026) downloaded and validated;
  - the user runs the rest (~3.3 days of quota).
- **Cowork Phase-6 sources reviewed:** 135 actions and 46 thresholds.
  - All 136 text-layer quotes were confirmed on the cited page from a fresh pdftotext extraction.
  - The 45 image-based quotes were spot-checked on 2 page images.
  - Issues went back to Cowork (Part C since done, see the Week 2 status below): 123/135 actions have no alert tier; two conflicting colour systems; 4- vs 5-day lead time; no Indian WBGT limit; energy is PARTIAL; medical-guidance conflicts.
  - Copyrighted non-government PDFs (ISO sample, Springer) must not be committed.

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
  2010 and 2015. Sourced 2026-10-05 (Part C): 1998 and 2010 are IMD sub-division/regional
  spells, 2015 is the IMD summary plus station data, and 2002 is station-only. None is an
  IMD Delhi declaration. Windows were updated to the sources; the labels flag all four.
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
- **BoM "WBGT" is a T–humidity index:** ~6 °C above a shade WBGT in July (superseded 2026-10-05: vs the physical Liljegren WBGT it is +2–3 °C in Jun–Sep, ≈0 otherwise). It is renamed
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
