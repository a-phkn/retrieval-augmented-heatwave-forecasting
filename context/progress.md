# Progress Tracker

Short, frequently-updated. See `build_plan.md` for full milestone detail.

## Done
- P1 data pipeline fully verified (not just read — executed): weather_daily
  parquet, event catalogue, chronological split, forecast windows, leakage
  tests (3/3 passing).
- Investigated and resolved (no code change) the 2019-zero-episodes question
  — see `decisions.md`.
- `context/` folder created and populated from verified facts.
- **P2 / Step 3 (baseline forecaster) CLOSED for now:**
  - `training/data.py`, `training/baselines.py`, `models/lstm.py`,
    `training/train_lstm.py` — canonical weighted-MSE LSTM, 5 seeds, all
    converged, checkpoints saved.
  - `training/evaluate_lstm.py` + `training/visualize_results.py` —
    narrowed to exactly 4 metrics after the "which metric matters"
    decision: global MAE/RMSE, stratified RMSE (headline), detection
    precision/recall/F2, bias diagnostic. Both files handed to the user
    directly.
  - `training/data.py`'s `build_split_target_stratum` — normal/unusual/
    extreme stratification, per the roadmap's own Section 4 definition.
  - **Headline result:** canonical LSTM beats persistence on ALL THREE
    strata, including extreme (RMSE 1.599 vs. persistence's 2.179, ~27%
    better). Climatology collapses on extreme (RMSE 5.031) despite a
    competitive global MAE — good illustrative panel.
  - **Explored and explicitly rejected:** a dual-head (focal-loss) model
    that hit a 75-80% recall target but was *worse* than canonical on
    extreme-stratum RMSE (bootstrap CI excludes zero) — canonical remains
    the baseline. Full reasoning in `decisions.md`/`ml_notes.md`.
  - `requirements.txt` added (pandas, pyarrow, pytest, torch).
- User may add further baseline model comparisons (e.g. gradient-boosted
  trees, plain feedforward net) in a later session — P2 is closed but not
  permanently sealed.

## Next
- P3 (retrieval system) — ready to start whenever the user is.
- When P3 starts: Step 4 (retrieval system) is independent of Step 3's
  output beyond needing the same `forecast_windows.parquet`/
  `all_daily.parquet` P1 artifacts already verified. Step 5 (joint
  integration) will need the LSTM baseline's encoder design from
  `models/lstm.py` as the starting point for the attention-fusion
  architecture, and MUST use the same weighted-loss objective
  (hot_weight=10) as the canonical baseline to keep the "same training
  procedure, retrieval on/off only" comparison valid.
- The bar RAG needs to clear: extreme-stratum RMSE of 1.599 (canonical
  LSTM, val split) — not a recall percentage.

## Blocked / waiting on user
- Nothing currently blocking Step 3.
- Step 5 (joint integration) blocked on P3, which is deferred by choice, not
  by a technical blocker.

## Last verified state
- Repo commit at session start: `c5d4cc0423260509ef33c31107c3784fc356aa55`.
- `pytest tests/test_no_leakage.py` → 3 passed, 0 failed (run this session).