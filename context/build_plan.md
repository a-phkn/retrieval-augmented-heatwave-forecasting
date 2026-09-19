# Build Plan

Milestones mapped from `Execution_Pipeline.md`. Status values: NOT STARTED /
IN PROGRESS / COMPLETED / BLOCKED. Never mark COMPLETED without having
actually run and verified the thing.

| Step | Owner | Description | Status |
|---|---|---|---|
| 0 | P4 | Dashboard scaffold (mock data) | NOT STARTED |
| 1 | P1 | Raw data acquisition + cleaning | **COMPLETED** (verified: ran code, checked schema/gaps) |
| 2 | P1 | Heatwave labels, chronological split, leakage rule + tests | **COMPLETED** (verified: `pytest tests/test_no_leakage.py` → 3/3 pass) |
| 3 | P2 | Baseline forecaster (persistence, climatology, LSTM, 5 seeds) | **CLOSED for now** (verified: extended evaluation done, stratified RMSE headline result established; user may reopen to add more baseline model comparisons later) |
| 4 | P3 | Retrieval system (statistical features, FAISS, eligibility filter, dedup) | NOT STARTED (deliberately deferred by user) |
| 5 | P2↔P3 | Joint integration (attention-fusion model) | NOT STARTED (blocked on Step 4) |
| 6 | P3 | Core result: stratified metrics + bootstrap CI | NOT STARTED (blocked on Step 5) |
| 7 | P2+P3 | Ablations (K-sweep, window length, representation, DB scope) | NOT STARTED (blocked on Step 6) |
| 8 | P1 | Conditional data fix, only if Step 7 flags an issue | NOT STARTED (conditional) |
| 9 | P4 | Wire real results into dashboard | NOT STARTED (blocked on Steps 6+7) |
| 10 | P4 | Final report, deck, demo | NOT STARTED (blocked on Step 9) |

## Step 3 sub-milestones (this session's actual granularity)

| Sub-milestone | Description | Status |
|---|---|---|
| 3a | `training/data.py` — window builder + normalization from `forecast_windows.parquet` | **COMPLETED** |
| 3b | `training/baselines.py` — persistence + climatology metrics | **COMPLETED** |
| 3c | `models/lstm.py` — LSTM architecture | **COMPLETED** |
| 3d | `training/train_lstm.py` — 5-seed training loop + checkpoints + val metrics | **COMPLETED** |

## Compute classification (Step 3) — actuals
- 3a, 3b: LOCAL, confirmed trivial (<5s each).
- 3c: LOCAL (module definition only, no compute).
- 3d: **LOCAL, confirmed cheap in practice** — 97.5s total for all 5 seeds on
  CPU, no GPU needed. Original "potentially heavy" classification was overly
  cautious; ~13k windows / 2-layer LSTM h=64 is small enough that CPU is fine.
  Not a candidate for Colab.

## Notes
- P3 (Step 4) and the resulting Step 5 are intentionally not started this
  session per explicit user instruction ("P3 we'll see later"). Step 3 can
  fully complete independently of Step 4.