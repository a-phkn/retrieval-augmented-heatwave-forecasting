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

## Next
- P3 is complete — Step 5 (joint integration: attention-fusion model using
  `retrieval/query.py`'s `query_analogues` interface) is next.
- Step 5 MUST use the same weighted-loss objective (hot_weight=10) as the
  canonical baseline to keep the "same training procedure, retrieval
  on/off only" comparison valid.
- The bar RAG needs to clear: extreme-stratum RMSE of 1.599 (canonical
  LSTM, val split) — not a recall percentage.
- Two things flagged during P3 that don't block Step 5 but are worth a
  decision at some point: (1) `min_days_apart=10` untuned, (2) "heatwave"
  episodes skew winter/early-spring rather than summer — see
  `decisions.md`.

## Blocked / waiting on user
- Nothing currently blocking. Step 5 is ready to start.

## Last verified state
- `pytest tests/` → 9 passed, 0 failed (3 in `test_no_leakage.py`, 6 in
  `test_retrieval_eligibility.py`), run this session.
- `retrieval/candidates.parquet` + `retrieval/faiss_index.bin` built over
  17,025 windows, verified via manual inspection + automated tests.

## 2026-09-22 update
- Canonical LSTM changed again: hot_weight 15 -> 20, after user correctly
  flagged the 15->20 marginal trade-off was better than 10->15 (see
  decisions.md for the full marginal analysis). Current canonical numbers:
  extreme RMSE=1.285, MAE=1.775, recall=0.339.
- Bootstrap CI on the 15-vs-20 extreme-RMSE gap not yet run -- offered, not
  requested. Worth doing before this number is treated as final.
