# Project Overview

## Project name
Retrieval-Augmented Heatwave Forecasting

## What the project does
Forecasts the next 5 days of daily maximum temperature (T_max) for Delhi/NCR
from the previous 14 days of weather, and tests whether augmenting an LSTM
forecaster with attention over retrieved historically-similar 14-day windows
("analogues") improves forecasting — specifically during extreme/heatwave
periods, where an LSTM's own training data is thinnest.

## Core problem / research question
Does retrieving historical analogue weather windows improve heatwave
forecasting vs. a conventional LSTM, and does retrieval quality change the
effect? Core controlled comparison: same LSTM, same data, same training —
retrieval OFF vs. ON.

## Motivation
Heatwaves have a recurring build/peak/decay shape even when magnitude
differs. A retriever can hand the LSTM a full worked historical example
exactly where its own training signal is sparse (extreme events are rare by
definition). But retrieval can also fail: chaotic divergence, non-stationarity
(old analogues too mild for a warming climate), low episode diversity, or
diluted signal at high K. The project is designed to measure *why* retrieval
helps or doesn't, not just whether it does.

## Overall architecture
```
Open-Meteo API (ERA5) → raw monthly JSON (9 grid cells)
    → process_weather.py → data/processed/weather_daily.parquet
    → prepare_datasets.py → events/event_catalogue.parquet
                           → datasets/{train,val,test}.parquet
                           → datasets/forecast_windows.parquet
                           → tests/test_no_leakage.py
    → [P2] baseline + retrieval-augmented LSTM training
    → [P3] FAISS retrieval system (statistical feature vectors, IndexFlatIP)
    → [P2+P3] attention-fusion model (LSTM encoder + attention over K analogues)
    → [P3] stratified evaluation + bootstrap CI
    → [P4] dashboard (React + Vite), final report/deck
```

## Main ML components
- Persistence baseline (sanity floor)
- Climatology baseline (sanity floor)
- 2-layer LSTM (hidden=64, dropout 0.2) → MLP head (64→32→5) — the real baseline
- Retrieval-augmented model: same LSTM encoder + scaled dot-product attention
  over K retrieved analogues' embeddings/outcomes → MLP head

## Retrieval components
- Representation: hand-engineered statistical feature vector per 14-day window
  (~15–20 dims: mean/slope/std of T_max, end-of-window anomaly,
  days-above-threshold-so-far, humidity trend, etc.) — NOT YET IMPLEMENTED
- Index: FAISS `IndexFlatIP` (exact cosine similarity) — NOT YET IMPLEMENTED
- Query-time eligibility filter (split rule + no-future-data + no-same-episode)
  — NOT YET IMPLEMENTED (explicitly deferred by P1 to P3)
- K=5 core, swept over {1,3,5,10,20} — NOT YET IMPLEMENTED

## Forecasting components
Input: 14-day window, 13 features (see `data.md`). Output: 5-day T_max,
deterministic regression. Loss: MSE on z-normalized targets; report is
de-normalized MAE/RMSE. 5 random seeds throughout, mean ± std reported.

## Input/output flow
One sample: `X: (14, 13)` z-normalized input window → `y: (5,)` next-5-day
T_max (de-normalized for reporting). Normalization stats fit on train split
only, frozen for val/test.

## Main datasets
See `context/data.md` for full detail. Summary: `weather_daily.parquet`
(17,051 daily rows, 1980-01-01 to 2026-09-06), `event_catalogue.parquet`
(102 heatwave episodes), `{train,val,test}.parquet` (chronological split),
`forecast_windows.parquet` (17,025 valid 14-in/5-out windows with the
retrieval leakage boundary precomputed).

## Technologies/frameworks
Python, pandas/numpy, PyTorch (models), FAISS-cpu (retrieval, not yet
installed/used), pytest (leakage tests), React + Vite + Recharts (dashboard,
not yet started), Google Colab (heavier compute if needed).

## Current state (as of this session)
- P1 (data/pipeline): **COMPLETE**, verified by running the actual code and
  tests, not just reading docs.
- P2 (baseline model): **CLOSED for now** — canonical weighted-MSE LSTM,
  beats persistence on all three severity strata (see `ml_notes.md`). May
  reopen later to add more baseline model comparisons (e.g. gradient-boosted
  trees). The retrieval-augmented half of P2 (attention-fusion model) is
  still not started — blocked on P3.
- P3 (retrieval system): **NOT STARTED** (deliberately deferred per user
  instruction).
- P4 (dashboard/integration): **NOT STARTED**.

## Major constraints
- Laptop-class compute assumed; classify every heavy step LOCAL vs. COLAB.
- Test split touched once, at the end, for final numbers only.
- Chronological split only — no random splitting of time-series data.

## Important assumptions
- "Present" end date for the processed data is whatever the API returned as
  complete at download time (currently 2026-09-06) — this will drift on
  re-download and is expected/fine per `process_weather.py`'s own logic.

## Important design decisions
See `context/decisions.md` for the full log. Headline one so far: the
2019 heatwave-episode gap (zero ≥3-day episodes in 2019 under the current
1.5σ/3-day definition, despite 2019 being a known severe-heatwave year) was
investigated and is **not a bug** — see decisions.md for the reasoning. No
threshold change made.