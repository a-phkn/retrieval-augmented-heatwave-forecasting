# Data Documentation

All paths relative to repo root. All verified directly (schema/shape read
from the actual parquet files), not copied from docs.

## `data/processed/weather_daily.parquet`
Area-averaged daily weather for Delhi/NCR (9 ERA5 grid cells around
Safdarjung, 28.58N/77.20E), truncated to the latest date complete across all
9 cells.
- **Rows:** 17,051 (1980-01-01 → 2026-09-06), no gaps, no duplicate dates.
- **Columns:** `date`, `t_max`, `t_min`, `t_mean`, `relative_humidity_mean`,
  `wind_speed_mean`, `surface_pressure_mean`, `shortwave_radiation_sum`,
  `doy_sin`, `doy_cos`, `years_since_1980`.
- **Not normalized** — raw units. z-scoring happens at model-input time,
  fit on train split only (owned by whoever builds the training pipeline —
  P2/P3, not P1).
- Gaps ≤2 days forward-filled during processing; longer gaps dropped and
  logged in `data/processing_log.json`.

## `events/event_catalogue.parquet`
One row per heatwave episode.
- **Columns:** `episode_id`, `episode_start`, `episode_end`, `duration_days`.
- **102 episodes total.** By split (mapped via episode start date):
  76 train / 9 val / 17 test.
- Definition: train-only (≤2015) ±7-day-of-year climatology mean/std,
  anomaly threshold >1.5σ, ≥3 consecutive days (1-day gap allowed) to count
  as one episode.
- **Known gap:** zero episodes in 2019 specifically — see
  `context/decisions.md` for why this is not a bug.

## `datasets/all_daily.parquet`
Full daily series with labels and split assignment, superset of
`weather_daily.parquet`.
- **Rows:** 17,051. Extra columns beyond `weather_daily.parquet`:
  `clim_mean_t_max`, `clim_std_t_max` (train-only ±7-day-of-year climatology),
  `t_max_anomaly` (= t_max − clim_mean), `hot_day` (bool, anomaly > 1.5σ),
  `heatwave_episode_id` (nullable int, FK into `event_catalogue.parquet`),
  `split` (`train`/`val`/`test`).
- **Split sizes (days):** train 13,149 / val 1,096 / test 2,806.
- Split boundaries: train ≤2015-12-31, val 2016-01-01–2018-12-31,
  test ≥2019-01-01. Chronological, no shuffling.

## `datasets/{train,val,test}.parquet`
Same schema as `all_daily.parquet`, pre-filtered by `split`. Convenience
copies — `all_daily.parquet` + the `split` column is the source of truth if
they ever disagree.

## `datasets/forecast_windows.parquet`
One row per valid 14-day-input → 5-day-forecast window.
- **Columns:** `query_date`, `input_start`, `input_end`, `forecast_start`,
  `forecast_end`, `split`, `target_episode_id` (nullable, FK into
  `event_catalogue.parquet` if the forecast window overlaps an episode),
  `target_is_heatwave` (bool — useful directly for stratified eval later),
  and the retrieval-candidate eligibility boundary
  (`candidate_latest_start` = query_date − 19 days) used by
  `tests/test_no_leakage.py`.
- **Split sizes (windows):** train 13,131 / val 1,092 / test 2,802.
- This is the file to use for building (X, y) training pairs — do not
  re-derive windowing logic elsewhere; join against `all_daily.parquet` for
  feature values.
- **Important:** this file only encodes the *split* and *dataset* leakage
  rule. It does NOT yet encode the *retrieval query-time* eligibility rule
  (same-split-only, no-same-episode) — that's P3's responsibility, applied
  at retrieval-query time, not baked into this file.

## `data/processing_log.json`
Log of gaps found/handled during `process_weather.py`. Small, human-readable,
check directly if data quality questions come up.

## Feature set for model input (13 features, per window-day)
`t_max, t_min, t_mean, relative_humidity_mean, wind_speed_mean,
surface_pressure_mean, shortwave_radiation_sum, doy_sin, doy_cos,
years_since_1980, clim_mean_t_max, clim_std_t_max, t_max_anomaly`

Target: next 5 days of `t_max` (raw °C for reporting; z-normalized for the
training loss).
