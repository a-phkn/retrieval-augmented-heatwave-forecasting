# P1 Handoff — Retrieval-Augmented Heatwave Forecasting

## 1. Purpose

P1 owns the data and pipeline foundation for the project:

- acquiring ERA5 weather data from the Open-Meteo Historical Weather API
- cleaning and validating the daily weather data
- defining heatwave episodes
- creating the chronological train/validation/test split
- defining and testing the retrieval leakage boundary
- producing the processed datasets used by P2 and P3

**P1 is complete and ready for handoff.**

## 2. Project Specification

- Region: Delhi/NCR
- Center: Safdarjung (28.58°N, 77.20°E)
- Spatial coverage: 3×3 ERA5 grid cells, 9 cells total
- Frequency: daily
- Input window: previous 14 days
- Forecast horizon: next 5 days
- Target: daily maximum temperature (`t_max`)
- Data period: 1980 to the latest complete date available from the API

## 3. Data Acquisition

Data was downloaded from the Open-Meteo Historical Weather API using the ERA5 model.

The downloader:

- requests all 9 grid cells
- stores monthly JSON responses under `data/raw/era5_monthly/`
- uses Asia/Kolkata timezone
- uses wind speed in m/s
- requests the weather variables required by the project
- can resume from the first missing monthly file
- retries failed/rate-limited requests

Raw API downloads are intentionally excluded from Git through `.gitignore`.

The reproducible downloader is:

`download_era5.py`

## 4. Data Processing

`process_weather.py`:

1. reads the monthly ERA5 JSON files
2. validates required weather variables
3. removes physically invalid values
4. averages the 9 ERA5 grid cells
5. creates the daily weather series
6. adds day-of-year sine/cosine and years-since-1980
7. stops at the latest date for which all required variables are complete across all 9 cells

The final processed weather data currently ends at **2026-09-06**. The incomplete API data after that date was deliberately excluded.

The final weather dataset contains:

- date
- t_max
- t_min
- shortwave_radiation_sum
- t_mean
- relative_humidity_mean
- wind_speed_mean
- surface_pressure_mean
- doy_sin
- doy_cos
- years_since_1980

No missing weather values remain in the final processed weather dataset.

## 5. Heatwave Definition

Heatwave thresholds are derived from the training period only.

For each day of year, the climatological mean and standard deviation of `t_max` are calculated using a ±7-day day-of-year window.

A day is considered hot when:

`anomaly > 1.5 × climatological standard deviation`

Consecutive hot days are grouped into episodes, allowing a one-day gap. Only episodes lasting at least 3 days are retained.

The resulting event catalogue is:

`events/event_catalogue.parquet`

## 6. Data Split

The split is chronological:

| Split      | Period       |
| ---------- | ------------ |
| Train      | 1980–2015    |
| Validation | 2016–2018    |
| Test       | 2019–present |

The test period is not used for model or retrieval tuning.

## 7. Leakage Prevention

The retrieval design requires candidate windows to be entirely in the past relative to the forecast query.

The P1 leakage test suite is:

`tests/test_no_leakage.py`

It currently checks the dataset-level eligibility boundaries and split constraints.

P3 must additionally enforce the complete query-time retrieval eligibility rule in the actual retrieval implementation, including the project requirements for split eligibility, future-data exclusion, and same-episode exclusion.

## 8. Validation Performed

P1 performed sanity checks including:

- completeness of the final daily weather series
- duplicate-date checks
- missing-date checks
- physical-range checks
- `t_max >= t_min`
- non-negative wind speed
- positive surface pressure
- non-negative shortwave radiation
- 2019 and 2022 temperature sanity checks
- leakage test suite

The heatwave sanity-check plot is:

`data/processed/heatwave_sanity_2019_2022.png`

## 9. Files Handed Off

### Data and Outputs

- `data/processed/weather_daily.parquet`
- `data/processing_log.json`
- `events/event_catalogue.parquet`
- `datasets/all_daily.parquet`
- `datasets/train.parquet`
- `datasets/val.parquet`
- `datasets/test.parquet`
- `datasets/forecast_windows.parquet`

### P1 Code

- `download_era5.py`
- `process_weather.py`
- `prepare_datasets.py`
- `check_heatwaves.py`

### Configuration and Tests

- `config/data_spec.yaml`
- `tests/test_no_leakage.py`

## 10. Notes for P2 — Modeling

P2 should use the processed Parquet outputs rather than modifying the raw API data.

The roadmap requires:

- persistence baseline
- climatology baseline
- 2-layer LSTM baseline
- MLP baseline
- 5 random seeds
- early stopping
- MSE on z-normalized targets
- final MAE/RMSE reported in original temperature units

Normalization statistics must be fitted on the training data only and then frozen for validation/test.

P1 does not own model architecture or training.

## 11. Notes for P3 — Retrieval

P3 should implement retrieval using the P1 datasets and event catalogue.

The project specifies:

- statistical feature vectors for 14-day windows
- exact cosine similarity using FAISS `IndexFlatIP`
- core K=5
- at most 2 retrieved windows from the same heatwave episode
- K sweep: 1, 3, 5, 10, 20

Most importantly, P3 must enforce retrieval eligibility at query time and verify that no retrieved candidate contains information from the future relative to the forecast query.

P1's leakage tests provide the dataset-level foundation; P3 must test the actual retrieval implementation as well.

## 12. What P1 Does Not Own

P1 does not own:

- LSTM implementation
- baseline model training
- FAISS retrieval implementation
- retrieval attention architecture
- retrieval quality metrics
- model ablations
- final evaluation
- dashboard/UI integration

If later experiments reveal a genuine data or feature problem, P1 can return to the pipeline and fix it. Otherwise, the P1 pipeline should remain frozen.

## 13. Current Status

**P1 COMPLETE — READY FOR HANDOFF**

The processed data, event catalogue, chronological splits, retrieval leakage framework, validation checks, and reproducible acquisition/processing scripts are committed to the shared Git repository.
