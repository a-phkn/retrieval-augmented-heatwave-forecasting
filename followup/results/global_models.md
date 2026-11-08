# Our forecasts vs operational global forecasts (P3-1, post-test)

Pre-registered 2026-11-05 in `context/decisions_phase3.md` (P3-1). Descriptive: the test years were already used, so nothing here is confirmatory. RMSE and bias in °C; the global models and C3-hop / LSTM ensemble means are single forecasts, 'per-seed' is the test's primary score (mean of the 10 seeds' RMSE). Delta = RMSE(global) - RMSE(C3-hop ensemble) on the same rows, with the paired cluster-jackknife 95% CI (negative = the global model is better).

## Tmax

### All days

| Model | Period (query dates) | Rows | Clusters | RMSE global | C3-hop | C3-hop per-seed | LSTM | Delta [95% CI] | Bias global | Bias C3-hop |
|---|---|---|---|---|---|---|---|---|---|---|
| IFS | 2024-02-05 to 2026-09-02 | 4705 | 9 | 1.763 | 2.126 | 2.173 | 2.238 | -0.363 [-0.830, +0.104], fragile (< 10 clusters) | -0.65 | +0.17 |
| AIFS | 2024-03-15 to 2025-03-10 | 1795 | 3 | 2.116 | 1.894 | 1.945 | 1.954 | +0.222 no CI (< 5 clusters), fragile (< 10 clusters) | -1.50 | -0.15 |
| GFS | 2024-01-20 to 2026-09-02 | 4785 | 9 | 3.730 | 2.120 | 2.167 | 2.230 | +1.611 [+1.151, +2.070], fragile (< 10 clusters) | +2.89 | +0.18 |
| GFS, Tmax since 2021 | 2021-03-25 to 2026-09-02 | 9825 | 17 | 3.433 | 2.185 | 2.231 | 2.331 | +1.247 [+0.904, +1.591] | +2.60 | +0.20 |

### Heatwave days (extreme stratum)

| Model | Period (query dates) | Rows | Clusters | RMSE global | C3-hop | C3-hop per-seed | LSTM | Delta [95% CI] | Bias global | Bias C3-hop |
|---|---|---|---|---|---|---|---|---|---|---|
| IFS | 2024-02-05 to 2026-09-02 | 245 | 3 | 1.584 | 2.384 | 2.428 | 2.599 | -0.800 no CI (< 5 clusters), fragile (< 10 clusters) | -0.85 | -2.04 |
| AIFS | 2024-03-15 to 2025-03-10 | 125 | 1 | 2.506 | 2.220 | 2.267 | 2.162 | not testable | -2.02 | -1.85 |
| GFS | 2024-01-20 to 2026-09-02 | 245 | 3 | 2.574 | 2.384 | 2.428 | 2.599 | +0.190 no CI (< 5 clusters), fragile (< 10 clusters) | +2.29 | -2.04 |
| GFS, Tmax since 2021 | 2021-03-25 to 2026-09-02 | 460 | 6 | 2.375 | 2.336 | 2.378 | 2.606 | +0.039 [-0.275, +0.353], fragile (< 10 clusters) | +2.09 | -1.99 |

### Heat season

| Model | Period (query dates) | Rows | Clusters | RMSE global | C3-hop | C3-hop per-seed | LSTM | Delta [95% CI] | Bias global | Bias C3-hop |
|---|---|---|---|---|---|---|---|---|---|---|
| IFS | 2024-02-05 to 2026-09-02 | 2085 | 3 | 1.752 | 2.476 | 2.520 | 2.623 | -0.724 no CI (< 5 clusters), fragile (< 10 clusters) | -0.34 | +0.17 |
| AIFS | 2024-03-15 to 2025-03-10 | 685 | 1 | 1.694 | 2.036 | 2.092 | 2.164 | not testable | -1.07 | -0.38 |
| GFS | 2024-01-20 to 2026-09-02 | 2085 | 3 | 4.418 | 2.476 | 2.520 | 2.623 | +1.942 no CI (< 5 clusters), fragile (< 10 clusters) | +3.67 | +0.17 |
| GFS, Tmax since 2021 | 2021-03-25 to 2026-09-02 | 4110 | 6 | 4.135 | 2.542 | 2.586 | 2.791 | +1.593 [+1.122, +2.064], fragile (< 10 clusters) | +3.44 | +0.26 |

### By lead (all days, RMSE)

| Model | | Day 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| IFS | global | 1.474 | 1.630 | 1.748 | 1.877 | 2.033 |
| IFS | C3-hop | 1.522 | 1.932 | 2.163 | 2.357 | 2.515 |
| AIFS | global | 1.907 | 2.037 | 2.130 | 2.210 | 2.279 |
| AIFS | C3-hop | 1.435 | 1.773 | 1.961 | 2.075 | 2.145 |
| GFS | global | 3.265 | 3.631 | 3.818 | 3.922 | 3.972 |
| GFS | C3-hop | 1.526 | 1.928 | 2.157 | 2.347 | 2.502 |
| GFS, Tmax since 2021 | global | 3.042 | 3.306 | 3.494 | 3.593 | 3.690 |
| GFS, Tmax since 2021 | C3-hop | 1.520 | 1.970 | 2.226 | 2.431 | 2.613 |

### Yes/no heatwave warnings (in-season rows where the global model has a value)

| Model rows | Rule | Hit rate | False-alarm ratio | ETS | Frequency bias |
|---|---|---|---|---|---|
| IFS (2085 rows, 260 events) | global model, R-point | 50% | 38% | 0.33 | 0.81 |
| IFS (2085 rows, 260 events) | C3-hop, R-point | 28% | 28% | 0.22 | 0.38 |
| IFS (2085 rows, 260 events) | C3-hop, R-prob (p* = 0.37) | 31% | 29% | 0.24 | 0.44 |
| IFS (2085 rows, 260 events) | C3-hop, R-shift (delta = 0.25 C) | 36% | 31% | 0.27 | 0.52 |
| IFS (2085 rows, 260 events) | LSTM, R-point | 29% | 41% | 0.20 | 0.49 |
| AIFS (685 rows, 125 events) | global model, R-point | 52% | 3% | 0.46 | 0.54 |
| AIFS (685 rows, 125 events) | C3-hop, R-point | 50% | 24% | 0.36 | 0.66 |
| AIFS (685 rows, 125 events) | C3-hop, R-prob (p* = 0.37) | 52% | 23% | 0.39 | 0.67 |
| AIFS (685 rows, 125 events) | C3-hop, R-shift (delta = 0.25 C) | 62% | 25% | 0.45 | 0.83 |
| AIFS (685 rows, 125 events) | LSTM, R-point | 45% | 27% | 0.32 | 0.62 |
| GFS (2085 rows, 260 events) | global model, R-point | 100% | 72% | 0.18 | 3.58 |
| GFS (2085 rows, 260 events) | C3-hop, R-point | 28% | 28% | 0.22 | 0.38 |
| GFS (2085 rows, 260 events) | C3-hop, R-prob (p* = 0.37) | 31% | 29% | 0.24 | 0.44 |
| GFS (2085 rows, 260 events) | C3-hop, R-shift (delta = 0.25 C) | 36% | 31% | 0.27 | 0.52 |
| GFS (2085 rows, 260 events) | LSTM, R-point | 29% | 41% | 0.20 | 0.49 |
| GFS, Tmax since 2021 (4110 rows, 500 events) | global model, R-point | 100% | 72% | 0.18 | 3.59 |
| GFS, Tmax since 2021 (4110 rows, 500 events) | C3-hop, R-point | 25% | 35% | 0.19 | 0.39 |
| GFS, Tmax since 2021 (4110 rows, 500 events) | C3-hop, R-prob (p* = 0.37) | 30% | 35% | 0.22 | 0.46 |
| GFS, Tmax since 2021 (4110 rows, 500 events) | C3-hop, R-shift (delta = 0.25 C) | 34% | 38% | 0.24 | 0.55 |
| GFS, Tmax since 2021 (4110 rows, 500 events) | LSTM, R-point | 28% | 47% | 0.18 | 0.53 |

## WBGT (physical)

### All days

| Model | Period (query dates) | Rows | Clusters | RMSE global | C3-hop | C3-hop per-seed | LSTM | Delta [95% CI] | Bias global | Bias C3-hop |
|---|---|---|---|---|---|---|---|---|---|---|
| IFS | 2024-03-07 to 2026-09-02 | 4533 | 9 | 1.969 | 2.217 | 2.244 | 2.367 | -0.249 [-0.476, -0.021], fragile (< 10 clusters) | -0.69 | +0.36 |
| GFS | 2024-01-20 to 2026-09-02 | 4781 | 9 | 2.646 | 2.209 | 2.236 | 2.377 | +0.436 [+0.211, +0.661], fragile (< 10 clusters) | -1.64 | +0.36 |

### Heatwave days (extreme stratum)

| Model | Period (query dates) | Rows | Clusters | RMSE global | C3-hop | C3-hop per-seed | LSTM | Delta [95% CI] | Bias global | Bias C3-hop |
|---|---|---|---|---|---|---|---|---|---|---|
| IFS | 2024-03-07 to 2026-09-02 | 145 | 4 | 3.277 | 3.077 | 3.092 | 3.134 | +0.200 no CI (< 5 clusters), fragile (< 10 clusters) | -2.81 | -2.72 |
| GFS | 2024-01-20 to 2026-09-02 | 145 | 4 | 4.967 | 3.077 | 3.092 | 3.134 | +1.891 no CI (< 5 clusters), fragile (< 10 clusters) | -4.57 | -2.72 |

### Heat season

| Model | Period (query dates) | Rows | Clusters | RMSE global | C3-hop | C3-hop per-seed | LSTM | Delta [95% CI] | Bias global | Bias C3-hop |
|---|---|---|---|---|---|---|---|---|---|---|
| IFS | 2024-03-07 to 2026-09-02 | 2856 | 6 | 2.001 | 2.195 | 2.224 | 2.315 | -0.195 [-0.511, +0.122], fragile (< 10 clusters) | -0.71 | +0.60 |
| GFS | 2024-01-20 to 2026-09-02 | 2866 | 6 | 2.552 | 2.193 | 2.222 | 2.313 | +0.359 [+0.125, +0.592], fragile (< 10 clusters) | -1.54 | +0.60 |

### By lead (all days, RMSE)

| Model | | Day 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| IFS | global | 1.643 | 1.867 | 1.969 | 2.101 | 2.215 |
| IFS | C3-hop | 1.950 | 2.154 | 2.246 | 2.327 | 2.383 |
| GFS | global | 2.483 | 2.545 | 2.689 | 2.701 | 2.798 |
| GFS | C3-hop | 1.946 | 2.148 | 2.235 | 2.319 | 2.373 |

### Yes/no heatwave warnings (in-season rows where the global model has a value)

| Model rows | Rule | Hit rate | False-alarm ratio | ETS | Frequency bias |
|---|---|---|---|---|---|
| IFS (2856 rows, 190 events) | global model, R-point | 16% | 77% | 0.08 | 0.68 |
| IFS (2856 rows, 190 events) | C3-hop, R-point | 8% | 75% | 0.05 | 0.31 |
| IFS (2856 rows, 190 events) | C3-hop, R-prob (p* = 0.13) | 54% | 85% | 0.08 | 3.57 |
| IFS (2856 rows, 190 events) | C3-hop, R-shift (delta = 1.2 C) | 47% | 83% | 0.09 | 2.81 |
| IFS (2856 rows, 190 events) | LSTM, R-point | 4% | 75% | 0.02 | 0.15 |
| GFS (2866 rows, 190 events) | global model, R-point | 4% | 75% | 0.02 | 0.15 |
| GFS (2866 rows, 190 events) | C3-hop, R-point | 8% | 75% | 0.05 | 0.31 |
| GFS (2866 rows, 190 events) | C3-hop, R-prob (p* = 0.13) | 54% | 85% | 0.08 | 3.57 |
| GFS (2866 rows, 190 events) | C3-hop, R-shift (delta = 1.2 C) | 47% | 83% | 0.09 | 2.81 |
| GFS (2866 rows, 190 events) | LSTM, R-point | 4% | 75% | 0.02 | 0.15 |

## Notes

- Timing: lead L uses `_previous_dayL`, the value predicted L x 24 h before its valid time, so every value comes from a run started no later than 12 UTC on the day before the query date (our models' inputs end at the end of that day, IST).
- Native output steps (Open-Meteo's documentation; hourly values are its interpolation, which can miss the afternoon peak, most for 6-hourly AIFS): IFS 3-hourly (6-hourly after 144 h); AIFS 6-hourly; GFS hourly (3-hourly after 120 h).
- WBGT inputs: IFS 0.25° gives only global radiation, so Open-Meteo estimates the diffuse part with a separation model; WBGT's direct / diffuse split is therefore partly modelled for IFS.
- ERA5 is produced with the IFS model, so ERA5 truth may favour IFS and our ERA5-trained models over GFS and AIFS (P3-3 repeats Tmax against the Safdarjung station).
- WBGT solver failures (hours dropped): {'IFS': 0, 'AIFS': 0, 'GFS': 0}.
- Grid points vs the ERA5 download: IFS: grid points the same as ERA5's, elevations the same; AIFS: grid points the same as ERA5's, elevations the same; GFS: grid points differ from ERA5's, elevations the same.
- Daily extract: `data/external/open_meteo_runs_daily.parquet`, SHA-256 8b51c2096533f87c544953d296e8fbc21f6b67513d0d72b3f6381424812a9b92.
- Timing check (IFS HRES 9 km, 12 UTC run): not complete yet (0 of 902 runs downloaded); python -m followup.open_meteo --single-runs resumes it.
