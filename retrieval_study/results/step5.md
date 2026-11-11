# Retrieval study, step 5: retrieval keyed on a global model's forecast (RS-5, Tmax)

Pre-registered in `context/decisions_retrieval.md` (RS-5) before any forecast value was downloaded. ECMWF IFS HRES (WeatherBench 2, 1.5°), runs started 12 UTC the day before; query dates 2016-01-02 to 2018-12-27 (1091 windows x 5 leads, fold f4's validation years). Analogue ensemble: the mean ERA5 Tmax that followed the 20 most similar past HRES forecasts of the same lead (other years, ±45 days). RMSE in °C. 9 year x season clusters on all days (fragile); heatwave days have too few for a CI. Descriptive.

| Forecast | RMSE all days | Bias all days | RMSE heatwave days | Bias heatwave days |
|---|---|---|---|---|
| C3-hop | 1.971 | -0.03 | 2.022 | -1.57 |
| HRES corrected | 1.589 | -0.65 | 1.438 | -1.25 |
| AnEn | 1.465 | +0.04 | 1.538 | -1.27 |
| AnEn-random | 2.657 | -0.00 | 5.276 | -5.18 |

| Comparison | All days | Heatwave days |
|---|---|---|
| AnEn vs C3-hop | -0.506 [-0.819, -0.192] **(CI < 0)** | -0.483 (no CI, 3 clusters) |
| AnEn vs AnEn-random | -1.192 [-1.501, -0.883] **(CI < 0)** | -3.738 (no CI, 3 clusters) |
| AnEn vs HRES corrected | -0.124 [-0.297, +0.049] | +0.101 (no CI, 3 clusters) |
| HRES corrected vs C3-hop | -0.382 [-0.630, -0.134] **(CI < 0)** | -0.584 (no CI, 3 clusters) |

| Lead | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| C3-hop | 1.363 | 1.792 | 2.032 | 2.214 | 2.307 |
| HRES corrected | 1.353 | 1.524 | 1.584 | 1.684 | 1.768 |
| AnEn | 1.215 | 1.383 | 1.481 | 1.553 | 1.655 |
| Key-value Spearman (forecast key) | +0.16 | +0.13 | +0.11 | +0.10 | +0.09 |

HRES box-mean extract: `retrieval_study/data/hres_2016_2018_delhi_t2m.parquet`, SHA-256 14a98de55ee84ef2c7d9b6f600a3840384df1b92b4544a7392a570bed1109024.
