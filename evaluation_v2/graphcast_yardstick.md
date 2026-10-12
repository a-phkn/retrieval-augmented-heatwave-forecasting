# External yardstick: GraphCast vs our models, Delhi Tmax, 2018 (paper only)

361 forecast windows (query dates in 2018, fold f4 validation block), 5 leads. GraphCast: WeatherBench 2 ERA5-initialised forecasts, 0.7° grid, started 12 UTC the day before the first forecast day; daily maximum from its 6-hourly values via a per-month correction fitted on our ERA5 1980-2015. Descriptive only (one year = 3 clusters). Our models: mean of per-seed RMSE (10 seeds), same windows. GraphCast starts from the full global atmosphere; ours from 28 points' daily data, so this measures the gap to a global AI weather model, not a like-for-like contest. Tmax only (no WBGT from GraphCast).

| Model | RMSE all | RMSE extreme | Bias all | Bias extreme | Lead 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|---|---|---|
| GraphCast (corrected) | 1.154 | 1.049 | -0.17 | -0.71 | 0.909 | 1.107 | 1.162 | 1.230 | 1.320 |
| perfect 6-hourly forecast (method floor) | 0.668 | 0.693 | -0.10 | -0.13 | 0.668 | 0.668 | 0.669 | 0.669 | 0.667 |
| control (A1prime_hw5) | 2.105 | 2.399 | -0.27 | -2.18 | 1.535 | 1.952 | 2.213 | 2.330 | 2.378 |
| C3-hop | 1.979 | 2.007 | +0.11 | -1.80 | 1.353 | 1.763 | 2.054 | 2.242 | 2.324 |
| C2 | 1.993 | 2.339 | -0.00 | -2.20 | 1.379 | 1.779 | 2.082 | 2.256 | 2.318 |
| U1 (tuned) | 2.091 | 1.403 | +0.36 | -1.05 | 1.530 | 1.872 | 2.166 | 2.331 | 2.425 |
| ridge | 2.021 | 2.771 | +0.17 | -2.55 | 1.443 | 1.868 | 2.166 | 2.249 | 2.258 |

Extreme-day rows: 20 (window x lead). The method floor is the error a perfect 6-hourly forecast would still have after the sampling correction; GraphCast is not expected to do better than it. With only 20 extreme rows, the extreme-day column is too noisy to rank models.
