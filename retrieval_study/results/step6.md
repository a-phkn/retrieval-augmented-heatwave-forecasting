# Retrieval study, step 6: the 500 hPa input on 2019-2021 (RS-6, post-test, descriptive)

Pre-registered in `context/decisions_retrieval.md` (RS-6) before any run. RS-4's model trained once on the `test2021` fold (fit 1980-2016, early stopping 2017-2018, forecasts for the windows from 2019-01-01 whose 5 days end by 2021-12-31), compared with C3-hop's test forecasts on the same windows and seeds. These years were used by the one-time test, so nothing here is confirmatory. RMSE in °C (mean of the seeds' RMSE); Δ = new minus old (negative = the 500 hPa input helps); paired cluster jackknife (no CI below 5 clusters).

## Tmax

1092 windows x 5 leads, 10 seeds.

| Compared with | Days | Its RMSE | RMSE with the 500 hPa input | Δ [95% CI] |
|---|---|---|---|---|
| C3-hop | all days | 2.239 | 2.156 | -0.083 [-0.158, -0.008], fragile: 9 clusters |
| C3-hop | heatwave days | 2.294 | 1.905 | -0.389 (no CI: 3 clusters) |

Reading, against C3-hop: all days, consistent with RS-4 (fragile: 9 clusters); heatwave days, no conclusion (no prediction was made); no CI with 3 clusters. P13 (all days below 0) holds; P14 (between RS-4's -0.057 and 0) fails.

By lead (descriptive):

| Lead | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| C3-hop | 1.576 | 2.044 | 2.299 | 2.472 | 2.645 |
| C3-hop + 500 hPa input | 1.528 | 1.957 | 2.207 | 2.397 | 2.541 |

Bias (mean error, forecast minus truth, °C; negative = too cold; descriptive):

| Model | All days | Heatwave days |
|---|---|---|
| C3-hop | +0.25 | -1.86 |
| C3-hop + 500 hPa input | +0.23 | -1.50 |

## WBGT

1092 windows x 5 leads, 10 seeds.

| Compared with | Days | Its RMSE | RMSE with the 500 hPa input | Δ [95% CI] |
|---|---|---|---|---|
| C3-hop | all days | 2.276 | 2.187 | -0.089 [-0.158, -0.020], fragile: 9 clusters |
| C3-hop | heatwave days | 3.065 | 2.904 | -0.161 [-0.717, +0.395], fragile: 5 clusters |
| C3-hop + Rz (retrieval) | all days | 2.272 | 2.187 | -0.085 [-0.137, -0.032], fragile: 9 clusters |
| C3-hop + Rz (retrieval) | heatwave days | 3.109 | 2.904 | -0.206 [-0.665, +0.253], fragile: 5 clusters |

Reading, against C3-hop: all days, consistent with RS-4 (fragile: 9 clusters); heatwave days, no conclusion (no prediction was made). P13 (all days below 0) holds; P14 (between RS-4's -0.083 and 0) fails.

By lead (descriptive):

| Lead | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| C3-hop | 1.950 | 2.210 | 2.299 | 2.419 | 2.464 |
| C3-hop + Rz (retrieval) | 1.956 | 2.211 | 2.293 | 2.405 | 2.459 |
| C3-hop + 500 hPa input | 1.922 | 2.127 | 2.174 | 2.286 | 2.397 |

Bias (mean error, forecast minus truth, °C; negative = too cold; descriptive):

| Model | All days | Heatwave days |
|---|---|---|
| C3-hop | +0.32 | -2.67 |
| C3-hop + Rz (retrieval) | +0.29 | -2.73 |
| C3-hop + 500 hPa input | +0.29 | -2.53 |

## Predictions

- P13 (all days below 0 in both families): held.
- P14 (smaller than in development in both families): failed.
- No prediction for heatwave days.

