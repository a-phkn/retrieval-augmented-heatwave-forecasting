# Retrieval study, step 4: C3-hop with the 500 hPa pattern as a direct input (RS-4)

Pre-registered in `context/decisions_retrieval.md` (RS-4) before any run. Not retrieval: the upper-air information given to the model directly. Development folds (validation 2007-2018 pooled), 10 seeds x 4 folds; RMSE in °C (mean of the 10 seeds' RMSE); Δ = new minus old (negative = the direct input helps); paired cluster jackknife with 10 seeds each.

## Tmax

| Compared with | Days | Its RMSE | RMSE with the 500 hPa input | Δ [95% CI] |
|---|---|---|---|---|
| C3-hop | all days | 2.065 | 2.008 | -0.057 [-0.091, -0.024] **(CI < 0)** |
| C3-hop | heatwave days | 1.827 | 1.603 | -0.224 [-0.409, -0.038] **(CI < 0)** |

| Lead | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| C3-hop | 1.443 | 1.859 | 2.120 | 2.318 | 2.433 |
| C3-hop + 500 hPa input | 1.430 | 1.808 | 2.059 | 2.237 | 2.367 |

Bias (mean error, forecast minus truth, °C; negative = too cold; descriptive):

| Model | All days | Heatwave days |
|---|---|---|
| C3-hop | +0.15 | -1.25 |
| C3-hop + 500 hPa input | +0.21 | -0.98 |

## WBGT

| Compared with | Days | Its RMSE | RMSE with the 500 hPa input | Δ [95% CI] |
|---|---|---|---|---|
| C3-hop | all days | 2.248 | 2.165 | -0.083 [-0.108, -0.057] **(CI < 0)** |
| C3-hop | heatwave days | 3.019 | 2.950 | -0.069 [-0.208, +0.071] |
| C3-hop + Rz (retrieval) | all days | 2.225 | 2.165 | -0.059 [-0.083, -0.035] **(CI < 0)** |
| C3-hop + Rz (retrieval) | heatwave days | 3.023 | 2.950 | -0.073 [-0.197, +0.052] |
| the shuffled-year control | all days | 2.245 | 2.165 | -0.080 [-0.108, -0.052] **(CI < 0)** |
| the shuffled-year control | heatwave days | 3.077 | 2.950 | -0.127 [-0.257, +0.003] |

| Lead | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| C3-hop | 1.928 | 2.198 | 2.282 | 2.370 | 2.428 |
| C3-hop + Rz (retrieval) | 1.913 | 2.167 | 2.255 | 2.345 | 2.410 |
| the shuffled-year control | 1.925 | 2.183 | 2.277 | 2.372 | 2.434 |
| C3-hop + 500 hPa input | 1.905 | 2.113 | 2.180 | 2.260 | 2.343 |

Bias (mean error, forecast minus truth, °C; negative = too cold; descriptive):

| Model | All days | Heatwave days |
|---|---|---|
| C3-hop | +0.38 | -2.69 |
| C3-hop + Rz (retrieval) | +0.35 | -2.69 |
| the shuffled-year control | +0.36 | -2.76 |
| C3-hop + 500 hPa input | +0.34 | -2.59 |

