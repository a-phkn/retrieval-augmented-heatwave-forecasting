# Retrieval study, step 4: C3-hop with the 500 hPa pattern as a direct input (RS-4)

Pre-registered in `context/decisions_retrieval.md` (RS-4) before any run. Not retrieval: the upper-air information given to the model directly. Development folds (validation 2007-2018 pooled), 10 seeds x 4 folds; RMSE in °C (mean of the 10 seeds' RMSE); Δ = new minus old (negative = the direct input helps); paired cluster jackknife with 10 seeds each.

## Tmax

Not complete yet (not complete).

## WBGT

| Compared with | Days | RMSE before | RMSE with the 500 hPa input | Δ [95% CI] |
|---|---|---|---|---|
| C3-hop | all days | 2.248 | 2.165 | -0.083 [-0.108, -0.057] **(CI < 0)** |
| C3-hop | heatwave days | 3.019 | 2.950 | -0.069 [-0.208, +0.071] |
| C3-hop + Rz (retrieval) | all days | 2.225 | 2.165 | -0.059 [-0.083, -0.035] **(CI < 0)** |
| C3-hop + Rz (retrieval) | heatwave days | 3.023 | 2.950 | -0.073 [-0.197, +0.052] |

| Lead | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| C3-hop | 1.928 | 2.198 | 2.282 | 2.370 | 2.428 |
| C3-hop + 500 hPa input | 1.905 | 2.113 | 2.180 | 2.260 | 2.343 |

