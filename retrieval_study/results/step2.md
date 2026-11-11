# Retrieval study, step 2: screens (RS-2)

Pre-registered in `context/decisions_retrieval.md` (RS-2) before any screen was run. Development folds only (validation 2007-2018 pooled). RMSE in °C; Δ = RMSE of the second minus the first (negative = helps); 95% CIs from the paired cluster jackknife. Skipped by RS-1's results: RAFT inside C3-hop, a cross-site pool, a learned similarity (reasons in the decision log).

## Tmax

### Information screen (ridge on B3: what C3-hop roughly sees)

| Signal | Days | vs B3 alone | vs its random control | direct version vs B3 alone | signal vs direct version |
|---|---|---|---|---|---|
| Rz (500 hPa pattern analogues) | all days | -0.004 [-0.009, +0.002] | -0.004 [-0.010, +0.001] | -0.038 [-0.051, -0.025] **(CI < 0)** | +0.034 [+0.021, +0.047] (CI > 0) |
| Rz (500 hPa pattern analogues) | heatwave days | -0.022 [-0.050, +0.006] | -0.022 [-0.051, +0.008] | -0.120 [-0.183, -0.057] **(CI < 0)** | +0.099 [+0.038, +0.159] (CI > 0) |
| LW (60-day season-to-date analogues) | all days | +0.001 [-0.001, +0.003] | +0.000 [-0.002, +0.002] | -0.003 [-0.007, +0.000] | +0.004 [+0.000, +0.008] (CI > 0) |
| LW (60-day season-to-date analogues) | heatwave days | +0.002 [-0.006, +0.011] | +0.003 [-0.008, +0.013] | -0.004 [-0.071, +0.062] | +0.007 [-0.060, +0.073] |

### Relevance screen (add-on to C3-hop's 10-seed forecast)

| Signal | Days | C3-hop RMSE | + signal vs alone | + signal vs + random | + direct version vs alone |
|---|---|---|---|---|---|
| Rz (500 hPa pattern analogues) | all days | 2.012 | -0.002 [-0.020, +0.015] | +0.002 [-0.003, +0.006] | -0.018 [-0.035, +0.000] |
| Rz (500 hPa pattern analogues) | heatwave days | 1.740 | +0.114 [+0.032, +0.195] (CI > 0) | -0.022 [-0.050, +0.007] | +0.044 [-0.154, +0.241] |
| LW (60-day season-to-date analogues) | all days | 2.012 | -0.003 [-0.023, +0.017] | +0.001 [-0.005, +0.007] | -0.005 [-0.022, +0.012] |
| LW (60-day season-to-date analogues) | heatwave days | 1.740 | +0.172 [+0.094, +0.249] (CI > 0) | +0.040 [+0.025, +0.055] (CI > 0) | +0.136 [+0.082, +0.189] (CI > 0) |

### KR: retrieval for 80% ranges around C3-hop's forecast

| Ranges from | Days | Coverage | Mean width | Interval score |
|---|---|---|---|---|
| all inner-block errors | all days | 79% | 4.78 | 7.111 |
| the 50 most similar situations (retrieval) | all days | 76% | 4.45 | 7.166 |
| 50 random situations (10 seeds) | all days | 77% | 4.59 | 7.300 |
| all inner-block errors | heatwave days | 79% | 4.76 | 6.523 |
| the 50 most similar situations (retrieval) | heatwave days | 70% | 5.08 | 7.831 |
| 50 random situations (10 seeds) | heatwave days | 76% | 4.60 | 6.616 |

| Comparison (mean interval score) | All days | Heatwave days |
|---|---|---|
| knn vs global | +0.056 [-0.120, +0.232] | +1.308 [+0.836, +1.779] (CI > 0) |
| knn vs random | -0.134 [-0.307, +0.040] | +1.214 [+0.723, +1.706] (CI > 0) |

### Pre-registered pass rule

| Candidate | Passes |
|---|---|
| Rz | no |
| LW | no |
| KR | no |

## WBGT (physical)

### Information screen (ridge on B3: what C3-hop roughly sees)

| Signal | Days | vs B3 alone | vs its random control | direct version vs B3 alone | signal vs direct version |
|---|---|---|---|---|---|
| Rz (500 hPa pattern analogues) | all days | -0.011 [-0.016, -0.005] **(CI < 0)** | -0.011 [-0.017, -0.005] **(CI < 0)** | -0.046 [-0.062, -0.031] **(CI < 0)** | +0.036 [+0.022, +0.050] (CI > 0) |
| Rz (500 hPa pattern analogues) | heatwave days | -0.034 [-0.078, +0.011] | -0.034 [-0.078, +0.011] | -0.004 [-0.105, +0.098] | -0.030 [-0.115, +0.054] |
| LW (60-day season-to-date analogues) | all days | -0.001 [-0.002, +0.000] | -0.001 [-0.002, -0.000] **(CI < 0)** | -0.001 [-0.002, +0.001] | -0.000 [-0.001, +0.001] |
| LW (60-day season-to-date analogues) | heatwave days | -0.005 [-0.013, +0.004] | -0.005 [-0.014, +0.005] | -0.005 [-0.014, +0.004] | +0.000 [-0.011, +0.012] |

### Relevance screen (add-on to C3-hop's 10-seed forecast)

| Signal | Days | C3-hop RMSE | + signal vs alone | + signal vs + random | + direct version vs alone |
|---|---|---|---|---|---|
| Rz (500 hPa pattern analogues) | all days | 2.216 | -0.038 [-0.070, -0.007] **(CI < 0)** | -0.012 [-0.020, -0.005] **(CI < 0)** | -0.041 [-0.074, -0.008] **(CI < 0)** |
| Rz (500 hPa pattern analogues) | heatwave days | 2.991 | +0.350 [+0.239, +0.461] (CI > 0) | -0.080 [-0.155, -0.005] **(CI < 0)** | +0.381 [+0.282, +0.479] (CI > 0) |
| LW (60-day season-to-date analogues) | all days | 2.216 | -0.025 [-0.059, +0.009] | +0.001 [-0.004, +0.005] | -0.026 [-0.061, +0.010] |
| LW (60-day season-to-date analogues) | heatwave days | 2.991 | +0.423 [+0.339, +0.506] (CI > 0) | -0.007 [-0.034, +0.020] | +0.420 [+0.332, +0.508] (CI > 0) |

### KR: retrieval for 80% ranges around C3-hop's forecast

| Ranges from | Days | Coverage | Mean width | Interval score |
|---|---|---|---|---|
| all inner-block errors | all days | 79% | 5.25 | 7.810 |
| the 50 most similar situations (retrieval) | all days | 75% | 4.86 | 7.921 |
| 50 random situations (10 seeds) | all days | 76% | 5.05 | 8.047 |
| all inner-block errors | heatwave days | 41% | 5.23 | 12.202 |
| the 50 most similar situations (retrieval) | heatwave days | 15% | 4.49 | 20.204 |
| 50 random situations (10 seeds) | heatwave days | 39% | 5.03 | 13.389 |

| Comparison (mean interval score) | All days | Heatwave days |
|---|---|---|
| knn vs global | +0.111 [-0.033, +0.254] | +8.003 [+6.238, +9.767] (CI > 0) |
| knn vs random | -0.127 [-0.272, +0.018] | +6.815 [+4.991, +8.640] (CI > 0) |

### Pre-registered pass rule

| Candidate | Passes |
|---|---|
| Rz | yes |
| LW | no |
| KR | no |

