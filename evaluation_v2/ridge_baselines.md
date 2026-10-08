# Ridge baselines (decisions.md 2026-10-08, reporting addition 1)

Out of fold 2007-2018; every ridge fitted per fold on training years only. `ridge` = the G-D0 ridge (Delhi's last anomaly + 27 upstream points' Tmax anomalies, 3 lags); `ridge_hw5` = the same, weighted like the neural loss (1 + 4 x hot). The ridges refit on ALL training years; the neural models never train on the last 2 (no refit). Δ = trained model minus ridge (negative = the trained model is better). Trained models: mean of per-seed RMSE, 10 seeds.

## Tmax

| Model | All | Extreme | Bias all | Bias extreme | Lead 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|---|---|---|
| ridge | 2.104 | 2.908 | +0.00 | -2.50 | 1.463 | 1.946 | 2.210 | 2.347 | 2.410 |
| ridge_hw5 | 2.138 | 2.456 | +0.27 | -1.98 | 1.474 | 1.977 | 2.246 | 2.388 | 2.453 |
| control | 2.175 | 2.153 | +0.09 | -1.71 | 1.597 | 2.020 | 2.268 | 2.398 | 2.478 |
| C2 | 2.051 | 1.845 | +0.12 | -1.34 | 1.444 | 1.851 | 2.109 | 2.294 | 2.409 |
| U1 | 2.199 | 1.767 | +0.19 | -1.23 | 1.760 | 2.012 | 2.231 | 2.401 | 2.508 |
| C3 | 2.147 | 1.904 | +0.17 | -1.43 | 1.554 | 1.983 | 2.236 | 2.373 | 2.462 |

| Comparison | Δ RMSE (95% CI) | p |
|---|---|---|
| control vs ridge (all) | +0.072 [+0.032, +0.112] | <0.001 |
| control vs ridge (extreme) | -0.754 [-1.052, -0.457] | <0.001 |
| C2 vs ridge (all) | -0.053 [-0.096, -0.011] | 0.015 |
| C2 vs ridge (extreme) | -1.062 [-1.344, -0.781] | <0.001 |
| U1 vs ridge (all) | +0.095 [+0.039, +0.151] | 0.001 |
| U1 vs ridge (extreme) | -1.141 [-1.457, -0.826] | <0.001 |
| C3 vs ridge (all) | +0.043 [+0.001, +0.085] | 0.046 |
| C3 vs ridge (extreme) | -1.004 [-1.294, -0.714] | <0.001 |
| control vs ridge_hw5 (all) | +0.038 [-0.002, +0.078] | 0.065 |
| control vs ridge_hw5 (extreme) | -0.303 [-0.577, -0.029] | 0.033 |
| C2 vs ridge_hw5 (all) | -0.087 [-0.123, -0.052] | <0.001 |
| C2 vs ridge_hw5 (extreme) | -0.611 [-0.861, -0.361] | <0.001 |
| U1 vs ridge_hw5 (all) | +0.061 [+0.009, +0.113] | 0.022 |
| U1 vs ridge_hw5 (extreme) | -0.690 [-0.985, -0.395] | <0.001 |
| C3 vs ridge_hw5 (all) | +0.009 [-0.030, +0.049] | 0.641 |
| C3 vs ridge_hw5 (extreme) | -0.553 [-0.840, -0.265] | 0.002 |

## WBGT (physical)

| Model | All | Extreme | Bias all | Bias extreme | Lead 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|---|---|---|
| ridge | 2.217 | 4.366 | -0.30 | -4.15 | 1.948 | 2.182 | 2.254 | 2.317 | 2.362 |
| ridge_hw5 | 2.212 | 3.866 | +0.17 | -3.58 | 1.963 | 2.180 | 2.244 | 2.302 | 2.350 |
| control | 2.373 | 3.274 | +0.44 | -3.02 | 2.085 | 2.361 | 2.439 | 2.472 | 2.486 |
| C2 | 2.254 | 2.925 | +0.42 | -2.62 | 1.931 | 2.186 | 2.293 | 2.382 | 2.443 |
| U1 | 2.270 | 3.122 | +0.35 | -2.79 | 2.056 | 2.209 | 2.292 | 2.367 | 2.411 |
| C3 | 2.346 | 3.213 | +0.37 | -2.96 | 2.026 | 2.314 | 2.408 | 2.470 | 2.483 |

| Comparison | Δ RMSE (95% CI) | p |
|---|---|---|
| control vs ridge (all) | +0.156 [+0.103, +0.209] | <0.001 |
| control vs ridge (extreme) | -1.092 [-1.299, -0.885] | <0.001 |
| C2 vs ridge (all) | +0.037 [-0.016, +0.089] | 0.166 |
| C2 vs ridge (extreme) | -1.441 [-1.620, -1.262] | <0.001 |
| U1 vs ridge (all) | +0.053 [+0.006, +0.100] | 0.028 |
| U1 vs ridge (extreme) | -1.244 [-1.527, -0.961] | <0.001 |
| C3 vs ridge (all) | +0.129 [+0.071, +0.186] | <0.001 |
| C3 vs ridge (extreme) | -1.152 [-1.354, -0.951] | <0.001 |
| control vs ridge_hw5 (all) | +0.161 [+0.113, +0.210] | <0.001 |
| control vs ridge_hw5 (extreme) | -0.592 [-0.847, -0.337] | <0.001 |
| C2 vs ridge_hw5 (all) | +0.042 [-0.005, +0.089] | 0.077 |
| C2 vs ridge_hw5 (extreme) | -0.941 [-1.145, -0.737] | <0.001 |
| U1 vs ridge_hw5 (all) | +0.058 [+0.016, +0.101] | 0.008 |
| U1 vs ridge_hw5 (extreme) | -0.744 [-1.032, -0.456] | <0.001 |
| C3 vs ridge_hw5 (all) | +0.134 [+0.081, +0.187] | <0.001 |
| C3 vs ridge_hw5 (extreme) | -0.652 [-0.895, -0.409] | <0.001 |

