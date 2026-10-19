# Test results, 2019-01-01 .. 2026-09-06 (one evaluation run under the frozen protocol)

Models trained once on 1980-2016 (early stopping 2017-2018); truth = ERA5 (last ~3 months ERA5T when downloaded). Δ = first-named minus second (negative = C3-hop better). Paired cluster-jackknife, year x season.

## Primary hypotheses (WBGT; Holm over H1-H4)

| | Comparison | Δ RMSE (95% CI) | p | Holm p | Supported |
|---|---|---|---|---|---|
| H1 | C3-hop vs Delhi-only control, all days | -0.127 [-0.180, -0.075] | <0.001 | <0.001 | yes |
| H2 | C3-hop vs control, heatwave days | -0.081 [-0.261, +0.100] | 0.337 | 0.337 | no |
| H3 | C3-hop vs ridge_hw5, heatwave days | -0.718 [-0.895, -0.540] | <0.001 | <0.001 | yes |
| H4 | LSTM + Rg vs Delhi-only control, all days | -0.042 [-0.080, -0.004] | 0.033 | 0.066 | no |

## Exploratory (no confirmatory claims)

### WBGT (physical)

| Model | RMSE all / bias / SD | RMSE extreme / bias / SD |
|---|---|---|
| C3-hop | 2.258 / +0.34 / 2.23 | 3.023 / -2.66 / 1.44 |
| control | 2.386 / +0.42 / 2.35 | 3.103 / -2.75 / 1.43 |
| ridge | 2.293 / -0.28 / 2.28 | 4.151 / -3.90 / 1.43 |
| ridge_hw5 | 2.299 / +0.13 / 2.30 | 3.740 / -3.42 / 1.51 |
| C2 | 2.249 / +0.23 / 2.24 | 3.116 / -2.77 / 1.43 |
| U1_tuned | 2.268 / +0.28 / 2.25 | 3.098 / -2.71 / 1.50 |
| C3pool_tuned | 2.246 / +0.24 / 2.23 | 3.102 / -2.76 / 1.42 |
| Rg | 2.344 / +0.26 / 2.33 | 3.279 / -2.94 / 1.46 |
| C3hop_Rg | 2.255 / +0.27 / 2.24 | 3.101 / -2.74 / 1.45 |

| Comparison | Δ (95% CI) |
|---|---|
| C3-hop vs control (all) | -0.127 [-0.180, -0.075] (p <0.001) |
| C3-hop vs ridge_hw5 (all) | -0.041 [-0.093, +0.012] (p 0.123) |
| C3-hop vs ridge (all) | -0.035 [-0.090, +0.021] (p 0.207) |
| C2 vs C3-hop (all) | -0.009 [-0.031, +0.012] (p 0.380) |
| U1_tuned vs C3-hop (all) | +0.010 [-0.025, +0.045] (p 0.566) |
| C3pool_tuned vs C3-hop (all) | -0.013 [-0.033, +0.008] (p 0.207) |
| LSTM + Rg vs control (all) | -0.042 [-0.080, -0.004] (p 0.033) |
| C3-hop + Rg vs C3-hop (all) | -0.003 [-0.023, +0.016] (p 0.744) |
| C3-hop vs control (extreme) | -0.081 [-0.261, +0.100] (p 0.337) |
| C3-hop vs ridge_hw5 (extreme) | -0.718 [-0.895, -0.540] (p <0.001) |
| C3-hop vs ridge (extreme) | -1.128 [-1.319, -0.938] (p <0.001) |
| C2 vs C3-hop (extreme) | +0.093 [-0.073, +0.259] (p 0.238) |
| U1_tuned vs C3-hop (extreme) | +0.075 [-0.109, +0.260] (p 0.380) |
| C3pool_tuned vs C3-hop (extreme) | +0.079 [-0.059, +0.217] (p 0.227) |
| LSTM + Rg vs control (extreme) | +0.176 [+0.009, +0.343] (p 0.041) |
| C3-hop + Rg vs C3-hop (extreme) | +0.079 [-0.071, +0.228] (p 0.265) |

By lead (C3-hop vs control, all days): day 1 -0.172 [-0.228, -0.117] (p <0.001); day 2 -0.179 [-0.239, -0.119] (p <0.001); day 3 -0.169 [-0.234, -0.104] (p <0.001); day 4 -0.084 [-0.144, -0.023] (p 0.009); day 5 -0.047 [-0.100, +0.007] (p 0.085)
- C3-hop: hot-day Brier 0.0702 (skill +0.073, 640 hot days); CRPS all 1.217, extreme 2.438; 80% coverage all 0.77, extreme 0.29
- control: hot-day Brier 0.0708 (skill +0.065, 640 hot days); CRPS all 1.292, extreme 2.441; 80% coverage all 0.78, extreme 0.33
- C3-hop: forecast hot days per seed 142, RMSE 2.49, bias +1.37
- control: forecast hot days per seed 96, RMSE 2.62, bias +1.44

Case studies (10-seed ensemble; RMSE / bias °C at leads 1, 3, 5):

- 2019 (late May - mid June): observed max 39.2 °C; C3-hop lead 1 1.89 / +0.73; C3-hop lead 3 2.37 / +0.75; C3-hop lead 5 2.42 / +0.54; control lead 1 1.85 / +0.51; control lead 3 2.52 / +0.60; control lead 5 2.73 / +0.43
- 2022 (April - mid May): observed max 38.3 °C; C3-hop lead 1 1.82 / +0.66; C3-hop lead 3 1.88 / +0.52; C3-hop lead 5 1.85 / +0.16; control lead 1 1.77 / +0.51; control lead 3 1.73 / +0.04; control lead 5 1.78 / -0.34
- 2024 (mid May - June): observed max 37.6 °C; C3-hop lead 1 1.83 / +1.03; C3-hop lead 3 1.99 / +1.46; C3-hop lead 5 1.94 / +1.41; control lead 1 1.72 / +1.04; control lead 3 2.03 / +1.50; control lead 5 1.95 / +1.37

### Tmax

| Model | RMSE all / bias / SD | RMSE extreme / bias / SD |
|---|---|---|
| C3-hop | 2.234 / +0.23 / 2.21 | 2.307 / -1.92 / 1.27 |
| control | 2.361 / +0.25 / 2.34 | 2.548 / -2.02 / 1.53 |
| ridge | 2.276 / +0.18 / 2.27 | 2.889 / -2.54 / 1.37 |
| ridge_hw5 | 2.317 / +0.40 / 2.28 | 2.528 / -2.12 / 1.38 |
| C2 | 2.226 / +0.19 / 2.22 | 2.330 / -1.90 / 1.34 |
| U1_tuned | 2.335 / +0.20 / 2.32 | 2.294 / -1.87 / 1.30 |
| C3pool_tuned | 2.227 / +0.23 / 2.21 | 2.280 / -1.87 / 1.30 |
| Rg | 2.337 / +0.19 / 2.33 | 2.647 / -2.20 / 1.47 |
| C3hop_Rg | 2.230 / +0.18 / 2.22 | 2.426 / -2.00 / 1.36 |

| Comparison | Δ (95% CI) |
|---|---|
| C3-hop vs control (all) | -0.127 [-0.209, -0.045] (p 0.004) |
| C3-hop vs ridge_hw5 (all) | -0.083 [-0.128, -0.038] (p <0.001) |
| C3-hop vs ridge (all) | -0.043 [-0.089, +0.004] (p 0.072) |
| C2 vs C3-hop (all) | -0.008 [-0.042, +0.026] (p 0.626) |
| U1_tuned vs C3-hop (all) | +0.101 [+0.048, +0.154] (p <0.001) |
| C3pool_tuned vs C3-hop (all) | -0.006 [-0.040, +0.027] (p 0.697) |
| LSTM + Rg vs control (all) | -0.024 [-0.059, +0.012] (p 0.179) |
| C3-hop + Rg vs C3-hop (all) | -0.004 [-0.037, +0.030] (p 0.818) |
| C3-hop vs control (extreme) | -0.242 [-0.648, +0.165] (p 0.203) |
| C3-hop vs ridge_hw5 (extreme) | -0.221 [-0.538, +0.096] (p 0.143) |
| C3-hop vs ridge (extreme) | -0.582 [-0.914, -0.251] (p 0.004) |
| C2 vs C3-hop (extreme) | +0.023 [-0.249, +0.295] (p 0.847) |
| U1_tuned vs C3-hop (extreme) | -0.013 [-0.343, +0.317] (p 0.928) |
| C3pool_tuned vs C3-hop (extreme) | -0.027 [-0.263, +0.210] (p 0.796) |
| LSTM + Rg vs control (extreme) | +0.099 [-0.291, +0.490] (p 0.567) |
| C3-hop + Rg vs C3-hop (extreme) | +0.119 [-0.172, +0.409] (p 0.365) |

By lead (C3-hop vs control, all days): day 1 -0.135 [-0.189, -0.082] (p <0.001); day 2 -0.149 [-0.228, -0.071] (p <0.001); day 3 -0.152 [-0.253, -0.051] (p 0.005); day 4 -0.138 [-0.242, -0.035] (p 0.011); day 5 -0.080 [-0.170, +0.009] (p 0.077)
- C3-hop: hot-day Brier 0.0740 (skill +0.281, 645 hot days); CRPS all 1.197, extreme 1.706; 80% coverage all 0.78, extreme 0.54
- control: hot-day Brier 0.0790 (skill +0.233, 645 hot days); CRPS all 1.277, extreme 1.833; 80% coverage all 0.78, extreme 0.56
- C3-hop: forecast hot days per seed 289, RMSE 2.32, bias +0.56
- control: forecast hot days per seed 385, RMSE 2.74, bias +1.14

Case studies (10-seed ensemble; RMSE / bias °C at leads 1, 3, 5):

- 2019 (late May - mid June): observed max 45.3 °C; C3-hop lead 1 1.54 / -0.38; C3-hop lead 3 2.07 / -0.55; C3-hop lead 5 1.97 / -0.31; control lead 1 1.52 / -0.64; control lead 3 2.36 / -0.74; control lead 5 2.23 / -0.68
- 2022 (April - mid May): observed max 44.3 °C; C3-hop lead 1 1.26 / +0.00; C3-hop lead 3 1.85 / -0.17; C3-hop lead 5 2.14 / -0.08; control lead 1 1.51 / +0.08; control lead 3 2.01 / -0.00; control lead 5 2.15 / -0.12
- 2024 (mid May - June): observed max 46.2 °C; C3-hop lead 1 1.31 / -0.62; C3-hop lead 3 2.20 / -1.11; C3-hop lead 5 2.38 / -1.20; control lead 1 1.49 / -0.55; control lead 3 2.10 / -1.01; control lead 5 2.43 / -1.50

Rz side-check (WBGT, 2019-2021, all days): C3-hop + Rz vs C3-hop -0.004 [-0.040, +0.032] (p 0.791) over 1092 windows
