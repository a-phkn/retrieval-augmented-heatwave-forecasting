# Bias split, recalibration and hot-day probability (decisions.md 2026-10-08, addition 2)

Out of fold 2007-2018. Recalibration: actual = a + b x forecast per fold, seed and lead, fitted on the fold's inner block (last 2 training years) and applied to the validation block. Probabilities: Gaussian around the 10-seed ensemble mean, SD from inner-block residuals; event = the hot label itself. Inner-block residuals come from the early-stopping block, so sigma is slightly optimistic (disclosed).

## Tmax

| Model | All RMSE / bias / SD | Extreme RMSE / bias / SD | Recalibrated all | Recalibrated extreme / bias | Ensemble all (recal) | Ensemble extreme (recal) |
|---|---|---|---|---|---|---|
| control | 2.175 / +0.09 / 2.17 | 2.153 / -1.71 / 1.31 | 2.150 | 2.726 / -2.35 | 2.144 (2.130) | 2.112 (2.708) |
| C2 | 2.051 / +0.12 / 2.05 | 1.845 / -1.34 / 1.27 | 2.019 | 2.337 / -1.91 | 2.020 (1.998) | 1.811 (2.319) |
| U1 | 2.199 / +0.19 / 2.19 | 1.767 / -1.23 / 1.26 | 2.160 | 2.357 / -1.97 | 2.122 (2.098) | 1.667 (2.304) |
| C3 | 2.147 / +0.17 / 2.14 | 1.904 / -1.43 / 1.25 | 2.119 | 2.507 / -2.11 | 2.114 (2.096) | 1.858 (2.485) |
| C4 | 2.142 / +0.15 / 2.14 | 1.996 / -1.52 / 1.29 | 2.115 | 2.559 / -2.16 | 2.110 (2.093) | 1.953 (2.537) |
| C4a | 2.138 / +0.20 / 2.13 | 1.881 / -1.38 / 1.27 | 2.108 | 2.469 / -2.06 | 2.106 (2.086) | 1.840 (2.448) |

| Model | Brier raw | BSS raw | Brier recal | BSS recal | Hot days |
|---|---|---|---|---|---|
| control | 0.0651 | +0.241 | 0.0616 | +0.282 | 790 |
| C2 | 0.0605 | +0.295 | 0.0567 | +0.339 | 790 |
| U1 | 0.0663 | +0.227 | 0.0605 | +0.294 | 790 |
| C3 | 0.0654 | +0.238 | 0.0602 | +0.298 | 790 |
| C4 | 0.0650 | +0.243 | 0.0603 | +0.298 | 790 |
| C4a | 0.0645 | +0.248 | 0.0595 | +0.307 | 790 |

| Comparison | Δ (95% CI) | p |
|---|---|---|
| C2 vs control, recalibrated (all) | -0.1310 [-0.161, -0.101] | <0.001 |
| C2 vs control, recalibrated (extreme) | -0.3898 [-0.526, -0.254] | <0.001 |
| C2 vs control, sqrt-Brier hot day (raw) | -0.0092 [-0.017, -0.001] | 0.031 |
| C2 vs control, sqrt-Brier hot day (recal) | -0.0102 [-0.017, -0.003] | 0.008 |
| U1 vs control, recalibrated (all) | +0.0100 [-0.031, +0.051] | 0.623 |
| U1 vs control, recalibrated (extreme) | -0.3692 [-0.550, -0.189] | 0.001 |
| U1 vs control, sqrt-Brier hot day (raw) | +0.0023 [-0.009, +0.013] | 0.656 |
| U1 vs control, sqrt-Brier hot day (recal) | -0.0022 [-0.009, +0.005] | 0.491 |
| C3 vs control, recalibrated (all) | -0.0318 [-0.051, -0.013] | 0.002 |
| C3 vs control, recalibrated (extreme) | -0.2194 [-0.358, -0.080] | 0.006 |
| C3 vs control, sqrt-Brier hot day (raw) | +0.0005 [-0.005, +0.006] | 0.824 |
| C3 vs control, sqrt-Brier hot day (recal) | -0.0028 [-0.007, +0.002] | 0.179 |
| C4 vs control, recalibrated (all) | -0.0351 [-0.053, -0.017] | <0.001 |
| C4 vs control, recalibrated (extreme) | -0.1671 [-0.337, +0.003] | 0.053 |
| C4 vs control, sqrt-Brier hot day (raw) | -0.0003 [-0.005, +0.005] | 0.897 |
| C4 vs control, sqrt-Brier hot day (recal) | -0.0028 [-0.007, +0.002] | 0.212 |
| C4a vs control, recalibrated (all) | -0.0419 [-0.063, -0.021] | <0.001 |
| C4a vs control, recalibrated (extreme) | -0.2575 [-0.396, -0.119] | 0.002 |
| C4a vs control, sqrt-Brier hot day (raw) | -0.0013 [-0.007, +0.004] | 0.624 |
| C4a vs control, sqrt-Brier hot day (recal) | -0.0043 [-0.010, +0.001] | 0.095 |

Reliability, control (recalibrated): 0.0-0.1: n 5450, forecast 0.02, observed 0.01; 0.1-0.3: n 1955, forecast 0.18, observed 0.15; 0.3-0.5: n 734, forecast 0.38, observed 0.38; 0.5-0.7: n 160, forecast 0.58, observed 0.68; 0.7-0.9: n 35, forecast 0.77, observed 0.91; 0.9-1.0: n 6, forecast 0.93, observed 1.00

## WBGT (physical)

| Model | All RMSE / bias / SD | Extreme RMSE / bias / SD | Recalibrated all | Recalibrated extreme / bias | Ensemble all (recal) | Ensemble extreme (recal) |
|---|---|---|---|---|---|---|
| control | 2.373 / +0.44 / 2.33 | 3.274 / -3.02 / 1.27 | 2.297 | 4.136 / -3.95 | 2.354 (2.284) | 3.263 (4.130) |
| C2 | 2.254 / +0.42 / 2.21 | 2.925 / -2.62 / 1.30 | 2.168 | 3.783 / -3.56 | 2.236 (2.155) | 2.911 (3.776) |
| U1 | 2.270 / +0.35 / 2.24 | 3.122 / -2.79 / 1.40 | 2.207 | 3.915 / -3.68 | 2.221 (2.169) | 3.081 (3.892) |
| C3 | 2.346 / +0.37 / 2.32 | 3.213 / -2.96 / 1.26 | 2.278 | 4.022 / -3.83 | 2.316 (2.255) | 3.192 (4.009) |
| C4 | 2.338 / +0.38 / 2.31 | 3.231 / -2.98 / 1.24 | 2.268 | 4.072 / -3.88 | 2.314 (2.251) | 3.213 (4.062) |
| C4a | 2.334 / +0.38 / 2.30 | 3.226 / -2.97 / 1.25 | 2.262 | 4.066 / -3.88 | 2.307 (2.242) | 3.205 (4.053) |

| Model | Brier raw | BSS raw | Brier recal | BSS recal | Hot days |
|---|---|---|---|---|---|
| control | 0.0633 | -0.077 | 0.0570 | +0.029 | 750 |
| C2 | 0.0649 | -0.104 | 0.0565 | +0.039 | 750 |
| U1 | 0.0649 | -0.105 | 0.0571 | +0.027 | 750 |
| C3 | 0.0634 | -0.079 | 0.0571 | +0.028 | 750 |
| C4 | 0.0635 | -0.080 | 0.0571 | +0.028 | 750 |
| C4a | 0.0635 | -0.081 | 0.0571 | +0.028 | 750 |

| Comparison | Δ (95% CI) | p |
|---|---|---|
| C2 vs control, recalibrated (all) | -0.1284 [-0.153, -0.104] | <0.001 |
| C2 vs control, recalibrated (extreme) | -0.3523 [-0.474, -0.231] | <0.001 |
| C2 vs control, sqrt-Brier hot day (raw) | +0.0031 [-0.002, +0.008] | 0.215 |
| C2 vs control, sqrt-Brier hot day (recal) | -0.0011 [-0.003, +0.001] | 0.188 |
| U1 vs control, recalibrated (all) | -0.0900 [-0.121, -0.059] | <0.001 |
| U1 vs control, recalibrated (extreme) | -0.2205 [-0.477, +0.036] | 0.086 |
| U1 vs control, sqrt-Brier hot day (raw) | +0.0032 [-0.002, +0.008] | 0.203 |
| U1 vs control, sqrt-Brier hot day (recal) | +0.0002 [-0.001, +0.002] | 0.755 |
| C3 vs control, recalibrated (all) | -0.0183 [-0.039, +0.002] | 0.076 |
| C3 vs control, recalibrated (extreme) | -0.1134 [-0.208, -0.019] | 0.023 |
| C3 vs control, sqrt-Brier hot day (raw) | +0.0001 [-0.003, +0.003] | 0.933 |
| C3 vs control, sqrt-Brier hot day (recal) | +0.0002 [-0.001, +0.001] | 0.764 |
| C4 vs control, recalibrated (all) | -0.0291 [-0.048, -0.010] | 0.003 |
| C4 vs control, recalibrated (extreme) | -0.0634 [-0.160, +0.033] | 0.177 |
| C4 vs control, sqrt-Brier hot day (raw) | +0.0003 [-0.003, +0.004] | 0.837 |
| C4 vs control, sqrt-Brier hot day (recal) | +0.0002 [-0.001, +0.001] | 0.708 |
| C4a vs control, recalibrated (all) | -0.0344 [-0.055, -0.014] | 0.002 |
| C4a vs control, recalibrated (extreme) | -0.0698 [-0.172, +0.032] | 0.163 |
| C4a vs control, sqrt-Brier hot day (raw) | +0.0004 [-0.003, +0.004] | 0.801 |
| C4a vs control, sqrt-Brier hot day (recal) | +0.0001 [-0.001, +0.001] | 0.767 |

Reliability, control (recalibrated): 0.0-0.1: n 9550, forecast 0.03, observed 0.05; 0.1-0.3: n 2402, forecast 0.14, observed 0.12; 0.3-0.5: n 48, forecast 0.36, observed 0.31

