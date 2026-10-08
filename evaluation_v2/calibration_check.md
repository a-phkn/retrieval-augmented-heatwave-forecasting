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
| control | 0.0242 | +0.261 | 0.0234 | +0.287 | 790 |
| C2 | 0.0224 | +0.316 | 0.0214 | +0.348 | 790 |
| U1 | 0.0246 | +0.248 | 0.0228 | +0.304 | 790 |
| C3 | 0.0242 | +0.263 | 0.0227 | +0.306 | 790 |
| C4 | 0.0240 | +0.266 | 0.0228 | +0.305 | 790 |
| C4a | 0.0238 | +0.274 | 0.0224 | +0.315 | 790 |

| Comparison | Δ (95% CI) | p |
|---|---|---|
| C2 vs control, recalibrated (all) | -0.1310 [-0.161, -0.101] | <0.001 |
| C2 vs control, recalibrated (extreme) | -0.3898 [-0.526, -0.254] | <0.001 |
| C2 vs control, sqrt-Brier hot day (raw) | -0.0059 [-0.011, -0.001] | 0.024 |
| C2 vs control, sqrt-Brier hot day (recal) | -0.0067 [-0.011, -0.002] | 0.004 |
| U1 vs control, recalibrated (all) | +0.0100 [-0.031, +0.051] | 0.623 |
| U1 vs control, recalibrated (extreme) | -0.3692 [-0.550, -0.189] | 0.001 |
| U1 vs control, sqrt-Brier hot day (raw) | +0.0014 [-0.005, +0.008] | 0.683 |
| U1 vs control, sqrt-Brier hot day (recal) | -0.0018 [-0.006, +0.002] | 0.357 |
| C3 vs control, recalibrated (all) | -0.0318 [-0.051, -0.013] | 0.002 |
| C3 vs control, recalibrated (extreme) | -0.2194 [-0.358, -0.080] | 0.006 |
| C3 vs control, sqrt-Brier hot day (raw) | -0.0001 [-0.003, +0.003] | 0.926 |
| C3 vs control, sqrt-Brier hot day (recal) | -0.0021 [-0.005, +0.001] | 0.113 |
| C4 vs control, recalibrated (all) | -0.0351 [-0.053, -0.017] | <0.001 |
| C4 vs control, recalibrated (extreme) | -0.1671 [-0.337, +0.003] | 0.053 |
| C4 vs control, sqrt-Brier hot day (raw) | -0.0005 [-0.003, +0.002] | 0.697 |
| C4 vs control, sqrt-Brier hot day (recal) | -0.0020 [-0.005, +0.001] | 0.152 |
| C4a vs control, recalibrated (all) | -0.0419 [-0.063, -0.021] | <0.001 |
| C4a vs control, recalibrated (extreme) | -0.2575 [-0.396, -0.119] | 0.002 |
| C4a vs control, sqrt-Brier hot day (raw) | -0.0013 [-0.004, +0.002] | 0.411 |
| C4a vs control, sqrt-Brier hot day (recal) | -0.0031 [-0.006, +0.000] | 0.057 |

Reliability, control (recalibrated): 0.0-0.1: n 19405, forecast 0.01, observed 0.01; 0.1-0.3: n 1620, forecast 0.18, observed 0.17; 0.3-0.5: n 609, forecast 0.38, observed 0.41; 0.5-0.7: n 157, forecast 0.58, observed 0.68; 0.7-0.9: n 37, forecast 0.78, observed 0.92; 0.9-1.0: n 7, forecast 0.94, observed 1.00

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
| control | 0.0358 | -0.108 | 0.0314 | +0.029 | 750 |
| C2 | 0.0365 | -0.129 | 0.0311 | +0.037 | 750 |
| U1 | 0.0363 | -0.123 | 0.0314 | +0.026 | 750 |
| C3 | 0.0357 | -0.105 | 0.0314 | +0.027 | 750 |
| C4 | 0.0358 | -0.108 | 0.0314 | +0.027 | 750 |
| C4a | 0.0358 | -0.108 | 0.0314 | +0.028 | 750 |

| Comparison | Δ (95% CI) | p |
|---|---|---|
| C2 vs control, recalibrated (all) | -0.1284 [-0.153, -0.104] | <0.001 |
| C2 vs control, recalibrated (extreme) | -0.3523 [-0.474, -0.231] | <0.001 |
| C2 vs control, sqrt-Brier hot day (raw) | +0.0018 [-0.002, +0.006] | 0.320 |
| C2 vs control, sqrt-Brier hot day (recal) | -0.0008 [-0.002, +0.000] | 0.205 |
| U1 vs control, recalibrated (all) | -0.0900 [-0.121, -0.059] | <0.001 |
| U1 vs control, recalibrated (extreme) | -0.2205 [-0.477, +0.036] | 0.086 |
| U1 vs control, sqrt-Brier hot day (raw) | +0.0013 [-0.003, +0.005] | 0.483 |
| U1 vs control, sqrt-Brier hot day (recal) | +0.0002 [-0.001, +0.001] | 0.704 |
| C3 vs control, recalibrated (all) | -0.0183 [-0.039, +0.002] | 0.076 |
| C3 vs control, recalibrated (extreme) | -0.1134 [-0.208, -0.019] | 0.023 |
| C3 vs control, sqrt-Brier hot day (raw) | -0.0002 [-0.003, +0.002] | 0.845 |
| C3 vs control, sqrt-Brier hot day (recal) | +0.0002 [-0.001, +0.001] | 0.702 |
| C4 vs control, recalibrated (all) | -0.0291 [-0.048, -0.010] | 0.003 |
| C4 vs control, recalibrated (extreme) | -0.0634 [-0.160, +0.033] | 0.177 |
| C4 vs control, sqrt-Brier hot day (raw) | +0.0000 [-0.003, +0.003] | 0.993 |
| C4 vs control, sqrt-Brier hot day (recal) | +0.0001 [-0.001, +0.001] | 0.769 |
| C4a vs control, recalibrated (all) | -0.0344 [-0.055, -0.014] | 0.002 |
| C4a vs control, recalibrated (extreme) | -0.0698 [-0.172, +0.032] | 0.163 |
| C4a vs control, sqrt-Brier hot day (raw) | +0.0000 [-0.003, +0.003] | 0.987 |
| C4a vs control, sqrt-Brier hot day (recal) | +0.0001 [-0.001, +0.001] | 0.791 |

Reliability, control (recalibrated): 0.0-0.1: n 18844, forecast 0.02, observed 0.02; 0.1-0.3: n 2930, forecast 0.15, observed 0.12; 0.3-0.5: n 61, forecast 0.35, observed 0.28

