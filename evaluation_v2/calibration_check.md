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
| U1_tuned | 2.142 / +0.16 / 2.13 | 1.709 / -1.18 / 1.23 | 2.099 | 2.259 / -1.86 | 2.072 (2.046) | 1.598 (2.200) |
| C3_tuned | 2.141 / +0.18 / 2.13 | 1.962 / -1.47 / 1.29 | 2.111 | 2.547 / -2.16 | 2.104 (2.087) | 1.910 (2.524) |
| C3pool_tuned | 2.062 / +0.10 / 2.06 | 1.768 / -1.24 / 1.26 | 2.025 | 2.278 / -1.86 | 2.033 (2.006) | 1.719 (2.257) |
| C3hop | 2.065 / +0.15 / 2.06 | 1.827 / -1.25 / 1.33 | 2.033 | 2.250 / -1.78 | 2.012 (1.994) | 1.740 (2.203) |

| Model | Brier raw | BSS raw | Brier recal | BSS recal | Hot days |
|---|---|---|---|---|---|
| control | 0.0651 | +0.241 | 0.0616 | +0.282 | 790 |
| C2 | 0.0605 | +0.295 | 0.0567 | +0.339 | 790 |
| U1 | 0.0663 | +0.227 | 0.0605 | +0.294 | 790 |
| C3 | 0.0654 | +0.238 | 0.0602 | +0.298 | 790 |
| C4 | 0.0650 | +0.243 | 0.0603 | +0.298 | 790 |
| C4a | 0.0645 | +0.248 | 0.0595 | +0.307 | 790 |
| U1_tuned | 0.0646 | +0.247 | 0.0590 | +0.312 | 790 |
| C3_tuned | 0.0641 | +0.253 | 0.0599 | +0.302 | 790 |
| C3pool_tuned | 0.0612 | +0.286 | 0.0564 | +0.343 | 790 |
| C3hop | 0.0595 | +0.307 | 0.0549 | +0.360 | 790 |

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
| U1_tuned vs control, recalibrated (all) | -0.0518 [-0.088, -0.016] | 0.006 |
| U1_tuned vs control, recalibrated (extreme) | -0.4674 [-0.657, -0.278] | <0.001 |
| U1_tuned vs control, sqrt-Brier hot day (raw) | -0.0010 [-0.013, +0.011] | 0.860 |
| U1_tuned vs control, sqrt-Brier hot day (recal) | -0.0054 [-0.013, +0.002] | 0.156 |
| C3_tuned vs control, recalibrated (all) | -0.0397 [-0.057, -0.022] | <0.001 |
| C3_tuned vs control, recalibrated (extreme) | -0.1793 [-0.291, -0.067] | 0.005 |
| C3_tuned vs control, sqrt-Brier hot day (raw) | -0.0020 [-0.006, +0.002] | 0.347 |
| C3_tuned vs control, sqrt-Brier hot day (recal) | -0.0036 [-0.006, -0.001] | 0.017 |
| C3pool_tuned vs control, recalibrated (all) | -0.1257 [-0.156, -0.095] | <0.001 |
| C3pool_tuned vs control, recalibrated (extreme) | -0.4486 [-0.623, -0.274] | <0.001 |
| C3pool_tuned vs control, sqrt-Brier hot day (raw) | -0.0078 [-0.018, +0.002] | 0.114 |
| C3pool_tuned vs control, sqrt-Brier hot day (recal) | -0.0107 [-0.020, -0.002] | 0.026 |
| C3hop vs control, recalibrated (all) | -0.1169 [-0.152, -0.082] | <0.001 |
| C3hop vs control, recalibrated (extreme) | -0.4763 [-0.633, -0.319] | <0.001 |
| C3hop vs control, sqrt-Brier hot day (raw) | -0.0113 [-0.020, -0.003] | 0.014 |
| C3hop vs control, sqrt-Brier hot day (recal) | -0.0139 [-0.023, -0.005] | 0.007 |

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
| U1_tuned | 2.247 / +0.31 / 2.22 | 3.192 / -2.88 / 1.36 | 2.186 | 3.945 / -3.72 | 2.199 (2.152) | 3.160 (3.926) |
| C3_tuned | 2.346 / +0.37 / 2.32 | 3.213 / -2.96 / 1.26 | 2.278 | 4.022 / -3.83 | 2.316 (2.255) | 3.192 (4.009) |
| C3pool_tuned | 2.252 / +0.39 / 2.22 | 2.918 / -2.61 / 1.30 | 2.172 | 3.775 / -3.56 | 2.237 (2.162) | 2.905 (3.768) |
| C3hop | 2.248 / +0.38 / 2.21 | 3.019 / -2.69 / 1.36 | 2.174 | 3.845 / -3.61 | 2.216 (2.150) | 2.991 (3.829) |

| Model | Brier raw | BSS raw | Brier recal | BSS recal | Hot days |
|---|---|---|---|---|---|
| control | 0.0633 | -0.077 | 0.0570 | +0.029 | 750 |
| C2 | 0.0649 | -0.104 | 0.0565 | +0.039 | 750 |
| U1 | 0.0649 | -0.105 | 0.0571 | +0.027 | 750 |
| C3 | 0.0634 | -0.079 | 0.0571 | +0.028 | 750 |
| C4 | 0.0635 | -0.080 | 0.0571 | +0.028 | 750 |
| C4a | 0.0635 | -0.081 | 0.0571 | +0.028 | 750 |
| U1_tuned | 0.0632 | -0.075 | 0.0570 | +0.031 | 750 |
| C3_tuned | 0.0634 | -0.079 | 0.0571 | +0.028 | 750 |
| C3pool_tuned | 0.0646 | -0.100 | 0.0564 | +0.040 | 750 |
| C3hop | 0.0649 | -0.104 | 0.0569 | +0.031 | 750 |

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
| U1_tuned vs control, recalibrated (all) | -0.1107 [-0.141, -0.081] | <0.001 |
| U1_tuned vs control, recalibrated (extreme) | -0.1913 [-0.389, +0.007] | 0.057 |
| U1_tuned vs control, sqrt-Brier hot day (raw) | -0.0003 [-0.004, +0.004] | 0.892 |
| U1_tuned vs control, sqrt-Brier hot day (recal) | -0.0002 [-0.001, +0.001] | 0.783 |
| C3_tuned vs control, recalibrated (all) | -0.0183 [-0.039, +0.002] | 0.076 |
| C3_tuned vs control, recalibrated (extreme) | -0.1134 [-0.208, -0.019] | 0.023 |
| C3_tuned vs control, sqrt-Brier hot day (raw) | +0.0001 [-0.003, +0.003] | 0.933 |
| C3_tuned vs control, sqrt-Brier hot day (recal) | +0.0002 [-0.001, +0.001] | 0.764 |
| C3pool_tuned vs control, recalibrated (all) | -0.1242 [-0.150, -0.098] | <0.001 |
| C3pool_tuned vs control, recalibrated (extreme) | -0.3612 [-0.502, -0.221] | <0.001 |
| C3pool_tuned vs control, sqrt-Brier hot day (raw) | +0.0026 [-0.002, +0.007] | 0.240 |
| C3pool_tuned vs control, sqrt-Brier hot day (recal) | -0.0014 [-0.003, +0.000] | 0.056 |
| C3hop vs control, recalibrated (all) | -0.1230 [-0.151, -0.095] | <0.001 |
| C3hop vs control, recalibrated (extreme) | -0.2904 [-0.458, -0.123] | 0.002 |
| C3hop vs control, sqrt-Brier hot day (raw) | +0.0031 [-0.002, +0.008] | 0.224 |
| C3hop vs control, sqrt-Brier hot day (recal) | -0.0003 [-0.002, +0.002] | 0.779 |

Reliability, control (recalibrated): 0.0-0.1: n 9550, forecast 0.03, observed 0.05; 0.1-0.3: n 2402, forecast 0.14, observed 0.12; 0.3-0.5: n 48, forecast 0.36, observed 0.31

