# Graph backbone gates G-D2 / G-D3 (pre-registered 2026-10-07)

**Backbone (BB\*): U1** ((a) passed in both families, (b) failed in at least one: the gain is the data).

Out of fold 2007-2018, 10 seeds x 4 folds. Δ = first minus second; negative = first better. Non-inferiority margin 0.05 °C on the upper end of the 95% CI.

## Tmax (control `A1prime_hw5`)

Candidate: **C3**; (a) vs control: True; (b) vs U1: True.

| Model | All RMSE | Extreme RMSE | G-D2 | finite | improved after epoch 1 | seed SD (control) |
|---|---|---|---|---|---|---|
| control | 2.175 | 2.153 | | | | |
| C2 | 2.051 | 1.845 | pass | True | 100% | 0.016 (0.013) |
| C3 | 2.147 | 1.904 | pass | True | 100% | 0.014 (0.013) |
| C4 | 2.142 | 1.996 | pass | True | 100% | 0.014 (0.013) |
| C4a | 2.138 | 1.881 | pass | True | 100% | 0.018 (0.013) |
| U1 | 2.199 | 1.767 | | | | |

| Comparison | Δ RMSE (95% CI) | p |
|---|---|---|
| C2 vs control (all) | -0.125 [-0.159, -0.091] | 0.000 |
| C2 vs control (extreme) | -0.308 [-0.445, -0.171] | 0.001 |
| C2 vs U1 (all) | -0.148 [-0.186, -0.111] | 0.000 |
| C2 vs U1 (extreme) | +0.079 [-0.082, +0.239] | 0.301 |
| C3 vs control (all) | -0.029 [-0.051, -0.006] | 0.014 |
| C3 vs control (extreme) | -0.249 [-0.390, -0.109] | 0.003 |
| C3 vs U1 (all) | -0.052 [-0.092, -0.013] | 0.011 |
| C3 vs U1 (extreme) | +0.137 [-0.006, +0.281] | 0.059 |
| C3 vs C2 (all) | +0.096 [+0.066, +0.126] | 0.000 |
| C3 vs C2 (extreme) | +0.059 [-0.089, +0.206] | 0.396 |
| C4 vs control (all) | -0.034 [-0.056, -0.012] | 0.004 |
| C4 vs control (extreme) | -0.157 [-0.314, -0.000] | 0.050 |
| C4 vs U1 (all) | -0.057 [-0.099, -0.016] | 0.008 |
| C4 vs U1 (extreme) | +0.230 [+0.081, +0.379] | 0.006 |
| C4 vs C2 (all) | +0.091 [+0.062, +0.120] | 0.000 |
| C4 vs C2 (extreme) | +0.151 [-0.010, +0.312] | 0.063 |
| C4a vs control (all) | -0.037 [-0.061, -0.013] | 0.004 |
| C4a vs control (extreme) | -0.273 [-0.393, -0.152] | 0.001 |
| C4a vs U1 (all) | -0.061 [-0.102, -0.019] | 0.005 |
| C4a vs U1 (extreme) | +0.114 [-0.054, +0.282] | 0.161 |
| C4a vs C2 (all) | +0.088 [+0.059, +0.117] | 0.000 |
| C4a vs C2 (extreme) | +0.035 [-0.110, +0.181] | 0.600 |
| U1 vs control (all) | +0.024 [-0.022, +0.069] | 0.305 |
| U1 vs control (extreme) | -0.387 [-0.543, -0.230] | 0.000 |

Structure claim (CI entirely below 0 vs both U1 and C2, all days): C3 no, C4 no, C4a no

## WBGT (physical) (control `A2Lr_hw5`)

Candidate: **C3**; (a) vs control: True; (b) vs U1: False.

| Model | All RMSE | Extreme RMSE | G-D2 | finite | improved after epoch 1 | seed SD (control) |
|---|---|---|---|---|---|---|
| control | 2.373 | 3.274 | | | | |
| C2 | 2.254 | 2.925 | pass | True | 100% | 0.011 (0.012) |
| C3 | 2.346 | 3.213 | pass | True | 100% | 0.015 (0.012) |
| C4 | 2.338 | 3.231 | pass | True | 100% | 0.019 (0.012) |
| C4a | 2.334 | 3.226 | pass | True | 100% | 0.007 (0.012) |
| U1 | 2.270 | 3.122 | | | | |

| Comparison | Δ RMSE (95% CI) | p |
|---|---|---|
| C2 vs control (all) | -0.119 [-0.148, -0.091] | 0.000 |
| C2 vs control (extreme) | -0.349 [-0.527, -0.171] | 0.001 |
| C2 vs U1 (all) | -0.016 [-0.041, +0.008] | 0.188 |
| C2 vs U1 (extreme) | -0.197 [-0.409, +0.015] | 0.066 |
| C3 vs control (all) | -0.027 [-0.054, -0.000] | 0.046 |
| C3 vs control (extreme) | -0.060 [-0.185, +0.065] | 0.315 |
| C3 vs U1 (all) | +0.076 [+0.042, +0.110] | 0.000 |
| C3 vs U1 (extreme) | +0.092 [-0.209, +0.393] | 0.521 |
| C3 vs C2 (all) | +0.092 [+0.069, +0.116] | 0.000 |
| C3 vs C2 (extreme) | +0.289 [+0.136, +0.442] | 0.001 |
| C4 vs control (all) | -0.036 [-0.063, -0.008] | 0.012 |
| C4 vs control (extreme) | -0.043 [-0.164, +0.079] | 0.463 |
| C4 vs U1 (all) | +0.067 [+0.035, +0.100] | 0.000 |
| C4 vs U1 (extreme) | +0.110 [-0.178, +0.398] | 0.425 |
| C4 vs C2 (all) | +0.084 [+0.060, +0.107] | 0.000 |
| C4 vs C2 (extreme) | +0.307 [+0.144, +0.469] | 0.001 |
| C4a vs control (all) | -0.040 [-0.066, -0.013] | 0.004 |
| C4a vs control (extreme) | -0.048 [-0.200, +0.104] | 0.506 |
| C4a vs U1 (all) | +0.063 [+0.029, +0.098] | 0.001 |
| C4a vs U1 (extreme) | +0.104 [-0.187, +0.395] | 0.453 |
| C4a vs C2 (all) | +0.080 [+0.058, +0.101] | 0.000 |
| C4a vs C2 (extreme) | +0.301 [+0.142, +0.461] | 0.001 |
| U1 vs control (all) | -0.103 [-0.140, -0.066] | 0.000 |
| U1 vs control (extreme) | -0.152 [-0.455, +0.150] | 0.296 |

Structure claim (CI entirely below 0 vs both U1 and C2, all days): C3 no, C4 no, C4a no

