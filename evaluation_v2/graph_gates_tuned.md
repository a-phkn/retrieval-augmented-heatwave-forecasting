# Backbone after the tuning round (amended rule, decisions.md 2026-10-08)

> **Note added 2026-10-25:** this report predates the final choice. The user chose **C3-hop** over the provisional U1
> on 2026-10-12 (`context/decisions.md`), and C3-hop was the model tested on 2019–2026.

**Provisional backbone: U1 (tuned)** (lowest WBGT all-days RMSE among the eligible ['U1', 'C2', 'C3pool', 'C3hop'] (ties within 0.02 °C to the simpler). PROVISIONAL: awaiting the user's approval).

With C3-hop: U1 (tuned). Without C3-hop (the five candidates declared before the review): U1 (tuned).

Out of fold 2007-2018, 10 seeds x 4 folds. U1 / C3 / C3-pool: hyperparameters selected on the inner 2-year blocks (`evaluation_v2/graph_tuning.md`); C3-hop uses tuned C3's; the control and C2 are not retuned (comparisons favour the tuned models; disclosed). Non-inferiority margin 0.05 °C on the upper end of the 95% CI. U1 below = tuned U1. Ridges: `evaluation/ridge_baselines.py` (reported, not eligible).

Eligible in both families: control no, U1 yes, C2 yes, C3 no, C3pool yes, C3hop yes

## Tmax (control `A1prime_hw5`)

| Model | All RMSE | Extreme RMSE | G-D2 |
|---|---|---|---|
| control | 2.175 | 2.153 |  |
| C2 | 2.051 | 1.845 | pass |
| U1 | 2.142 | 1.709 |  |
| C3 | 2.141 | 1.962 | pass |
| C3pool | 2.062 | 1.768 | pass |
| C3hop | 2.065 | 1.827 | pass |
| ridge | 2.104 | 2.908 |  |
| ridge_hw5 | 2.138 | 2.456 |  |

Structure claim (CI entirely below 0 vs both U1 and C2, all days): C3 no, C3pool no, C3hop no

| Comparison | Δ RMSE (95% CI) | p | Selection-free years: Δ all (95% CI) |
|---|---|---|---|
| C2 vs control (all) | -0.125 [-0.159, -0.091] | 0.000 | -0.131 [-0.174, -0.089] |
| C2 vs control (extreme) | -0.308 [-0.445, -0.171] | 0.001 |  |
| C2 vs U1 (all) | -0.091 [-0.118, -0.064] | 0.000 | -0.065 [-0.102, -0.028] |
| C2 vs U1 (extreme) | +0.136 [-0.048, +0.320] | 0.131 |  |
| C3 vs control (all) | -0.035 [-0.056, -0.014] | 0.002 | -0.048 [-0.075, -0.022] |
| C3 vs control (extreme) | -0.191 [-0.330, -0.053] | 0.012 |  |
| C3 vs U1 (all) | -0.001 [-0.039, +0.037] | 0.961 | +0.018 [-0.039, +0.074] |
| C3 vs U1 (extreme) | +0.253 [-0.031, +0.536] | 0.075 |  |
| C3pool vs control (all) | -0.114 [-0.147, -0.080] | 0.000 | -0.113 [-0.155, -0.070] |
| C3pool vs control (extreme) | -0.385 [-0.574, -0.197] | 0.001 |  |
| C3pool vs U1 (all) | -0.080 [-0.108, -0.052] | 0.000 | -0.047 [-0.083, -0.010] |
| C3pool vs U1 (extreme) | +0.059 [-0.168, +0.285] | 0.576 |  |
| C3hop vs control (all) | -0.110 [-0.147, -0.074] | 0.000 | -0.127 [-0.183, -0.071] |
| C3hop vs control (extreme) | -0.327 [-0.535, -0.119] | 0.006 |  |
| C3hop vs U1 (all) | -0.076 [-0.106, -0.047] | 0.000 | -0.061 [-0.096, -0.026] |
| C3hop vs U1 (extreme) | +0.117 [-0.011, +0.246] | 0.070 |  |
| U1 vs control (all) | -0.034 [-0.075, +0.008] | 0.106 | -0.066 [-0.120, -0.011] |
| U1 vs control (extreme) | -0.444 [-0.655, -0.234] | 0.001 |  |
| control vs U1 (all) | +0.034 [-0.008, +0.075] | 0.106 | +0.066 [+0.011, +0.120] |
| control vs U1 (extreme) | +0.444 [+0.234, +0.655] | 0.001 |  |
| C3pool vs C3 (all) | -0.079 [-0.110, -0.048] | 0.000 | -0.064 [-0.107, -0.022] |
| C3pool vs C3 (extreme) | -0.194 [-0.423, +0.034] | 0.088 |  |
| C3pool vs C2 (all) | +0.011 [-0.006, +0.028] | 0.185 | +0.019 [-0.003, +0.041] |
| C3pool vs C2 (extreme) | -0.077 [-0.198, +0.044] | 0.185 |  |
| C3 vs C2 (all) | +0.090 [+0.059, +0.121] | 0.000 | +0.083 [+0.041, +0.125] |
| C3 vs C2 (extreme) | +0.117 [-0.076, +0.310] | 0.208 |  |
| C3hop vs C3 (all) | -0.075 [-0.107, -0.044] | 0.000 | -0.079 [-0.131, -0.026] |
| C3hop vs C3 (extreme) | -0.136 [-0.410, +0.139] | 0.297 |  |
| C3hop vs C2 (all) | +0.015 [-0.009, +0.038] | 0.214 | +0.004 [-0.025, +0.034] |
| C3hop vs C2 (extreme) | -0.019 [-0.180, +0.143] | 0.802 |  |
| control vs ridge (all) | +0.072 [+0.032, +0.112] | 0.001 | +0.073 [+0.004, +0.142] |
| control vs ridge (extreme) | -0.754 [-1.052, -0.457] | 0.000 |  |
| control vs ridge_hw5 (all) | +0.038 [-0.002, +0.078] | 0.065 | +0.036 [-0.028, +0.101] |
| control vs ridge_hw5 (extreme) | -0.303 [-0.577, -0.029] | 0.033 |  |
| C2 vs ridge (all) | -0.053 [-0.096, -0.011] | 0.015 | -0.058 [-0.132, +0.015] |
| C2 vs ridge (extreme) | -1.062 [-1.344, -0.781] | 0.000 |  |
| C2 vs ridge_hw5 (all) | -0.087 [-0.123, -0.052] | 0.000 | -0.095 [-0.155, -0.035] |
| C2 vs ridge_hw5 (extreme) | -0.611 [-0.861, -0.361] | 0.000 |  |
| U1 vs ridge (all) | +0.038 [-0.012, +0.088] | 0.136 | +0.007 [-0.077, +0.091] |
| U1 vs ridge (extreme) | -1.198 [-1.486, -0.911] | 0.000 |  |
| U1 vs ridge_hw5 (all) | +0.004 [-0.040, +0.048] | 0.852 | -0.030 [-0.093, +0.034] |
| U1 vs ridge_hw5 (extreme) | -0.747 [-0.968, -0.526] | 0.000 |  |
| C3 vs ridge (all) | +0.037 [-0.002, +0.076] | 0.063 | +0.025 [-0.041, +0.091] |
| C3 vs ridge (extreme) | -0.945 [-1.291, -0.600] | 0.000 |  |
| C3 vs ridge_hw5 (all) | +0.003 [-0.035, +0.042] | 0.870 | -0.012 [-0.074, +0.050] |
| C3 vs ridge_hw5 (extreme) | -0.494 [-0.846, -0.143] | 0.011 |  |
| C3pool vs ridge (all) | -0.042 [-0.083, -0.001] | 0.046 | -0.040 [-0.109, +0.030] |
| C3pool vs ridge (extreme) | -1.140 [-1.452, -0.828] | 0.000 |  |
| C3pool vs ridge_hw5 (all) | -0.076 [-0.110, -0.041] | 0.000 | -0.076 [-0.134, -0.019] |
| C3pool vs ridge_hw5 (extreme) | -0.688 [-0.971, -0.406] | 0.000 |  |
| C3hop vs ridge (all) | -0.039 [-0.085, +0.008] | 0.104 | -0.054 [-0.137, +0.029] |
| C3hop vs ridge (extreme) | -1.081 [-1.344, -0.818] | 0.000 |  |
| C3hop vs ridge_hw5 (all) | -0.072 [-0.112, -0.033] | 0.001 | -0.091 [-0.156, -0.026] |
| C3hop vs ridge_hw5 (extreme) | -0.630 [-0.838, -0.422] | 0.000 |  |

## WBGT (physical) (control `A2Lr_hw5`)

| Model | All RMSE | Extreme RMSE | G-D2 |
|---|---|---|---|
| control | 2.373 | 3.274 |  |
| C2 | 2.254 | 2.925 | pass |
| U1 | 2.247 | 3.192 |  |
| C3 | 2.346 | 3.213 | pass |
| C3pool | 2.252 | 2.918 | pass |
| C3hop | 2.248 | 3.019 | pass |
| ridge | 2.217 | 4.366 |  |
| ridge_hw5 | 2.212 | 3.866 |  |

Structure claim (CI entirely below 0 vs both U1 and C2, all days): C3 no, C3pool no, C3hop no

| Comparison | Δ RMSE (95% CI) | p | Selection-free years: Δ all (95% CI) |
|---|---|---|---|
| C2 vs control (all) | -0.119 [-0.148, -0.091] | 0.000 | -0.118 [-0.160, -0.076] |
| C2 vs control (extreme) | -0.349 [-0.527, -0.171] | 0.001 |  |
| C2 vs U1 (all) | +0.007 [-0.018, +0.031] | 0.580 | -0.004 [-0.041, +0.033] |
| C2 vs U1 (extreme) | -0.267 [-0.416, -0.118] | 0.002 |  |
| C3 vs control (all) | -0.027 [-0.054, -0.000] | 0.046 | -0.030 [-0.071, +0.011] |
| C3 vs control (extreme) | -0.060 [-0.185, +0.065] | 0.315 |  |
| C3 vs U1 (all) | +0.099 [+0.064, +0.134] | 0.000 | +0.084 [+0.041, +0.127] |
| C3 vs U1 (extreme) | +0.022 [-0.200, +0.244] | 0.837 |  |
| C3pool vs control (all) | -0.121 [-0.152, -0.091] | 0.000 | -0.118 [-0.164, -0.073] |
| C3pool vs control (extreme) | -0.356 [-0.562, -0.150] | 0.002 |  |
| C3pool vs U1 (all) | +0.005 [-0.020, +0.029] | 0.688 | -0.004 [-0.043, +0.035] |
| C3pool vs U1 (extreme) | -0.274 [-0.415, -0.133] | 0.001 |  |
| C3hop vs control (all) | -0.125 [-0.159, -0.092] | 0.000 | -0.123 [-0.171, -0.075] |
| C3hop vs control (extreme) | -0.255 [-0.480, -0.031] | 0.029 |  |
| C3hop vs U1 (all) | +0.001 [-0.023, +0.025] | 0.955 | -0.009 [-0.047, +0.029] |
| C3hop vs U1 (extreme) | -0.173 [-0.343, -0.004] | 0.046 |  |
| U1 vs control (all) | -0.126 [-0.163, -0.089] | 0.000 | -0.114 [-0.164, -0.064] |
| U1 vs control (extreme) | -0.082 [-0.325, +0.161] | 0.478 |  |
| control vs U1 (all) | +0.126 [+0.089, +0.163] | 0.000 | +0.114 [+0.064, +0.164] |
| control vs U1 (extreme) | +0.082 [-0.161, +0.325] | 0.478 |  |
| C3pool vs C3 (all) | -0.094 [-0.118, -0.070] | 0.000 | -0.088 [-0.121, -0.056] |
| C3pool vs C3 (extreme) | -0.296 [-0.472, -0.119] | 0.003 |  |
| C3pool vs C2 (all) | -0.002 [-0.014, +0.010] | 0.758 | -0.000 [-0.015, +0.014] |
| C3pool vs C2 (extreme) | -0.007 [-0.083, +0.069] | 0.849 |  |
| C3 vs C2 (all) | +0.092 [+0.069, +0.116] | 0.000 | +0.088 [+0.057, +0.119] |
| C3 vs C2 (extreme) | +0.289 [+0.136, +0.442] | 0.001 |  |
| C3hop vs C3 (all) | -0.098 [-0.126, -0.071] | 0.000 | -0.093 [-0.132, -0.054] |
| C3hop vs C3 (extreme) | -0.195 [-0.397, +0.007] | 0.057 |  |
| C3hop vs C2 (all) | -0.006 [-0.023, +0.011] | 0.470 | -0.005 [-0.029, +0.019] |
| C3hop vs C2 (extreme) | +0.094 [-0.042, +0.230] | 0.160 |  |
| control vs ridge (all) | +0.156 [+0.103, +0.209] | 0.000 | +0.103 [+0.036, +0.169] |
| control vs ridge (extreme) | -1.092 [-1.299, -0.885] | 0.000 |  |
| control vs ridge_hw5 (all) | +0.161 [+0.113, +0.210] | 0.000 | +0.141 [+0.080, +0.202] |
| control vs ridge_hw5 (extreme) | -0.592 [-0.847, -0.337] | 0.000 |  |
| C2 vs ridge (all) | +0.037 [-0.016, +0.089] | 0.166 | -0.016 [-0.070, +0.039] |
| C2 vs ridge (extreme) | -1.441 [-1.620, -1.262] | 0.000 |  |
| C2 vs ridge_hw5 (all) | +0.042 [-0.005, +0.089] | 0.077 | +0.023 [-0.028, +0.073] |
| C2 vs ridge_hw5 (extreme) | -0.941 [-1.145, -0.737] | 0.000 |  |
| U1 vs ridge (all) | +0.030 [-0.017, +0.077] | 0.207 | -0.012 [-0.052, +0.028] |
| U1 vs ridge (extreme) | -1.174 [-1.380, -0.967] | 0.000 |  |
| U1 vs ridge_hw5 (all) | +0.035 [-0.007, +0.078] | 0.099 | +0.026 [-0.014, +0.067] |
| U1 vs ridge_hw5 (extreme) | -0.674 [-0.893, -0.455] | 0.000 |  |
| C3 vs ridge (all) | +0.129 [+0.071, +0.186] | 0.000 | +0.072 [+0.017, +0.128] |
| C3 vs ridge (extreme) | -1.152 [-1.354, -0.951] | 0.000 |  |
| C3 vs ridge_hw5 (all) | +0.134 [+0.081, +0.187] | 0.000 | +0.111 [+0.062, +0.159] |
| C3 vs ridge_hw5 (extreme) | -0.652 [-0.895, -0.409] | 0.000 |  |
| C3pool vs ridge (all) | +0.035 [-0.017, +0.086] | 0.182 | -0.016 [-0.072, +0.041] |
| C3pool vs ridge (extreme) | -1.448 [-1.652, -1.244] | 0.000 |  |
| C3pool vs ridge_hw5 (all) | +0.040 [-0.006, +0.086] | 0.084 | +0.022 [-0.029, +0.074] |
| C3pool vs ridge_hw5 (extreme) | -0.948 [-1.173, -0.723] | 0.000 |  |
| C3hop vs ridge (all) | +0.031 [-0.021, +0.082] | 0.239 | -0.020 [-0.075, +0.034] |
| C3hop vs ridge (extreme) | -1.347 [-1.532, -1.163] | 0.000 |  |
| C3hop vs ridge_hw5 (all) | +0.036 [-0.008, +0.080] | 0.108 | +0.018 [-0.031, +0.066] |
| C3hop vs ridge_hw5 (extreme) | -0.847 [-1.056, -0.639] | 0.000 |  |

Selection-free years = target dates in (2007, 2010, 2013, 2016, 2017, 2018): never an inner block of any fold, so never used for early stopping or tuning selection (the f2-f4 inner blocks 2008-09, 2011-12, 2014-15 lie inside 2007-2018).
