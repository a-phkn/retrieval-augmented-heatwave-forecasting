# Retrieval ladder screen: R2-R4 on Rg (2026-10-07)

Specified in `context/decisions.md` before it was run. No training; training years only (each fold's last 2 training years held out). Δ = RMSE (°C) of the query-only linear forecast with the rung's analogue signal minus the comparison; negative = the rung helps. 95% CI: bootstrap over the 8 held-out years (fragile).

Trained only if the gain vs the baseline, vs its random control AND vs Rg are all significant.

## Tmax

Drift-gate firing rate (R3 power check): 26.5%. Trend beta (R4): +0.097, +0.086, +0.065, +0.041 SD/decade per fold. Mean pairwise similarity of the 5 analogues (redundancy): Rg 0.790, R2 0.701.

| Rung | vs query-only baseline | vs its random control | vs Rg | Train? |
|---|---|---|---|---|
| R2 | -0.037 [-0.045, -0.028] | -0.037 [-0.045, -0.029] (R0-rand) | -0.009 [-0.016, -0.001] | yes |
| R3 | -0.026 [-0.035, -0.016] | -0.026 [-0.035, -0.016] (R3-rand) | +0.002 [-0.000, +0.005] | no |
| R4 | -0.028 [-0.037, -0.017] | -0.028 [-0.037, -0.017] (R4-rand) | -0.000 [-0.002, +0.002] | no |
| Rg (reference) | -0.028 [-0.037, -0.017] | -0.028 [-0.038, -0.018] (R0-rand) | | |

## WBGT (physical)

Drift-gate firing rate (R3 power check): 11.6%. Trend beta (R4): +0.135, +0.136, +0.118, +0.104 SD/decade per fold. Mean pairwise similarity of the 5 analogues (redundancy): Rg 0.734, R2 0.633.

| Rung | vs query-only baseline | vs its random control | vs Rg | Train? |
|---|---|---|---|---|
| R2 | -0.049 [-0.060, -0.039] | -0.049 [-0.060, -0.040] (R0-rand) | -0.010 [-0.015, -0.006] | yes |
| R3 | -0.040 [-0.050, -0.030] | -0.040 [-0.050, -0.030] (R3-rand) | -0.001 [-0.003, +0.001] | no |
| R4 | -0.037 [-0.052, -0.024] | -0.037 [-0.050, -0.024] (R4-rand) | +0.002 [-0.001, +0.005] | no |
| Rg (reference) | -0.039 [-0.051, -0.028] | -0.039 [-0.051, -0.028] (R0-rand) | | |

Notes:
- R4-rand = R0-rand with the same climate-shift adjustment; R3-rand = random draws from R3's gated pool.
- A linear check: a null says the simple analogue-mean information is absent, not that a network could not use it.
