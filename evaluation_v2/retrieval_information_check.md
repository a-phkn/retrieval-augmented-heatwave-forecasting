# Retrieval information check (2026-10-07)

Specified in `context/decisions.md` before it was run. No training; training years only (each fold's last 2 training years held out; no validation block used).

Question: do the analogues' own next 5 days add information beyond the query's own 14 input days? RMSE in °C, pooled over 4 folds x leads 1-5. Δ = augmented minus comparison; negative = analogues help. 95% CI: bootstrap over the 8 held-out years (few clusters: fragile).

## Tmax

Held-out query-days: 14530 (extreme: 385); fit queries: 41439.

**Information present (pre-specified rule):** R0 no, R1 no

| Comparison | All days Δ RMSE (95% CI) | Extreme days Δ RMSE (95% CI, descriptive) |
|---|---|---|
| R0 vs query-only baseline | -0.001 [-0.002, +0.000] | -0.002 [-0.007, +0.003] |
| R0-rand vs query-only baseline | +0.000 [-0.000, +0.000] | +0.000 [-0.001, +0.001] |
| R1 vs query-only baseline | -0.000 [-0.001, +0.000] | -0.002 [-0.006, +0.004] |
| R1-rand vs query-only baseline | +0.000 [-0.000, +0.001] | -0.000 [-0.001, +0.001] |
| R0 vs R0-rand | -0.001 [-0.002, +0.000] | -0.002 [-0.008, +0.003] |
| R1 vs R1-rand | -0.001 [-0.001, +0.000] | -0.001 [-0.006, +0.004] |

Query-only baseline RMSE: all 2.185 °C, extreme 2.847 °C.

By lead (RMSE °C, all days) and correlation of the analogue signal with the truth (held-out):

| Lead | Baseline | R0 | R0-rand | R1 | R1-rand | corr R0 | corr R0-rand | corr R1 | corr R1-rand |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.555 | 1.555 | 1.556 | 1.555 | 1.556 | +0.694 | +0.005 | +0.686 | -0.009 |
| 2 | 2.071 | 2.071 | 2.071 | 2.071 | 2.071 | +0.469 | -0.004 | +0.464 | -0.015 |
| 3 | 2.300 | 2.301 | 2.301 | 2.301 | 2.301 | +0.320 | +0.003 | +0.317 | -0.020 |
| 4 | 2.406 | 2.406 | 2.406 | 2.407 | 2.406 | +0.249 | +0.008 | +0.255 | -0.020 |
| 5 | 2.467 | 2.464 | 2.467 | 2.465 | 2.468 | +0.153 | +0.006 | +0.165 | -0.008 |

## WBGT (physical)

Held-out query-days: 14530 (extreme: 205); fit queries: 41439.

**Information present (pre-specified rule):** R0 no, R1 no

| Comparison | All days Δ RMSE (95% CI) | Extreme days Δ RMSE (95% CI, descriptive) |
|---|---|---|
| R0 vs query-only baseline | +0.000 [-0.002, +0.003] | -0.012 [-0.021, -0.001] |
| R0-rand vs query-only baseline | -0.000 [-0.000, +0.000] | -0.001 [-0.001, +0.000] |
| R1 vs query-only baseline | +0.000 [-0.002, +0.002] | -0.009 [-0.018, +0.002] |
| R1-rand vs query-only baseline | +0.000 [+0.000, +0.000] | -0.001 [-0.002, -0.000] |
| R0 vs R0-rand | +0.000 [-0.002, +0.003] | -0.011 [-0.021, -0.001] |
| R1 vs R1-rand | -0.000 [-0.002, +0.002] | -0.008 [-0.017, +0.002] |

Query-only baseline RMSE: all 2.253 °C, extreme 4.539 °C.

By lead (RMSE °C, all days) and correlation of the analogue signal with the truth (held-out):

| Lead | Baseline | R0 | R0-rand | R1 | R1-rand | corr R0 | corr R0-rand | corr R1 | corr R1-rand |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.973 | 1.972 | 1.973 | 1.972 | 1.973 | +0.277 | +0.002 | +0.267 | -0.002 |
| 2 | 2.237 | 2.237 | 2.237 | 2.237 | 2.238 | +0.135 | +0.008 | +0.126 | -0.009 |
| 3 | 2.314 | 2.315 | 2.314 | 2.315 | 2.314 | +0.060 | -0.005 | +0.050 | -0.007 |
| 4 | 2.349 | 2.350 | 2.349 | 2.350 | 2.350 | +0.037 | +0.003 | +0.043 | -0.006 |
| 5 | 2.368 | 2.369 | 2.369 | 2.369 | 2.369 | +0.030 | -0.002 | +0.043 | -0.005 |

Notes:
- Random versions (R0-rand, R1-rand): 10 seeds; RMSE is the mean over seeds, as in the main comparison.
- The signal enters a linear model. A network could in principle extract non-linear information this misses; a null here says the simple, AnEn-style information is absent.
- The analogue features are v1's 17 Tmax-based features for both families (as in the runs).
