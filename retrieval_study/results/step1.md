# Retrieval study, step 1: why retrieval did not help (RS-1)

Pre-registered in `context/decisions_retrieval.md` (RS-1) before anything was run. Development folds only (validation blocks 2007-2018 pooled; nothing from 2019 on is read). RMSE in °C. Δ = RMSE of the second minus RMSE of the first; negative = the retrieval helps. 95% CIs: paired cluster jackknife over year x season clusters (36 on all days, 12 on heatwave days). Random arms: 10 seeds.

Bases: **B0** own history only; **B1** + season; **B2** + all Delhi inputs; **B3** + region. Arms: **RAFT** as published (Han et al. 2025); **R0** our own retrieval as in the runs; each with its random control.

## Tmax

### The information ladder, all days

| Base | RMSE, no retrieval | RAFT vs none | RAFT vs its random control | R0 vs none | R0 vs its random control |
|---|---|---|---|---|---|
| B0 (own history only) | 2.421 | +0.001 [-0.003, +0.004] | -0.000 [-0.003, +0.003] | -0.148 [-0.170, -0.126] **(CI < 0)** | -0.134 [-0.155, -0.112] **(CI < 0)** |
| B1 (+ season) | 2.168 | +0.000 [-0.002, +0.002] | -0.000 [-0.002, +0.002] | +0.003 [-0.002, +0.009] | +0.005 [-0.001, +0.010] |
| B2 (+ all Delhi inputs) | 2.127 | +0.000 [-0.002, +0.002] | -0.001 [-0.002, +0.001] | +0.001 [-0.002, +0.004] | +0.001 [-0.001, +0.004] |
| B3 (+ region) | 2.069 | -0.000 [-0.002, +0.002] | -0.001 [-0.002, +0.001] | +0.001 [-0.001, +0.004] | +0.002 [-0.002, +0.006] |

### The information ladder, heatwave days

| Base | RMSE, no retrieval | RAFT vs none | RAFT vs its random control | R0 vs none | R0 vs its random control |
|---|---|---|---|---|---|
| B0 (own history only) | 2.610 | +0.006 [-0.012, +0.024] | +0.006 [-0.011, +0.023] | +0.065 [-0.081, +0.211] | -0.185 [-0.344, -0.027] **(CI < 0)** |
| B1 (+ season) | 3.012 | +0.007 [-0.011, +0.024] | +0.006 [-0.011, +0.024] | -0.035 [-0.084, +0.014] | -0.025 [-0.083, +0.033] |
| B2 (+ all Delhi inputs) | 2.675 | +0.002 [-0.014, +0.018] | +0.003 [-0.014, +0.019] | +0.002 [-0.016, +0.020] | -0.001 [-0.024, +0.023] |
| B3 (+ region) | 2.542 | +0.002 [-0.010, +0.013] | +0.003 [-0.010, +0.015] | +0.001 [-0.018, +0.019] | -0.002 [-0.029, +0.024] |

### RAFT's add-on mode on our networks (10-seed ensemble mean)

| Network | Days | RMSE alone | + RAFT vs alone | + RAFT vs + RAFT-rand |
|---|---|---|---|---|
| C3-hop | all days | 2.012 | -0.006 [-0.026, +0.013] | -0.003 [-0.007, +0.001] |
| C3-hop | heatwave days | 1.740 | +0.154 [+0.064, +0.243] (CI > 0) | +0.013 [-0.000, +0.025] |
| Delhi-only LSTM | all days | 2.144 | -0.004 [-0.024, +0.016] | -0.002 [-0.007, +0.003] |
| Delhi-only LSTM | heatwave days | 2.112 | +0.191 [+0.117, +0.266] (CI > 0) | +0.018 [-0.008, +0.043] |

### Diagnostics

4367 validation queries, 271 with a heatwave day among their 5 target days.

**Does a similar past mean a similar future?** Spearman correlation between key similarity and value similarity (minus the RMSE between the 5-day changes), over 200 random candidates per query. RAFT reports 0.60 on its ETTh1 data (definition not given, so the comparison is rough).

| Key | All queries | Heatwave queries |
|---|---|---|
| RAFT (Pearson of the 14 days) | +0.018 | +0.026 |
| R0 (cosine of the 17 features) | +0.080 | -0.041 |

**Can the key find the right analogues?** The 5 candidates whose actual 5-day changes were closest to the query's (the oracle analogues): their percentile rank under each key (50 = chance, 100 = ranked first).

| Key | Median, all | Median, heatwave queries | Mean, all |
|---|---|---|---|
| RAFT | 52.4 | 57.0 | 51.6 |
| R0 | 60.0 | 58.0 | 56.8 |

**The retrievals' own forecasts** (no model; oracle = last value + the oracle analogues' mean change, an upper bound that uses the truth):

| Forecast | RMSE all | Bias all | RMSE heatwave days | Bias heatwave days |
|---|---|---|---|---|
| Oracle analogues (uses the truth) | 0.290 | -0.00 | 0.315 | -0.08 |
| RAFT, change-based (last value + weighted change) | 2.863 | -0.00 | 3.057 | -1.92 |
| R0, level-based (analogues' mean raw future) | 2.578 | -0.33 | 3.305 | -2.78 |

**Cold days** (heatwave-day rows):

| Lead | Rows | R0 analogues' raw future - truth | RAFT change-based - truth | No candidate as hot as the truth |
|---|---|---|---|---|
| 1 | 146 | -1.56 | -0.61 | 0% |
| 2 | 146 | -2.40 | -1.45 | 0% |
| 3 | 146 | -2.95 | -2.00 | 0% |
| 4 | 146 | -3.36 | -2.60 | 0% |
| 5 | 146 | -3.63 | -2.96 | 0% |

Query's last input value minus its R0 analogues' mean last value: +0.20 °C over all queries, +0.25 °C for queries with a heatwave target day.

### The predictions (automatic reading)

| Prediction | Holds |
|---|---|
| P1 | yes |
| P2 (B0 gain and beats random) | yes |
| P2 (shrinks B0 -> B1) | yes |
| P2 (no gain on B2, B3) | yes |
| P2 holds | yes |
| P3 | yes |
| P4 | yes |
| P5 | yes |
| P6 | yes |
| P7 | yes |

## WBGT (physical)

### The information ladder, all days

| Base | RMSE, no retrieval | RAFT vs none | RAFT vs its random control | R0 vs none | R0 vs its random control |
|---|---|---|---|---|---|
| B0 (own history only) | 2.557 | -0.002 [-0.007, +0.002] | -0.003 [-0.008, +0.001] | -0.143 [-0.166, -0.120] **(CI < 0)** | -0.133 [-0.156, -0.110] **(CI < 0)** |
| B1 (+ season) | 2.355 | -0.001 [-0.002, +0.001] | -0.001 [-0.003, +0.001] | -0.009 [-0.014, -0.003] **(CI < 0)** | -0.009 [-0.014, -0.003] **(CI < 0)** |
| B2 (+ all Delhi inputs) | 2.277 | +0.000 [-0.001, +0.001] | -0.001 [-0.002, +0.000] | -0.001 [-0.003, +0.001] | -0.001 [-0.004, +0.001] |
| B3 (+ region) | 2.161 | +0.000 [-0.001, +0.001] | -0.000 [-0.001, +0.001] | -0.000 [-0.003, +0.002] | -0.001 [-0.003, +0.002] |

### The information ladder, heatwave days

| Base | RMSE, no retrieval | RAFT vs none | RAFT vs its random control | R0 vs none | R0 vs its random control |
|---|---|---|---|---|---|
| B0 (own history only) | 4.301 | -0.004 [-0.027, +0.020] | -0.004 [-0.026, +0.018] | -0.031 [-0.200, +0.138] | -0.183 [-0.345, -0.021] **(CI < 0)** |
| B1 (+ season) | 4.375 | +0.003 [-0.012, +0.018] | +0.001 [-0.013, +0.016] | -0.026 [-0.065, +0.012] | -0.025 [-0.065, +0.014] |
| B2 (+ all Delhi inputs) | 4.023 | -0.002 [-0.013, +0.008] | -0.006 [-0.016, +0.004] | -0.007 [-0.018, +0.005] | -0.007 [-0.018, +0.004] |
| B3 (+ region) | 3.917 | +0.001 [-0.004, +0.007] | -0.000 [-0.007, +0.007] | -0.009 [-0.018, +0.001] | -0.006 [-0.015, +0.003] |

### RAFT's add-on mode on our networks (10-seed ensemble mean)

| Network | Days | RMSE alone | + RAFT vs alone | + RAFT vs + RAFT-rand |
|---|---|---|---|---|
| C3-hop | all days | 2.216 | -0.025 [-0.060, +0.009] | +0.001 [-0.001, +0.003] |
| C3-hop | heatwave days | 2.991 | +0.446 [+0.368, +0.523] (CI > 0) | +0.015 [-0.012, +0.042] |
| Delhi-only LSTM | all days | 2.354 | -0.041 [-0.080, -0.002] **(CI < 0)** | +0.001 [-0.002, +0.003] |
| Delhi-only LSTM | heatwave days | 3.263 | +0.508 [+0.448, +0.567] (CI > 0) | +0.012 [-0.016, +0.040] |

### Diagnostics

4367 validation queries, 201 with a heatwave day among their 5 target days.

**Does a similar past mean a similar future?** Spearman correlation between key similarity and value similarity (minus the RMSE between the 5-day changes), over 200 random candidates per query. RAFT reports 0.60 on its ETTh1 data (definition not given, so the comparison is rough).

| Key | All queries | Heatwave queries |
|---|---|---|
| RAFT (Pearson of the 14 days) | +0.062 | +0.051 |
| R0 (cosine of the 17 features) | +0.023 | +0.034 |

**Can the key find the right analogues?** The 5 candidates whose actual 5-day changes were closest to the query's (the oracle analogues): their percentile rank under each key (50 = chance, 100 = ranked first).

| Key | Median, all | Median, heatwave queries | Mean, all |
|---|---|---|---|
| RAFT | 55.3 | 58.8 | 53.8 |
| R0 | 53.7 | 50.9 | 52.3 |

**The retrievals' own forecasts** (no model; oracle = last value + the oracle analogues' mean change, an upper bound that uses the truth):

| Forecast | RMSE all | Bias all | RMSE heatwave days | Bias heatwave days |
|---|---|---|---|---|
| Oracle analogues (uses the truth) | 0.366 | -0.00 | 0.421 | -0.22 |
| RAFT, change-based (last value + weighted change) | 2.735 | +0.06 | 4.291 | -3.83 |
| R0, level-based (analogues' mean raw future) | 2.724 | -0.53 | 4.851 | -4.64 |

**Cold days** (heatwave-day rows):

| Lead | Rows | R0 analogues' raw future - truth | RAFT change-based - truth | No candidate as hot as the truth |
|---|---|---|---|---|
| 1 | 72 | -4.42 | -2.85 | 0% |
| 2 | 72 | -4.58 | -3.75 | 0% |
| 3 | 72 | -4.61 | -4.22 | 0% |
| 4 | 72 | -4.87 | -4.11 | 0% |
| 5 | 72 | -4.70 | -4.23 | 0% |

Query's last input value minus its R0 analogues' mean last value: +0.34 °C over all queries, +1.35 °C for queries with a heatwave target day.

### The predictions (automatic reading)

| Prediction | Holds |
|---|---|
| P1 | yes |
| P2 (B0 gain and beats random) | yes |
| P2 (shrinks B0 -> B1) | yes |
| P2 (no gain on B2, B3) | yes |
| P2 holds | yes |
| P3 | yes |
| P4 | no |
| P5 | yes |
| P6 | yes |
| P7 | yes |

## Settings chosen on the inner block (per fold)

Ridge strength and RAFT's m were chosen on each fold's last 2 training years; details in `step1.json`.

