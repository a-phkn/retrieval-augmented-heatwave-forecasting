# Rz screen: retrieval on the 500 hPa weather pattern (rule written 2026-10-17, before any data)

No training; training years only (each fold's last 2 training years held out). RMSE in °C pooled over 4 folds x leads 1-5; Δ = first minus second (negative = the first is better); 95% CI by bootstrap over the 8 held-out years. Baseline = what C3-hop already sees, as a linear model (query-only features + Rg's regional readings + C3-hop's upstream inputs on the last input day).

## Tmax: **fail: not trained**

Held-out query-days: 14530 (extreme: 385); fit queries: 42645. Baseline RMSE: all 2.049, extreme 2.496 °C.

| Comparison | All days Δ (95% CI) | Extreme days Δ (95% CI, descriptive) |
|---|---|---|
| Rz vs baseline (rule) | -0.002 [-0.005, +0.001] | -0.016 [-0.035, +0.009] |
| Rz vs R0-rand (rule) | -0.002 [-0.005, +0.001] | -0.016 [-0.035, +0.011] |
| Rg vs baseline | +0.000 [-0.000, +0.001] | -0.002 [-0.003, -0.001] |
| R0-rand vs baseline | +0.000 [-0.000, +0.000] | -0.000 [-0.001, +0.001] |
| Rz vs Rg | -0.002 [-0.005, +0.001] | -0.014 [-0.033, +0.012] |

Correlation of the analogue signal with the truth, leads 1-5 (held out): Rz +0.40, +0.35, +0.29, +0.23, +0.19; R0-rand +0.00, -0.00, +0.00, +0.01, +0.01; Rg +0.74, +0.57, +0.43, +0.33, +0.25
Analogues within ±30 days of the query's time of year: Rz 24%; R0-rand 17%; Rg 34%

## WBGT (physical): **PASS: train C3hop_Rz**

Held-out query-days: 14530 (extreme: 205); fit queries: 42644. Baseline RMSE: all 2.095, extreme 4.165 °C.

| Comparison | All days Δ (95% CI) | Extreme days Δ (95% CI, descriptive) |
|---|---|---|
| Rz vs baseline (rule) | -0.004 [-0.006, -0.002] | -0.012 [-0.044, +0.014] |
| Rz vs R0-rand (rule) | -0.004 [-0.006, -0.002] | -0.013 [-0.044, +0.015] |
| Rg vs baseline | -0.000 [-0.000, +0.000] | +0.002 [-0.000, +0.004] |
| R0-rand vs baseline | +0.000 [-0.000, +0.000] | +0.001 [+0.000, +0.001] |
| Rz vs Rg | -0.004 [-0.006, -0.002] | -0.015 [-0.045, +0.012] |

Correlation of the analogue signal with the truth, leads 1-5 (held out): Rz +0.32, +0.31, +0.29, +0.26, +0.21; R0-rand +0.00, +0.01, -0.01, +0.00, -0.00; Rg +0.47, +0.34, +0.25, +0.17, +0.10
Analogues within ±30 days of the query's time of year: Rz 24%; R0-rand 17%; Rg 41%

Notes: the signal enters a linear model, so a null says the simple analogue-mean information is absent beyond the baseline. R0-rand: 10 seeds, RMSE averaged over seeds.
