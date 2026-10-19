# Screen S2: new information for WBGT heatwave days (rule written 2026-10-18, before the new data)

No training; training years only (each fold's last 2 training years held out). Ridge weighted like the neural loss (1 + 4 x hot). RMSE in °C pooled over 4 folds x leads 1-5; Δ = first minus second (negative = the arm helps); 95% CI by bootstrap over the 8 held-out years. Baseline = what C3-hop already sees, as a linear model. Arms run: A (humid-air direct), B (Rh retrieval (humid-air analogues)), C (heatwave-onset matching), D (upper-air (500 hPa) direct).

## Tmax (descriptive)

Held-out query-days: 14530 (extreme: 385); fit queries: 42648. Baseline RMSE: all 2.109, extreme 1.715 °C.

| Comparison | Extreme days Δ (95% CI) | All days Δ (95% CI) |
|---|---|---|
| A (humid-air direct) vs baseline | -0.123 [-0.286, -0.027] | -0.008 [-0.031, +0.017] |
| B (Rh retrieval (humid-air analogues)) vs baseline | -0.021 [-0.036, -0.011] | +0.002 [-0.000, +0.004] |
| C (heatwave-onset matching) vs baseline | -0.022 [-0.031, -0.014] | -0.001 [-0.005, +0.003] |
| D (upper-air (500 hPa) direct) vs baseline | -0.079 [-0.159, +0.031] | -0.031 [-0.047, -0.016] |
| B (Rh retrieval (humid-air analogues)) vs B-rand | -0.021 [-0.036, -0.012] | +0.001 [-0.000, +0.003] |
| C (heatwave-onset matching) vs C-rand | -0.006 [-0.011, +0.003] | -0.000 [-0.002, +0.002] |

## WBGT (physical) (decides)

Held-out query-days: 14530 (extreme: 205); fit queries: 42648. Baseline RMSE: all 2.225, extreme 3.326 °C.

| Comparison | Extreme days Δ (95% CI) | All days Δ (95% CI) |
|---|---|---|
| A (humid-air direct) vs baseline | +0.024 [-0.047, +0.114] | +0.005 [-0.009, +0.019] |
| B (Rh retrieval (humid-air analogues)) vs baseline | +0.004 [-0.003, +0.011] | +0.000 [-0.000, +0.001] |
| C (heatwave-onset matching) vs baseline | -0.042 [-0.060, -0.023] | +0.005 [+0.002, +0.007] |
| D (upper-air (500 hPa) direct) vs baseline | +0.053 [-0.058, +0.136] | -0.040 [-0.059, -0.021] |
| B (Rh retrieval (humid-air analogues)) vs B-rand | +0.003 [-0.004, +0.011] | +0.000 [-0.000, +0.001] |
| C (heatwave-onset matching) vs C-rand | -0.028 [-0.044, -0.011] | +0.003 [+0.001, +0.005] |

**Decision:** A fail (False, True); B fail (False, True, False); C fail (True, False, True); D fail (False, True)

