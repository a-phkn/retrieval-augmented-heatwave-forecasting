# Screen S2: new information for WBGT heatwave days (rule written 2026-10-18, before the new data)

No training; training years only (each fold's last 2 training years held out). Ridge weighted like the neural loss (1 + 4 x hot). RMSE in °C pooled over 4 folds x leads 1-5; Δ = first minus second (negative = the arm helps); 95% CI by bootstrap over the 8 held-out years. Baseline = what C3-hop already sees, as a linear model. Arms run: D (upper-air (500 hPa) direct).

## Tmax (descriptive)

Held-out query-days: 14530 (extreme: 385); fit queries: 42969. Baseline RMSE: all 2.109, extreme 1.721 °C.

| Comparison | Extreme days Δ (95% CI) | All days Δ (95% CI) |
|---|---|---|
| D (upper-air (500 hPa) direct) vs baseline | -0.081 [-0.161, +0.029] | -0.031 [-0.047, -0.015] |

## WBGT (physical) (decides)

Held-out query-days: 14530 (extreme: 205); fit queries: 42969. Baseline RMSE: all 2.224, extreme 3.323 °C.

| Comparison | Extreme days Δ (95% CI) | All days Δ (95% CI) |
|---|---|---|
| D (upper-air (500 hPa) direct) vs baseline | +0.053 [-0.045, +0.140] | -0.040 [-0.058, -0.021] |

**Decision:** D fail (False, True)

