# Rz on C3-hop, WBGT (rule written 2026-10-17, before any z500 data)

Out of fold 2007-2018, 10 seeds x 4 folds, paired cluster-jackknife; Δ = C3hop_Rz minus comparison (negative = Rz better).

| RMSE all: Rz / C3-hop | Extreme: Rz / C3-hop | (1) vs C3-hop, all | (2) vs C3-hop, extreme | (3) vs random, extreme | Result | Final test model |
|---|---|---|---|---|---|---|
| 2.225 / 2.248 | 3.023 / 3.019 | -0.023 [-0.041, -0.005] (p 0.012) | +0.004 [-0.120, +0.128] (p 0.945) | -0.013 [-0.124, +0.097] (p 0.798) | fail (True, False, False) | `A2Lr_hw5_C3hop` |

Rz vs random, all days: -0.020 [-0.040, -0.000] (p 0.050). Extreme-day detectable effect: 0.17 °C. Extreme bias Rz / C3-hop: -2.69 / -2.69.

## Descriptive

**Per lead (all days, Rz - C3-hop):** day 1 -0.016 [-0.039, +0.008] (p 0.181); day 2 -0.030 [-0.054, -0.007] (p 0.011); day 3 -0.027 [-0.047, -0.007] (p 0.010); day 4 -0.025 [-0.050, +0.000] (p 0.051); day 5 -0.018 [-0.042, +0.005] (p 0.125)

**Rz vs Rg (same backbone):** all days -0.026 [-0.045, -0.007] (p 0.010); extreme +0.022 [-0.084, +0.129] (p 0.659)

**Attention:** Rz max weight 0.325, normalised entropy 0.927; Rg 0.276 / 0.967 (uniform: entropy 1).
