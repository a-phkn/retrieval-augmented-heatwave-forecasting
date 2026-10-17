# G-R*: retrieval (Rg) on the C3-hop backbone (rule written 2026-10-13, before any run)

Out of fold 2007-2018, 10 seeds x 4 folds, paired cluster-jackknife; Δ = first-named minus second (negative = C3hop_Rg better). Retrieval acts only at C3-hop's Delhi readout.

## Decision (G3's three conditions)

| Family | RMSE all: Rg / C3-hop | Extreme: Rg / C3-hop | (1) vs C3-hop, all | (2) vs C3-hop, extreme | (3) vs random, extreme | Result | Final test model |
|---|---|---|---|---|---|---|---|
| Tmax | 2.083 / 2.065 | 1.806 / 1.827 | +0.018 [-0.006, +0.042] (p 0.140) | -0.021 [-0.199, +0.157] (p 0.800) | +0.028 [-0.139, +0.195] (p 0.716) | fail (True, False, False) | `A1prime_hw5_C3hop` |
| WBGT (physical) | 2.250 / 2.248 | 3.000 / 3.019 | +0.002 [-0.013, +0.017] (p 0.748) | -0.018 [-0.106, +0.070] (p 0.663) | -0.036 [-0.117, +0.045] (p 0.359) | fail (True, False, False) | `A2Lr_hw5_C3hop` |

Also: C3hop_Rg vs random, all days: Tmax +0.006 [-0.013, +0.026] (p 0.510); WBGT (physical) +0.006 [-0.011, +0.023] (p 0.505)
Random analogues vs C3-hop, all days: Tmax +0.012 [-0.009, +0.033] (p 0.270); WBGT (physical) -0.003 [-0.018, +0.012] (p 0.654)
Extreme-day detectable effect (80% power): Tmax 0.25 °C; WBGT (physical) 0.12 °C
Extreme-day bias Rg / C3-hop: Tmax -1.20 / -1.25; WBGT (physical) -2.69 / -2.69

## Diagnostics (exploratory, no claims)

### Tmax

**When (per lead, all days, Rg - C3-hop):** day 1 +0.026 [+0.001, +0.050] (p 0.040); day 2 +0.023 [+0.001, +0.044] (p 0.037); day 3 +0.016 [-0.012, +0.044] (p 0.246); day 4 +0.018 [-0.015, +0.051] (p 0.282); day 5 +0.013 [-0.024, +0.049] (p 0.485)

**Where (upstream heat arriving = top quarter of upstream Tmax anomaly on the last input day):** upstream heat (1212 windows) +0.021 [-0.016, +0.059] (p 0.259); other days (3155 windows) +0.017 [-0.007, +0.040] (p 0.159)

**How (attention):** C3hop_Rg max weight 0.290, normalised entropy 0.960; LSTM Rg 0.266 / 0.971 (uniform: max 0.200, entropy 1).

**Overlap, all days:** Rg's gain on the LSTM -0.003, on C3-hop +0.018; difference +0.021 [-0.006, +0.048] (p 0.121) (positive = smaller gain on the graph model).
**Overlap, extreme days:** Rg's gain on the LSTM +0.045, on C3-hop -0.021; difference -0.066 [-0.180, +0.049] (p 0.232) (positive = smaller gain on the graph model).

### WBGT (physical)

**When (per lead, all days, Rg - C3-hop):** day 1 -0.009 [-0.029, +0.011] (p 0.376); day 2 -0.010 [-0.031, +0.011] (p 0.330); day 3 +0.003 [-0.015, +0.021] (p 0.743); day 4 +0.014 [-0.008, +0.035] (p 0.200); day 5 +0.011 [-0.011, +0.033] (p 0.301)

**Where (upstream heat arriving = top quarter of upstream Tmax anomaly on the last input day):** upstream heat (1212 windows) -0.004 [-0.022, +0.014] (p 0.664); other days (3155 windows) +0.005 [-0.011, +0.021] (p 0.535)

**How (attention):** C3hop_Rg max weight 0.276, normalised entropy 0.967; LSTM Rg 0.232 / 0.994 (uniform: max 0.200, entropy 1).

**Overlap, all days:** Rg's gain on the LSTM -0.044, on C3-hop +0.002; difference +0.046 [+0.023, +0.069] (p <0.001) (positive = smaller gain on the graph model).
**Overlap, extreme days:** Rg's gain on the LSTM -0.160, on C3-hop -0.018; difference +0.142 [+0.012, +0.272] (p 0.035) (positive = smaller gain on the graph model).

