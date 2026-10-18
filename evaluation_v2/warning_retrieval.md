# W: retrieval for heatwave warnings (rule written 2026-10-18, before it was run)

Hot-day probability of C3-hop, in season only, out of fold 2007-2018. Brier score (lower is better); BSS = skill vs the training-years base rate. Blends fitted on each fold's inner block. Tests: paired cluster-jackknife on sqrt-Brier (Δ in sqrt-probability units; negative = the blend is better).

## Tmax

| Variant | Brier | BSS | Hot days | Hits / season (P >= 0.5) | False alarms / season |
|---|---|---|---|---|---|
| model-only | 0.0544 | +0.366 | 790 | 24.2 | 9.3 |
| Rg | 0.0552 | +0.357 | 790 | 24.3 | 10.2 |
| Rz | 0.0550 | +0.359 | 790 | 24.1 | 9.3 |
| random | 0.0547 | +0.363 | 7900 | 24.0 | 9.2 |

| Arm | vs model-only | vs random-analogue blend | Result |
|---|---|---|---|
| Rg | +0.0016 [-0.000, +0.004] (p 0.108) | +0.0011 [-0.001, +0.003] (p 0.211) | fail (False, False) |
| Rz | +0.0012 [-0.002, +0.005] (p 0.496) | +0.0006 [-0.003, +0.004] (p 0.722) | fail (False, False) |

Random-analogue blend vs model-only: +0.0005 [+0.000, +0.001] (p 0.048).

## WBGT (physical)

| Variant | Brier | BSS | Hot days | Hits / season (P >= 0.5) | False alarms / season |
|---|---|---|---|---|---|
| model-only | 0.0571 | +0.028 | 750 | 0.1 | 0.5 |
| Rg | 0.0571 | +0.028 | 750 | 0.2 | 0.5 |
| Rz | 0.0571 | +0.028 | 750 | 0.0 | 0.4 |
| random | 0.0572 | +0.027 | 7500 | 0.1 | 0.5 |

| Arm | vs model-only | vs random-analogue blend | Result |
|---|---|---|---|
| Rg | +0.0000 [-0.000, +0.000] (p 0.950) | -0.0001 [-0.000, +0.000] (p 0.490) | fail (False, False) |
| Rz | +0.0001 [-0.000, +0.000] (p 0.407) | -0.0000 [-0.000, +0.000] (p 0.794) | fail (False, False) |

Random-analogue blend vs model-only: +0.0001 [+0.000, +0.000] (p 0.007).

