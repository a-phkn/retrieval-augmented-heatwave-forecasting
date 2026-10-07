# Retrieval mechanism metrics (2026-10-07)

Descriptive (plan v5 Week 4). Validation queries of all 4 folds (out of fold, 2007-2018), from the analogue files each run saved. Random runs: mean over 10 seeds. Redundancy is measured in the same two spaces for every run (Delhi's 17 features; Rg's regional pattern), so rungs are comparable.

## Tmax

| Rung | Age (years) | Older than 20 y | Season gap (days) | Within ±30 d | Redundancy, Delhi space | Redundancy, regional space | Attention unevenness | Attention vs similarity (Spearman) |
|---|---|---|---|---|---|---|---|---|
| R0 | 16.5 | 35% | 12 | 95% | 0.880 | 0.300 | 0.005 | +0.04 |
| R0-rand | 17.2 | 39% | 91 | 17% | 0.003 | 0.001 | 0.039 | +0.46 |
| R1 | 16.5 | 35% | 11 | 100% | 0.879 | 0.298 | 0.004 | +0.04 |
| R1-rand | 17.2 | 39% | 15 | 100% | 0.369 | 0.001 | 0.011 | +0.17 |
| Rg | 15.9 | 33% | 66 | 35% | 0.341 | 0.791 | 0.029 | +0.04 |

Outcome signal: correlation of the analogues' mean standardised outcome with the query's truth, by lead (all days / observed-extreme days):

| Rung | Lead 1 | Lead 2 | Lead 3 | Lead 4 | Lead 5 |
|---|---|---|---|---|---|
| R0 | +0.70 / +0.49 | +0.48 / +0.30 | +0.36 / +0.22 | +0.29 / +0.22 | +0.22 / +0.19 |
| R0-rand | +0.00 / +0.01 | +0.00 / +0.03 | +0.00 / -0.03 | +0.01 / -0.02 | -0.00 / -0.03 |
| R1 | +0.70 / +0.44 | +0.47 / +0.29 | +0.35 / +0.19 | +0.28 / +0.21 | +0.22 / +0.17 |
| R1-rand | +0.00 / -0.05 | +0.00 / -0.02 | +0.00 / +0.02 | -0.00 / -0.00 | -0.00 / -0.04 |
| Rg | +0.74 / +0.41 | +0.58 / +0.36 | +0.45 / +0.32 | +0.35 / +0.29 | +0.27 / +0.17 |

## WBGT (physical)

| Rung | Age (years) | Older than 20 y | Season gap (days) | Within ±30 d | Redundancy, Delhi space | Redundancy, regional space | Attention unevenness | Attention vs similarity (Spearman) |
|---|---|---|---|---|---|---|---|---|
| R0 | 16.5 | 35% | 12 | 95% | 0.880 | 0.238 | 0.004 | +0.03 |
| R0-rand | 17.2 | 39% | 91 | 17% | 0.003 | 0.000 | 0.008 | +0.15 |
| R1 | 16.5 | 35% | 11 | 100% | 0.879 | 0.234 | 0.004 | +0.03 |
| R1-rand | 17.2 | 39% | 15 | 100% | 0.369 | -0.001 | 0.007 | +0.12 |
| Rg | 14.0 | 25% | 57 | 41% | 0.375 | 0.738 | 0.006 | +0.08 |

Outcome signal: correlation of the analogues' mean standardised outcome with the query's truth, by lead (all days / observed-extreme days):

| Rung | Lead 1 | Lead 2 | Lead 3 | Lead 4 | Lead 5 |
|---|---|---|---|---|---|
| R0 | +0.30 / +0.13 | +0.16 / -0.10 | +0.10 / -0.05 | +0.06 / -0.07 | +0.05 / -0.05 |
| R0-rand | +0.00 / -0.03 | +0.00 / +0.11 | -0.00 / +0.02 | +0.01 / +0.01 | -0.00 / +0.02 |
| R1 | +0.30 / +0.17 | +0.16 / -0.07 | +0.09 / -0.05 | +0.06 / -0.12 | +0.05 / -0.12 |
| R1-rand | +0.01 / +0.03 | +0.01 / +0.01 | +0.01 / -0.10 | +0.00 / -0.06 | +0.00 / -0.05 |
| Rg | +0.47 / -0.08 | +0.36 / -0.05 | +0.28 / -0.05 | +0.21 / -0.02 | +0.14 / -0.04 |

Notes:
- Attention unevenness 0 = the network weights its 5 analogues equally; a network that trusts some analogues more than others scores higher.
- The outcome-signal correlation is not 'skill beyond the query's own inputs'; that is `evaluation_v2/retrieval_information_check.md`.
