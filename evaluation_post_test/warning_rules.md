# Better yes/no heat warnings from the same forecasts (post-test, descriptive)

Pre-registered in `context/decisions.md` (2026-10-27) before any number was computed. Warning levels (p*, delta) chosen on 2007-2018 by the highest equitable threat score (ETS), then applied unchanged to 2019-2026. Rows = 10-seed ensemble forecasts, one per (query date, lead), in season. Hit rate = share of heatwave days warned; false-alarm ratio = share of warnings that were wrong; ETS: 0 = no better than chance, 1 = perfect; frequency bias = warnings / heatwave days. 95% CIs: cluster bootstrap over year x season blocks.

Event definition (as pre-registered): every day that meets the heatwave label rule, including isolated hot days. The RMSE tables and `test_period_extras.md` count only hot days inside a heatwave run, so the current rule's hit rate is a little different there (WBGT 9% vs 6% here, Tmax 27% vs 24%).

## WBGT (physical) (640 heatwave rows of 7870)

| Model and rule | Chosen on 2007-2018 | Hit rate (95% CI) | False-alarm ratio | ETS (95% CI) | Frequency bias |
|---|---|---|---|---|---|
| C3pool_tuned R-shift | delta = 1.20 C | 34% (22% to 43%) | 77% | 0.11 (0.06 to 0.15) | 1.50 |
| C2 R-shift | delta = 1.50 C | 43% (30% to 53%) | 80% | 0.10 (0.06 to 0.14) | 2.15 |
| C3-hop R-shift | delta = 1.20 C | 38% (25% to 47%) | 79% | 0.10 (0.06 to 0.15) | 1.81 |
| U1_tuned R-shift | delta = 1.35 C | 42% (28% to 52%) | 80% | 0.10 (0.06 to 0.15) | 2.08 |
| C3-hop R-prob | p* = 0.13 | 47% (32% to 57%) | 82% | 0.09 (0.05 to 0.13) | 2.54 |
| Delhi-only LSTM R-shift | delta = 1.20 C | 35% (23% to 42%) | 81% | 0.09 (0.05 to 0.12) | 1.80 |
| Delhi-only LSTM R-prob | p* = 0.12 | 54% (36% to 65%) | 83% | 0.08 (0.05 to 0.11) | 3.14 |
| ridge R-shift | delta = 2.50 C | 37% (26% to 45%) | 82% | 0.08 (0.04 to 0.11) | 2.10 |
| ridge_hw5 R-shift | delta = 2.10 C | 39% (27% to 48%) | 83% | 0.08 (0.04 to 0.11) | 2.26 |
| persistence |  | 19% (11% to 25%) | 81% | 0.06 (0.02 to 0.09) | 1.00 |
| U1_tuned R-point |  | 6% (3% to 10%) | 60% | 0.05 (0.02 to 0.08) | 0.16 |
| C3-hop R-point |  | 6% (3% to 8%) | 64% | 0.04 (0.02 to 0.06) | 0.17 |
| C2 R-point |  | 6% (2% to 8%) | 61% | 0.04 (0.02 to 0.06) | 0.14 |
| C3pool_tuned R-point |  | 5% (2% to 8%) | 63% | 0.04 (0.02 to 0.06) | 0.15 |
| Delhi-only LSTM R-point |  | 4% (2% to 6%) | 68% | 0.03 (0.01 to 0.04) | 0.12 |
| ridge_hw5 R-point |  | 3% (1% to 4%) | 56% | 0.02 (0.01 to 0.03) | 0.06 |
| ridge R-point |  | 0% (0% to 0%) | 80% | 0.00 (-0.00 to 0.00) | 0.01 |

**Paired differences (first minus second, same days):**

| Comparison | ETS difference (95% CI) | Hit-rate difference (95% CI) |
|---|---|---|
| C3-hop R-prob minus C3-hop R-point | +0.03 to +0.07 | +28 to +49 points |
| C3-hop R-shift minus C3-hop R-point | +0.04 to +0.09 | +22 to +39 points |
| C3-hop R-prob minus Delhi-only LSTM R-prob | -0.01 to +0.02 | -15 to +1 points |
| C3-hop R-prob minus persistence | +0.00 to +0.06 | +18 to +36 points |
| C3-hop R-shift minus Delhi-only LSTM R-shift | -0.01 to +0.04 | -3 to +8 points |
| C3-hop R-shift minus C2 R-shift | -0.02 to +0.01 | -10 to -1 points |
| C3-hop R-shift minus U1_tuned R-shift | -0.02 to +0.02 | -8 to +0 points |
| C3-hop R-shift minus C3pool_tuned R-shift | -0.02 to +0.01 | +1 to +7 points |
| C3-hop R-shift minus ridge R-shift | +0.00 to +0.05 | -5 to +5 points |
| C3-hop R-shift minus ridge_hw5 R-shift | +0.00 to +0.05 | -6 to +4 points |

**C3-hop, probability rule at other levels (descriptive trade-off; not chosen on the test):**

| Warn if probability >= | Hit rate | False-alarm ratio | ETS |
|---|---|---|---|
| 0.1 | 59% | 83% | 0.08 |
| 0.2 | 26% | 77% | 0.10 |
| 0.3 | 9% | 68% | 0.06 |
| 0.5 | 0% | 71% | 0.00 |

## Tmax (645 heatwave rows of 5560)

| Model and rule | Chosen on 2007-2018 | Hit rate (95% CI) | False-alarm ratio | ETS (95% CI) | Frequency bias |
|---|---|---|---|---|---|
| ridge_hw5 R-shift | delta = 1.05 C | 52% (39% to 61%) | 55% | 0.26 (0.18 to 0.31) | 1.17 |
| Delhi-only LSTM R-prob | p* = 0.24 | 54% (40% to 62%) | 57% | 0.25 (0.18 to 0.32) | 1.24 |
| ridge R-shift | delta = 1.45 C | 49% (37% to 57%) | 56% | 0.24 (0.17 to 0.30) | 1.10 |
| U1_tuned R-shift | delta = 0.25 C | 36% (23% to 45%) | 45% | 0.23 (0.15 to 0.31) | 0.65 |
| C3-hop R-shift | delta = 0.25 C | 32% (17% to 46%) | 39% | 0.23 (0.13 to 0.33) | 0.53 |
| C2 R-shift | delta = 0.20 C | 32% (18% to 44%) | 40% | 0.22 (0.13 to 0.31) | 0.54 |
| Delhi-only LSTM R-shift | delta = 0.60 C | 43% (29% to 52%) | 56% | 0.22 (0.14 to 0.30) | 0.98 |
| C3pool_tuned R-shift | delta = 0.15 C | 32% (17% to 45%) | 42% | 0.22 (0.12 to 0.31) | 0.55 |
| C3-hop R-prob | p* = 0.37 | 29% (17% to 39%) | 37% | 0.21 (0.13 to 0.30) | 0.45 |
| persistence |  | 42% (31% to 51%) | 58% | 0.21 (0.14 to 0.28) | 1.00 |
| U1_tuned R-point |  | 29% (18% to 36%) | 39% | 0.20 (0.13 to 0.26) | 0.47 |
| C3pool_tuned R-point |  | 27% (13% to 40%) | 40% | 0.20 (0.10 to 0.29) | 0.46 |
| C2 R-point |  | 26% (14% to 39%) | 38% | 0.19 (0.11 to 0.29) | 0.42 |
| C3-hop R-point |  | 24% (13% to 36%) | 37% | 0.18 (0.10 to 0.27) | 0.39 |
| Delhi-only LSTM R-point |  | 28% (16% to 36%) | 50% | 0.18 (0.10 to 0.24) | 0.55 |
| ridge_hw5 R-point |  | 22% (15% to 27%) | 41% | 0.16 (0.11 to 0.19) | 0.37 |
| ridge R-point |  | 14% (10% to 18%) | 34% | 0.11 (0.08 to 0.14) | 0.21 |

**Paired differences (first minus second, same days):**

| Comparison | ETS difference (95% CI) | Hit-rate difference (95% CI) |
|---|---|---|
| C3-hop R-prob minus C3-hop R-point | +0.02 to +0.04 | +3 to +5 points |
| C3-hop R-shift minus C3-hop R-point | +0.03 to +0.06 | +4 to +10 points |
| C3-hop R-prob minus Delhi-only LSTM R-prob | -0.08 to -0.00 | -29 to -18 points |
| C3-hop R-prob minus persistence | -0.03 to +0.03 | -17 to -10 points |
| C3-hop R-shift minus Delhi-only LSTM R-shift | -0.03 to +0.05 | -14 to -5 points |
| C3-hop R-shift minus C2 R-shift | -0.01 to +0.03 | -2 to +2 points |
| C3-hop R-shift minus U1_tuned R-shift | -0.04 to +0.03 | -8 to +2 points |
| C3-hop R-shift minus C3pool_tuned R-shift | -0.01 to +0.03 | -3 to +3 points |
| C3-hop R-shift minus ridge R-shift | -0.07 to +0.05 | -24 to -8 points |
| C3-hop R-shift minus ridge_hw5 R-shift | -0.09 to +0.04 | -29 to -9 points |

**C3-hop, probability rule at other levels (descriptive trade-off; not chosen on the test):**

| Warn if probability >= | Hit rate | False-alarm ratio | ETS |
|---|---|---|---|
| 0.1 | 86% | 64% | 0.26 |
| 0.2 | 64% | 54% | 0.30 |
| 0.3 | 41% | 45% | 0.26 |
| 0.5 | 13% | 29% | 0.11 |

