# Test period: extra descriptive numbers (post-test, 2026-10-25)

Asked for by the critical reviews. Descriptive only: the confirmatory result is `evaluation_test/test_results.md`.
RMSE in °C (lower is better). Ensemble = the average of the 10 seeds' forecasts.

## WBGT (physical)

| Model | Ensemble RMSE all days | Ensemble RMSE heatwave days | Hit rate | False-alarm ratio | ETS |
|---|---|---|---|---|---|
| C3-hop | 2.232 | 3.006 | 0.09 | 0.68 | 0.07 |
| Delhi-only LSTM | 2.370 | 3.094 | 0.06 | 0.68 | 0.05 |
| ridge | 2.293 | 4.151 | 0.00 | 0.80 | 0.00 |
| ridge_hw5 | 2.299 | 3.740 | 0.04 | 0.56 | 0.04 |

Observed heatwave forecasts (window x lead rows): 395. Hit rate = share of observed heatwave days that were forecast; false-alarm ratio = share of forecast heatwave days that did not happen; ETS = equitable threat score (0 = chance, 1 = perfect).

Heatwave days, C3-hop vs ridge_hw5: MSE 9.16 vs 13.99, of which bias² 7.09 vs 11.71; **96% of the gap is bias**.

C3-hop vs the plain ridge, all days: -0.035 [-0.090, +0.021] (p 0.207) (a tie).

## Tmax

| Model | Ensemble RMSE all days | Ensemble RMSE heatwave days | Hit rate | False-alarm ratio | ETS |
|---|---|---|---|---|---|
| C3-hop | 2.187 | 2.262 | 0.27 | 0.37 | 0.22 |
| Delhi-only LSTM | 2.322 | 2.499 | 0.30 | 0.50 | 0.22 |
| ridge | 2.276 | 2.889 | 0.15 | 0.35 | 0.13 |
| ridge_hw5 | 2.317 | 2.528 | 0.23 | 0.43 | 0.19 |

Observed heatwave forecasts (window x lead rows): 585. Hit rate = share of observed heatwave days that were forecast; false-alarm ratio = share of forecast heatwave days that did not happen; ETS = equitable threat score (0 = chance, 1 = perfect).

Heatwave days, C3-hop vs ridge_hw5: MSE 5.35 vs 6.39, of which bias² 3.72 vs 4.48; **73% of the gap is bias**.

C3-hop vs the plain ridge, all days: -0.043 [-0.089, +0.004] (p 0.072) (a tie).

## H2 power

Realised minimum detectable effect (80% power): **0.25 °C** (the protocol's pre-test estimate was 0.45). The observed difference was -0.081, about a third of the development value (-0.255): the test was precise enough to detect the development-sized effect, so the null is not only a power problem; development probably overstated the heatwave-day gain.

## How much of each all-day WBGT gain is a change in the average error (bias²)

- H1 (C3-hop vs control): MSE 5.101 vs 5.693; bias² 0.122 vs 0.182; **10%** of the gain is bias
- H4 (LSTM + Rg vs control): MSE 5.495 vs 5.693; bias² 0.083 vs 0.182; **50%** of the gain is bias

## Without the ERA5T months (target dates from 2026-06-01 removed)

- H1: -0.134 [-0.185, -0.082] (p <0.001)
- H3: -0.700 [-0.885, -0.515] (p <0.001)
- H4: -0.044 [-0.083, -0.006] (p 0.026)
