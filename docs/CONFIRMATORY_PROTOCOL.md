# Confirmatory test protocol (FROZEN 2026-10-19)

Status: **frozen** (user approval 2026-10-19), as the commit tagged `test-lock`, before the locked test period
(2019-01-01 to 2026-09-06) was read for the first time. Drafted 2026-10-08 on the independent ML review's advice
(finding M5: many decisions were made on the same 2007-2018 years, and the test period is short).

## 1. What is frozen

- **Models:** the final backbone **C3-hop** (`A2Lr_hw5_C3hop` for WBGT, `A1prime_hw5_C3hop` for Tmax; user
  decision 2026-10-12, disclosed as chosen after results) and the control LSTM (`A2Lr_hw5` for WBGT,
  `A1prime_hw5` for Tmax), with their configs, code hash and commit; for H4, the LSTM + Rg (`A2Lr_hw5_Rg`).
  Also the two ridge baselines (`evaluation/ridge_baselines.py`). **No retrieval rung and no physics head enter the final models:** every
  retrieval rung failed its development rule (G3; G-R* 2026-10-16; Rz 2026-10-18; the analog correction and the
  warning retrieval 2026-10-18) and the physics head was dropped by its keep rule (2026-10-12). The humid-air screen S2
  (2026-10-19) found no arm that passes its rule, so nothing from it enters either.
- **Code:** the test is run by `training/test_run.py` and scored by `evaluation/test_evaluation.py`, exactly
  as frozen; `training/test_lock.py` refuses any 2019+ read unless HEAD contains the tag `test-lock` and the
  code under training/ models/ pipeline/ retrieval/ evaluation/ configs/ is unchanged since the tag.
- **Training for the test:** one "test fold". It trains on 1980-2016 and early-stops on 2017-2018
  (the same inner_2y recipe), with seeds 0-9. The ridges are fitted on 1980-2018 as in G-D0.
  Climatology, labels and normalisation come from 1980-2018 only.
- **Scored on:** 2019-01-01 to 2026-09-06, every window whose forecast days lie inside it. The truth
  is ERA5. The last ~3 months were preliminary ERA5T when downloaded, which is disclosed.
- **Primary score:** the mean of per-seed RMSE, as in development. The 10-seed ensemble mean is the
  deliverable forecast and is reported next to it.
- **Test:** the same paired cluster-jackknife (year x season clusters, `evaluation/stats.py`).
- **One evaluation run.** No model, rule or threshold is changed after the test numbers have been
  seen. Anything run after that is labelled post-test exploration.

## 2. Primary hypotheses (four; Holm correction, family-wise alpha 0.05)

Confirmed at the freeze (BB\* = C3-hop):

| | Hypothesis | Comparison | Stratum | Supported if |
|---|---|---|---|---|
| H1 | The final backbone forecasts WBGT better than the Delhi-only LSTM | BB\* vs control | all days | Holm-adjusted p < 0.05 and Δ < 0 |
| H2 | ... and better on heatwave days | BB\* vs control | WBGT extreme days | as H1. The bias / SD split and the hot-day Brier score are reported next to it, because extreme-day RMSE rewards warm bias |
| H3 | The final backbone beats a linear upstream model on heatwave days | BB\* vs `ridge_hw5` (the ridge weighted like the neural loss, 1 + 4 x hot) | WBGT extreme days | as H1. Changed 2026-10-18 (user, option d) from "vs `ridge`, WBGT all days", where development showed a tie (+0.031, n.s.); that comparison is still reported |
| H4 | Analogue retrieval (Rg) improves the Delhi-only LSTM | `A2Lr_hw5_Rg` vs control | WBGT all days | as H1. Added 2026-10-19 (user): the one retrieval effect found in development (-0.044, p = 0.003); it failed G3 (no heatwave-day gain), so this tests the all-days effect only |

Retrieval (the plan's H-A) failed every development rule (G3, G-R\*, Rz, analog correction, warning
retrieval, humid-air retrieval), so no retrieval rung is part of the final models. H4 tests the one retrieval effect
found in development, on all days, on the Delhi-only LSTM.

## 3. Power: what the test period can detect

Smallest true RMSE difference detected with 80% power (two-sided 5%, before Holm). These are
scaled from the development CIs by the number of clusters. That assumes similar between-cluster
variance and, for extremes, heatwaves in every test hot season; rough estimates only.

| Hypothesis (comparison, stratum) | Development effect (95% CI) | Development clusters, MDE | Test clusters, MDE |
|---|---|---|---|
| H1: C3-hop vs control, WBGT all days | -0.125 (-0.159, -0.092) | 36, 0.047 °C | ~23, ~0.06 °C |
| H2: C3-hop vs control, WBGT extreme days | -0.255 (-0.480, -0.031) | 14, 0.32 °C | ~8, ~0.45 °C |
| H3: C3-hop vs ridge_hw5, WBGT extreme days | -0.847 (-1.056, -0.639) | 14, 0.29 °C | ~8, ~0.42 °C |
| H4: LSTM + Rg vs control, WBGT all days | -0.044 (-0.071, -0.016) | 36, 0.039 °C | ~23, ~0.05 °C |
| (Tmax, exploratory) C3-hop vs control, all days | -0.110 (-0.147, -0.074) | 36, 0.052 °C | ~23, ~0.07 °C |
| (Tmax, exploratory) C3-hop vs control, extreme days | -0.327 (-0.535, -0.119) | 11, 0.29 °C | ~8, ~0.36 °C |

(Recomputed 2026-10-19 for the final backbone C3-hop; the test MDE scales the development CI half-width by
sqrt(G_dev / G_test) with t quantiles at G_test - 1; test clusters: ~7.7 years x 3 seasons, ~8 hot seasons with
heatwaves.)

Consequence: H1 and H3 can detect effects of the size seen in development; H4 (0.044 °C) is close to the
detectable effect (~0.05 °C), so it has roughly even odds. H2 (development effect 0.26 °C)
is below the test's detectable effect (~0.45 °C), so a null H2 will mostly mean "not detectable", not "no
effect"; this is stated in advance.

## 4. Reported but exploratory (no confirmatory claim)

- Tmax: the same comparisons. Tmax is the negative control for the drift story.
- By lead, the bias / SD split, recalibrated scores, hot-day Brier and reliability
  (`evaluation/calibration_check.py`), and the forecast-conditioned stratum.
- Other models trained on the test fold by `training/test_run.py --stage extra`: C2, U1 (tuned), C3-pool
  (tuned), LSTM + Rg and C3-hop + Rg (both families); C3 / C4 / C4a are not re-trained.
- **Rz side-check (user decision 2026-10-18):** C3-hop + Rz (500 hPa pattern retrieval) for WBGT failed its
  development rule (no heatwave-day gain) but improved all days by 0.023 °C. It is re-checked on the test years
  as exploratory only, on **2019-2021** (the WeatherBench 2 ERA5 2.8° data ends 2021-12-31). The 2019-2021 z500
  extract is downloaded only after this protocol is frozen, as part of the test run.
- Case studies: the 2019, 2022 and 2024 heat events, on windows fixed before the test period was opened:
  2019-05-25 to 2019-06-15, 2022-04-01 to 2022-05-15, 2024-05-15 to 2024-06-20 (10-seed ensemble; RMSE and bias
  at leads 1, 3 and 5).
- Range forecasts (CRPS, interval coverage) and hot-day Brier, recalibrated on the early-stopping block (2017-2018).
- Graph-structure claims, still under the structure-claim rule (CI below 0 against both U1 and C2).

## 5. Safeguards

- The test lock stays closed until this file is frozen as a tagged commit. The code already refuses
  to read 2019+ data in development (`training/folds.py`).
- A refit-on-all-training-years check, if run, happens before the freeze, on development data only.
- Every deviation from this protocol after the freeze is listed in the paper's limitations.
