# Confirmatory test protocol (DRAFT, 2026-10-08)

Status: **draft**. It is frozen as a tagged commit before the locked test period (2019-01-01 to 2026-09-06)
is read for the first time. Items marked *[at freeze]* are filled in then, after the user approves the
final backbone (decisions.md 2026-10-08: nothing is final until the user approves it). Written on the
independent ML review's advice (finding M5: many decisions were made on the same 2007-2018 years, and
the test period is short).

## 1. What is frozen

- **Models:** the final backbone *[at freeze]* and the control LSTM (`A2Lr_hw5` for WBGT,
  `A1prime_hw5` for Tmax), with their configs, code hash and commit. Also the two ridge baselines
  (`evaluation/ridge_baselines.py`). Retrieval and the physics head enter only if they have passed
  their own rules by the freeze *[at freeze]*.
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

## 2. Primary hypotheses (at most 3; Holm correction, family-wise alpha 0.05)

Proposed; confirmed *[at freeze]*:

| | Hypothesis | Comparison | Stratum | Supported if |
|---|---|---|---|---|
| H1 | The final backbone forecasts WBGT better than the Delhi-only LSTM | BB\* vs control | all days | Holm-adjusted p < 0.05 and Δ < 0 |
| H2 | ... and better on heatwave days | BB\* vs control | WBGT extreme days | as H1. The bias / SD split and the hot-day Brier score are reported next to it, because extreme-day RMSE rewards warm bias |
| H3 | The final backbone adds skill beyond a linear upstream model | BB\* vs `ridge` | WBGT all days | as H1 |

Retrieval (the plan's H-A) failed its development rule (G3: none), so it is not a confirmatory
hypothesis unless a retrieval rung passes G-R\* before the freeze.

## 3. Power: what the test period can detect

Smallest true RMSE difference detected with 80% power (two-sided 5%, before Holm). These are
scaled from the development CIs by the number of clusters. That assumes similar between-cluster
variance and, for extremes, heatwaves in every test hot season; rough estimates only.

| Comparison (development estimate) | Development clusters, MDE | Test clusters, MDE |
|---|---|---|
| WBGT all days (C2 vs control) | 36, 0.040 °C | ~24, ~0.05 °C |
| WBGT extreme days (C2 vs control) | 14, 0.25 °C | ~8, ~0.36 °C |
| Tmax all days (C2 vs control) | 36, 0.048 °C | ~24, ~0.06 °C |
| Tmax extreme days (C2 vs control) | 11, 0.19 °C | ~8, ~0.24 °C |

Consequence: the all-days hypotheses (H1, H3) can detect differences of the size seen in
development (C2 vs control: -0.12 °C). The extreme-day hypothesis (H2) can detect only large
effects, so a null H2 will mostly mean "not detectable".

## 4. Reported but exploratory (no confirmatory claim)

- Tmax: the same comparisons. Tmax is the negative control for the drift story.
- By lead, the bias / SD split, recalibrated scores, hot-day Brier and reliability
  (`evaluation/calibration_check.py`), and the forecast-conditioned stratum.
- Every other model: U1, C2, the C3 / C4 / C4a / C3-pool graphs, and retrieval rungs.
- Case studies: the 2019, 2022 and 2024 heat events.
- Graph-structure claims, still under the structure-claim rule (CI below 0 against both U1 and C2).

## 5. Safeguards

- The test lock stays closed until this file is frozen as a tagged commit. The code already refuses
  to read 2019+ data in development (`training/folds.py`).
- A refit-on-all-training-years check, if run, happens before the freeze, on development data only.
- Every deviation from this protocol after the freeze is listed in the paper's limitations.
