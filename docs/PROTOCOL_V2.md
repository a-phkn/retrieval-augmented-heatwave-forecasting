# v2 study protocol: physics-constrained forecasting of unsafe working hours (DRAFT, 23 Nov 2026)

Status: **draft for team review. Not frozen.** It is frozen by committing it on branch `experiments-v2`; the
test-period tag `test-lock-v2` comes later (Section 8). All earlier [DECIDE] and [CHECK] items were resolved on
23 Nov 2026 (Section 10, with the source for each). Four items remain **[DATA]**: files that must be obtained,
not decisions. No v2 model may be trained before this file is committed.

Every rule below names its source: a repository file, a verified catalogue row (`sources/phase6_thresholds.json`,
`sources/phase6_candidate_actions.json`, each row has a page and a verbatim excerpt), or a publication.

---

## 0. What was already seen before this protocol (disclosure)

This study is not blind, and the paper must say so.

1. **The v1 test period (2019-01-01 to 2026-09-06) was opened on 2026-10-19** under `docs/CONFIRMATORY_PROTOCOL.md`.
   Its results are in `evaluation_test/test_results.md`. In particular, the motivation for H-C below comes from
   that test: C3-hop's 80% intervals covered 0.29 of WBGT extreme days (`evaluation_test/test_results.md`, line 50).
   Any v2 result on 2019+ is therefore a **reuse of seen years** and is reported as secondary (Section 5).
2. **The physics head was trained and dropped on 2026-10-12** (`evaluation_v2/late_rungs.md`): WBGT
   +0.041 [+0.001, +0.082], Tmax +0.132 [+0.077, +0.187], out of fold 2007-2018. Its rule is not loosened (Section 4).
3. **A first look at fixed windows was run on 2026-11-23**, on 1980-2018 only, from daily peak hours
   (`results/go_no_go_fixed_windows.md` in the project files). It motivated H-A. It is reported as the reason for
   H-A, not as evidence for it.

The only fully unseen data available to this study is (a) any city added in Section 7, and (b) a prospective
2027 season (Section 9, optional).

---

## 1. Definitions (all sourced)

### 1.1 Heat stress measure
- **WBGT (outdoor, sun):** `WBGT = 0.7 Tnwb + 0.2 Tg + 0.1 Ta`, computed hourly by the Liljegren model as
  implemented in `pipeline/wbgt_liljegren.py` (docstring line 4; Liljegren et al. 2008, adapted to reanalysis
  following Kong & Huber 2022, doi:10.1029/2021EF002334, cited in that file). The same weights are the outdoor
  formula used by OSHA (OSHA Technical Manual, Section III Chapter 4).
- **Inputs:** ERA5 hourly via Open-Meteo, 9 Delhi cells (`pipeline/download_era5_v2.py`, `CELLS`), IST timestamps
  (`timezone: Asia/Kolkata` in both downloaders). Radiation is the mean of the preceding hour
  (`pipeline/build_wbgt_liljegren.py` docstring). **[DATA] Requires `data/raw/era5_v2/`, which is not on the
  laptop used for this draft: copy it from the teammate who ran the download, or re-run
  `python -m pipeline.download_era5_v2` (resumable, about 1-2 days by its own estimate).**
- An hourly WBGT is treated as that hour's 1-hour time-weighted average. NIOSH expresses exposure "as 1-hour
  time-weighted averages (TWAs)" (NIOSH 2016, p.2). Assumption, disclosed: the worker works the whole hour.

### 1.2 Exposure limits (ISO 7243:2017, read directly)
- **Primary limits: ISO 7243:2017 Table A.1** (Annex A, p.7), reference WBGT for ordinary work clothes (CAV = 0,
  Annex F Table F.1, p.16), in °C, acclimatised / unacclimatised:
  class 1 low (180 W) 30 / 29; **class 2 moderate (300 W) 28 / 26**; class 3 high (415 W) 26 / 23; class 4 very
  high (520 W) 25 / 20. Class ranges from Table E.1 (p.15, from ISO 8996): moderate = 235-360 W.
- **Primary limit: acclimatised, class 2 moderate, WBGT 28 °C.** Reason: Table E.1's moderate class covers
  sustained hand-and-arm and walking work typical of construction and street labour; workers in a city's heat
  season are mostly acclimatised. Sensitivities: unacclimatised 26 °C (new or migrant workers), and classes 1 and 3.
- **Equation form (sensitivity):** ISO 7243:2017 Annex A (p.8) also gives
  `WBGT_ref = 56.7 - 11.5 log10(M)` (acclimatised) and `59.9 - 14.1 log10(M)` (unacclimatised), M in W,
  115 < M < 520, and says they "may differ from the table values". These are the same equations as the NIOSH
  REL / RAL (Ricketts 2022, *Professional Safety*, Aug 2022). The table is primary because it is the standard's
  stated reference value; the acclimatised equation agrees with it within 0.3 °C, the unacclimatised one does not
  (30.8 vs 32 °C at 115 W).
- **Time base:** WBGT is a time-weighted mean over "about 1 h" (ISO 7243:2017 clause 5, p.3; clause 9.2, p.5),
  matching NIOSH's 1-hour TWA (NIOSH 2016, p.2). An hourly WBGT value is treated as that hour's mean, assuming the
  worker works the whole hour (disclosed).
- **Outdoor formula:** ISO 7243:2017 formula (2), p.3: `WBGT = 0.7 tnw + 0.2 tg + 0.1 ta`, identical to the
  repository's (Section 1.1).
- **Unsafe hour:** an hour from 06:00 to 20:00 IST whose WBGT exceeds the primary limit.
- **Unsafe day:** a day with at least **2 consecutive** unsafe hours. Reason: NIOSH 2016 (Section 1.1.2, p.3) asks
  for conditions to be changed when "two such sequential measurements exceed the applicable RAL or REL".

### 1.3 Official alert schemes for Delhi (from `sources/phase6_thresholds.json`, every row VERIFIED there)
- **S1, IMD heat wave (station rule):** Tmax ≥ 40 °C and departure from normal ≥ 4.5 °C, or Tmax ≥ 45 °C
  (IMD_HW_FAQ p.1). IMD declares it when met at ≥ 2 stations for ≥ 2 consecutive days, on the second day
  (IMD_HW_FAQ p.2). With one station this study applies the 2-consecutive-day condition at Safdarjung only;
  disclosed as an approximation.
- **S2, Delhi HAP absolute table:** Yellow Tmax ≥ 40 °C; Orange and Red both printed as ≥ 45 °C
  (DELHI_HAP_2025 p.10; the duplication is recorded in the catalogue).
- **S3, Delhi HAP departure table:** Orange +4 to 5 °C above normal Tmax; Red ≥ +6 °C (DELHI_HAP_2025 p.45).
- **Station data:** Safdarjung GHCN-Daily Tmax, 1944-01-01 to 2025-08-23, with gaps (`sources/data/README_DATA.md`).
  **[DATA]** the CSVs are gitignored (`/sources/data/*.csv`) and not on the laptop used for this draft; copy them
  from a teammate or re-download GHCN-Daily station IN022021900.
- **Normal Tmax:** Safdarjung 1981-2010 day-of-year mean, 31-day smoothed. Reason: no source found that states
  IMD's normal period (catalogue and a web search, 23 Nov); 1981-2010 keeps every statistic inside the years
  before the test period, the rule of CONFIRMATORY_PROTOCOL §1 ("Climatology, labels and normalisation come from
  1980-2018 only"). Sensitivity: 1991-2020, the current WMO standard normal.
- S1-S3 are applied retrospectively to every year as rules, not as
  historical declarations. This is a counterfactual audit of the rules, not of what was declared.

### 1.4 Fixed time windows (from `sources/phase6_candidate_actions.json`, every row VERIFIED there)
- **W1 work, 12:00-16:00:** "Change the shift of work as far as possible, to avoid peak hours (12 - 4 PM)"
  (DELHI_HAP_2025 p.155; trigger: IMD forecast of above-normal temperature, Apr-Jun 2025).
- **W2 work, 13:00-17:00:** "shift outdoor workers' schedules away from the peak afternoon hours (1pm - 5pm) during
  a heat alert" (DELHI_HAP_2025 p.81).
- **W3 public, 12:00-15:00:** "Avoid going out in the sun, especially between 12.00 noon and 3.00 p.m."
  (IMD_HW_FAQ p.10). **This is public advice, not a work rule**, and is reported separately.
- **W4 school, 12:00-16:00** when a heat wave is declared (DELHI_HAP_2025 p.58), sub-analysis.
- **W5 work, 13:00-16:00 (2026 directive):** Delhi CM's directive under the Heat Wave Action Plan 2026 that
  workers "must be given rest from 1 PM to 4 PM" (NewKerala, 2 May 2026; Republic World, 27 Apr 2026, whose own
  directive list also says "noon to 3-4 PM"). News reports only; no official order text found. Reported as a
  secondary window.
- **W6 work, 12:00-15:00 (2024 order):** Delhi LG's order of paid leave for construction workers "from 12 noon to
  3 pm" during extreme heat (News On AIR, 29 May 2024). Historical, reported as a secondary window.
- **Primary work window: W1 (12-4 pm)**, the only work window in the official plan text with a verified page and
  excerpt that names hours for all outdoor workers. W2, W5 and W6 are reported next to it.
- **Inside a window:** radiation at stamp t is the mean over (t-1, t] (`pipeline/build_wbgt_liljegren.py`), so the
  hour stamped t is treated as the interval (t-1, t]. It is inside a window if start < t ≤ end (for W1: stamps
  13:00-16:00). Sensitivity: start ≤ t < end (stamps 12:00-15:00). Both are reported, because the 23 Nov first look
  used a third convention (stamps 12:00-15:00 counted inside 12-3) and the choice moves the share by several points.

---

## 2. Data, splits and folds (unchanged from v1 unless stated)

- **Development:** the four rolling-origin folds f1-f4 (validation 2007-09, 2010-12, 2013-15, 2016-18; training
  every earlier year), each with its inner 2-year early-stopping block (`training/folds.py`). Climatology, labels
  and normalisation are per fold from training years only (same file).
- **Labels:** v2 Tmax and WBGT labels and strata from `pipeline/labels_v2.py` (WBGT: 95th percentile of in-season
  daily max over the fold's training years, Mar 15-Sep 30).
- **Statistics:** paired cluster-jackknife with year x season clusters, t(G-1), no CI below 5 clusters
  (`evaluation/stats.py`; MacKinnon, Nielsen & Webb 2023).
- **Seeds:** 10 per model; the metric is the mean of per-seed metrics (v1 convention, CONFIRMATORY_PROTOCOL §1).

---

## 3. Questions and hypotheses

### H-A. Do fixed work windows miss unsafe hours? (exposure; no model)
- **Population:** S2-Yellow days (Tmax ≥ 40 °C at Safdarjung), Mar 15-Sep 30. S1 and S3 days as sensitivities.
- **Primary estimate:** the share of unsafe hours (Section 1.2) that fall outside W1, pooled over alert days,
  9-cell mean WBGT, with a 95% CI from the year-cluster jackknife (`evaluation/stats.py` clusters).
- **Period:** 1980-2018 primary. 2019-2025 reported separately (descriptive; reads 2019+ only through
  `v2/test_lock_v2.require()` after the tag).
- **Supported if:** the best fixed 4-hour window, chosen on 1980-2006 alert days as the one that covers the most
  unsafe hours, covers more unsafe hours than W1 on 2007-2018 alert days, with the year-cluster jackknife CI of the
  difference above 0. Reason: this compares the rule with the best rule of the same length that could have been
  written from past data, so no arbitrary cut-off is needed. The share outside W1 is reported as the effect size.
- **Reported:** the same for W2 and W3; per work class; by decade (trend); unsafe hours per season; and
  "unsafe days with no alert in any scheme".
- **Validation of the diurnal cycle [DATA]:** compare ERA5's hourly temperature and humidity cycle (cell nearest
  Safdarjung) with NOAA's hourly station record for New Delhi / Safdarjung (WMO 42182, ICAO VIDD; NOAA
  Integrated Surface Database / GHCN-hourly). Metric: hour of daily maximum temperature and mean RH by hour,
  hot season, 1980-2018. If ERA5's peak hour differs by more than one hour on average, H-A is reported with that
  caveat in the abstract.

### H-B. Can missed humid-heat days be forecast? (models)
- **Event:** an unsafe day (Section 1.2: at least 2 consecutive unsafe hours) with no alert under S1, S2 or S3.
- **Forecasts:** existing out-of-fold forecasts from the v1 folds (`predictions_v2/`) converted to the event
  through the physics head (H-N) or the direct models with the climatological diurnal shape (baseline).
- **Scores:** hit rate, false-alarm ratio, equitable threat score, leads 1-5, out of fold 2007-2018. Exploratory
  unless the event has at least 25 episodes pooled over the validation blocks (the v1 minimum-sample rule,
  `pipeline/labels_v2.py` docstring).

### H-C. Calibrated ranges on heatwave days (models)
- **Baseline:** the existing Gaussian ranges recalibrated on the inner block (`evaluation/calibration_check.py`).
- **What development already shows** (`evaluation_v2/range_forecasts.md`, out of fold 2007-2018, C3-hop WBGT):
  80% coverage 0.81 on all days but **0.23 on extreme days** (Gaussian) and 0.25 with the empirical inner-block
  residuals. The empirical method is in effect plain split conformal, so plain split conformal is already known
  to fail here and is not a new arm.
- **Method:** conformal prediction conditioned on heat level (Mondrian: separate inner-block residual quantiles
  for days whose forecast is in the top decile of in-season forecasts, and for other days). Calibration set: each
  fold's inner 2-year block only.
- **Primary stratum: forecast-hot days** (forecast in the top decile), not observed extreme days. Reason: coverage
  conditional on the *observed* outcome cannot be guaranteed by any method that only sees the forecast, the
  forecaster's dilemma (Lerch et al. 2017, Statistical Science 32(1), already cited in
  `evaluation/calibration_check.py`). Observed-extreme coverage is reported next to it, with this caveat.
- **Supported if:** 80% coverage on forecast-hot days and on all days are both within 0.75-0.85, out of fold
  2007-2018. Reason: the existing Gaussian ranges on all days (0.81) are the repo's working standard for
  "calibrated"; ±0.05 is the range the v1 results already treat as near-nominal (0.77-0.78 on the test years,
  `evaluation_test/test_results.md`). Interval width is always reported next to coverage.
- **Known limit, stated in advance:** conformal guarantees assume exchangeability, which serial correlation
  breaks; Asch et al. (2026, arXiv:2606.19642) found conformal "improves, but [does] not perfect, coverage of
  extreme temperature". A failure is reported as a result.

### H-N. Physics head v2 (engineering; models)
- **Model:** C3-hop encoder + the physics head, moved into `v2/` (copy of `models/physics_head.py`), predicting the
  7 ingredients for every hour 06:00-20:00 per cell, WBGT through `pipeline/wbgt_liljegren_torch.py`
  (implicit-function gradients; tests in `tests/test_wbgt_liljegren_torch.py`).
- **Keep rule (unchanged from v1, written 2026-10-07):** dropped if the all-days CI upper end is ≥ +0.05 °C for
  daily max WBGT or Tmax against the direct C3-hop, out of fold 2007-2018 (`evaluation_v2/late_rungs.md`).
- **Second rule for the new target:** unsafe-hour count MAE against the direct C3-hop with the climatological
  diurnal shape; dropped if the CI upper end of the difference is ≥ 2% of the baseline's MAE. Reason: the v1
  rule's 0.05 °C is 2.1% of the direct model's WBGT RMSE (2.373, `evaluation_v2/late_rungs.md`), so the same
  relative margin is applied to the new target.
- **E4 (soft-count loss):** a sigmoid sum over hours with sharpness annealed in training. It is a separate arm
  and must pass the same two rules.
- **Reported, not tested:** physical consistency (share of forecasts where WBGT, Tmax and unsafe hours
  contradict), and the ingredient error breakdown.

### H-D. Value of the forecast (exploratory)
- Relative economic value for the event "unsafe day" (Section 1.2), over cost/loss ratios 0.05-0.5 (Richardson 2000,
  QJRMS 126). Exploratory only.

### Multiple testing
H-A, H-C and H-N are the primary family: Holm correction, family-wise alpha 0.05 (as v1). H-B and H-D are exploratory.

---

## 4. Rules that do not change

- The v1 physics-head rule is not loosened. The paper states that the daily head failed it.
- Nothing in `training/ models/ pipeline/ retrieval/ evaluation/ configs/` is edited on any branch
  (`training/test_lock.py`). Changes are copies in `v2/`.
- Labels, thresholds, windows and limits are the ones in Section 1. A change after the tag is a deviation and is
  listed in the paper.

---

## 5. Test period (2019+)

- Read only after `test-lock-v2` exists, via `v2/test_lock_v2.require()` (which also runs the v1 lock).
- **Secondary, because seen (Section 0).** H-C and H-N are scored on 2019-01-01 to 2026-09-06 once, as in v1,
  and reported next to the development result. They cannot overturn a development failure.
- Power: before the tag, compute the test MDE for each primary hypothesis the v1 way (CONFIRMATORY_PROTOCOL §3:
  scale the development CI half-width by sqrt(G_dev / G_test), t quantiles at G_test - 1) and add the table here.

---

## 6. Outputs and folders

- Code: `v2/`. Configs: `configs_v2/`. Results: `results_v2/` (never inside the locked v2 folders, which would
  break `v2/test_lock_v2.py`). Do not write into `evaluation_v2/`, which holds the v1 development results.

---

## 7. Other cities (exploratory; the only unseen data)

**Selection criteria, fixed before any city data is downloaded:** (1) climate contrast with Delhi (one humid
coastal, one hot-dry inland), so a result can be shown to generalise; (2) the city has a heat action plan with an
alert rule that can be audited; (3) published evidence of heat-health risk or a plan evaluation; (4) a large
outdoor workforce. Cities are not chosen or swapped after seeing any WBGT result.

- **Chennai (humid):** the IMD-authored hazard analysis in the catalogue says heat-stress and heat-stroke risk
  are maximum in Chennai "due to high temperature along with higher RH" (HOT_WEATHER_HAZARD_ANALYSIS_2022 p.2).
  Greater Chennai Corporation runs a city heat plan (news reports 2024-2026); **[DATA]** its alert thresholds
  must be added to the catalogue with page and verbatim excerpt before Chennai's alert audit is run.
- **Ahmedabad (dry):** South Asia's first heat action plan (2013; Knowlton et al. 2014, IJERPH 11(4), 3473),
  with a published mortality evaluation (Hess et al. 2018, PMC6236972), and the only other city with a verified
  alert table in the catalogue (AHMEDABAD_HAP_2018 p.3: Yellow 41.1-43, Orange 43.1-44.9, Red ≥ 45 °C).
- Considered and not chosen: Hyderabad and Nagpur (dry inland; plans exist but thresholds not verified, and
  Ahmedabad has the stronger evaluation record); Kolkata and Mumbai (humid; Chennai has the IMD-authored risk
  statement).
- H-A is repeated with the same definitions; city alert rules use the IMD criteria (S1) unless a city plan is
  added to the threshold catalogue with page and verbatim excerpt.

---

## 8. Order of work and freeze

1. Obtain the [DATA] items (Section 10); commit this file on `experiments-v2`.
2. Development folds only (2007-2018). Apply the rules in Section 3.
3. Add the power table (Section 5); commit.
4. `git tag test-lock-v2` and `git push origin test-lock-v2` straight away. The tag is created once and never
   moved (`git tag -f` is not used); the push timestamp on GitHub is the record.
5. Score 2019+ once. Anything after that is post-test exploration.

## 9. Optional: prospective 2027 season

The only test no one has seen. Daily forecasts for Mar 15-Sep 30 2027, committed to the repository each day before
the forecast day starts, scored at season end with the same definitions.

---

## 10. Decisions and checks resolved on 23 Nov 2026

| Item | Decision | Source |
|---|---|---|
| Exposure limits | ISO 7243:2017 Table A.1; primary acclimatised moderate 28 °C | ISO 7243:2017 Annex A p.7, Table E.1 p.15, Table F.1 p.16 |
| NIOSH equation discrepancy (27.8 vs 27.5 °C) | Moot: the table is primary; equations kept as sensitivity | ISO 7243:2017 Annex A p.8 ("may differ from the table values") |
| Work classes | ISO 8996 classes as reproduced in ISO 7243 Table E.1 | ISO 7243:2017 p.15 |
| Unsafe day | ≥ 2 consecutive unsafe hours | NIOSH 2016 §1.1.2 p.3 |
| Normal Tmax | Safdarjung 1981-2010; 1991-2020 sensitivity | CONFIRMATORY_PROTOCOL §1; WMO standard normals |
| Work windows | W1 12-4 pm primary; 1-5, 1-4 (2026), 12-3 (2024) secondary | DELHI_HAP_2025 p.155, p.81; news reports 2024, 2026 |
| H-A rule | Beats-the-best-past-window test, no fixed cut-off | Section 3 |
| H-C stratum and rule | Forecast-hot days, coverage 0.75-0.85 | range_forecasts.md; Lerch et al. 2017 |
| H-N new rule | 2% of baseline MAE, as v1 | late_rungs.md |
| Cities | Chennai, Ahmedabad | phase6_thresholds.json rows |
| Diurnal check | NOAA hourly record, WMO 42182 | NOAA ISD / GHCN-hourly |

Remaining **[DATA]**: `data/raw/era5_v2/`; Safdarjung GHCN-Daily CSVs; NOAA hourly record for 42182; ERA5 for
Chennai and Ahmedabad.

---

## References
- IMD_HW_FAQ, DELHI_HAP_2025, HOT_WEATHER_HAZARD_ANALYSIS_2022: see `sources/phase6_thresholds.json` and
  `sources/phase6_candidate_actions.json` (page and verbatim excerpt per row).
- ISO 7243:2017. Ergonomics of the thermal environment: assessment of heat stress using the WBGT index.
- Lerch, S. et al. (2017). Forecaster's dilemma: extreme events and forecast evaluation. Statistical Science 32(1).
- NIOSH (2016). Criteria for a Recommended Standard: Occupational Exposure to Heat and Hot Environments. DHHS
  (NIOSH) Publication 2016-106.
- Ricketts (2022). The case of the heat-stressed tree feller. Professional Safety, August 2022 (Math Toolbox).
- OSHA Technical Manual, Section III Chapter 4, Heat Stress.
- Liljegren et al. (2008), J. Occup. Environ. Hyg. 5(10); Kong & Huber (2022), Earth's Future,
  doi:10.1029/2021EF002334.
- MacKinnon, Nielsen & Webb (2023), J. Econometrics 232(2).
- Asch et al. (2026), arXiv:2606.19642.
- Richardson (2000), Q. J. R. Meteorol. Soc. 126.
