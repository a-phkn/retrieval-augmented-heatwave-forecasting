# v2 study protocol: physics-constrained forecasting of unsafe working hours (DRAFT, 23 Nov 2026; updated 24 Nov)

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
4. **Code check on Ahmedabad development data, 2026-11-24:** while testing `v2/build_city_wbgt.py`, three months of
   one Ahmedabad cell (cell 5: May 2010, June 2010, May 2016) were computed and their daily maximum WBGT and
   hour of the WBGT peak were looked at. Most peaks were at 14:00-15:00 IST in those months. Nothing else of the
   city data has been looked at. Ahmedabad is therefore not fully unseen for H-A; it is reported as such.

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
- **Missing station days (added 24 Nov, on receiving the file, before any H-A computation):** Safdarjung GHCN-Daily
  TMAX (station IN022021900, `sources/data/safdarjung_daily_tidy.csv`) lacks 660 of 7,800 season days in
  1980-2018 (8.5%), concentrated in 1992 (81), 2016 (101), 2017 (106) and 2018 (80). Days without a station Tmax
  cannot be classified and are excluded; the count is reported per year. Sensitivity: the same test with alert
  days from the ERA5 9-cell Tmax proxy (the go/no-go note's rule), which has no gaps. Values flagged in TMAX_QFLAG
  (4 days in the whole record) are excluded.
- **Implementation details fixed before any H-A result (24 Nov, `v2/window_test.py`):** S1's two-day condition is
  unknown (excluded) when either day lacks a station value, unless the other day is a known non-criterion day, which
  settles "no alert". The ERA5 proxy's daily Tmax is the mean over the 9 cells of each cell's daily maximum hourly
  2 m temperature, and it keeps the go/no-go rule's own season (Mar 15-Jul 31). Normals are a circular 31-day
  moving average of the day-of-year mean. The by-decade share outside each window is reported for S2 only.
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
- **Event:** an unsafe day (Section 1.2: at least 2 consecutive unsafe hours) with no alert under S1, S2 or S3,
  and (added 24 Nov) no day counted toward any insurance trigger in Section 3 H-E. This merges the "missed days"
  question for alerts and for insurance into one event.
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
- **Secondary arm, added 24 Nov: trigger with a bounded miss rate (conformal risk control).** The decision "unsafe
  day tomorrow" is issued when the forecast score exceeds a threshold λ. λ is chosen on each fold's inner block
  with conformal risk control (Angelopoulos et al., ICLR 2024) so that the expected share of unsafe days with no
  trigger is at most α = 0.20 (matching the 80% level used above; α = 0.10 reported as a sensitivity).
  **Reported:** realised miss rate and false-trigger rate out of fold 2007-2018, leads 1-5. **Supported if** the
  realised miss rate is at most α + 0.05 (the same ±0.05 tolerance as above). **Superseded by amendment A2
  (Section 12): E5 is now a primary hypothesis in the Holm family, with the event and test defined there.**
  The same exchangeability caveat applies.

### H-N. Physics head v2 (engineering; models)
- **Model:** C3-hop encoder + the physics head, moved into `v2/` (copy of `models/physics_head.py`), predicting the
  6 learned ingredients (temperature, RH, pressure, 10 m wind, global irradiance, direct fraction) for every hour
  06:00-20:00 per cell (`v2/physics_head_hourly.py`). The sun angle is not predicted: mean sunlit cos(zenith) is
  computed from date, hour and location (`pipeline.wbgt_liljegren.mean_sunlit_cosz`) and passed in, because its
  true value is known in advance. WBGT through `v2/wbgt_smooth.py` (E2: the stability-class wind exponent made
  smooth; exact at zero width; at the default widths the bias against the frozen port is mean 0.0025 °C, 99th
  percentile 0.048 °C, maximum 0.125 °C on 20,000 random hours, `tests/test_v2_wbgt_smooth.py`), which calls the
  frozen implicit-gradient solvers of `pipeline/wbgt_liljegren_torch.py`.
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

### H-D. Value of the forecast: a pay-ahead heat payout (exploratory)
- Replaces the generic cost-loss analysis (24 Nov) with a concrete user: a payout announced 1-5 days ahead
  when the forecast triggers "unsafe day" (the H-C trigger), so a worker can rest without losing the day's wage.
- **Compared with:** the observed-temperature insurance triggers of H-E, on the same days and cells.
- **Reported:** unsafe days paid in advance, unsafe days missed, payouts on days that were not unsafe, payouts per
  season, and relative economic value over cost/loss ratios 0.05-0.5 (Richardson 2000, QJRMS 126).
- **Why:** current Indian heat payouts arrive 6-8 weeks after the heat (NPR, 16 Jul 2025), and workers "are
  unlikely to forgo wages based on a forecast that might not trigger a payout" (Down To Earth, May 2026).
  Closest existing practice: Start Network's Pakistan heatwave model releases funds to NGOs on a temperature
  forecast (startnetwork.org; HPN, Mar 2025), with no published hit or miss record.
- **Second design, added 24 Nov: advance plus top-up.** A fixed share of the payout is paid on the forecast
  trigger; the rest is paid after the day only if it was observed unsafe; an unsafe day with no forecast trigger is
  paid in full after the event. Shares 25%, 50% and 75% are reported; none is chosen as best in advance. Same
  outcomes as above, plus the share of all payout money that reached workers before an unsafe day.
- **Per-shift output (reported only):** unsafe hours forecast for 06-10, 10-14 and 14-18 h, the form a shift-based
  user would need. No test.
- Exploratory only.

### H-E. How often do heat-insurance triggers miss unsafe working days? (exposure; no model; descriptive)
- **Triggers audited**, applied to ERA5 daily maximum 2 m temperature per cell, as the schemes themselves use ERA5:
  - VimoSEWA, Ahmedabad: at least 7 days (not necessarily consecutive) above 43.2 °C, 1 Apr-31 Jul (VimoSEWA x
    SCF report, Dec 2025, pp. 4, 21-22).
  - SEWA with ICICI Lombard and Swiss Re, Ahmedabad: 2 consecutive days above 43.6 °C and above 44.1 °C (NPR, 16 Jul
    2025).
  - Mahila Housing Trust, Delhi: 45.27, 46.53 and 47 °C, 1 May-31 Jul (Down To Earth, May 2026). The source gives
    no day-count rule for Delhi; a single day above the threshold is assumed. **[CHECK]**
  - For the two SEWA triggers the risk period is not stated in the source; VimoSEWA's 1 Apr-31 Jul is used. **[CHECK]**
  - "Exceeding" is read as strictly greater than. Daily maximum = highest of the 24 hourly ERA5 values of the
    local (IST) day; the schemes' own definition and grid point are not stated. **[CHECK]**
  - **Unit:** each ERA5 cell separately, for both the trigger and the unsafe-day count, because a worker is
    insured and exposed at one place. Results are pooled over the 9 cells and also given per cell.
  **[CHECK]** these come from a programme report and news, not the contracts. Confirm with the schemes if possible;
  the paper says where each value came from.
- **Reported, per city, 1980-2018 (2019-2025 separately, via the lock):** share of unsafe days (Section 1.2) inside
  each scheme's risk period that count toward no trigger; unsafe hours on those days; seasons with no payout but
  above-median unsafe hours; and the same by decade.
- **Why descriptive, not confirmatory:** a trigger built from WBGT unsafe hours would catch WBGT unsafe days by
  construction, so "WBGT trigger beats temperature trigger" would be circular. The audit measures exposure basis
  risk only.
- **Limit:** no wage or health records. The paper says "exposure basis risk", never "income loss".
- **Physiological sensitivity, added 24 Nov:** the same audit with "unsafe" defined by ISO 7933 Predicted Heat
  Strain (predicted rectal temperature above 38 °C within the 06-20 h working day), computed from the same hourly
  ingredients, globe temperature from the Liljegren solver, metabolic rate per ISO 7243 work class, acclimatised,
  drinking freely. **[CHECK]** the ISO 7933 limit values and the implementation (e.g. pythermalcomfort's `phs`)
  against the standard before running. Reported next to the WBGT result; it answers the question whether WBGT,
  a screening index, changes the conclusion when a body-heat model is used.
- **Closest prior work:** Risks (MDPI) 2026, 14(12), 271 optimises US insurance triggers on economic-loss indices,
  with no heat variable and no forecasts.

### Multiple testing
H-A, H-C and H-N are the primary family: Holm correction, family-wise alpha 0.05 (as v1). H-B, H-D, H-E and the
H-C trigger arm are exploratory or descriptive. **Amended by A2 and A2.5 (Section 12): the family is H-A, H-C and E5 (H-N tested alone),
with the p-value for each defined there.**

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
  **Alert rule (checked 23 Nov):** no Chennai-specific threshold was found. Greater Chennai Corporation's plan
  is reported in news, but no threshold table was found. Tamil Nadu's Heat Wave Action Plan 2019 (TNSDMA)
  reproduces the IMD criteria, so Chennai's rule is IMD's coastal criterion: Tmax ≥ 37 °C and departure ≥ 4.5 °C
  (IMD_HW_FAQ p.2, already VERIFIED in `sources/phase6_thresholds.json`).
  **Window:** TN HAP 2019 advises "Avoid working outside between 12 noon and 3 p.m." (read from a transcription
  of the plan; **[DATA]** confirm the page in the TNSDMA PDF before adding it to the catalogue). A reported
  10:00-16:00 construction-work restriction (2024) could not be traced to a primary notification: the one 2024
  Tamil Nadu gazette found (Extraordinary No. 336, 15 Oct 2024, G.O.Ms. No. 419) declares heat wave a
  state-specific disaster and sets no work hours. It is excluded unless the notification itself is found.
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

**Added 24 Nov 2026, before any insurance trigger was computed on any data:** H-E (trigger audit), H-D re-scoped
to a pay-ahead payout, H-B event widened to insurance triggers, and the conformal-risk-control arm of H-C. Dropped
after review the same day: an urban-heat-island analysis (ERA5's ~25 km cells cannot resolve it, and working
hours are daytime). Source notes: `research/sdg13_heat_insurance_gap.md` in the project files.
Later on 24 Nov, from a teammate's list: added the advance-plus-top-up payout design and per-shift output (H-D) and
the ISO 7933 sensitivity (H-E). Not adopted: wearable or phone telemetry, street-level downscaling and sensor
crowdsourcing, smart-contract or e-Rupee payment rails, per-shift insurance pricing, and smart-meter cooling
triggers. Each needs new data collection, hardware or commercial partners outside this study.

Remaining **[DATA]**: `data/raw/era5_v2/`; Safdarjung GHCN-Daily CSVs; NOAA hourly record for 42182; ERA5 for
Chennai and Ahmedabad.

---

## 11. Amendment A1 (24 Nov 2026, written AFTER the first H-A and H-E results; exploratory)
**What was seen first.** H-A was run on 1980-2018 (commit 69dd9b3, before results) and was not supported:
best past window 10-14 vs 12-4 pm, +0.003 [-0.000, +0.007], 12 clusters; S1, S3 and ERA5-proxy populations
agree. The H-E audit for Delhi showed the binary "unsafe day" is saturated: at 28 C, 31,640 of 32,292 May-July
cell-days qualify. H-A stays reported as pre-registered and not supported. Nothing below replaces it.

**Why.** A yes/no "unsafe" label at the continuous-work limit cannot tell windows, triggers or forecasts apart in
Delhi. A graded scale can.

**A1.1 Rest-needed scale (`v2/work_rest.py`).** Required rest share per working hour from the ACGIH work-rest
screening table (TLV = acclimatised, Action Limit = unacclimatised; 4 work classes; values checked 24 Nov in two
copies that agree: Cal/DOSH copy of the ACGIH chapter, Table 2, and CCOHS OSH Answers Table 1 citing the 2026
TLVs and BEIs p.242). WBGT at or below a row's value allows that row; rest share 0 / 0.25 / 0.50 / 0.75, or 1
(stop) above the 0-25 % row; rows printed "-" are never allowed. Daily rest-hours = sum over stamps 06-20.
The table is a screening tool "not intended to prescribe work and recovery periods"; it is used as an exposure
scale only. Amendment primary row: moderate work, TLV, sun WBGT.

**A1.2 Shade WBGT (`v2/shade_wbgt.py`).** Same frozen Liljegren functions; 2 m wind from the actual global
radiation; globe and natural wet bulb with diffuse radiation only (no direct beam); WBGT = 0.7 Tnwb + 0.3 Tg
(ACGIH/CCOHS shaded formula). Sun WBGT is the upper bracket, shade the lower. Approximation: the ground-reflected
beam is dropped with the beam.

**A1.3 H-A1 (exploratory).** On S2 alert days 1980-2018: share of required rest-hours outside W1, W2, W3
(jackknife CI as H-A); the H-A best-past-window test with hours weighted by rest share; mean rest-hours per alert
day. Reported for every class, TLV and AL, sun and shade.

**A1.4 H-E1 (descriptive).** Per insurance trigger, per ERA5 cell, within the risk period: share of rest-hours
on days that do not count toward a payout; share of seasons paid; share of top-quartile rest-hour seasons that
were unpaid; Spearman correlation across cell-seasons of season rest-hours vs days counted toward a payout.

**A1.5 Consequence for H-D (fixed now, before any forecast result).** The H-D event becomes "tomorrow's
rest-hours (moderate, TLV, sun) >= R", with R the 75th percentile of in-season (Mar 15-Sep 30) daily rest-hours
over 1980-2006 ERA5 9-cell-mean days, computed once and recorded here before H-D is run. The unsafe-day event
is kept as a reported secondary.

None of A1 enters the Holm family (H-A, H-C, H-N).

**A1.6 Recorded after the first A1 run (commit 9df47dc, 24 Nov), before any forecast result:**
- **R = 9.0 rest-hours** (moderate, TLV, sun WBGT; 75th percentile of in-season daily rest-hours, 9-cell mean,
  1980-2006, 5,400 days). This fixes the H-D event "tomorrow's rest-hours >= 9.0".
- **Shade WBGT correction.** The first run used the no-solar-load formula (0.7 Tnwb + 0.3 Tg) for shade. A diagnostic
  printed by the builder showed shade exceeding sun by up to 1.62 C (bright overcast hours, globe far above air
  temperature). Shade now uses the same outdoor weights as sun (0.7 Tnwb + 0.2 Tg + 0.1 Ta), which guarantees
  shade <= sun. All shade rows are re-run; the first-run shade numbers are kept in results_v2 history and noted in
  the paper as superseded. Sun rows are unaffected.
- **H-E1 Spearman CI added:** delete-one-season-year jackknife (the 9 cells of a year are not independent), t(G-1).


## 12. Amendment A2 (24 Nov 2026, evening; before any model, calibration or trigger result)
**Status at writing:** H-A has been run (not supported, commit 69dd9b3) and A1's exposure results exist (9df47dc).
No forecast, calibration, trigger or physics-head result exists: Sudarshan's C3-hop model files and
`predictions_v2/inner/` have not been received. A2 changes only hypotheses that have not been run.

**A2.1 E5 becomes a primary hypothesis (Holm family).**
- **Event:** tomorrow's rest-hours >= R = 9.0 (A1.5/A1.6; moderate, TLV, sun WBGT, 9-cell mean, stamps 06-20),
  in season (Mar 15-Sep 30). Not the binary "unsafe day": in Delhi that holds on ~98% of May-July days (H-E run).
- **Score:** the C3-hop daily maximum WBGT forecast at each lead 1-5 (exists without E4). If H-N passes, the
  physics head's forecast rest-hours is reported as a second score; the test uses the C3-hop score only.
- **Threshold:** conformal risk control on each fold's inner block, α = 0.20 (α = 0.10 as sensitivity), as in H-C.
- **Test:** out of fold 2007-2018, pooled over folds, lead 1 primary (leads 2-5 reported). Realised miss rate =
  share of event days with no trigger. H0: miss rate >= 0.25 (α + 0.05). One-sided p-value from the
  year x season cluster jackknife t statistic (t(G-1), as H-A). **Supported if** the Holm-adjusted p < 0.05.
  False-trigger rate and triggers per season are always reported next to it.
- **Wording:** superseded by A2.5 ("finite-sample miss-rate bound"); never "guaranteed" or "certified".
  Conformal risk control bounds the expected miss rate only under exchangeability; serial correlation and the
  warming trend can break it (H-C known limit).

**A2.2 p-values for the whole Holm family** (the earlier text gave pass rules as CIs or bands, which Holm cannot
use directly):
- **H-A:** two-sided jackknife t p-value of coverage(best) - coverage(W1), matching the original rule (95% CI above
  0). Already run; it is not relaxed to a one-sided p after the result, and Holm can only make it stricter.
- **H-C:** equivalence (two one-sided tests) of 80% coverage against 0.75 and 0.85, on forecast-hot days and on all
  days; p = the largest of the four one-sided jackknife p-values.
- **H-N:** non-inferiority, one-sided jackknife p for each pre-set margin (WBGT and Tmax 0.05 C, unsafe-hour error
  2%); p = the largest.
- **E5:** as A2.1.
Holm over these four at family-wise 0.05. (Superseded by A2.5: the family is H-A, H-C, E5.)

**A2.3 E4 is not load-bearing.** The paper's main claims rest on the audit (H-A, H-E/H-E1), the calibrated
trigger (H-C, E5) and the pay-ahead design (H-D, exploratory). H-N/E4 is reported pass or fail; if it fails, the
WBGT layer is reported as an enabling tool only.

**A2.4 Novelty wording (prior-work checks of 24 Nov, `research/sdg13_heat_insurance_gap.md`).** Conformal methods
already exist for heat decisions on air temperature (Berk, arXiv 2508.12886), for core-body temperature
(Communications Engineering 2025, doi 10.1038/s44172-025-00548-6) and for parametric-insurance payout obligations
on rainfall (Kazlauskaite, arXiv 2606.06391). The claim is: "we found no conformal risk control applied to an
hourly WBGT exposure trigger or to the timing of a heat payout."

**A2.5 Final additions to A2 (24 Nov, evening; still before any model, calibration or trigger result).**
- **E5 is two-part.** Supported only if (i) the Holm-adjusted p for "miss rate < 0.25" is < 0.05, and (ii) the
  false-alarm rate (share of non-event in-season days on which the trigger fires) is <= 0.40. Reason for (ii): CRC
  bounds misses only; a weak score can meet that bound by firing on most days. 0.40 is a pre-specified choice, not
  sourced; it is base-rate free (lift is base-rate dependent) and means the trigger is right about at least 60% of
  quiet days. Reported next to it: base rate, fire rate, precision, lift over base rate, Peirce skill score
  (1 - miss - false alarm), triggers per season, number of event days.
- **Severe subset (secondary, reported).** Miss rate on event days with rest-hours >= the 90th percentile of in-season
  daily rest-hours over 1980-2006 (9-cell mean, moderate, TLV, sun), so the paper shows whether misses concentrate
  on the worst days.
- **No hot-day conditioning of E5's miss rate.** A suggestion was to report E5's miss rate "on forecast-hot days".
  Rejected: the miss rate is already computed only on observed event days; conditioning on the forecast would
  exclude exactly the days the forecast under-predicts and hide misses. The severe subset above answers the concern.
  E5 keeps the raw C3-hop score: CRC needs only the ranking of the score, not its calibration, so E5 does not
  depend on H-C.
- **Rest-hours is not "unsafe hours" relabelled.** It is the sum over stamps 06-20 of the ACGIH required rest share
  (0, 0.25, 0.5, 0.75, 1) per hour (A1.1), so an hour just over the continuous-work limit counts 0.25, not 1.
  **R = 9.0** is the 75th percentile of in-season daily rest-hours over 1980-2006 (A1.5): a pre-specified
  percentile, not an external standard. The event base rate over 1980-2018 and the number of event days in
  2007-2018 and in the test years are reported and go into the power table.
- **Holm family = H-A, H-C, E5.** H-N leaves the family (A2.3 makes it not load-bearing; keeping it would only raise
  the bar for the tests that are). H-N is tested on its own at one-sided 0.05 with the non-inferiority p of A2.2,
  and reported as the engineering result, pass or fail.
- **Pass rules are p-values only.** Each hypothesis passes on its Holm-adjusted p (< 0.05) as defined in A2.2; CIs
  are reported at 95% and labelled unadjusted. H-A additionally needs its point estimate > 0 (a significant negative
  difference does not support it). H-C is the TOST equivalence test of A2.2, margin 0.05 around 0.80 (the v1
  near-nominal range, H-C "Supported if"). H-N's "unsafe-hour error 2%" rule is its one-sided non-inferiority p.
- **H-C's Holm test is pinned down** (the H-C text named neither model nor lead): the WBGT model A2Lr_hw5_C3hop,
  lead 1, pooled over folds f1-f4 (each fold calibrated on its own inner block, which is also its early-stopping
  block: `training/train_unified.py` `fit_and_stop_sets`, `early_stop: inner_2y`). Leads 2-5 and the Tmax model
  A1prime_hw5_C3hop are reported, not tested. E5 likewise uses lead 1 for the test.
- **Power table** is computed at the Holm thresholds for three tests (0.0167 for the smallest p, 0.025, 0.05) and
  reports the number of event days, not just days.
- **Wording.** E5 is "a trigger with a finite-sample miss-rate bound (under exchangeability)"; "calibrated" is kept
  for H-C coverage only. The claim reads: "we found no conformal risk control applied to (a) an hourly WBGT exposure
  trigger or (b) the timing of a heat-insurance payout."
- **Prior-work search record.** Web searches on 24 Nov 2026 (general web search; Google Scholar/Scopus pending,
  team plan task 5), strings: "adaptive conformal prediction heat action temperature threshold", "conformal deep
  learning core body temperature heat stress", "conformal risk control drought index insurance trigger payout
  error", "conformal" + "WBGT", "conformal" + "heat insurance". Neighbours found: Berk arXiv 2508.12886;
  Communications Engineering 2025 doi 10.1038/s44172-025-00548-6; Kazlauskaite arXiv 2606.06391. Full notes:
  `research/sdg13_heat_insurance_gap.md`. Repeated before submission.

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
- Angelopoulos, A. N., Bates, S., Fisch, A., Lei, L., Schuster, T. (2024). Conformal risk control. ICLR 2024.
- VimoSEWA and SEWA Cooperative Federation (Dec 2025). Parametric heat and rainfall insurance for informal women
  workers. sewafederation.org.
- NPR (16 Jul 2025). Heat insurance for India's informal workers. npr.org/sections/goats-and-soda.
- Down To Earth (May 2026). Can parametric insurance shield India's informal women workers from extreme heat and
  wage loss? downtoearth.org.in.
- Start Network. Pakistan heatwave model (2022); Amad & Riaz (2025), Humanitarian Practice Network.
- Parametric weather insurance design under extreme event risk (2026). Risks 14(12), 271.
