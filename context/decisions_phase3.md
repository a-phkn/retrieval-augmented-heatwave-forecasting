# Decision Log: Phase 3 extras (branch `phase3-extras`)

Pre-registrations and results for the follow-up work after the one-time test. Kept separate from
`context/decisions.md` so the two branches never edit the same log; a summary goes into the main log when this
branch is merged.

Rules for this file:
- An entry is **DRAFT** until the user approves it; it is **PRE-REGISTERED** once committed and pushed, before
  anything it describes is run on the 2019-2026 test years. GitHub's push time is the timestamp.
- Frozen entries are never edited. A later change is a new dated **DEVIATION** line under the entry; results are
  added as dated **RESULT** lines.
- Everything computed is reported, including results that go against us.
- The less-data (data-scarcity) experiment, if it is run, gets its own entry before it starts.

**Common to every entry below. PRE-REGISTERED 2026-11-05 (user approved).**
- Status: post-test and descriptive. Nothing here changes H1-H4, the frozen models or the advisory. The test years
  were already used once (`docs/CONFIRMATORY_PROTOCOL.md`), so no result here is confirmatory and there is no Holm
  family.
- Code: new files only, in `followup/` (results in `followup/results/`, tests in `tests/test_followup_*.py`). No file
  under training/ models/ pipeline/ retrieval/ evaluation/ configs/ changes, so the test lock stays valid; new code
  may import the frozen modules. Every script calls `training/test_lock.require()` before reading 2019+ data. No new
  dependencies (HTTP via the pinned `requests`).
- Our forecasts: the committed test-fold forecasts in `predictions_test/` (10 seeds; trained 1980-2016, early-stopped
  on 2017-2018). Against a single deterministic forecast we use the 10-seed ensemble mean (the deliverable) and also
  report the mean of per-seed RMSE, the test's primary score.
- Truth: ERA5, as in the test. Tmax = 9-cell mean of each cell's daily maximum (IST day); WBGT = 9-cell mean of the
  daily maximum of hourly Liljegren WBGT (`wbgt_lj_max`). The station truth is P3-3.
- Strata: all days; the test's "extreme" stratum (heatwave days by the label rule, as in H2 and H3); the heat season
  (Tmax 15 Mar - 31 Jul, WBGT 15 Mar - 30 Sep). Every score also by lead 1-5.
- Statistics: the paired cluster jackknife of `evaluation/stats.py` (year x season clusters). No CI below 5 clusters,
  flagged as fragile below 10.
- **Seen before writing these entries (exploratory, development year 2018 only, recorded so it cannot be
  reinterpreted later):**
  - Blending C3-hop's Tmax ensemble mean with GraphCast (current max-only correction): error correlation 0.37-0.47
    by lead; blend weight on C3-hop 0.12-0.21; RMSE 1.154 -> 1.129 (weights fitted leaving each month out), heat
    season 1.209 -> 1.198, the 20 extreme rows 1.049 -> 1.123.
  - A four-value daily-maximum map (P3-2a): floor 0.671 -> 0.542; GraphCast 1.150 -> 1.069 (my rebuild of the
    max-only figure gives 1.150 vs the official 1.154); with the C3-hop blend 1.062.
  - These motivated P3-2a and P3-2d, so 2018 cannot count as evidence for them.
  - Also seen: one Open-Meteo probe (2024-05-28, 12:00-16:00 IST: which variables exist, and the three models'
    values for that day; nothing scored against truth), each model's first archived day, and the WeatherBench 2
    GraphCast 2020 file list.

**P3-1: Our forecasts vs operational global forecasts (IFS, AIFS, GFS). PRE-REGISTERED 2026-11-05 (user approved).**
- Question: for Delhi, how do the forecasts that are actually available every day compare with C3-hop on Tmax and
  WBGT, days 1-5?
- Source: Open-Meteo's Previous Runs API. For lead L we use the variable's `_previous_dayL` version: the value
  predicted L x 24 h before its valid time. Models: `ecmwf_ifs025` (ECMWF IFS 0.25°), `ecmwf_aifs025` (ECMWF's AI
  model AIFS, 0.25°), `gfs_seamless` (NCEP GFS). Hourly, timezone Asia/Kolkata, wind in m/s, at the 9 ERA5 cell
  centres (lat 28.25 / 28.5 / 28.75 x lon 77.0 / 77.25 / 77.5).
- Timing: for query date q and lead L the target day is D = q + L - 1. Every hour of D then comes from a run started
  at least L x 24 h earlier, at the latest 12 UTC on q - 1. That is before our model's input cutoff (the end of
  q - 1 IST), so no global model has information ours lacks.
- Tmax: daily maximum of the 24 hourly 2 m temperatures of the IST day, per cell, then the 9-cell mean (the ERA5
  target's definition). If a model's native output is coarser than hourly, the hourly values are Open-Meteo's
  interpolation and can miss the afternoon peak. We record each model's native time step and disclose this; it is not
  corrected here.
- WBGT: hourly Liljegren WBGT from temperature_2m, relative_humidity_2m, surface_pressure, wind_speed_10m,
  shortwave_radiation and diffuse_radiation. The steps are those of `pipeline/build_wbgt_liljegren.py`: direct =
  global - diffuse; the mean sunlit cosine zenith of the preceding hour (Open-Meteo radiation is the past-hour mean,
  as for ERA5); `urban=True`. Then the daily maximum per cell and the 9-cell mean. IFS and GFS only: AIFS has no
  radiation in this archive, so AIFS is Tmax only.
- Periods: from each model's first complete archived day to windows ending 2026-09-06. That is IFS from 2024-02-04,
  AIFS from 2024-03-14 and GFS (all variables) from 2024-01-19. GFS Tmax also gets a longer comparison from its first
  archived day (Open-Meteo states March 2021). A (query date, lead) row whose target day lacks any hour or cell for a
  model is dropped from every comparison with that model.
- Compared with: C3-hop (`A1prime_hw5_C3hop`, `A2Lr_hw5_C3hop`) and the control LSTM (`A1prime_hw5`, `A2Lr_hw5`).
- Scores: RMSE and bias per stratum and lead. Yes/no warnings use R-point (warn if the forecast reaches the label
  threshold) for every model, reported as hit rate, false-alarm ratio, ETS and frequency bias (post-test 3
  definitions; event = a day meeting the label rule). C3-hop's pre-chosen rules (R-prob, R-shift) are shown for
  reference.
- Test: the paired cluster jackknife of RMSE(global model) - RMSE(C3-hop), per model, target and stratum.
- Expected (written now): IFS and AIFS have a lower Tmax error than C3-hop at every lead, as GraphCast did in 2018.
  WBGT is open.
- Caveat stated now: ERA5 is produced with the IFS model, so ERA5 truth may favour IFS and our ERA5-trained models
  over GFS and AIFS. P3-3 repeats Tmax against the station.
- Timing sensitivity (IFS only): ECMWF IFS HRES 9 km from Open-Meteo's Single Runs API, the run started at 12 UTC on
  q - 1 for every lead (the GraphCast yardstick's timing), from 2024-03-14, Tmax and WBGT. Reported next to the main
  result.
- Code: `followup/global_models.py` -> `followup/results/global_models.{md,json}`. Raw API replies are kept in
  `data/raw/open_meteo_runs/` (gitignored, like other raw downloads); the tidy daily extract is committed with its
  SHA-256.
- **DEVIATION 2026-11-06 (a clarification, written before any data was downloaded):**
  - Requests use the ERA5 download's own coordinates and settings (`pipeline/download_era5_v2.py`: `CELLS`,
    `cell_selection=nearest`), not the rounded cell centres.
  - A probe showed both snap to the same 0.25° grid points (cell 1: 28.25° N, 77.0° E) with the same elevation for
    Open-Meteo's height adjustment (274 m), so the global forecasts and the ERA5 truth are adjusted identically.
  - IFS HRES 9 km (Open-Meteo model id `ecmwf_ifs`) uses its nearest 9 km grid point to the same coordinates.
  - The Single Runs API labels hours at XX:30 IST (whole UTC hours), unlike XX:00 IST elsewhere. The 24 values whose
    timestamps fall in the IST day are used, as for the other models.
- **RESULT 2026-11-08** (`followup/results/global_models.md`). Descriptive, with few clusters: IFS has 9 on all days and
  3 in the heat season and on heatwave days.
  - **Tmax, IFS vs C3-hop: IFS is better.**
    - RMSE on all days 1.76 vs 2.13 °C (delta -0.36, CI -0.83 to +0.10).
    - Heat season 1.75 vs 2.48; heatwave days 1.58 vs 2.38 (no CI).
    - The gap grows with lead: day 1 1.47 vs 1.52 (about even), day 5 2.03 vs 2.52.
    - On heatwave days IFS is 0.85 °C too cold and C3-hop 2.04 °C too cold.
  - **Tmax, AIFS** (the archive id `ecmwf_aifs025` ends 2025-03-10: one heat season, 3 clusters):
    - worse than C3-hop on all days (2.12 vs 1.89), 1.50 °C too cold, as expected from 6-hourly output missing the
      afternoon peak;
    - better in the heat season (1.69 vs 2.04); worse on heatwave days (2.51 vs 2.22).
  - **Tmax, GFS: much worse than C3-hop.**
    - 3.73 vs 2.12; since 2021, 3.43 vs 2.19 (delta +1.25, CI +0.90 to +1.59, 17 clusters).
    - 2.6-2.9 °C warmer than ERA5 on all days, 3.4-3.7 °C in the heat season. Whether ERA5 or GFS is off is a
      question for P3-3 (the station).
  - **WBGT, IFS vs C3-hop: IFS is better on all days, C3-hop slightly better on heatwave days.**
    - All days 1.97 vs 2.22 (delta -0.25, CI -0.48 to -0.02, fragile); IFS is better at every lead.
    - Heatwave days 3.28 vs 3.08 in C3-hop's favour (no CI); both are about 2.7-2.8 °C too cold there.
  - **WBGT, GFS: worse than C3-hop.** All days 2.65 vs 2.21 (delta +0.44, CI +0.21 to +0.66). Heatwave days 4.97 vs
    3.08, 4.6 °C too cold, because GFS's air is too dry.
  - **Warnings (ETS):**
    - Tmax: IFS 0.33 vs C3-hop 0.24-0.27 (its pre-chosen rules); AIFS 0.46 vs 0.39-0.45; GFS 0.18 (it warns on far
      too many days).
    - WBGT: every model poor and about equal (0.02-0.09).
    - The GFS and LSTM WBGT R-point counts are the same (28 warnings, 7 hits) by coincidence: they warn on different
      days (1 in common).
  - **Reading.** The forecasts Delhi can get from ECMWF beat C3-hop on all days, mostly beyond day 1, for both Tmax
    and WBGT. C3-hop beats GFS on both. C3-hop is slightly better than IFS only for WBGT on heatwave days, and that
    is not testable with 3 clusters.
  - **Seen while scoring:** the archive changed AIFS's model id.
    - `ecmwf_aifs025` (the experimental AIFS) ends 2025-03-10.
    - The operational AIFS Single is `ecmwf_aifs025_single` (probe 2026-11-08). It has radiation, so WBGT would be
      possible.
    - It is not included: adding it would change this pre-registration after these results were seen. Decision
      pending (user).
  - Not run yet: the IFS timing check (single runs; download pending).
- **DEVIATION 2026-11-10 (user approved). Written after the 2026-11-08 results were seen, and before any value of
  the new model was downloaded for scoring:**
  - The operational AIFS Single (Open-Meteo model id `ecmwf_aifs025_single`) is added as a fourth model, named
    "AIFS Single". It is reported as its own model and is never joined to the experimental AIFS series.
  - Why this is a deviation: the choice to add it was made after seeing that the experimental AIFS series stops
    after one heat season, and after seeing how IFS, AIFS and GFS scored. Every table marks it as "added after the
    first results".
  - Seen of the new model so far, probes only (2026-11-08 and 2026-11-10, cell 1; no value compared with truth):
    - all six WBGT inputs are archived;
    - lead 1 starts on 2025-02-18 and lead 5 on 2025-02-22; the last month (to 2026-09-06) has no gaps;
    - the grid point and elevation are ERA5's (28.25° N, 77.0° E, 274 m).
  - Period: from its first complete archived day per lead (2025-02-19 for lead 1, 2025-02-23 for lead 5) to windows
    ending 2026-09-06. That is two heat seasons (2025 and 2026).
  - Targets: Tmax and WBGT, with the definitions, strata, scores, warnings and test of the other models. Its native
    output is 6-hourly, so its hourly values are Open-Meteo's interpolation, as for the experimental AIFS.
  - Expected (written now):
    - Tmax: too cold on all days, as the experimental AIFS was (the 6-hourly output misses the afternoon peak), and
      so worse than C3-hop on all days;
    - WBGT: open. The same 6-hourly limit applies to every input.
  - It counts as one of "the P3-1 models" wherever P3-2 and P3-3 use that phrase.
- **RESULT 2026-11-11** (`followup/results/global_models.md`, re-run with both downloads complete; daily extract
  SHA-256 58ceaaaca724b4dbffbd79ea21717d9aa322efddc4ec5cec8b9d6e64660c4ccd). The 2026-11-08 numbers are unchanged.
  - **IFS timing check** (IFS HRES 9 km, the run started 12 UTC on q - 1; 901 of 902 runs are in the archive; 8
    clusters on all days):
    - Tmax: IFS is better than C3-hop at every lead, day 1 included: 1.37 vs 1.54 at day 1, 1.98 vs 2.53 at day 5.
    - Tmax, all days 1.69 vs 2.14 (delta -0.45, CI -0.92 to +0.01, fragile); heatwave days 1.63 vs 2.38 (3
      clusters, no CI).
    - WBGT, all days 1.85 vs 2.22 (delta -0.37, CI -0.53 to -0.20, fragile); heat season 1.87 vs 2.19 (delta -0.32,
      CI -0.54 to -0.11, fragile).
    - WBGT, heatwave days 3.18 vs 3.08: still slightly in C3-hop's favour (4 clusters, no CI).
    - Reading: the "about even at day 1" of the main result came from the older runs behind `_previous_day1`. With
      the latest run that starts before our input cutoff, IFS is ahead at day 1 too.
  - **AIFS Single** (added after the first results; query dates 2025-02-19 to 2026-09-02; 6 clusters on all days, 2
    on heatwave days):
    - Tmax, all days: 1.31 °C too cold, as expected, but still better than C3-hop: 1.89 vs 2.27 (delta -0.39, CI
      -1.04 to +0.27, fragile). **The expectation "worse than C3-hop on all days" was wrong.** About even at day 1
      (1.66 vs 1.62), better from day 2.
    - Tmax, heat season 2.02 vs 2.66 (no CI). Heatwave days 2.64 vs 2.49, C3-hop slightly better (110 rows, no CI);
      AIFS Single is 2.51 °C too cold there.
    - Tmax warnings: R-point almost never fires for AIFS Single (hit rate 1%, ETS 0.01) because of its cold bias.
      C3-hop on the same rows: ETS 0.07-0.10. These two seasons have few events (123 rows).
    - WBGT: the best forecast in this comparison. All days 1.68 vs 2.26 (delta -0.58, CI -0.71 to -0.45, fragile);
      heat season 1.68 vs 2.24 (4 clusters, no CI).
    - WBGT, heatwave days 2.69 vs 3.08 (95 rows, 2 clusters, no CI); 1.88 °C too cold, C3-hop 2.71. Warnings: ETS
      0.17 vs C3-hop's 0.07-0.11.
    - Reading: C3-hop's one edge over IFS (WBGT on heatwave days) does not hold against AIFS Single. With 2
      clusters that is not testable either way.
- **DEVIATION 2026-11-11 (a clarification of where files live; user approved; no number changes):**
  - The committed daily extract now lives in `followup/data/open_meteo_runs_daily.parquet`. It moved from
    `data/external/`; its content is unchanged (SHA-256 58ceaaaca724b4dbffbd79ea21717d9aa322efddc4ec5cec8b9d6e64660c4ccd).
  - Why: `data/external/` holds the inputs of the one-time test. Every tracked file there must be in the test's
    fingerprint list (`data/TEST_MANIFEST.json`, checked by `tests/test_test_manifest.py`), and this file was never
    one of the test's inputs. Committing it there made that check fail on this branch from commit 94bf43a until now.

**P3-2: A corrector on top of a global forecast. PRE-REGISTERED 2026-11-05 (user approved).**
- Question: does a cheap Delhi corrector make global forecasts better, and does C3-hop add information on top?
- Variants, all fixed now:
  - (a) Four-value daily maximum, GraphCast only (it gives 6-hourly values):
    - Tmax = a_m + b_m T00 + c_m T06 + d_m T12 + e_m T18 per calendar month m, where Txx is the 2 m temperature at
      xx UTC of the IST day.
    - Fitted by least squares on ERA5 1980-2015; each ERA5 value is the mean of the two bracketing hours, as in
      `evaluation/graphcast_yardstick.py`.
    - Applied to WeatherBench 2 GraphCast 2020 (ERA5-initialised, the version trained to 2019, run started 12 UTC on
      q - 1, native 0.25° grid, so the 9-cell box mean is exact).
    - Downloaded for query dates 2020-01-14 to 2020-07-27 (about 3,900 chunks of 2 m temperature): the first 60 days
      only feed the running windows of (b)-(d). Scored on the heat-season windows, query dates 2020-03-15 to
      2020-07-27.
    - Compared with the current max-only correction and with C3-hop.
  - (b) Running bias correction:
    - corrected = raw minus the model's mean error (same lead and target) over the previous 60 days with target date
      <= q - 1 (our model's information cutoff);
    - needs at least 20 such days, otherwise the forecast stays raw;
    - applied to every P3-1 model and to C3-hop.
  - (c) Tail-focused bias correction (replaces retraining with a tail-weighted loss):
    - on forecast-hot rows (the model itself forecasts a heatwave day, the "forecast-conditioned" definition of
      `evaluation/compare_v2.py`), subtract the mean error over ALL earlier forecast-hot rows of that model and lead
      with target date <= q - 1;
    - needs at least 10 such rows, otherwise (b) is used;
    - all other rows as (b);
    - conditioning on the forecast, not the outcome, avoids simply rewarding warm bias.
  - (d) Blend:
    - forecast = (1 - w) x the (b)-corrected global forecast + w x the (b)-corrected C3-hop ensemble mean;
    - w per model, lead and target, by least squares over the previous 60 days (target date <= q - 1), clipped to
      [0, 1];
    - at least 20 days, otherwise w = 0.
- The running windows are used for every model, GraphCast 2020 included. The 2018 GraphCast extract (0.7° grid) is
  not used to fit anything, because its grid differs from the 2020 file.
- WBGT: (b)-(d) for IFS and GFS WBGT, with `A2Lr_hw5_C3hop`.
- Scores and test: as P3-1, plus RMSE on forecast-hot rows, and Brier score and warnings for (c).
- Reading rule (fixed now):
  - A variant improves a forecast if its all-day RMSE is lower with the 95% CI below 0, and its extreme-stratum RMSE
    is not higher by more than 0.05 °C.
  - "C3-hop adds information" means (d) beats (b) by the same rule.
  - With fewer than 5 clusters the comparison is descriptive and makes no claim.
- Expected: (a) improves GraphCast 2020 (2018: -7%); (b) helps most for the most biased model; (d) gives a small gain
  (2018: under 1% on top of (a)).
- Code: `followup/corrector.py`; the GraphCast 2020 extract (box means only, cached) goes to
  `data/external/graphcast_2020_delhi_t2m.parquet`.
- **DEVIATION 2026-11-11 (a clarification, written before any P3-2 code or download):** the GraphCast 2020 extract
  goes to `followup/data/graphcast_2020_delhi_t2m.parquet` instead, for the reason given under P3-1 on the same date.
- **DEVIATION 2026-11-21 (user approved; written before any P3-2 code or download):** variant (a), the four-value
  daily maximum for GraphCast 2020, is dropped, and with it the GraphCast 2020 download. Variants (b)-(d) run as
  registered, on the P3-1 forecasts already downloaded. Why: the journal plan (2026-11-20) needs the blend's answer
  (does C3-hop add anything to a corrected global forecast?), and (b)-(d) give it without new data.

**P3-3: Safdarjung station check. PRE-REGISTERED 2026-11-05 (user approved).**
- Data: GHCN-Daily IN022021900 TMAX (`sources/data/safdarjung_daily_tidy.csv`; to 2025-08-23; gaps; quality-flagged
  values dropped). SHA-256 of the file: a8fe98424ff2829c12601dbbe67de02fe07639cdf68a4ff9fdb4d78c24bbbe1b.
- Station heatwave days, two rules, both 15 Mar - 31 Jul:
  - the project's label rule: Tmax >= max(40, normal + 3) or >= 45 °C;
  - IMD's station criterion: departure >= 4.5 °C with Tmax >= 40, or Tmax >= 45 °C.
  - Normal = `pipeline.climatology.doy_climatology` on the station's own Tmax, training years 1980-2018.
- Analyses:
  - (i) ERA5 9-cell Tmax vs the station, 1980-2018 and 2019-2025: bias and RMSE on all days and on station heatwave
    days, and how often the ERA5 and station labels agree.
  - (ii) The frozen test forecasts (C3-hop, the control, ridge_hw5, Tmax family) vs the station, 2019-2025: RMSE and
    bias by lead, R-point and R-prob warnings against station heatwave days.
  - (iii) GraphCast 2018 and 2020, the P3-1 models and the P3-2 variants vs the station.
  - (iv) A station mapping fitted on 1980-2018 only (per month: station = a + b x ERA5 9-cell Tmax), applied to every
    forecast and reported as "station-mapped".
- Reading:
  - does the model ranking change when the truth is the station?
  - how much of the heatwave-day cold bias is ERA5's own (ERA5 - station on station heatwave days, set beside
    forecast - station)?
- WBGT: GHCN-Daily has no humidity, so WBGT is not part of this check.
- Code: `followup/station_check.py`.
- **DEVIATION 2026-11-10 (a clarification, written before any ERA5 or forecast value was compared with the
  station):**
  - Seen of the station so far: its coverage per year (about 350 usable days a year to 2015, 145-290 from 2016), and
    the Phase 6 summaries of known heatwave windows (`sources/data/phase6_label_windows_ghcn.json`). Nothing else.
  - Station days: a day counts if it has a TMAX value and no quality flag (4 values dropped). No gap is filled. A day
    without a station value is left out of every station comparison.
  - Station normal: `doy_climatology` on the usable station days of 1980-2018.
  - Station heatwave day: any single day that meets the rule. No run-length condition, because the gaps make runs
    unreliable. Thresholds: label rule min(45, max(40, normal + 3)); IMD rule min(45, max(40, normal + 4.5)).
  - Rows: forecasts whose target day has a station value. Every row is scored against both truths (ERA5 and the
    station), so the two rankings are on the same rows. Test forecasts: 2019-01-01 to 2025-08-23.
  - Strata: all days; the heat season; station heatwave days (label rule, and IMD rule); and the test's "extreme"
    stratum (ERA5 label) for reference.
  - (i) also gets the mirror view: station minus ERA5 on ERA5 heatwave days. Stated now: picking days by one
    record's heat makes the other record look colder on those days (selection, as in the forecaster's dilemma), so
    the two views are read together.
  - Warnings, in-season rows, events = station heatwave days (each rule):
    - raw forecasts: R-point with the ERA5 label threshold, as issued; R-prob with the test report's probabilities
      and the pre-chosen p* (C3-hop 0.37, the LSTM 0.24). ridge_hw5 and the global models have no probability, so
      R-point only.
    - station-mapped forecasts: R-point with the station's own threshold for that rule. No R-prob.
  - (iv) The mapping is fitted on the usable station days of 1980-2018 and applied by the target day's month. It is
    also applied to ERA5 itself. That gives the error a perfect forecast of ERA5 would still have against the
    station, raw and mapped.
  - Tests (paired cluster jackknife, RMSE against the station, same rows; raw and station-mapped): C3-hop vs the
    control and vs ridge_hw5; each global model vs C3-hop. Strata: all days, heat season, station heatwave days
    (label rule).
  - Reading rule for the ranking: both orderings are shown side by side. A change of order between two models is
    called clear only if their paired CIs exclude 0 with opposite signs under the two truths.
  - ERA5's share of the cold bias: on station heatwave days, per lead, forecast - station = (forecast - ERA5) +
    (ERA5 - station). The three means are reported, and the share is (ERA5 - station) / (forecast - station).
  - (iii) in two steps:
    - now: GraphCast 2018 (the committed extract with the yardstick's max-only correction, fold f4's 2018 windows,
      our development forecasts on the same windows) and the P3-1 models from the committed daily extract (AIFS
      Single included; the IFS timing check once its download is complete);
    - after P3-2 has run: GraphCast 2020 and the P3-2 variants.
  - Expected (written now): ERA5's 9-cell mean is colder than the station on station heatwave days (an area mean
    damps a station's peaks), and this explains part, not all, of the forecasts' cold bias. The ranking of our
    models does not change.
- **RESULT 2026-11-11** (`followup/results/station_check.md`; the P3-2 parts of (iii) are not run yet).
  - **The registered comparison cannot answer its questions for 2019-2025: the station file's recent values are not
    dated to the day they occurred.** This was found after the registered numbers were seen, by a check that was not
    registered (exploratory). The check uses the station and ERA5 only, no forecast:
    - 1980-2008: a value dated D matches ERA5 of day D. The SD of ERA5 - station is 1.09 °C on the same day and 1.95
      against the day before. Missing days look random. ERA5 is 0.42 °C below the station.
    - 2011-2015: the value matches the larger of the maxima of D - 1 and D best (SD 1.21, same day 1.43). 22% of
      consecutive values are identical (5% in 1980-2008): a hot day's maximum is repeated on the next, cooler day.
      2009-2010 lie in between (14% identical).
    - 2016-2025: values exist on 56-64% of days. 63-72% of them fall on a day cooler than the day before; 46-48% of
      all days are such days. The day before fits far better than the same day (2019-2025: SD 1.02 vs 1.74).
    - All values carry GHCN-Daily's source flag S (Global Summary of the Day). Its readme (v3.35, read 2026-11-11)
      says S values are "derived from hourly synoptic reports" and "may differ significantly from 'true' daily data
      ... (i.e., use with caution)". It says nothing on how an S-source TMAX is dated.
    - Consequence: in 2019-2025 most station values dated D are the maximum of D - 1, so a forecast for D is scored
      against the wrong day. The numbers of (ii), (iii) and the 2019-2025 half of (i) are in the report exactly as
      registered, and no conclusion is drawn from them.
  - **The registered numbers, for the record (not interpretable for 2019-2025):**
    - (i) ERA5 - station: -0.63 °C in 1980-2018 (RMSE 1.41) and -1.83 °C in 2019-2025 (RMSE 2.53). Matched to the
      larger of the two days, the 2019-2025 gap is -0.91 °C.
    - (i) ERA5 labels 75% of the station's heatwave days (label rule) in 1980-2018 and 53% in 2019-2025.
    - (ii) Against the station, ridge_hw5 scores better than C3-hop on all days: 2.13 vs 2.28 (delta +0.15, CI +0.07
      to +0.22). Against ERA5 on the same rows the order is the reverse (delta -0.14, CI -0.18 to -0.09). By the
      registered rule this is a clear change of order. It cannot be read as one: a truth that is mostly the day
      before favours a forecast that leans on the last observed days.
    - (ii) C3-hop vs the LSTM keeps its order against the station (delta -0.06, CI -0.18 to +0.06).
    - (ii) Cold-bias split, C3-hop on station heatwave days: -2.21 = -0.54 (forecast - ERA5) + -1.67 (ERA5 -
      station), ERA5's share 76%. The ERA5 - station part carries the dating effect.
    - (iii) On all days, IFS (both versions), AIFS Single and GraphCast 2018 are better than C3-hop against ERA5 and
      worse against the station; the experimental AIFS is worse against both. GFS is 0.9 °C warmer than the station
      (2.9 °C warmer than ERA5 in P3-1). The same caveat applies to all of it.
  - **What can be said:**
    - Where the dates are right (1980-2008), ERA5's 9-cell mean runs 0.42 °C below the station on average, and the
      two differ by about 1.1 °C from day to day.
    - The expected damping of peaks is not visible in the registered 1980-2018 numbers: ERA5 - station is -0.61 °C
      on station heatwave days and -0.63 °C on all days. (The affected years 2011-2018 are inside that period.)
    - The mapping (iv) does not carry over to recent years with this file: it leaves -1.20 °C in 2019-2025. The
      advisory's reserve option C (station mapping) would need a station record with correct dates.
  - **Expectations:** not testable. The registered numbers go against "the ranking of our models does not change",
    but for a reason that lies in the data, not in the models.
  - **Not decided (user):** how to get a usable station truth for 2019-2025. Nothing further is run on the station
    until this is decided, and P3-2's station part waits on it. Options:
    - (a) rebuild the station's daily maxima from NOAA's hourly archive (Integrated Surface Database), dated by IST
      day: a new data source, with its own pre-registration;
    - (b) a re-dated comparison with this file: score each forecast's two-day maximum against the station value,
      with ERA5's two-day maximum as the reference. Post hoc, and only on the days the station reports;
    - (c) leave P3-3 as a documented negative result.
- **DECISION 2026-11-12 (user): option (a).** The station's daily maxima are rebuilt from NOAA's Integrated Surface
  Database, under their own pre-registration, written and pushed before any ISD temperature is downloaded. Options
  (b) and (c) are not taken.
  - 2026-11-21 (user approved): Palam, the airport station, is rebuilt the same way, as a second station.
- **DECISION 2026-11-24 (user approved; before the rebuild's pre-registration and before any ISD temperature is
  downloaded):** the rebuild's scope is extended. For both stations it also produces hourly temperature and dew point,
  and a daily minimum. Why:
  - the dew point checks ERA5's humidity around 2000-01 (the caveat in `evaluation_v2/premise_report.md`);
  - the station maxima let Delhi's alert schemes run on the readings they are defined on, for a planned heat-stress
    analysis (humid-heat days the alerts miss);
  - the minimum serves a possible hot-nights target.
  - NOAA's station list (`isd-history.csv`, downloaded 2026-11-24 with the format document; no temperatures)
    confirms 421820-99999 SAFDARJUNG and 421810-99999 INDIRA GANDHI INTL (Palam). Both records end 2025-08-24. The
    rebuild's own pre-registration fixes every rule.
- **DECISION 2026-11-24, later (user approved):** the planned heat-stress analyses are retired. They are humid-heat
  days the alerts miss, unsafe working hours against the fixed work windows, and forecasting unsafe hours. All three
  are covered by the teammate's pre-registered study on the branch `experiments-v2` (`docs/PROTOCOL_V2.md`: H-A,
  H-B, H-E and H-N; tag `test-lock-v2`; results in `results_v2/`). None of them was started here.
  - The rebuild now also serves that study. Its station comparisons use GHCN files: Safdarjung's is misdated after
    2010 (the RESULT above), and Ahmedabad's and Chennai's come from the same source. So the rebuild adds Ahmedabad
    and Chennai airport, defined in its own pre-registration.

**P3-4: Conformal forecast ranges. PRE-REGISTERED 2026-11-05 (user approved).**
- Base: the test report's ranges: the recalibrated ensemble mean +/- a Gaussian width from the stop-block residual SD
  per lead, at 80% and 50% (`evaluation/test_evaluation.py::probabilistic`).
- Calibration data: the 2017-2018 stop-block forecasts (`predictions_test/<run>/test_stop.parquet`) after the same
  recalibration. Disclosed: the stop block also chose the early-stopping epoch and the recalibration, so its
  residuals are slightly optimistic.
- Methods:
  - (A) Split conformal per lead: half-width = the ceil((n + 1)(1 - alpha))-th smallest absolute stop-block residual,
    for alpha = 0.2 and 0.5.
  - (B) Mondrian conformal: (A) separately within hot-probability groups p < 0.1, 0.1 <= p < p*, p >= p* (p* = 0.37
    Tmax, 0.13 WBGT, from post-test 3). Out-of-season rows join the first group. If a group has fewer than 30
    residuals at a lead, its leads are pooled.
  - (C) Adaptive conformal (Gibbs & Candès 2021): start from (A) and run in date order through 2019-2026, per lead.
    Each time a truth becomes available (target date <= q - 1): alpha_next = alpha_now + gamma x (alpha - miss),
    where miss = 1 if that truth fell outside its range and gamma = 0.005.
- Models: C3-hop and the control, both targets.
- Scores: coverage at 80% and 50% (all days, extreme stratum, heat season, each hot-probability group); mean width;
  interval score (Gneiting & Raftery 2007). The point forecast and CRPS are unchanged.
- Stated now: no method can guarantee coverage on days that turn out to be heatwaves, because the forecast does not
  know that in advance (the forecaster's dilemma, Lerch et al. 2017).
- Success:
  - all-day 80% coverage between 0.78 and 0.82, with a lower interval score than the base;
  - (B) and (C) are judged by their coverage in the p >= p* group.
- Code: `followup/conformal.py`.
- **DEVIATION 2026-11-21 (user approved; written before any P3-4 code):** P3-4 is no longer parked, and it gains one
  descriptive output: reliability tables for the test's recalibrated hot-day probability (observed hot-day
  frequency against the forecast probability in 10 equal-width bins, with counts), for the same models and targets.
  No pass rule. Why: the journal plan (2026-11-20) lists calibrated probabilities and range coverage among its fixes.
- **DEVIATION 2026-11-24 (user approved; written before any P3-4 code):** P3-4 gains one more descriptive output: the
  cost-loss value of acting on the test's recalibrated hot-day probability. This is the relative economic value
  (Richardson 2000), with the user acting when the probability is at least the cost/loss ratio, over ratios 0.01 to
  0.99 in steps of 0.01. It covers the same models and targets, and shows C3-hop's pre-chosen yes/no warning rules
  as single points. No pass rule. Why: the heat-stress proposal (2026-11-22) asks for results stated in terms an
  official can act on. The comparison with the official alert schemes waits for the planned alert analysis.
- **DROPPED 2026-11-24 (user approved), before any P3-4 code:** P3-4 is not run, nor are its two later additions
  (reliability tables, cost-loss value).
  - The same question was pre-registered and already scored on the branch `experiments-v2`, as H-C in
    `docs/PROTOCOL_V2.md`: conformal ranges conditioned on heat level, judged on forecast-hot days, with the same
    forecaster's-dilemma caveat. Its result is in `results_v2/test_v2.json`.
  - That study's H-D includes the cost-loss value (relative economic value, Richardson 2000).
  - Running P3-4 as well would test one question twice, with two answers to reconcile.
  - If reliability tables are wanted for the retrieval paper later, they come back as a new entry.

**P3-5: Benchmark, leaderboard and error budget. PRE-REGISTERED 2026-11-05 (user approved).**
- No new hypothesis. A package so others can score their own models the same way.
- Splits: training 1980-2016, early stopping 2017-2018, development folds 2007-2018, test 2019-01-01 to 2026-09-06,
  prospective 2027 (P3-6).
- Truths: ERA5 9-cell Tmax and WBGT; Safdarjung Tmax.
- Scoring script: `followup/benchmark/score.py`. It takes a CSV of forecasts (query date, lead, target, value,
  optional probability) and returns every score used above.
- Leaderboard: every model already scored, each with its period and caveats: persistence, ridge, ridge_hw5, the LSTM,
  C2, U1, C3-pool, C3-hop, Chronos-2, TimesFM, GraphCast 2018 and 2020, IFS, AIFS, GFS and the P3-2 variants. All
  numbers come from committed results; nothing is left out.
- Error budget (development year 2018, exploratory):
  - the 6-hourly floor (0.67 max-only, 0.54 four-value);
  - GraphCast's gap to C3-hop by lead;
  - the error correlation and blend weights.
- Before release: check the licence terms of each data source.
- **DROPPED 2026-11-21 (user approved), before any P3-5 code:** the leaderboard and the error budget are not built.
  The journal plan (2026-11-20) asks only for a public code release with a DOI; a scoring script, if one is written,
  goes into that release. The licence check stays, before the release.

**P3-6: Prospective 2027 season. PRE-REGISTERED 2026-11-05 (user approved).**
- Freeze by 2027-02-28: a commit tagged `prospective-2027` holding the test-fold models (checkpoint SHA-256s), the
  P3-2 rules and the scoring script. Nothing changes after.
- Period: query dates whose target days fall in 2027-03-15 to 2027-07-31 (Tmax) or 2027-09-30 (WBGT).
- Inputs: ERA5 via Open-Meteo as before, downloaded after the season, each forecast made as if issued on its query
  date (inputs up to q - 1). Global models: the P3-1 archive. Station: GHCN if available.
- Scores: as the test (all days, extreme, by lead, warnings, ranges), plus the P3-1 and P3-2 comparisons.
  Comparisons repeated:
  - H1-H4, descriptively: one season gives 2-3 clusters, below the 5 needed for a CI;
  - C3-hop vs IFS, AIFS and GFS;
  - the corrector variants.
- Scored once, after 2027-10-15 (when ERA5T for September exists), and reported whatever the result.
- **DECISION 2026-11-21 (user approved): kept only if the upper-air model (RS-4 in `context/decisions_retrieval.md`,
  branch `retrieval-improvements`) becomes the paper's main model.** That is decided before the 2027-02-28 freeze.
  - If kept, a dated change slims it before the freeze: the frozen models plus the upper-air model, the P3-1 global
    models, the P3-2 variants only if P3-2 has run, and the rebuilt ISD station record in place of GHCN. The upper-air
    model also needs 500 hPa data for 2027.
  - If not kept, a DROPPED line is added here.

**P3-7: Station records rebuilt from NOAA ISD (Safdarjung, Palam), and a dating check of the experiments-v2 station
files (Ahmedabad, Chennai). PRE-REGISTERED 2026-11-25 (user approved; drafted 2026-11-24, revised 2026-11-25):
no ISD file downloaded.**
- **Why.**
  - P3-3 found the GHCN-Daily file for Safdarjung misdated after 2010 (RESULT 2026-11-11).
  - On 2026-11-12 the user chose option (a): rebuild the station record from NOAA's Integrated Surface Database (ISD).
  - Palam was added on 2026-11-21, and hourly temperature, dew point and daily minima on 2026-11-24.
  - The teammate's study on `experiments-v2` uses GHCN files from the same source for Ahmedabad and Chennai.
  - On 2026-11-25 the user chose a staged plan: check those two files first, and rebuild a city from ISD only if its
    file shows the fault (part B).
- **Seen so far.**
  - The ISD station list and format document (downloaded 2026-11-24), and response headers for a few yearly files.
    No ISD report, temperature or dew point has been opened.
  - The Safdarjung GHCN file was seen in P3-3.
  - The `experiments-v2` station results (`results_v2/station_checks.json`, read 2026-11-24) use GHCN files for the
    three cities. ERA5 minus station is -0.71 °C for Delhi and -1.28 °C for Ahmedabad.
  - The two GHCN files that study used were received from the teammate on 2026-11-25. Only their headers were
    opened before this revision was written.
- **Data.**
  - ISD's yearly raw files, `https://www.ncei.noaa.gov/pub/data/noaa/<YEAR>/<USAF>-<WBAN>-<YEAR>.gz`, 1980-2025 (both
    records end 2025-08-24). That is 92 files, about 20-30 MB:
    - 421820-99999, Safdarjung;
    - 421810-99999, Indira Gandhi International (Palam).
  - Saved to `data/raw/noaa_isd/<USAF>-<WBAN>/` (gitignored). A committed manifest gives each file's URL, size and
    SHA-256.
  - The two GHCN-Daily files that `experiments-v2` used, as received from the teammate, are kept in
    `data/raw/ghcn_experiments_v2/` (gitignored). The manifest records their SHA-256:
    - IN005010600 (Ahmedabad): 0f09273e57ed7a74b9315309511eb6412605508e0e936115a60894498fbccdf8;
    - IN020040900 (Chennai, Minambakkam): e2f6e651c8aca0332b99a6097a5b75e4a388b694b8cda994c6c133f878feb381.
- **Reading the reports** (positions from the ISD format document):
  - Fields read:
    - the date and UTC time (16-27) and the report type (42-46);
    - the air temperature (88-92, tenths of °C, +9999 missing) and its quality code (93);
    - the dew point (94-98) and its quality code (99);
    - the extreme-temperature groups KA1-KA4: period in tenths of hours; code M (maximum) or N (minimum);
      temperature in tenths of °C; quality code.
  - A value is used if its quality code is 0, 1, 4, 5, 9, A, C, I, M, P, R or U. Codes 2, 3, 6 and 7 (suspect or
    erroneous) are dropped. Estimated extremes (codes O and P) are not used.
  - Several reports at one UTC time: FM-12 (SYNOP) is taken first, then FM-15 (METAR), then FM-16 (SPECI), then any
    other type. Within a type, the first in file order is taken.
- **Days and hours.**
  - A day is the IST calendar day (UTC + 5:30), as for the ERA5 target.
  - The hourly series keeps only reports at minute 00 UTC, the same instants as ERA5's hours. In IST they fall at half
    past the hour.
- **Daily values, two kinds of each:**
  - Reported values:
    - Tmax: the code M group of that date's 12 UTC report (17:30 IST), whatever its period. This is the
      maximum-thermometer reading behind IMD's and GHCN's values.
    - Tmin: the code N group of that date's 03 UTC report (08:30 IST), IMD's reading time for the minimum.
  - Hourly values:
    - Tmax: the highest of the 24 hourly readings of the IST day. It needs at least 18 of them, including all six from
      06 to 11 UTC (11:30-16:30 IST). This is like-for-like with ERA5's daily maximum.
    - Tmin: the lowest reading from 21 UTC the day before to 03 UTC (02:30-08:30 IST). It needs at least 6 of those 7.
  - Dew point: the hourly series, and the IST-day mean (at least 18 hours).
  - No gap is filled. A day without a value is left out.
- **Step 1, before any value is read: a census.**
  - For each station, it counts the reports per year, report type, UTC hour and minute, and the KA groups per UTC
    hour, code and period.
  - If a station has no 12 UTC maximum groups or no 03 UTC minimum groups, its reported values are not available.
    A DEVIATION is written before going on.
- **Step 2, validation.**
  - (a) Each station against itself.
    - Its reported Tmax must match its own hourly Tmax best at lag 0: the mean absolute difference is smaller than with
      the dates shifted one day either way.
    - The reported Tmax may fall more than 0.5 °C below the hourly Tmax on at most 5% of days. A maximum thermometer
      cannot read lower than the hourly readings; 0.5 °C allows for whole-degree METAR values.
    - A station that fails is reported, and left out of the steps below.
  - (b) Safdarjung against GHCN, 1980-2008 (the years P3-3 found correctly dated), on the days both records have.
    The reported Tmax passes if all three hold:
    - its mean absolute difference from GHCN TMAX is smaller at lag 0 than with the dates shifted one day either way;
    - the mean difference is within ±0.3 °C;
    - at least 80% of days agree within 0.5 °C.
    - The reported Tmin is compared with GHCN TMIN the same way, descriptive only.
    - If Safdarjung fails, the rebuild stops: the census and the comparisons are reported, and nothing below runs
      without a new entry.
- **Step 3, if Safdarjung passes: P3-3 rerun (Delhi).**
  - P3-3's analyses (i), (ii) and (iv) are rerun exactly as registered, with the 2026-11-10 clarification. So is the
    part of (iii) that is already run.
  - Safdarjung's reported Tmax (1980 to 2025-08-24) replaces the GHCN file.
  - The 2019-2025 forecast comparisons stay post-test and descriptive.
  - The hourly Tmax is reported beside it, as the like-for-like comparison with ERA5.
  - Palam gets the same tables, as a second station, descriptive.
- **Part B: the `experiments-v2` station files (Ahmedabad, Chennai; exploratory).**
  - **The check.** P3-3's dating check, on each file's usable TMAX values (quality-flagged values dropped), per block
    (1980-2008, 2009-2010, 2011-2015, 2016-2025) and per year.
    - It reports the coverage, and the share of consecutive reporting days with exactly the same value.
    - If the teammate sends her ERA5 city files (`data/raw/cities/_wbgt/<city>_hourly_dev.parquet`), it also reports
      how day D's station value matches ERA5 of D, of D - 1, and the larger of the two.
  - **Flagging.** A block after 2008 is flagged if any of these holds:
    - its share of identical consecutive values is at least twice the 1980-2008 share;
    - its coverage is below 75%;
    - with ERA5: the station matches ERA5 of D - 1, or the larger of D - 1 and D, more closely than ERA5 of D (a
      smaller SD of ERA5 minus station).
  - **What follows.**
    - A city's safe years end before its first flagged block. They go to the teammate, who reruns her station
      numbers on them under her own protocol.
    - If a city has a flagged block, it is rebuilt from ISD like Delhi (its 46 yearly files), after a dated change to
      this entry. If not, its file is used as it is.
- **Step 4: Delhi's ERA5 humidity, 1980-2018 (descriptive).**
  - For each heat season (15 Mar - 31 Jul): the mean over days of ERA5 minus the station's daily-mean dew point, for
    Safdarjung and Palam. ERA5 is taken as the 9-cell mean and as the nearest cell. ERA5's dew point is computed from
    its hourly temperature and relative humidity (Magnus formula, Alduchov and Eskridge 1996 constants).
  - Reported:
    - the difference per year;
    - its change from 1990-1999 to 2002-2011, with a 95% CI by bootstrap over years;
    - each record's 1980-2018 trend.
  - No pass rule and no prediction.
- **Expected (written now):**
  - Every station passes its self-check, and Safdarjung passes the GHCN check.
  - After 2010 Delhi's ERA5 - station Tmax gap returns to about its 1980-2008 size (-0.42 °C), not the misdated
    file's -1.83 °C.
  - ERA5 is colder than the station on station heatwave days, and the ranking of our models does not change (as P3-3
    expected).
  - Ahmedabad's and Chennai's GHCN files show the same fault after 2010 as Safdarjung's.
  - No expectation for the humidity check.
- **Code and outputs.**
  - `followup/isd_rebuild.py` does the downloads with their manifest, the parsing, the census and the daily series.
  - Committed: `followup/data/isd_manifest.json` (it also lists the two received GHCN files) and
    `followup/data/isd_daily_{safdarjung,palam}.csv`. The hourly series are rebuilt from the raw files.
  - `followup/station_check.py` gains the ISD source.
  - Reports: `followup/results/station_rebuild.md` (census, validation, part B, humidity) and
    `followup/results/station_check_isd.md` (the P3-3 rerun).
  - Tests: `tests/test_followup_isd.py`, on synthetic report lines (positions, quality codes, KA groups, the IST day,
    coverage rules, the self-check).
- **RESULT 2026-11-25: part B, the station-only half (exploratory).**
  - Timing, disclosed: it ran on 2026-11-25 after part B's rule was written into the draft, and before this entry was
    committed. The rule's text has not changed since. The ERA5 half waits for the teammate's ERA5 city files.
  - Usable TMAX per block (coverage, and the share of identical consecutive values):
    - Ahmedabad (IN005010600):
      - 1980-2008: 85%, 6.2%;
      - 2009-2010: 82%, 4.1%;
      - 2011-2015: 25%, 7.3%, flagged for coverage;
      - 2016-2025: 2%, 6.7%, flagged for coverage.
      - The file is nearly empty after 2010.
    - Chennai (IN020040900):
      - 1980-2008: 90%, 6.9%;
      - 2009-2010: 84%, 7.1%;
      - 2011-2015: 91%, 31.9%, flagged for repeats (Safdarjung's two-day signature);
      - 2016-2025: 60%, 11.0%, flagged for coverage.
  - Safe years for both: 1980-2010. Sent to the teammate.
  - The expectation "the same fault after 2010 as Safdarjung's" holds for Chennai. Ahmedabad's fault is missing data,
    not repeats.
- **DEVIATION 2026-11-25 (by part B's rule; user approved; before any ISD file was downloaded):** Ahmedabad
  (426470-99999, Sardar Vallabhbhai Patel International) and Chennai International (432790-99999) are rebuilt like
  Delhi.
  - Steps 1 and 2(a) apply to them. Steps 2(b), 3 and 4 stay Delhi-only.
  - This adds 92 yearly files, so 184 in all, about 40-60 MB.
  - Outputs: `followup/data/isd_daily_{ahmedabad,chennai}.csv`, handed to the `experiments-v2` study.
  - Part B's ERA5 half still runs if the teammate's files arrive.
- **Download, 2026-11-25** (`followup/isd_rebuild.py --download`; manifest `followup/data/isd_manifest.json`):
  - 173 of the 184 station-years exist, 39.9 MB in all.
  - Palam's 1980-1985 and 1991-1995 are not in the archive (404, rechecked).
  - Safdarjung, Ahmedabad and Chennai are complete.
  - No design change: a missing year is simply absent from Palam's descriptive tables. Safdarjung's GHCN check is
    unaffected.
  - The two received GHCN files match their recorded SHA-256.
- **RESULT 2026-11-25, steps 1 and 2.** Sources: `followup/results/station_rebuild.{md,json}`, the census in
  `followup/results/station_rebuild_census.json`, and the daily files `followup/data/isd_daily_<station>.csv`.
  - **Step 1, the census** (before any value was read):
    - All four stations have 12 UTC maximum groups: in 79% of the selected 12 UTC reports (Palam 32%).
    - All four have 03 UTC minimum groups: in 96-99% of the 03 UTC reports (Palam 46%).
    - No report has two of either, and the periods are mostly 12 h (24 h at Palam).
    - The 03 UTC reports also carry a 24-hour maximum (Safdarjung: 4,162 groups).
    - Hourly reports on the median day: none at Safdarjung (3-hourly only); Palam from 2003; Ahmedabad and Chennai
      from 2016.
    - No DEVIATION was needed.
  - **Step 2(b) passes.** Safdarjung's reported Tmax equals GHCN TMAX on 1980-2008:
    - mean absolute difference at lags -1 / 0 / +1: 1.45 / 0.00 / 1.44 °C;
    - 99.9% of 9,971 days agree within 0.5 °C.
    - Tmin is equal as well (descriptive).
  - **Step 2(a).**
    - Safdarjung cannot be checked: no day has both a reported and an hourly value.
    - Palam, Ahmedabad and Chennai fail.
      - Lag 0 is clearly best everywhere: 0.70, 0.64 and 0.57 °C, against 1.25-1.70 °C one day off.
      - But the reported value is more than 0.5 °C below the hourly one on 7.2%, 38.1% and 15.8% of days. The limit
        is 5%.
      - Almost all the days both values exist are from 2016 on, and carry 24-hour maximum groups. Ahmedabad's 2022
        alone gives 141 of its 210 days.
    - By the rule, these three stations are left out of the steps below until a decision is recorded here.
  - **Descriptive, after the checks.**
    - GHCN against the rebuilt Safdarjung series:
      - 1980-2008: GHCN equals the rebuilt same-day value on 100% of days;
      - 2011-2015: on 72% of days, and equals the higher of the previous and the same day on 84%.
    - From 2016 the 12 UTC maximum is mostly absent from ISD for Safdarjung: 245 days in 2016-2025.
- **DEVIATION 2026-11-25, after the step 1-2 results (user approved).** It is post hoc: written after the step 2
  results were seen, and before the checks below were run.
  - **(A) Safdarjung, 2016-2025.**
    - The fill: days without a 12 UTC maximum, from 2016-01-01, take the code-M group with a 24-hour period from the
      next day's 03 UTC report (08:30 IST). Those 24 hours run from 08:30 IST of the day to 08:30 IST of the next, so
      they hold that day's afternoon.
    - Validation first. Against the 12 UTC series, on every day both exist (1980-2025), the 2(b) rule must hold:
      - the lag-0 mean absolute difference is the smallest;
      - the mean difference is within ±0.3 °C;
      - at least 80% of days agree within 0.5 °C.
    - If the validation fails, nothing is filled.
  - **(B) Palam, Ahmedabad and Chennai.**
    - Their reported Tmax keeps only 12 UTC groups with a 12-hour period. 24-hour and unknown-period groups are
      dropped, because the failing days of step 2(a) were 24-hour groups.
    - Validation:
      - step 2(a) is rerun on the kept values;
      - Ahmedabad and Chennai also take the 2(b) rule against their GHCN files on 1980-2008. These are the files
        received from the `experiments-v2` study, which part B found correctly dated to 2010.
    - A station that fails is left out, as before.
    - Palam's reported values are almost all 24-hour groups, so under (B) it keeps almost none. Its hourly Tmax (from
      2003) stays, for the descriptive Palam tables of step 3.
  - The registered columns of the daily files are unchanged. The deviation adds three columns:
    `tmax_24h_next_morning`, `tmax` (the series used from here on) and `tmax_source`.
- **RESULT 2026-11-25, the deviation** (`followup/results/station_rebuild.md`, last section).
  - **(A) passes.** On the 940 days both exist, the next morning's 24-hour maximum matches the 12 UTC maximum:
    - mean absolute difference at lags -1 / 0 / +1: 1.52 / 0.02 / 1.51 °C;
    - mean difference -0.01 °C, with 99.5% of days within 0.5 °C.
    - 3,192 days are filled, so Safdarjung now has a Tmax on 3,437 of the 3,524 days of 2016-2025.
  - **(B).**
    - Ahmedabad's and Chennai's 12-hour groups equal their GHCN files on 1980-2008: mean absolute difference 0.00 °C
      at lag 0 against 1.08-1.09 °C a day off, on 8,343 and 8,805 days, with 99.9% and 100% within 0.5 °C. Both pass.
    - The self-check could not run for Chennai or Palam (no day with both values). For Ahmedabad it passes, but on
      only 6 days.
    - No station is left out.
    - Coverage of the series used:
      - Ahmedabad and Chennai: 840 of 1,826 days in 2011-2015 (about half the 12 UTC groups then were 24-hour ones),
        and none from 2016;
      - Palam: 32 days in all, so only its hourly Tmax remains.
