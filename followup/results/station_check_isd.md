# The station check on the rebuilt station records (P3-7 step 3, post-test)

Pre-registered: P3-3 on 2026-11-05 (details fixed 2026-11-10) and P3-7 step 3 on 2026-11-25, in `context/decisions_phase3.md`; the records used follow the deviation of 2026-11-25. Descriptive: the test years were already used, so nothing here is confirmatory. The analyses and their reading rules are P3-3's, unchanged; only the station record differs. ERA5: the 9-cell mean the models were trained and tested on. Bias = forecast - truth; 'ERA5 - station' is ERA5's own offset. Station-mapped = the forecast put through a per-month straight line fitted on 1980-2018 (station = a + b x ERA5). The registered run on the GHCN file is kept as it was (`followup/results/station_check.md`).

## Summary

| | GHCN file (registered run) | Safdarjung, rebuilt | Palam, hourly maximum |
|---|---|---|---|
| Days with a station value, 1980-2018 | 13093 | 13664 | 3213 |
| Days with a station value, 2019-2025 | 1544 | 2361 | 2261 |
| ERA5 - station, all days, 1980-2018 (°C) | -0.63 | -0.50 | -0.41 |
| ERA5 - station, all days, 2019-2025 (°C) | -1.83 | -0.88 | +0.08 |
| ERA5 - station, station heatwave days (label rule), 1980-2018 (°C) | -0.61 | -0.40 | -1.58 |
| ERA5 - station, station heatwave days (label rule), 2019-2025 (°C) | -1.67 | -0.49 | -0.85 |
| C3-hop: RMSE vs ERA5 / vs station, test rows (°C) | 2.273 / 2.280 | 2.164 / 2.446 | 2.138 / 2.368 |
| Delhi-only LSTM: RMSE vs ERA5 / vs station, test rows (°C) | 2.408 / 2.346 | 2.308 / 2.573 | 2.274 / 2.500 |
| ridge_hw5: RMSE vs ERA5 / vs station, test rows (°C) | 2.410 / 2.134 | 2.292 / 2.471 | 2.272 / 2.500 |
| Clear changes of model order | 1 of 27 | 0 of 27 | 0 of 27 |
| ERA5's share of C3-hop's cold bias on station heatwave days (label rule) | 76% | 22% | 34% |

- Each column uses its own rows: the days that record has a value.
- The RMSEs are of the ensemble means, on the test rows whose target day has a station value.
- A clear change of order is P3-3's reading rule: the paired CIs of two models exclude 0 with opposite signs under the two truths. The comparisons are the paired tests of (ii) and (iii), each on three strata.
- ERA5's share = (ERA5 - station) / (forecast - station) on station heatwave days, all leads together.
- The GHCN column is the registered run. Its values from 2011 on are misdated (its report explains how), which affects 2011-2018 in the first period and all of 2019-2025. It is shown only to see what the rebuild changes.
- Clear changes of order, GHCN file (registered run): C3-hop ensemble minus ridge_hw5, All days.

## Safdarjung, rebuilt

The record used is `tmax`: the maximum-thermometer reading of the 12 UTC (17:30 IST) report, which matches the GHCN file on 1980-2008 (P3-7 step 2(b)). From 2016, on days without it, it is the 24-hour maximum of the next morning's 03 UTC (08:30 IST) report, which passed its check against the 12 UTC reading (part (A) of the deviation; `followup/results/station_rebuild.md`). Safdarjung sends 3-hourly reports only, so the hourly maximum that step 3 shows beside the reported one does not exist for it.

### Station data

| Period | Usable days | In the heat season | Heatwave days, label rule | Heatwave days, IMD rule | Last day |
|---|---|---|---|---|---|
| 1980-2018 | 13664 | 5189 | 562 | 221 | 2018-12-31 |
| 2019-2025 | 2361 | 940 | 127 | 46 | 2025-08-22 |

A usable day has a value in the record used. Gaps are not filled; a day without a station value is left out of every comparison. A station heatwave day is a single in-season day at or above min(45, max(40, normal + 3)) (label rule) or min(45, max(40, normal + 4.5)) (IMD rule), with the station's own 1980-2018 normal.

### Exploratory: are the rebuilt dates right? (not registered)

The check that found the GHCN file's fault, run on this record. It uses the station and ERA5 only, no forecast. For the station value dated D: the mean and SD of ERA5 - station when ERA5 is taken on the same day D, on the day before, or as the larger of the two. The smallest SD shows which day the station value belongs to.

| Years | Days with a value | Same day: mean / SD | Day before: mean / SD | Larger of the two: mean / SD | Reporting days cooler than the day before (all days) | Consecutive values that are identical |
|---|---|---|---|---|---|---|
| 1980-2008 | 95% | -0.41 / 1.11 | -0.41 / 1.97 | +0.18 / 1.57 | 46% (46%) | 5% |
| 2009-2010 | 98% | -0.45 / 1.04 | -0.45 / 1.96 | +0.14 / 1.50 | 45% (45%) | 4% |
| 2011-2015 | 97% | -0.87 / 1.07 | -0.88 / 2.03 | -0.26 / 1.55 | 44% (45%) | 3% |
| 2016-2018 | 98% | -0.80 / 0.98 | -0.80 / 1.92 | -0.18 / 1.46 | 47% (48%) | 4% |
| 2019-2025 | 97% | -0.88 / 1.06 | -0.90 / 2.02 | -0.25 / 1.58 | 45% (46%) | 3% |

If the dates are right, the same day has the smallest SD in every block with values. Identical consecutive values stay rare too, unless the readings are in whole degrees (values in whole degrees: 1980-2008 21%; 2009-2010 18%; 2011-2015 12%; 2016-2018 11%; 2019-2025 11%).

### (i) ERA5 against the station

| Period | Days | ERA5 - station | RMSE | After the mapping: bias | RMSE |
|---|---|---|---|---|---|
| 1980-2018, all days | 13664 | -0.50 | 1.212 | +0.00 | 1.040 |
| 1980-2018, heat season | 5189 | -0.38 | 1.230 | +0.01 | 1.094 |
| 1980-2018, station heatwave days, label rule | 562 | -0.40 | 0.812 | -0.30 | 0.748 |
| 1980-2018, station heatwave days, IMD rule | 221 | -0.50 | 0.906 | -0.36 | 0.815 |
| 1980-2018, ERA5 heatwave days (mirror view) | 510 | -0.04 | 0.859 | +0.05 | 0.855 |
| 2019-2025, all days | 2361 | -0.88 | 1.374 | -0.40 | 1.101 |
| 2019-2025, heat season | 940 | -0.78 | 1.308 | -0.40 | 1.065 |
| 2019-2025, station heatwave days, label rule | 127 | -0.49 | 0.775 | -0.44 | 0.725 |
| 2019-2025, station heatwave days, IMD rule | 46 | -0.44 | 0.676 | -0.40 | 0.621 |
| 2019-2025, ERA5 heatwave days (mirror view) | 109 | -0.20 | 0.545 | -0.16 | 0.525 |

The mapping is fitted on 1980-2018, so its 1980-2018 rows are in-sample. Picking days by one record's heat makes the other record look colder on those days (selection), so the station-heatwave rows and the mirror row are read together.

**Do the heatwave labels agree?** In-season days with a station value. Events = station heatwave days; 'warnings' = ERA5 heatwave days.

| Period | Station rule | ERA5 side | Both | Station only | ERA5 only | Station days ERA5 also has | ERA5 days the station lacks | ETS | ERA5 days / station days |
|---|---|---|---|---|---|---|---|---|---|
| 1980-2018 (5189 days) | label rule (normal + 3) | ERA5 label | 434 | 128 | 76 | 77% | 15% | 0.65 | 0.91 |
| 1980-2018 (5189 days) | label rule (normal + 3) | mapped ERA5 vs the station threshold | 439 | 123 | 70 | 78% | 14% | 0.67 | 0.91 |
| 1980-2018 (5189 days) | IMD rule (normal + 4.5) | ERA5 label | 206 | 15 | 304 | 93% | 60% | 0.37 | 2.31 |
| 1980-2018 (5189 days) | IMD rule (normal + 4.5) | mapped ERA5 vs the station threshold | 157 | 64 | 41 | 71% | 21% | 0.59 | 0.90 |
| 2019-2025 (940 days) | label rule (normal + 3) | ERA5 label | 95 | 32 | 14 | 75% | 13% | 0.64 | 0.86 |
| 2019-2025 (940 days) | label rule (normal + 3) | mapped ERA5 vs the station threshold | 93 | 34 | 8 | 73% | 8% | 0.65 | 0.80 |
| 2019-2025 (940 days) | IMD rule (normal + 4.5) | ERA5 label | 41 | 5 | 68 | 89% | 62% | 0.33 | 2.37 |
| 2019-2025 (940 days) | IMD rule (normal + 4.5) | mapped ERA5 vs the station threshold | 34 | 12 | 5 | 74% | 13% | 0.65 | 0.85 |

### (iv) The station mapping

station = a + b x ERA5, least squares per calendar month on the usable station days of 1980-2018.

| Month | a | b | Days |
|---|---|---|---|
| 1 | -0.88 | 1.064 | 1158 |
| 2 | +0.89 | 0.989 | 1058 |
| 3 | +1.17 | 0.975 | 1166 |
| 4 | +2.11 | 0.946 | 1120 |
| 5 | -0.45 | 1.014 | 1148 |
| 6 | +2.12 | 0.952 | 1121 |
| 7 | +4.54 | 0.898 | 1158 |
| 8 | +3.28 | 0.934 | 1157 |
| 9 | +5.00 | 0.875 | 1130 |
| 10 | +2.55 | 0.934 | 1159 |
| 11 | +0.45 | 0.995 | 1132 |
| 12 | -0.96 | 1.058 | 1157 |

### (ii) Our test forecasts against the station

Target days 2019-01-01 to 2025-08-22 with a station value: 11795 rows (query date x lead), 21 year x season clusters. The networks are shown as the 10-seed ensemble mean and as the mean of the 10 seeds' RMSE (the test's primary score).

**All days**

| Forecast | Rows | RMSE vs ERA5 | RMSE vs station | RMSE station-mapped | Bias vs ERA5 | Bias vs station | Bias station-mapped |
|---|---|---|---|---|---|---|---|
| C3-hop (ensemble mean) | 11795 | 2.164 | 2.446 | 2.302 | +0.19 | -0.68 | -0.20 |
| C3-hop (mean of 10 seeds) | 11795 | 2.212 | 2.488 | 2.345 | +0.19 | -0.68 | -0.20 |
| Delhi-only LSTM (ensemble mean) | 11795 | 2.308 | 2.573 | 2.434 | +0.23 | -0.65 | -0.16 |
| Delhi-only LSTM (mean of 10 seeds) | 11795 | 2.348 | 2.608 | 2.470 | +0.23 | -0.65 | -0.16 |
| ridge_hw5 | 11795 | 2.292 | 2.471 | 2.393 | +0.37 | -0.51 | -0.04 |

**Heat season (15 Mar - 31 Jul)**

| Forecast | Rows | RMSE vs ERA5 | RMSE vs station | RMSE station-mapped | Bias vs ERA5 | Bias vs station | Bias station-mapped |
|---|---|---|---|---|---|---|---|
| C3-hop (ensemble mean) | 4700 | 2.455 | 2.611 | 2.504 | +0.25 | -0.53 | -0.15 |
| C3-hop (mean of 10 seeds) | 4700 | 2.501 | 2.654 | 2.545 | +0.25 | -0.53 | -0.15 |
| Delhi-only LSTM (ensemble mean) | 4700 | 2.719 | 2.847 | 2.753 | +0.39 | -0.40 | -0.02 |
| Delhi-only LSTM (mean of 10 seeds) | 4700 | 2.768 | 2.894 | 2.798 | +0.39 | -0.40 | -0.02 |
| ridge_hw5 | 4700 | 2.629 | 2.699 | 2.637 | +0.34 | -0.45 | -0.08 |

**Station heatwave days, label rule**

| Forecast | Rows | RMSE vs ERA5 | RMSE vs station | RMSE station-mapped | Bias vs ERA5 | Bias vs station | Bias station-mapped |
|---|---|---|---|---|---|---|---|
| C3-hop (ensemble mean) | 635 | 2.137 | 2.546 | 2.423 | -1.73 | -2.22 | -2.11 |
| C3-hop (mean of 10 seeds) | 635 | 2.186 | 2.587 | 2.462 | -1.73 | -2.22 | -2.11 |
| Delhi-only LSTM (ensemble mean) | 635 | 2.450 | 2.878 | 2.731 | -1.92 | -2.41 | -2.29 |
| Delhi-only LSTM (mean of 10 seeds) | 635 | 2.504 | 2.922 | 2.773 | -1.92 | -2.41 | -2.29 |
| ridge_hw5 | 635 | 2.399 | 2.775 | 2.641 | -1.90 | -2.39 | -2.28 |

**Station heatwave days, IMD rule**

| Forecast | Rows | RMSE vs ERA5 | RMSE vs station | RMSE station-mapped | Bias vs ERA5 | Bias vs station | Bias station-mapped |
|---|---|---|---|---|---|---|---|
| C3-hop (ensemble mean) | 230 | 2.400 | 2.835 | 2.696 | -2.04 | -2.48 | -2.37 |
| C3-hop (mean of 10 seeds) | 230 | 2.442 | 2.870 | 2.730 | -2.04 | -2.48 | -2.37 |
| Delhi-only LSTM (ensemble mean) | 230 | 2.719 | 3.175 | 3.002 | -2.29 | -2.73 | -2.60 |
| Delhi-only LSTM (mean of 10 seeds) | 230 | 2.757 | 3.208 | 3.033 | -2.29 | -2.73 | -2.60 |
| ridge_hw5 | 230 | 2.801 | 3.207 | 3.045 | -2.37 | -2.81 | -2.67 |

**ERA5 heatwave days (the test's extreme stratum)**

| Forecast | Rows | RMSE vs ERA5 | RMSE vs station | RMSE station-mapped | Bias vs ERA5 | Bias vs station | Bias station-mapped |
|---|---|---|---|---|---|---|---|
| C3-hop (ensemble mean) | 490 | 2.241 | 2.447 | 2.333 | -1.89 | -2.08 | -1.97 |
| C3-hop (mean of 10 seeds) | 490 | 2.287 | 2.490 | 2.373 | -1.89 | -2.08 | -1.97 |
| Delhi-only LSTM (ensemble mean) | 490 | 2.501 | 2.729 | 2.591 | -2.02 | -2.21 | -2.10 |
| Delhi-only LSTM (mean of 10 seeds) | 490 | 2.551 | 2.774 | 2.634 | -2.02 | -2.21 | -2.10 |
| ridge_hw5 | 490 | 2.571 | 2.762 | 2.629 | -2.16 | -2.35 | -2.23 |

**By lead, all days (ensemble means)**

| Forecast | Score | Day 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| C3-hop | RMSE vs station | 1.992 | 2.322 | 2.479 | 2.630 | 2.740 |
| C3-hop | Bias vs station | -0.85 | -0.85 | -0.69 | -0.61 | -0.41 |
| C3-hop | RMSE station-mapped vs station | 1.793 | 2.142 | 2.336 | 2.502 | 2.640 |
| C3-hop | Bias station-mapped vs station | -0.36 | -0.36 | -0.20 | -0.12 | +0.07 |
| Delhi-only LSTM | RMSE vs station | 2.106 | 2.424 | 2.641 | 2.771 | 2.854 |
| Delhi-only LSTM | Bias vs station | -0.81 | -0.68 | -0.64 | -0.58 | -0.51 |
| Delhi-only LSTM | RMSE station-mapped vs station | 1.919 | 2.275 | 2.502 | 2.644 | 2.740 |
| Delhi-only LSTM | Bias station-mapped vs station | -0.33 | -0.20 | -0.16 | -0.09 | -0.02 |
| ridge_hw5 | RMSE vs station | 1.965 | 2.327 | 2.527 | 2.687 | 2.766 |
| ridge_hw5 | Bias vs station | -0.76 | -0.60 | -0.48 | -0.38 | -0.33 |
| ridge_hw5 | RMSE station-mapped vs station | 1.808 | 2.225 | 2.457 | 2.638 | 2.726 |
| ridge_hw5 | Bias station-mapped vs station | -0.29 | -0.13 | -0.00 | +0.09 | +0.14 |

**By lead, station heatwave days, label rule (ensemble means)**

| Forecast | Score | Day 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| C3-hop | RMSE vs station | 1.647 | 2.413 | 2.622 | 2.864 | 2.967 |
| C3-hop | Bias vs station | -1.38 | -2.13 | -2.34 | -2.59 | -2.66 |
| C3-hop | RMSE station-mapped vs station | 1.558 | 2.294 | 2.493 | 2.731 | 2.829 |
| C3-hop | Bias station-mapped vs station | -1.30 | -2.03 | -2.23 | -2.47 | -2.53 |
| Delhi-only LSTM | RMSE vs station | 1.908 | 2.527 | 2.957 | 3.221 | 3.500 |
| Delhi-only LSTM | Bias vs station | -1.54 | -2.06 | -2.51 | -2.81 | -3.12 |
| Delhi-only LSTM | RMSE station-mapped vs station | 1.801 | 2.397 | 2.805 | 3.059 | 3.325 |
| Delhi-only LSTM | Bias station-mapped vs station | -1.46 | -1.96 | -2.39 | -2.68 | -2.97 |
| ridge_hw5 | RMSE vs station | 1.430 | 2.257 | 2.704 | 3.248 | 3.675 |
| ridge_hw5 | Bias vs station | -1.15 | -1.93 | -2.44 | -2.99 | -3.45 |
| ridge_hw5 | RMSE station-mapped vs station | 1.347 | 2.140 | 2.568 | 3.096 | 3.508 |
| ridge_hw5 | Bias station-mapped vs station | -1.08 | -1.83 | -2.33 | -2.85 | -3.29 |

**Paired tests** (delta = RMSE of the first minus RMSE of the second, same rows; negative = the first is better; paired cluster jackknife).

| Comparison | Stratum | Delta vs ERA5 [95% CI] | Delta vs station [95% CI] | Delta station-mapped [95% CI] | Clear change of order |
|---|---|---|---|---|---|
| C3-hop minus Delhi-only LSTM (10 seeds each) | All days | -0.136 [-0.229, -0.043] | -0.121 [-0.215, -0.026] | -0.125 [-0.208, -0.042] | no |
| C3-hop minus Delhi-only LSTM (10 seeds each) | Heat season (15 Mar - 31 Jul) | -0.267 [-0.402, -0.132], fragile (< 10 clusters) | -0.240 [-0.373, -0.106], fragile (< 10 clusters) | -0.253 [-0.372, -0.134], fragile (< 10 clusters) | no |
| C3-hop minus Delhi-only LSTM (10 seeds each) | Station heatwave days, label rule | -0.318 [-0.752, +0.116], fragile (< 10 clusters) | -0.336 [-0.773, +0.102], fragile (< 10 clusters) | -0.311 [-0.725, +0.103], fragile (< 10 clusters) | no |
| C3-hop ensemble minus ridge_hw5 | All days | -0.128 [-0.174, -0.082] | -0.025 [-0.097, +0.047] | -0.092 [-0.146, -0.038] | no |
| C3-hop ensemble minus ridge_hw5 | Heat season (15 Mar - 31 Jul) | -0.174 [-0.229, -0.119], fragile (< 10 clusters) | -0.088 [-0.190, +0.014], fragile (< 10 clusters) | -0.133 [-0.197, -0.070], fragile (< 10 clusters) | no |
| C3-hop ensemble minus ridge_hw5 | Station heatwave days, label rule | -0.262 [-0.523, -0.001], fragile (< 10 clusters) | -0.229 [-0.513, +0.055], fragile (< 10 clusters) | -0.218 [-0.483, +0.047], fragile (< 10 clusters) | no |

**Yes/no warnings** (in-season rows). R-point and R-prob use the forecast as issued, against the ERA5 label threshold. Station-mapped R-point uses the mapped forecast against the station's own threshold of that rule.

| Forecast | Events | Rule | Hits | Misses | False alarms | Hit rate | False-alarm ratio | ETS | Frequency bias |
|---|---|---|---|---|---|---|---|---|---|
| C3-hop (4700 rows) | station, label rule (normal + 3) (635) | R-point | 156 | 479 | 66 | 25% | 30% | 0.19 | 0.35 |
| C3-hop (4700 rows) | station, label rule (normal + 3) (635) | R-prob (p* = 0.37) | 182 | 453 | 79 | 29% | 30% | 0.22 | 0.41 |
| C3-hop (4700 rows) | station, label rule (normal + 3) (635) | station-mapped R-point | 148 | 487 | 64 | 23% | 30% | 0.18 | 0.33 |
| C3-hop (4700 rows) | station, IMD rule (normal + 4.5) (230) | R-point | 109 | 121 | 113 | 47% | 51% | 0.30 | 0.97 |
| C3-hop (4700 rows) | station, IMD rule (normal + 4.5) (230) | R-prob (p* = 0.37) | 115 | 115 | 146 | 50% | 56% | 0.28 | 1.13 |
| C3-hop (4700 rows) | station, IMD rule (normal + 4.5) (230) | station-mapped R-point | 26 | 204 | 4 | 11% | 13% | 0.11 | 0.13 |
| C3-hop (4700 rows) | ERA5, label rule (reference) (545) | R-point | 145 | 400 | 77 | 27% | 35% | 0.20 | 0.41 |
| C3-hop (4700 rows) | ERA5, label rule (reference) (545) | R-prob (p* = 0.37) | 168 | 377 | 93 | 31% | 36% | 0.23 | 0.48 |
| Delhi-only LSTM (4700 rows) | station, label rule (normal + 3) (635) | R-point | 163 | 472 | 153 | 26% | 48% | 0.16 | 0.50 |
| Delhi-only LSTM (4700 rows) | station, label rule (normal + 3) (635) | R-prob (p* = 0.24) | 328 | 307 | 367 | 52% | 53% | 0.26 | 1.09 |
| Delhi-only LSTM (4700 rows) | station, label rule (normal + 3) (635) | station-mapped R-point | 154 | 481 | 150 | 24% | 49% | 0.15 | 0.48 |
| Delhi-only LSTM (4700 rows) | station, IMD rule (normal + 4.5) (230) | R-point | 98 | 132 | 218 | 43% | 69% | 0.19 | 1.37 |
| Delhi-only LSTM (4700 rows) | station, IMD rule (normal + 4.5) (230) | R-prob (p* = 0.24) | 151 | 79 | 544 | 66% | 78% | 0.16 | 3.02 |
| Delhi-only LSTM (4700 rows) | station, IMD rule (normal + 4.5) (230) | station-mapped R-point | 26 | 204 | 11 | 11% | 30% | 0.10 | 0.16 |
| Delhi-only LSTM (4700 rows) | ERA5, label rule (reference) (545) | R-point | 160 | 385 | 156 | 29% | 49% | 0.19 | 0.58 |
| Delhi-only LSTM (4700 rows) | ERA5, label rule (reference) (545) | R-prob (p* = 0.24) | 301 | 244 | 394 | 55% | 57% | 0.26 | 1.28 |
| ridge_hw5 (4700 rows) | station, label rule (normal + 3) (635) | R-point | 136 | 499 | 71 | 21% | 34% | 0.16 | 0.33 |
| ridge_hw5 (4700 rows) | station, label rule (normal + 3) (635) | station-mapped R-point | 118 | 517 | 70 | 19% | 37% | 0.14 | 0.30 |
| ridge_hw5 (4700 rows) | station, IMD rule (normal + 4.5) (230) | R-point | 81 | 149 | 126 | 35% | 61% | 0.20 | 0.90 |
| ridge_hw5 (4700 rows) | station, IMD rule (normal + 4.5) (230) | station-mapped R-point | 28 | 202 | 4 | 12% | 12% | 0.11 | 0.14 |
| ridge_hw5 (4700 rows) | ERA5, label rule (reference) (545) | R-point | 123 | 422 | 84 | 23% | 41% | 0.16 | 0.38 |

**How much of the cold bias on station heatwave days is ERA5's own?** Forecast - station = (forecast - ERA5) + (ERA5 - station), means over the rows; ensemble means.

| Forecast | Station rule | Lead | Rows | Forecast - station | = forecast - ERA5 | + ERA5 - station | ERA5's share |
|---|---|---|---|---|---|---|---|
| C3-hop | label rule (normal + 3) | all | 635 | -2.22 | -1.73 | -0.49 | 22% |
| C3-hop | label rule (normal + 3) | 1 | 127 | -1.38 | -0.89 | -0.49 | 35% |
| C3-hop | label rule (normal + 3) | 2 | 127 | -2.13 | -1.65 | -0.49 | 23% |
| C3-hop | label rule (normal + 3) | 3 | 127 | -2.34 | -1.85 | -0.49 | 21% |
| C3-hop | label rule (normal + 3) | 4 | 127 | -2.59 | -2.10 | -0.49 | 19% |
| C3-hop | label rule (normal + 3) | 5 | 127 | -2.66 | -2.17 | -0.49 | 18% |
| C3-hop | IMD rule (normal + 4.5) | all | 230 | -2.48 | -2.04 | -0.44 | 18% |
| C3-hop | IMD rule (normal + 4.5) | 1 | 46 | -1.42 | -0.98 | -0.44 | 31% |
| C3-hop | IMD rule (normal + 4.5) | 2 | 46 | -2.24 | -1.80 | -0.44 | 20% |
| C3-hop | IMD rule (normal + 4.5) | 3 | 46 | -2.62 | -2.18 | -0.44 | 17% |
| C3-hop | IMD rule (normal + 4.5) | 4 | 46 | -3.06 | -2.61 | -0.44 | 14% |
| C3-hop | IMD rule (normal + 4.5) | 5 | 46 | -3.08 | -2.64 | -0.44 | 14% |
| Delhi-only LSTM | label rule (normal + 3) | all | 635 | -2.41 | -1.92 | -0.49 | 20% |
| Delhi-only LSTM | label rule (normal + 3) | 1 | 127 | -1.54 | -1.05 | -0.49 | 32% |
| Delhi-only LSTM | label rule (normal + 3) | 2 | 127 | -2.06 | -1.57 | -0.49 | 24% |
| Delhi-only LSTM | label rule (normal + 3) | 3 | 127 | -2.51 | -2.02 | -0.49 | 19% |
| Delhi-only LSTM | label rule (normal + 3) | 4 | 127 | -2.81 | -2.32 | -0.49 | 17% |
| Delhi-only LSTM | label rule (normal + 3) | 5 | 127 | -3.12 | -2.63 | -0.49 | 16% |
| Delhi-only LSTM | IMD rule (normal + 4.5) | all | 230 | -2.73 | -2.29 | -0.44 | 16% |
| Delhi-only LSTM | IMD rule (normal + 4.5) | 1 | 46 | -1.71 | -1.27 | -0.44 | 26% |
| Delhi-only LSTM | IMD rule (normal + 4.5) | 2 | 46 | -2.30 | -1.86 | -0.44 | 19% |
| Delhi-only LSTM | IMD rule (normal + 4.5) | 3 | 46 | -2.89 | -2.45 | -0.44 | 15% |
| Delhi-only LSTM | IMD rule (normal + 4.5) | 4 | 46 | -3.26 | -2.81 | -0.44 | 14% |
| Delhi-only LSTM | IMD rule (normal + 4.5) | 5 | 46 | -3.49 | -3.05 | -0.44 | 13% |
| ridge_hw5 | label rule (normal + 3) | all | 635 | -2.39 | -1.90 | -0.49 | 20% |
| ridge_hw5 | label rule (normal + 3) | 1 | 127 | -1.15 | -0.66 | -0.49 | 42% |
| ridge_hw5 | label rule (normal + 3) | 2 | 127 | -1.93 | -1.44 | -0.49 | 25% |
| ridge_hw5 | label rule (normal + 3) | 3 | 127 | -2.44 | -1.96 | -0.49 | 20% |
| ridge_hw5 | label rule (normal + 3) | 4 | 127 | -2.99 | -2.50 | -0.49 | 16% |
| ridge_hw5 | label rule (normal + 3) | 5 | 127 | -3.45 | -2.96 | -0.49 | 14% |
| ridge_hw5 | IMD rule (normal + 4.5) | all | 230 | -2.81 | -2.37 | -0.44 | 16% |
| ridge_hw5 | IMD rule (normal + 4.5) | 1 | 46 | -1.28 | -0.84 | -0.44 | 35% |
| ridge_hw5 | IMD rule (normal + 4.5) | 2 | 46 | -2.16 | -1.72 | -0.44 | 21% |
| ridge_hw5 | IMD rule (normal + 4.5) | 3 | 46 | -2.92 | -2.48 | -0.44 | 15% |
| ridge_hw5 | IMD rule (normal + 4.5) | 4 | 46 | -3.65 | -3.21 | -0.44 | 12% |
| ridge_hw5 | IMD rule (normal + 4.5) | 5 | 46 | -4.06 | -3.61 | -0.44 | 11% |

### (iii) Global models and GraphCast 2018 against the station

Each outside forecast is paired with C3-hop's ensemble mean on the rows both have and the station observed. The global models are scored on the test period (to 2025-08-22, where the station record ends); GraphCast 2018 on fold f4's 2018 windows with C3-hop's development forecasts (2018 lies inside the mapping's fitting years).

**All days**

| Forecast | Target days | Rows | Clusters | RMSE vs ERA5: it / C3-hop | RMSE vs station: it / C3-hop | Station-mapped: it / C3-hop | Bias vs station: it / C3-hop |
|---|---|---|---|---|---|---|---|
| AIFS | 2024-03-15 to 2025-03-10 | 1675 | 3 | 2.133 / 1.908 | 3.212 / 2.452 | 2.836 / 2.178 | -2.69 / -1.34 |
| AIFS Single (added after the first results) | 2025-02-19 to 2025-08-22 | 870 | 3 | 1.842 / 2.242 | 2.574 / 2.681 | 2.206 / 2.531 | -2.03 / -0.68 |
| GFS | 2024-01-20 to 2025-08-22 | 2730 | 6 | 3.672 / 1.996 | 3.103 / 2.477 | 3.230 / 2.257 | +1.70 / -1.07 |
| GFS, Tmax since 2021 | 2021-03-25 to 2025-08-22 | 7695 | 14 | 3.321 / 2.149 | 2.882 / 2.466 | 3.031 / 2.315 | +1.62 / -0.73 |
| IFS | 2024-02-05 to 2025-08-22 | 2650 | 6 | 1.824 / 2.004 | 2.697 / 2.488 | 2.253 / 2.262 | -1.72 / -1.09 |
| IFS HRES 9 km, 12 UTC run | 2024-03-15 to 2025-08-22 | 2455 | 5 | 1.720 / 2.033 | 2.579 / 2.532 | 2.160 / 2.306 | -1.61 / -1.10 |
| GraphCast 2018 | 2018-01-01 to 2018-12-31 | 1775 | 3 | 1.151 / 1.932 | 1.729 / 2.188 | 1.491 / 2.058 | -1.04 / -0.77 |

**Heat season (15 Mar - 31 Jul)**

| Forecast | Target days | Rows | Clusters | RMSE vs ERA5: it / C3-hop | RMSE vs station: it / C3-hop | Station-mapped: it / C3-hop | Bias vs station: it / C3-hop |
|---|---|---|---|---|---|---|---|
| AIFS | 2024-03-15 to 2025-03-10 | 605 | 1 | 1.683 / 2.084 | 2.525 / 2.414 | 2.192 / 2.124 | -1.99 / -1.31 |
| AIFS Single (added after the first results) | 2025-02-19 to 2025-08-22 | 655 | 1 | 1.956 / 2.407 | 2.486 / 2.740 | 2.182 / 2.624 | -1.92 / -0.45 |
| GFS | 2024-01-20 to 2025-08-22 | 1270 | 2 | 4.302 / 2.252 | 3.748 / 2.581 | 3.830 / 2.390 | +2.63 / -0.86 |
| GFS, Tmax since 2021 | 2021-03-25 to 2025-08-22 | 3280 | 5 | 4.005 / 2.451 | 3.549 / 2.656 | 3.653 / 2.536 | +2.59 / -0.53 |
| IFS | 2024-02-05 to 2025-08-22 | 1270 | 2 | 1.778 / 2.252 | 2.332 / 2.581 | 1.978 / 2.390 | -1.03 / -0.86 |
| IFS HRES 9 km, 12 UTC run | 2024-03-15 to 2025-08-22 | 1260 | 2 | 1.625 / 2.257 | 2.241 / 2.588 | 1.902 / 2.397 | -1.01 / -0.86 |
| GraphCast 2018 | 2018-01-01 to 2018-12-31 | 680 | 1 | 1.206 / 2.236 | 1.778 / 2.248 | 1.604 / 2.142 | -0.92 / -0.59 |

**Station heatwave days, label rule**

| Forecast | Target days | Rows | Clusters | RMSE vs ERA5: it / C3-hop | RMSE vs station: it / C3-hop | Station-mapped: it / C3-hop | Bias vs station: it / C3-hop |
|---|---|---|---|---|---|---|---|
| AIFS | 2024-03-15 to 2025-03-10 | 130 | 1 | 2.417 / 2.193 | 2.700 / 2.309 | 2.590 / 2.217 | -2.20 / -2.01 |
| AIFS Single (added after the first results) | 2025-02-19 to 2025-08-22 | 60 | 1 | 2.134 / 2.446 | 2.661 / 2.886 | 2.585 / 2.789 | -2.53 / -2.61 |
| GFS | 2024-01-20 to 2025-08-22 | 190 | 2 | 2.793 / 2.276 | 2.435 / 2.506 | 2.431 / 2.412 | +2.06 / -2.20 |
| GFS, Tmax since 2021 | 2021-03-25 to 2025-08-22 | 475 | 5 | 2.623 / 2.163 | 2.218 / 2.473 | 2.199 / 2.364 | +1.86 / -2.13 |
| IFS | 2024-02-05 to 2025-08-22 | 190 | 2 | 1.312 / 2.276 | 1.353 / 2.506 | 1.315 / 2.412 | -0.46 / -2.20 |
| IFS HRES 9 km, 12 UTC run | 2024-03-15 to 2025-08-22 | 190 | 2 | 1.328 / 2.276 | 1.549 / 2.506 | 1.484 / 2.412 | -1.01 / -2.20 |
| GraphCast 2018 | 2018-01-01 to 2018-12-31 | 90 | 1 | 0.848 / 1.825 | 1.578 / 2.666 | 1.479 / 2.449 | -1.34 / -2.44 |

**Station heatwave days, IMD rule**

| Forecast | Target days | Rows | Clusters | RMSE vs ERA5: it / C3-hop | RMSE vs station: it / C3-hop | Station-mapped: it / C3-hop | Bias vs station: it / C3-hop |
|---|---|---|---|---|---|---|---|
| AIFS | 2024-03-15 to 2025-03-10 | 65 | 1 | 2.485 / 2.594 | 2.635 / 2.751 | 2.552 / 2.657 | -2.24 / -2.47 |
| AIFS Single (added after the first results) | 2025-02-19 to 2025-08-22 | 15 | 1 | 1.402 / 1.655 | 2.079 / 2.422 | 2.052 / 2.375 | -1.93 / -2.26 |
| GFS | 2024-01-20 to 2025-08-22 | 80 | 2 | 2.342 / 2.445 | 2.047 / 2.692 | 2.049 / 2.606 | +1.68 / -2.43 |
| GFS, Tmax since 2021 | 2021-03-25 to 2025-08-22 | 180 | 4 | 2.343 / 2.436 | 1.962 / 2.813 | 1.925 / 2.689 | +1.67 / -2.43 |
| IFS | 2024-02-05 to 2025-08-22 | 80 | 2 | 0.668 / 2.445 | 0.773 / 2.692 | 0.783 / 2.606 | -0.20 / -2.43 |
| IFS HRES 9 km, 12 UTC run | 2024-03-15 to 2025-08-22 | 80 | 2 | 0.909 / 2.445 | 1.130 / 2.692 | 1.078 / 2.606 | -0.92 / -2.43 |
| GraphCast 2018 | 2018-01-01 to 2018-12-31 | 20 | 1 | 0.988 / 1.722 | 1.757 / 2.712 | 1.603 / 2.524 | -1.64 / -2.63 |

**ERA5 heatwave days (the test's extreme stratum)**

| Forecast | Target days | Rows | Clusters | RMSE vs ERA5: it / C3-hop | RMSE vs station: it / C3-hop | Station-mapped: it / C3-hop | Bias vs station: it / C3-hop |
|---|---|---|---|---|---|---|---|
| AIFS | 2024-03-15 to 2025-03-10 | 120 | 1 | 2.486 / 2.259 | 2.700 / 2.364 | 2.592 / 2.273 | -2.16 / -2.07 |
| AIFS Single (added after the first results) | 2025-02-19 to 2025-08-22 | 30 | 1 | 2.322 / 2.868 | 2.599 / 3.060 | 2.481 / 2.917 | -2.40 / -2.70 |
| GFS | 2024-01-20 to 2025-08-22 | 150 | 2 | 2.438 / 2.393 | 2.297 / 2.519 | 2.338 / 2.415 | +1.95 / -2.20 |
| GFS, Tmax since 2021 | 2021-03-25 to 2025-08-22 | 365 | 5 | 2.259 / 2.327 | 2.106 / 2.513 | 2.082 / 2.398 | +1.79 / -2.16 |
| IFS | 2024-02-05 to 2025-08-22 | 150 | 2 | 1.358 / 2.393 | 1.443 / 2.519 | 1.396 / 2.415 | -0.49 / -2.20 |
| IFS HRES 9 km, 12 UTC run | 2024-03-15 to 2025-08-22 | 150 | 2 | 1.449 / 2.393 | 1.653 / 2.519 | 1.578 / 2.415 | -1.10 / -2.20 |
| GraphCast 2018 | 2018-01-01 to 2018-12-31 | 20 | 1 | 1.049 / 1.919 | 1.703 / 2.730 | 1.555 / 2.527 | -1.55 / -2.64 |

**Paired tests** (delta = RMSE of the outside forecast minus RMSE of C3-hop; negative = the outside forecast is better).

| Comparison | Stratum | Delta vs ERA5 [95% CI] | Delta vs station [95% CI] | Delta station-mapped [95% CI] | Clear change of order |
|---|---|---|---|---|---|
| AIFS minus C3-hop | All days | +0.226 no CI (< 5 clusters), fragile (< 10 clusters) | +0.761 no CI (< 5 clusters), fragile (< 10 clusters) | +0.658 no CI (< 5 clusters), fragile (< 10 clusters) | no |
| AIFS minus C3-hop | Heat season (15 Mar - 31 Jul) | not testable | not testable | not testable | no |
| AIFS minus C3-hop | Station heatwave days, label rule | not testable | not testable | not testable | no |
| AIFS Single (added after the first results) minus C3-hop | All days | -0.400 no CI (< 5 clusters), fragile (< 10 clusters) | -0.107 no CI (< 5 clusters), fragile (< 10 clusters) | -0.325 no CI (< 5 clusters), fragile (< 10 clusters) | no |
| AIFS Single (added after the first results) minus C3-hop | Heat season (15 Mar - 31 Jul) | not testable | not testable | not testable | no |
| AIFS Single (added after the first results) minus C3-hop | Station heatwave days, label rule | not testable | not testable | not testable | no |
| GFS minus C3-hop | All days | +1.676 [+0.947, +2.406], fragile (< 10 clusters) | +0.626 [-0.291, +1.544], fragile (< 10 clusters) | +0.973 [+0.162, +1.784], fragile (< 10 clusters) | no |
| GFS minus C3-hop | Heat season (15 Mar - 31 Jul) | +2.050 no CI (< 5 clusters), fragile (< 10 clusters) | +1.168 no CI (< 5 clusters), fragile (< 10 clusters) | +1.440 no CI (< 5 clusters), fragile (< 10 clusters) | no |
| GFS minus C3-hop | Station heatwave days, label rule | +0.517 no CI (< 5 clusters), fragile (< 10 clusters) | -0.071 no CI (< 5 clusters), fragile (< 10 clusters) | +0.019 no CI (< 5 clusters), fragile (< 10 clusters) | no |
| GFS, Tmax since 2021 minus C3-hop | All days | +1.172 [+0.765, +1.579] | +0.415 [+0.030, +0.801] | +0.716 [+0.358, +1.073] | no |
| GFS, Tmax since 2021 minus C3-hop | Heat season (15 Mar - 31 Jul) | +1.554 [+0.946, +2.162], fragile (< 10 clusters) | +0.893 [+0.467, +1.319], fragile (< 10 clusters) | +1.117 [+0.645, +1.589], fragile (< 10 clusters) | no |
| GFS, Tmax since 2021 minus C3-hop | Station heatwave days, label rule | +0.460 [-0.272, +1.192], fragile (< 10 clusters) | -0.255 [-0.974, +0.463], fragile (< 10 clusters) | -0.164 [-0.855, +0.527], fragile (< 10 clusters) | no |
| IFS minus C3-hop | All days | -0.180 [-0.727, +0.366], fragile (< 10 clusters) | +0.209 [-0.393, +0.811], fragile (< 10 clusters) | -0.009 [-0.577, +0.559], fragile (< 10 clusters) | no |
| IFS minus C3-hop | Heat season (15 Mar - 31 Jul) | -0.474 no CI (< 5 clusters), fragile (< 10 clusters) | -0.248 no CI (< 5 clusters), fragile (< 10 clusters) | -0.411 no CI (< 5 clusters), fragile (< 10 clusters) | no |
| IFS minus C3-hop | Station heatwave days, label rule | -0.964 no CI (< 5 clusters), fragile (< 10 clusters) | -1.152 no CI (< 5 clusters), fragile (< 10 clusters) | -1.097 no CI (< 5 clusters), fragile (< 10 clusters) | no |
| IFS HRES 9 km, 12 UTC run minus C3-hop | All days | -0.314 [-0.891, +0.263], fragile (< 10 clusters) | +0.047 [-0.592, +0.686], fragile (< 10 clusters) | -0.146 [-0.743, +0.452], fragile (< 10 clusters) | no |
| IFS HRES 9 km, 12 UTC run minus C3-hop | Heat season (15 Mar - 31 Jul) | -0.633 no CI (< 5 clusters), fragile (< 10 clusters) | -0.348 no CI (< 5 clusters), fragile (< 10 clusters) | -0.495 no CI (< 5 clusters), fragile (< 10 clusters) | no |
| IFS HRES 9 km, 12 UTC run minus C3-hop | Station heatwave days, label rule | -0.948 no CI (< 5 clusters), fragile (< 10 clusters) | -0.957 no CI (< 5 clusters), fragile (< 10 clusters) | -0.928 no CI (< 5 clusters), fragile (< 10 clusters) | no |
| GraphCast 2018 minus C3-hop | All days | -0.781 no CI (< 5 clusters), fragile (< 10 clusters) | -0.459 no CI (< 5 clusters), fragile (< 10 clusters) | -0.567 no CI (< 5 clusters), fragile (< 10 clusters) | no |
| GraphCast 2018 minus C3-hop | Heat season (15 Mar - 31 Jul) | not testable | not testable | not testable | no |
| GraphCast 2018 minus C3-hop | Station heatwave days, label rule | not testable | not testable | not testable | no |

**By lead, all days**

| Forecast | Score | Day 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| AIFS | RMSE vs station | 3.064 | 3.186 | 3.235 | 3.271 | 3.301 |
| C3-hop on AIFS's rows | RMSE vs station | 2.135 | 2.387 | 2.485 | 2.617 | 2.606 |
| AIFS Single (added after the first results) | RMSE vs station | 2.257 | 2.583 | 2.666 | 2.666 | 2.679 |
| C3-hop on AIFS Single's rows | RMSE vs station | 2.319 | 2.571 | 2.621 | 2.864 | 2.988 |
| GFS | RMSE vs station | 2.561 | 2.956 | 3.200 | 3.320 | 3.407 |
| C3-hop on GFS's rows | RMSE vs station | 2.156 | 2.400 | 2.480 | 2.636 | 2.678 |
| GFS, Tmax since 2021 | RMSE vs station | 2.488 | 2.720 | 2.928 | 3.040 | 3.182 |
| C3-hop on GFS, Tmax since 2021's rows | RMSE vs station | 2.002 | 2.329 | 2.490 | 2.661 | 2.777 |
| IFS | RMSE vs station | 2.556 | 2.612 | 2.676 | 2.726 | 2.902 |
| C3-hop on IFS's rows | RMSE vs station | 2.152 | 2.410 | 2.492 | 2.650 | 2.699 |
| IFS HRES 9 km, 12 UTC run | RMSE vs station | 2.369 | 2.515 | 2.555 | 2.635 | 2.805 |
| C3-hop on IFS HRES 9 km, 12 UTC run's rows | RMSE vs station | 2.189 | 2.455 | 2.538 | 2.698 | 2.745 |
| GraphCast 2018 | RMSE vs station | 1.631 | 1.724 | 1.700 | 1.753 | 1.828 |
| C3-hop on GraphCast 2018's rows | RMSE vs station | 1.770 | 2.021 | 2.223 | 2.384 | 2.467 |

**Yes/no warnings** (in-season rows; R-point only, no probabilities for these comparisons).

| Forecast | Events | Rule | Hits | Misses | False alarms | Hit rate | False-alarm ratio | ETS | Frequency bias |
|---|---|---|---|---|---|---|---|---|---|
| AIFS (605 rows) | station, label rule (normal + 3) (130) | R-point | 65 | 65 | 2 | 50% | 3% | 0.43 | 0.52 |
| AIFS (605 rows) | station, label rule (normal + 3) (130) | station-mapped R-point | 65 | 65 | 2 | 50% | 3% | 0.43 | 0.52 |
| AIFS (605 rows) | station, IMD rule (normal + 4.5) (65) | R-point | 45 | 20 | 22 | 69% | 33% | 0.47 | 1.03 |
| AIFS (605 rows) | station, IMD rule (normal + 4.5) (65) | station-mapped R-point | 19 | 46 | 0 | 29% | 0% | 0.27 | 0.29 |
| AIFS (605 rows) | ERA5, label rule (reference) (120) | R-point | 65 | 55 | 2 | 54% | 3% | 0.48 | 0.56 |
| C3-hop on AIFS's rows (605 rows) | station, label rule (normal + 3) (130) | R-point | 58 | 72 | 19 | 45% | 25% | 0.31 | 0.59 |
| C3-hop on AIFS's rows (605 rows) | station, label rule (normal + 3) (130) | station-mapped R-point | 61 | 69 | 19 | 47% | 24% | 0.33 | 0.62 |
| C3-hop on AIFS's rows (605 rows) | station, IMD rule (normal + 4.5) (65) | R-point | 45 | 20 | 32 | 69% | 42% | 0.41 | 1.18 |
| C3-hop on AIFS's rows (605 rows) | station, IMD rule (normal + 4.5) (65) | station-mapped R-point | 6 | 59 | 0 | 9% | 0% | 0.08 | 0.09 |
| C3-hop on AIFS's rows (605 rows) | ERA5, label rule (reference) (120) | R-point | 57 | 63 | 20 | 48% | 26% | 0.33 | 0.64 |
| AIFS Single (added after the first results) (655 rows) | station, label rule (normal + 3) (60) | R-point | 0 | 60 | 0 | 0% | n/a | 0.00 | 0.00 |
| AIFS Single (added after the first results) (655 rows) | station, label rule (normal + 3) (60) | station-mapped R-point | 0 | 60 | 0 | 0% | n/a | 0.00 | 0.00 |
| AIFS Single (added after the first results) (655 rows) | station, IMD rule (normal + 4.5) (15) | R-point | 0 | 15 | 0 | 0% | n/a | 0.00 | 0.00 |
| AIFS Single (added after the first results) (655 rows) | station, IMD rule (normal + 4.5) (15) | station-mapped R-point | 0 | 15 | 0 | 0% | n/a | 0.00 | 0.00 |
| AIFS Single (added after the first results) (655 rows) | ERA5, label rule (reference) (40) | R-point | 0 | 40 | 0 | 0% | n/a | 0.00 | 0.00 |
| C3-hop on AIFS Single's rows (655 rows) | station, label rule (normal + 3) (60) | R-point | 4 | 56 | 0 | 7% | 0% | 0.06 | 0.07 |
| C3-hop on AIFS Single's rows (655 rows) | station, label rule (normal + 3) (60) | station-mapped R-point | 1 | 59 | 0 | 2% | 0% | 0.02 | 0.02 |
| C3-hop on AIFS Single's rows (655 rows) | station, IMD rule (normal + 4.5) (15) | R-point | 0 | 15 | 4 | 0% | 100% | -0.00 | 0.27 |
| C3-hop on AIFS Single's rows (655 rows) | station, IMD rule (normal + 4.5) (15) | station-mapped R-point | 0 | 15 | 0 | 0% | n/a | 0.00 | 0.00 |
| C3-hop on AIFS Single's rows (655 rows) | ERA5, label rule (reference) (40) | R-point | 2 | 38 | 2 | 5% | 50% | 0.04 | 0.10 |
| GFS (1270 rows) | station, label rule (normal + 3) (190) | R-point | 190 | 0 | 351 | 100% | 65% | 0.24 | 2.85 |
| GFS (1270 rows) | station, label rule (normal + 3) (190) | station-mapped R-point | 189 | 1 | 365 | 99% | 66% | 0.22 | 2.92 |
| GFS (1270 rows) | station, IMD rule (normal + 4.5) (80) | R-point | 80 | 0 | 461 | 100% | 85% | 0.09 | 6.76 |
| GFS (1270 rows) | station, IMD rule (normal + 4.5) (80) | station-mapped R-point | 80 | 0 | 336 | 100% | 81% | 0.14 | 5.20 |
| GFS (1270 rows) | ERA5, label rule (reference) (160) | R-point | 160 | 0 | 381 | 100% | 70% | 0.19 | 3.38 |
| C3-hop on GFS's rows (1270 rows) | station, label rule (normal + 3) (190) | R-point | 62 | 128 | 19 | 33% | 23% | 0.25 | 0.43 |
| C3-hop on GFS's rows (1270 rows) | station, label rule (normal + 3) (190) | station-mapped R-point | 62 | 128 | 19 | 33% | 23% | 0.25 | 0.43 |
| C3-hop on GFS's rows (1270 rows) | station, IMD rule (normal + 4.5) (80) | R-point | 45 | 35 | 36 | 56% | 44% | 0.36 | 1.01 |
| C3-hop on GFS's rows (1270 rows) | station, IMD rule (normal + 4.5) (80) | station-mapped R-point | 6 | 74 | 0 | 8% | 0% | 0.07 | 0.07 |
| C3-hop on GFS's rows (1270 rows) | ERA5, label rule (reference) (160) | R-point | 59 | 101 | 22 | 37% | 27% | 0.28 | 0.51 |
| GFS, Tmax since 2021 (3280 rows) | station, label rule (normal + 3) (475) | R-point | 471 | 4 | 931 | 99% | 66% | 0.22 | 2.95 |
| GFS, Tmax since 2021 (3280 rows) | station, label rule (normal + 3) (475) | station-mapped R-point | 470 | 5 | 961 | 99% | 67% | 0.21 | 3.01 |
| GFS, Tmax since 2021 (3280 rows) | station, IMD rule (normal + 4.5) (180) | R-point | 180 | 0 | 1222 | 100% | 87% | 0.08 | 7.79 |
| GFS, Tmax since 2021 (3280 rows) | station, IMD rule (normal + 4.5) (180) | station-mapped R-point | 180 | 0 | 852 | 100% | 83% | 0.13 | 5.73 |
| GFS, Tmax since 2021 (3280 rows) | ERA5, label rule (reference) (400) | R-point | 400 | 0 | 1002 | 100% | 71% | 0.19 | 3.50 |
| C3-hop on GFS, Tmax since 2021's rows (3280 rows) | station, label rule (normal + 3) (475) | R-point | 125 | 350 | 52 | 26% | 29% | 0.20 | 0.37 |
| C3-hop on GFS, Tmax since 2021's rows (3280 rows) | station, label rule (normal + 3) (475) | station-mapped R-point | 120 | 355 | 52 | 25% | 30% | 0.19 | 0.36 |
| C3-hop on GFS, Tmax since 2021's rows (3280 rows) | station, IMD rule (normal + 4.5) (180) | R-point | 89 | 91 | 88 | 49% | 50% | 0.31 | 0.98 |
| C3-hop on GFS, Tmax since 2021's rows (3280 rows) | station, IMD rule (normal + 4.5) (180) | station-mapped R-point | 24 | 156 | 4 | 13% | 14% | 0.12 | 0.16 |
| C3-hop on GFS, Tmax since 2021's rows (3280 rows) | ERA5, label rule (reference) (400) | R-point | 114 | 286 | 63 | 28% | 36% | 0.21 | 0.44 |
| IFS (1270 rows) | station, label rule (normal + 3) (190) | R-point | 143 | 47 | 55 | 75% | 28% | 0.53 | 1.04 |
| IFS (1270 rows) | station, label rule (normal + 3) (190) | station-mapped R-point | 140 | 50 | 53 | 74% | 27% | 0.52 | 1.02 |
| IFS (1270 rows) | station, IMD rule (normal + 4.5) (80) | R-point | 72 | 8 | 126 | 90% | 64% | 0.31 | 2.48 |
| IFS (1270 rows) | station, IMD rule (normal + 4.5) (80) | station-mapped R-point | 62 | 18 | 37 | 78% | 37% | 0.50 | 1.24 |
| IFS (1270 rows) | ERA5, label rule (reference) (160) | R-point | 120 | 40 | 78 | 75% | 39% | 0.45 | 1.24 |
| C3-hop on IFS's rows (1270 rows) | station, label rule (normal + 3) (190) | R-point | 62 | 128 | 19 | 33% | 23% | 0.25 | 0.43 |
| C3-hop on IFS's rows (1270 rows) | station, label rule (normal + 3) (190) | station-mapped R-point | 62 | 128 | 19 | 33% | 23% | 0.25 | 0.43 |
| C3-hop on IFS's rows (1270 rows) | station, IMD rule (normal + 4.5) (80) | R-point | 45 | 35 | 36 | 56% | 44% | 0.36 | 1.01 |
| C3-hop on IFS's rows (1270 rows) | station, IMD rule (normal + 4.5) (80) | station-mapped R-point | 6 | 74 | 0 | 8% | 0% | 0.07 | 0.07 |
| C3-hop on IFS's rows (1270 rows) | ERA5, label rule (reference) (160) | R-point | 59 | 101 | 22 | 37% | 27% | 0.28 | 0.51 |
| IFS HRES 9 km, 12 UTC run (1260 rows) | station, label rule (normal + 3) (190) | R-point | 128 | 62 | 44 | 67% | 26% | 0.49 | 0.91 |
| IFS HRES 9 km, 12 UTC run (1260 rows) | station, label rule (normal + 3) (190) | station-mapped R-point | 128 | 62 | 43 | 67% | 25% | 0.49 | 0.90 |
| IFS HRES 9 km, 12 UTC run (1260 rows) | station, IMD rule (normal + 4.5) (80) | R-point | 69 | 11 | 103 | 86% | 60% | 0.34 | 2.15 |
| IFS HRES 9 km, 12 UTC run (1260 rows) | station, IMD rule (normal + 4.5) (80) | station-mapped R-point | 47 | 33 | 21 | 59% | 31% | 0.44 | 0.85 |
| IFS HRES 9 km, 12 UTC run (1260 rows) | ERA5, label rule (reference) (160) | R-point | 115 | 45 | 57 | 72% | 33% | 0.48 | 1.07 |
| C3-hop on IFS HRES 9 km, 12 UTC run's rows (1260 rows) | station, label rule (normal + 3) (190) | R-point | 62 | 128 | 19 | 33% | 23% | 0.25 | 0.43 |
| C3-hop on IFS HRES 9 km, 12 UTC run's rows (1260 rows) | station, label rule (normal + 3) (190) | station-mapped R-point | 62 | 128 | 19 | 33% | 23% | 0.25 | 0.43 |
| C3-hop on IFS HRES 9 km, 12 UTC run's rows (1260 rows) | station, IMD rule (normal + 4.5) (80) | R-point | 45 | 35 | 36 | 56% | 44% | 0.36 | 1.01 |
| C3-hop on IFS HRES 9 km, 12 UTC run's rows (1260 rows) | station, IMD rule (normal + 4.5) (80) | station-mapped R-point | 6 | 74 | 0 | 8% | 0% | 0.07 | 0.07 |
| C3-hop on IFS HRES 9 km, 12 UTC run's rows (1260 rows) | ERA5, label rule (reference) (160) | R-point | 59 | 101 | 22 | 37% | 27% | 0.28 | 0.51 |
| GraphCast 2018 (680 rows) | station, label rule (normal + 3) (90) | R-point | 29 | 61 | 3 | 32% | 9% | 0.28 | 0.36 |
| GraphCast 2018 (680 rows) | station, label rule (normal + 3) (90) | station-mapped R-point | 32 | 58 | 4 | 36% | 11% | 0.31 | 0.40 |
| GraphCast 2018 (680 rows) | station, IMD rule (normal + 4.5) (20) | R-point | 14 | 6 | 18 | 70% | 56% | 0.35 | 1.60 |
| GraphCast 2018 (680 rows) | station, IMD rule (normal + 4.5) (20) | station-mapped R-point | 0 | 20 | 4 | 0% | 100% | -0.00 | 0.20 |
| GraphCast 2018 (680 rows) | ERA5, label rule (reference) (30) | R-point | 20 | 10 | 12 | 67% | 38% | 0.46 | 1.07 |
| C3-hop on GraphCast 2018's rows (680 rows) | station, label rule (normal + 3) (90) | R-point | 2 | 88 | 2 | 2% | 50% | 0.02 | 0.04 |
| C3-hop on GraphCast 2018's rows (680 rows) | station, label rule (normal + 3) (90) | station-mapped R-point | 3 | 87 | 3 | 3% | 50% | 0.02 | 0.07 |
| C3-hop on GraphCast 2018's rows (680 rows) | station, IMD rule (normal + 4.5) (20) | R-point | 0 | 20 | 4 | 0% | 100% | -0.00 | 0.20 |
| C3-hop on GraphCast 2018's rows (680 rows) | station, IMD rule (normal + 4.5) (20) | station-mapped R-point | 0 | 20 | 0 | 0% | n/a | 0.00 | 0.00 |
| C3-hop on GraphCast 2018's rows (680 rows) | ERA5, label rule (reference) (30) | R-point | 0 | 30 | 4 | 0% | 100% | -0.01 | 0.13 |

**Cold bias on station heatwave days**

| Forecast | Station rule | Lead | Rows | Forecast - station | = forecast - ERA5 | + ERA5 - station | ERA5's share |
|---|---|---|---|---|---|---|---|
| AIFS | label rule (normal + 3) | all | 130 | -2.20 | -1.92 | -0.28 | 13% |
| AIFS | label rule (normal + 3) | 1 | 26 | -1.00 | -0.72 | -0.28 | 28% |
| AIFS | label rule (normal + 3) | 2 | 26 | -1.69 | -1.41 | -0.28 | 16% |
| AIFS | label rule (normal + 3) | 3 | 26 | -2.36 | -2.08 | -0.28 | 12% |
| AIFS | label rule (normal + 3) | 4 | 26 | -2.86 | -2.58 | -0.28 | 10% |
| AIFS | label rule (normal + 3) | 5 | 26 | -3.08 | -2.80 | -0.28 | 9% |
| AIFS | IMD rule (normal + 4.5) | all | 65 | -2.24 | -2.06 | -0.18 | 8% |
| AIFS | IMD rule (normal + 4.5) | 1 | 13 | -0.92 | -0.74 | -0.18 | 19% |
| AIFS | IMD rule (normal + 4.5) | 2 | 13 | -1.73 | -1.56 | -0.18 | 10% |
| AIFS | IMD rule (normal + 4.5) | 3 | 13 | -2.42 | -2.24 | -0.18 | 7% |
| AIFS | IMD rule (normal + 4.5) | 4 | 13 | -2.90 | -2.72 | -0.18 | 6% |
| AIFS | IMD rule (normal + 4.5) | 5 | 13 | -3.22 | -3.04 | -0.18 | 5% |
| C3-hop on AIFS's rows | label rule (normal + 3) | all | 130 | -2.01 | -1.73 | -0.28 | 14% |
| C3-hop on AIFS's rows | label rule (normal + 3) | 1 | 26 | -1.31 | -1.03 | -0.28 | 21% |
| C3-hop on AIFS's rows | label rule (normal + 3) | 2 | 26 | -1.96 | -1.68 | -0.28 | 14% |
| C3-hop on AIFS's rows | label rule (normal + 3) | 3 | 26 | -2.20 | -1.92 | -0.28 | 13% |
| C3-hop on AIFS's rows | label rule (normal + 3) | 4 | 26 | -2.32 | -2.04 | -0.28 | 12% |
| C3-hop on AIFS's rows | label rule (normal + 3) | 5 | 26 | -2.27 | -1.99 | -0.28 | 12% |
| C3-hop on AIFS's rows | IMD rule (normal + 4.5) | all | 65 | -2.47 | -2.29 | -0.18 | 7% |
| C3-hop on AIFS's rows | IMD rule (normal + 4.5) | 1 | 13 | -1.37 | -1.20 | -0.18 | 13% |
| C3-hop on AIFS's rows | IMD rule (normal + 4.5) | 2 | 13 | -2.33 | -2.16 | -0.18 | 8% |
| C3-hop on AIFS's rows | IMD rule (normal + 4.5) | 3 | 13 | -2.64 | -2.46 | -0.18 | 7% |
| C3-hop on AIFS's rows | IMD rule (normal + 4.5) | 4 | 13 | -3.05 | -2.88 | -0.18 | 6% |
| C3-hop on AIFS's rows | IMD rule (normal + 4.5) | 5 | 13 | -2.96 | -2.78 | -0.18 | 6% |
| AIFS Single (added after the first results) | label rule (normal + 3) | all | 60 | -2.53 | -1.97 | -0.56 | 22% |
| AIFS Single (added after the first results) | label rule (normal + 3) | 1 | 12 | -2.22 | -1.66 | -0.56 | 25% |
| AIFS Single (added after the first results) | label rule (normal + 3) | 2 | 12 | -2.61 | -2.05 | -0.56 | 21% |
| AIFS Single (added after the first results) | label rule (normal + 3) | 3 | 12 | -2.73 | -2.17 | -0.56 | 21% |
| AIFS Single (added after the first results) | label rule (normal + 3) | 4 | 12 | -2.67 | -2.11 | -0.56 | 21% |
| AIFS Single (added after the first results) | label rule (normal + 3) | 5 | 12 | -2.40 | -1.84 | -0.56 | 23% |
| AIFS Single (added after the first results) | IMD rule (normal + 4.5) | all | 15 | -1.93 | -1.11 | -0.83 | 43% |
| AIFS Single (added after the first results) | IMD rule (normal + 4.5) | 1 | 3 | -1.57 | -0.74 | -0.83 | 53% |
| AIFS Single (added after the first results) | IMD rule (normal + 4.5) | 2 | 3 | -1.90 | -1.07 | -0.83 | 44% |
| AIFS Single (added after the first results) | IMD rule (normal + 4.5) | 3 | 3 | -2.17 | -1.34 | -0.83 | 38% |
| AIFS Single (added after the first results) | IMD rule (normal + 4.5) | 4 | 3 | -2.17 | -1.35 | -0.83 | 38% |
| AIFS Single (added after the first results) | IMD rule (normal + 4.5) | 5 | 3 | -1.86 | -1.04 | -0.83 | 44% |
| C3-hop on AIFS Single's rows | label rule (normal + 3) | all | 60 | -2.61 | -2.05 | -0.56 | 21% |
| C3-hop on AIFS Single's rows | label rule (normal + 3) | 1 | 12 | -1.41 | -0.85 | -0.56 | 40% |
| C3-hop on AIFS Single's rows | label rule (normal + 3) | 2 | 12 | -2.59 | -2.03 | -0.56 | 22% |
| C3-hop on AIFS Single's rows | label rule (normal + 3) | 3 | 12 | -2.75 | -2.19 | -0.56 | 20% |
| C3-hop on AIFS Single's rows | label rule (normal + 3) | 4 | 12 | -3.05 | -2.49 | -0.56 | 18% |
| C3-hop on AIFS Single's rows | label rule (normal + 3) | 5 | 12 | -3.28 | -2.72 | -0.56 | 17% |
| C3-hop on AIFS Single's rows | IMD rule (normal + 4.5) | all | 15 | -2.26 | -1.44 | -0.83 | 36% |
| C3-hop on AIFS Single's rows | IMD rule (normal + 4.5) | 1 | 3 | -1.18 | -0.35 | -0.83 | 70% |
| C3-hop on AIFS Single's rows | IMD rule (normal + 4.5) | 2 | 3 | -2.04 | -1.21 | -0.83 | 41% |
| C3-hop on AIFS Single's rows | IMD rule (normal + 4.5) | 3 | 3 | -2.16 | -1.34 | -0.83 | 38% |
| C3-hop on AIFS Single's rows | IMD rule (normal + 4.5) | 4 | 3 | -2.85 | -2.02 | -0.83 | 29% |
| C3-hop on AIFS Single's rows | IMD rule (normal + 4.5) | 5 | 3 | -3.10 | -2.27 | -0.83 | 27% |
| GFS | label rule (normal + 3) | all | 190 | +2.06 | +2.43 | -0.37 | -18% |
| GFS | label rule (normal + 3) | 1 | 38 | +1.91 | +2.28 | -0.37 | -19% |
| GFS | label rule (normal + 3) | 2 | 38 | +2.22 | +2.58 | -0.37 | -17% |
| GFS | label rule (normal + 3) | 3 | 38 | +2.18 | +2.54 | -0.37 | -17% |
| GFS | label rule (normal + 3) | 4 | 38 | +1.97 | +2.33 | -0.37 | -19% |
| GFS | label rule (normal + 3) | 5 | 38 | +2.03 | +2.39 | -0.37 | -18% |
| GFS | IMD rule (normal + 4.5) | all | 80 | +1.68 | +1.98 | -0.30 | -18% |
| GFS | IMD rule (normal + 4.5) | 1 | 16 | +1.72 | +2.02 | -0.30 | -17% |
| GFS | IMD rule (normal + 4.5) | 2 | 16 | +1.99 | +2.28 | -0.30 | -15% |
| GFS | IMD rule (normal + 4.5) | 3 | 16 | +1.80 | +2.10 | -0.30 | -17% |
| GFS | IMD rule (normal + 4.5) | 4 | 16 | +1.34 | +1.64 | -0.30 | -22% |
| GFS | IMD rule (normal + 4.5) | 5 | 16 | +1.54 | +1.84 | -0.30 | -19% |
| C3-hop on GFS's rows | label rule (normal + 3) | all | 190 | -2.20 | -1.83 | -0.37 | 17% |
| C3-hop on GFS's rows | label rule (normal + 3) | 1 | 38 | -1.34 | -0.98 | -0.37 | 27% |
| C3-hop on GFS's rows | label rule (normal + 3) | 2 | 38 | -2.16 | -1.79 | -0.37 | 17% |
| C3-hop on GFS's rows | label rule (normal + 3) | 3 | 38 | -2.37 | -2.00 | -0.37 | 15% |
| C3-hop on GFS's rows | label rule (normal + 3) | 4 | 38 | -2.55 | -2.18 | -0.37 | 14% |
| C3-hop on GFS's rows | label rule (normal + 3) | 5 | 38 | -2.59 | -2.22 | -0.37 | 14% |
| C3-hop on GFS's rows | IMD rule (normal + 4.5) | all | 80 | -2.43 | -2.13 | -0.30 | 12% |
| C3-hop on GFS's rows | IMD rule (normal + 4.5) | 1 | 16 | -1.34 | -1.04 | -0.30 | 22% |
| C3-hop on GFS's rows | IMD rule (normal + 4.5) | 2 | 16 | -2.28 | -1.98 | -0.30 | 13% |
| C3-hop on GFS's rows | IMD rule (normal + 4.5) | 3 | 16 | -2.55 | -2.25 | -0.30 | 12% |
| C3-hop on GFS's rows | IMD rule (normal + 4.5) | 4 | 16 | -3.01 | -2.72 | -0.30 | 10% |
| C3-hop on GFS's rows | IMD rule (normal + 4.5) | 5 | 16 | -2.99 | -2.69 | -0.30 | 10% |
| GFS, Tmax since 2021 | label rule (normal + 3) | all | 475 | +1.86 | +2.26 | -0.40 | -21% |
| GFS, Tmax since 2021 | label rule (normal + 3) | 1 | 95 | +1.66 | +2.06 | -0.40 | -24% |
| GFS, Tmax since 2021 | label rule (normal + 3) | 2 | 95 | +1.94 | +2.34 | -0.40 | -20% |
| GFS, Tmax since 2021 | label rule (normal + 3) | 3 | 95 | +1.90 | +2.30 | -0.40 | -21% |
| GFS, Tmax since 2021 | label rule (normal + 3) | 4 | 95 | +1.87 | +2.27 | -0.40 | -21% |
| GFS, Tmax since 2021 | label rule (normal + 3) | 5 | 95 | +1.92 | +2.32 | -0.40 | -21% |
| GFS, Tmax since 2021 | IMD rule (normal + 4.5) | all | 180 | +1.67 | +2.05 | -0.38 | -22% |
| GFS, Tmax since 2021 | IMD rule (normal + 4.5) | 1 | 36 | +1.61 | +1.98 | -0.38 | -23% |
| GFS, Tmax since 2021 | IMD rule (normal + 4.5) | 2 | 36 | +1.85 | +2.23 | -0.38 | -20% |
| GFS, Tmax since 2021 | IMD rule (normal + 4.5) | 3 | 36 | +1.80 | +2.17 | -0.38 | -21% |
| GFS, Tmax since 2021 | IMD rule (normal + 4.5) | 4 | 36 | +1.55 | +1.93 | -0.38 | -24% |
| GFS, Tmax since 2021 | IMD rule (normal + 4.5) | 5 | 36 | +1.56 | +1.93 | -0.38 | -24% |
| C3-hop on GFS, Tmax since 2021's rows | label rule (normal + 3) | all | 475 | -2.13 | -1.73 | -0.40 | 19% |
| C3-hop on GFS, Tmax since 2021's rows | label rule (normal + 3) | 1 | 95 | -1.30 | -0.90 | -0.40 | 31% |
| C3-hop on GFS, Tmax since 2021's rows | label rule (normal + 3) | 2 | 95 | -1.99 | -1.59 | -0.40 | 20% |
| C3-hop on GFS, Tmax since 2021's rows | label rule (normal + 3) | 3 | 95 | -2.25 | -1.85 | -0.40 | 18% |
| C3-hop on GFS, Tmax since 2021's rows | label rule (normal + 3) | 4 | 95 | -2.52 | -2.12 | -0.40 | 16% |
| C3-hop on GFS, Tmax since 2021's rows | label rule (normal + 3) | 5 | 95 | -2.59 | -2.19 | -0.40 | 15% |
| C3-hop on GFS, Tmax since 2021's rows | IMD rule (normal + 4.5) | all | 180 | -2.43 | -2.06 | -0.38 | 15% |
| C3-hop on GFS, Tmax since 2021's rows | IMD rule (normal + 4.5) | 1 | 36 | -1.33 | -0.95 | -0.38 | 28% |
| C3-hop on GFS, Tmax since 2021's rows | IMD rule (normal + 4.5) | 2 | 36 | -2.16 | -1.78 | -0.38 | 17% |
| C3-hop on GFS, Tmax since 2021's rows | IMD rule (normal + 4.5) | 3 | 36 | -2.55 | -2.18 | -0.38 | 15% |
| C3-hop on GFS, Tmax since 2021's rows | IMD rule (normal + 4.5) | 4 | 36 | -3.02 | -2.64 | -0.38 | 12% |
| C3-hop on GFS, Tmax since 2021's rows | IMD rule (normal + 4.5) | 5 | 36 | -3.12 | -2.74 | -0.38 | 12% |
| IFS | label rule (normal + 3) | all | 190 | -0.46 | -0.09 | -0.37 | 80% |
| IFS | label rule (normal + 3) | 1 | 38 | -1.02 | -0.66 | -0.37 | 36% |
| IFS | label rule (normal + 3) | 2 | 38 | -0.81 | -0.45 | -0.37 | 45% |
| IFS | label rule (normal + 3) | 3 | 38 | -0.37 | -0.01 | -0.37 | 99% |
| IFS | label rule (normal + 3) | 4 | 38 | -0.16 | +0.21 | -0.37 | 231% |
| IFS | label rule (normal + 3) | 5 | 38 | +0.06 | +0.43 | -0.37 | -586% |
| IFS | IMD rule (normal + 4.5) | all | 80 | -0.20 | +0.10 | -0.30 | 150% |
| IFS | IMD rule (normal + 4.5) | 1 | 16 | -0.69 | -0.39 | -0.30 | 44% |
| IFS | IMD rule (normal + 4.5) | 2 | 16 | -0.48 | -0.18 | -0.30 | 62% |
| IFS | IMD rule (normal + 4.5) | 3 | 16 | -0.13 | +0.17 | -0.30 | 236% |
| IFS | IMD rule (normal + 4.5) | 4 | 16 | -0.00 | +0.29 | -0.30 | 6143% |
| IFS | IMD rule (normal + 4.5) | 5 | 16 | +0.31 | +0.60 | -0.30 | -98% |
| C3-hop on IFS's rows | label rule (normal + 3) | all | 190 | -2.20 | -1.83 | -0.37 | 17% |
| C3-hop on IFS's rows | label rule (normal + 3) | 1 | 38 | -1.34 | -0.98 | -0.37 | 27% |
| C3-hop on IFS's rows | label rule (normal + 3) | 2 | 38 | -2.16 | -1.79 | -0.37 | 17% |
| C3-hop on IFS's rows | label rule (normal + 3) | 3 | 38 | -2.37 | -2.00 | -0.37 | 15% |
| C3-hop on IFS's rows | label rule (normal + 3) | 4 | 38 | -2.55 | -2.18 | -0.37 | 14% |
| C3-hop on IFS's rows | label rule (normal + 3) | 5 | 38 | -2.59 | -2.22 | -0.37 | 14% |
| C3-hop on IFS's rows | IMD rule (normal + 4.5) | all | 80 | -2.43 | -2.13 | -0.30 | 12% |
| C3-hop on IFS's rows | IMD rule (normal + 4.5) | 1 | 16 | -1.34 | -1.04 | -0.30 | 22% |
| C3-hop on IFS's rows | IMD rule (normal + 4.5) | 2 | 16 | -2.28 | -1.98 | -0.30 | 13% |
| C3-hop on IFS's rows | IMD rule (normal + 4.5) | 3 | 16 | -2.55 | -2.25 | -0.30 | 12% |
| C3-hop on IFS's rows | IMD rule (normal + 4.5) | 4 | 16 | -3.01 | -2.72 | -0.30 | 10% |
| C3-hop on IFS's rows | IMD rule (normal + 4.5) | 5 | 16 | -2.99 | -2.69 | -0.30 | 10% |
| IFS HRES 9 km, 12 UTC run | label rule (normal + 3) | all | 190 | -1.01 | -0.65 | -0.37 | 36% |
| IFS HRES 9 km, 12 UTC run | label rule (normal + 3) | 1 | 38 | -1.23 | -0.86 | -0.37 | 30% |
| IFS HRES 9 km, 12 UTC run | label rule (normal + 3) | 2 | 38 | -1.19 | -0.82 | -0.37 | 31% |
| IFS HRES 9 km, 12 UTC run | label rule (normal + 3) | 3 | 38 | -1.03 | -0.66 | -0.37 | 36% |
| IFS HRES 9 km, 12 UTC run | label rule (normal + 3) | 4 | 38 | -0.98 | -0.61 | -0.37 | 37% |
| IFS HRES 9 km, 12 UTC run | label rule (normal + 3) | 5 | 38 | -0.65 | -0.28 | -0.37 | 57% |
| IFS HRES 9 km, 12 UTC run | IMD rule (normal + 4.5) | all | 80 | -0.92 | -0.62 | -0.30 | 33% |
| IFS HRES 9 km, 12 UTC run | IMD rule (normal + 4.5) | 1 | 16 | -1.05 | -0.76 | -0.30 | 28% |
| IFS HRES 9 km, 12 UTC run | IMD rule (normal + 4.5) | 2 | 16 | -1.01 | -0.71 | -0.30 | 29% |
| IFS HRES 9 km, 12 UTC run | IMD rule (normal + 4.5) | 3 | 16 | -0.95 | -0.65 | -0.30 | 32% |
| IFS HRES 9 km, 12 UTC run | IMD rule (normal + 4.5) | 4 | 16 | -0.87 | -0.57 | -0.30 | 34% |
| IFS HRES 9 km, 12 UTC run | IMD rule (normal + 4.5) | 5 | 16 | -0.70 | -0.41 | -0.30 | 42% |
| C3-hop on IFS HRES 9 km, 12 UTC run's rows | label rule (normal + 3) | all | 190 | -2.20 | -1.83 | -0.37 | 17% |
| C3-hop on IFS HRES 9 km, 12 UTC run's rows | label rule (normal + 3) | 1 | 38 | -1.34 | -0.98 | -0.37 | 27% |
| C3-hop on IFS HRES 9 km, 12 UTC run's rows | label rule (normal + 3) | 2 | 38 | -2.16 | -1.79 | -0.37 | 17% |
| C3-hop on IFS HRES 9 km, 12 UTC run's rows | label rule (normal + 3) | 3 | 38 | -2.37 | -2.00 | -0.37 | 15% |
| C3-hop on IFS HRES 9 km, 12 UTC run's rows | label rule (normal + 3) | 4 | 38 | -2.55 | -2.18 | -0.37 | 14% |
| C3-hop on IFS HRES 9 km, 12 UTC run's rows | label rule (normal + 3) | 5 | 38 | -2.59 | -2.22 | -0.37 | 14% |
| C3-hop on IFS HRES 9 km, 12 UTC run's rows | IMD rule (normal + 4.5) | all | 80 | -2.43 | -2.13 | -0.30 | 12% |
| C3-hop on IFS HRES 9 km, 12 UTC run's rows | IMD rule (normal + 4.5) | 1 | 16 | -1.34 | -1.04 | -0.30 | 22% |
| C3-hop on IFS HRES 9 km, 12 UTC run's rows | IMD rule (normal + 4.5) | 2 | 16 | -2.28 | -1.98 | -0.30 | 13% |
| C3-hop on IFS HRES 9 km, 12 UTC run's rows | IMD rule (normal + 4.5) | 3 | 16 | -2.55 | -2.25 | -0.30 | 12% |
| C3-hop on IFS HRES 9 km, 12 UTC run's rows | IMD rule (normal + 4.5) | 4 | 16 | -3.01 | -2.72 | -0.30 | 10% |
| C3-hop on IFS HRES 9 km, 12 UTC run's rows | IMD rule (normal + 4.5) | 5 | 16 | -2.99 | -2.69 | -0.30 | 10% |
| GraphCast 2018 | label rule (normal + 3) | all | 90 | -1.34 | -0.22 | -1.12 | 84% |
| GraphCast 2018 | label rule (normal + 3) | 1 | 18 | -1.25 | -0.12 | -1.12 | 90% |
| GraphCast 2018 | label rule (normal + 3) | 2 | 18 | -1.20 | -0.08 | -1.12 | 94% |
| GraphCast 2018 | label rule (normal + 3) | 3 | 18 | -1.37 | -0.25 | -1.12 | 82% |
| GraphCast 2018 | label rule (normal + 3) | 4 | 18 | -1.42 | -0.29 | -1.12 | 79% |
| GraphCast 2018 | label rule (normal + 3) | 5 | 18 | -1.45 | -0.33 | -1.12 | 77% |
| GraphCast 2018 | IMD rule (normal + 4.5) | all | 20 | -1.64 | -0.61 | -1.02 | 63% |
| GraphCast 2018 | IMD rule (normal + 4.5) | 1 | 4 | -1.25 | -0.22 | -1.02 | 82% |
| GraphCast 2018 | IMD rule (normal + 4.5) | 2 | 4 | -1.33 | -0.30 | -1.02 | 77% |
| GraphCast 2018 | IMD rule (normal + 4.5) | 3 | 4 | -1.68 | -0.66 | -1.02 | 61% |
| GraphCast 2018 | IMD rule (normal + 4.5) | 4 | 4 | -1.90 | -0.88 | -1.02 | 54% |
| GraphCast 2018 | IMD rule (normal + 4.5) | 5 | 4 | -2.03 | -1.00 | -1.02 | 51% |
| C3-hop on GraphCast 2018's rows | label rule (normal + 3) | all | 90 | -2.44 | -1.32 | -1.12 | 46% |
| C3-hop on GraphCast 2018's rows | label rule (normal + 3) | 1 | 18 | -1.76 | -0.64 | -1.12 | 64% |
| C3-hop on GraphCast 2018's rows | label rule (normal + 3) | 2 | 18 | -2.22 | -1.10 | -1.12 | 50% |
| C3-hop on GraphCast 2018's rows | label rule (normal + 3) | 3 | 18 | -2.53 | -1.41 | -1.12 | 44% |
| C3-hop on GraphCast 2018's rows | label rule (normal + 3) | 4 | 18 | -2.67 | -1.54 | -1.12 | 42% |
| C3-hop on GraphCast 2018's rows | label rule (normal + 3) | 5 | 18 | -3.01 | -1.89 | -1.12 | 37% |
| C3-hop on GraphCast 2018's rows | IMD rule (normal + 4.5) | all | 20 | -2.63 | -1.60 | -1.02 | 39% |
| C3-hop on GraphCast 2018's rows | IMD rule (normal + 4.5) | 1 | 4 | -2.08 | -1.05 | -1.02 | 49% |
| C3-hop on GraphCast 2018's rows | IMD rule (normal + 4.5) | 2 | 4 | -2.73 | -1.71 | -1.02 | 37% |
| C3-hop on GraphCast 2018's rows | IMD rule (normal + 4.5) | 3 | 4 | -2.95 | -1.92 | -1.02 | 35% |
| C3-hop on GraphCast 2018's rows | IMD rule (normal + 4.5) | 4 | 4 | -2.52 | -1.49 | -1.02 | 41% |
| C3-hop on GraphCast 2018's rows | IMD rule (normal + 4.5) | 5 | 4 | -2.87 | -1.85 | -1.02 | 36% |

## Palam, hourly maximum

The record used is `tmax_hourly`, from 2002-06-24: the highest of the IST day's hourly readings (at least 18 of the 24, including all six from 11:30 to 16:30 IST). Palam's maximum-thermometer readings are almost all 24-hour ones, which part (B) of the deviation dropped, so only the hourly maximum is left. It is like-for-like with ERA5's daily maximum, but it can miss a peak between two readings, so it reads a little below a thermometer maximum. The station's normal and thresholds come from the same series. Descriptive, as registered.

### Station data

| Period | Usable days | In the heat season | Heatwave days, label rule | Heatwave days, IMD rule | Last day |
|---|---|---|---|---|---|
| 1980-2018 | 3213 | 1328 | 188 | 99 | 2018-12-31 |
| 2019-2025 | 2261 | 901 | 107 | 50 | 2025-08-24 |

A usable day has a value in the record used. Gaps are not filled; a day without a station value is left out of every comparison. A station heatwave day is a single in-season day at or above min(45, max(40, normal + 3)) (label rule) or min(45, max(40, normal + 4.5)) (IMD rule), with the station's own 1980-2018 normal.

### Exploratory: are the rebuilt dates right? (not registered)

The check that found the GHCN file's fault, run on this record. It uses the station and ERA5 only, no forecast. For the station value dated D: the mean and SD of ERA5 - station when ERA5 is taken on the same day D, on the day before, or as the larger of the two. The smallest SD shows which day the station value belongs to.

| Years | Days with a value | Same day: mean / SD | Day before: mean / SD | Larger of the two: mean / SD | Reporting days cooler than the day before (all days) | Consecutive values that are identical |
|---|---|---|---|---|---|---|
| 1980-2008 | 10% | -0.65 / 1.27 | -0.61 / 2.13 | -0.03 / 1.73 | 47% (46%) | 28% |
| 2009-2010 | 46% | +0.49 / 1.08 | +0.57 / 2.09 | +1.17 / 1.58 | 46% (45%) | 27% |
| 2011-2015 | 44% | -0.15 / 1.14 | -0.20 / 2.16 | +0.46 / 1.61 | 45% (45%) | 24% |
| 2016-2018 | 93% | -0.66 / 1.16 | -0.67 / 2.10 | -0.06 / 1.64 | 48% (48%) | 12% |
| 2019-2025 | 93% | +0.08 / 1.07 | +0.07 / 2.03 | +0.71 / 1.54 | 45% (46%) | 13% |

If the dates are right, the same day has the smallest SD in every block with values. Identical consecutive values stay rare too, unless the readings are in whole degrees (values in whole degrees: 1980-2008 100%; 2009-2010 100%; 2011-2015 99%; 2016-2018 63%; 2019-2025 64%).

### (i) ERA5 against the station

| Period | Days | ERA5 - station | RMSE | After the mapping: bias | RMSE |
|---|---|---|---|---|---|
| 1980-2018, all days | 3213 | -0.41 | 1.306 | +0.00 | 1.144 |
| 1980-2018, heat season | 1328 | -0.68 | 1.403 | +0.01 | 1.193 |
| 1980-2018, station heatwave days, label rule | 188 | -1.58 | 1.961 | -0.59 | 1.333 |
| 1980-2018, station heatwave days, IMD rule | 99 | -1.79 | 2.200 | -0.78 | 1.528 |
| 1980-2018, ERA5 heatwave days (mirror view) | 121 | -0.87 | 1.557 | +0.20 | 1.311 |
| 2019-2025, all days | 2261 | +0.08 | 1.075 | +0.41 | 1.123 |
| 2019-2025, heat season | 901 | -0.31 | 1.022 | +0.35 | 1.052 |
| 2019-2025, station heatwave days, label rule | 107 | -0.85 | 1.144 | +0.25 | 0.866 |
| 2019-2025, station heatwave days, IMD rule | 50 | -0.82 | 1.144 | +0.33 | 0.928 |
| 2019-2025, ERA5 heatwave days (mirror view) | 105 | -0.46 | 0.903 | +0.66 | 1.017 |

The mapping is fitted on 1980-2018, so its 1980-2018 rows are in-sample. Picking days by one record's heat makes the other record look colder on those days (selection), so the station-heatwave rows and the mirror row are read together.

**Do the heatwave labels agree?** In-season days with a station value. Events = station heatwave days; 'warnings' = ERA5 heatwave days.

| Period | Station rule | ERA5 side | Both | Station only | ERA5 only | Station days ERA5 also has | ERA5 days the station lacks | ETS | ERA5 days / station days |
|---|---|---|---|---|---|---|---|---|---|
| 1980-2018 (1328 days) | label rule (normal + 3) | ERA5 label | 101 | 87 | 20 | 54% | 17% | 0.44 | 0.64 |
| 1980-2018 (1328 days) | label rule (normal + 3) | mapped ERA5 vs the station threshold | 124 | 64 | 32 | 66% | 21% | 0.51 | 0.83 |
| 1980-2018 (1328 days) | IMD rule (normal + 4.5) | ERA5 label | 69 | 30 | 52 | 70% | 43% | 0.42 | 1.22 |
| 1980-2018 (1328 days) | IMD rule (normal + 4.5) | mapped ERA5 vs the station threshold | 54 | 45 | 16 | 55% | 23% | 0.44 | 0.71 |
| 2019-2025 (901 days) | label rule (normal + 3) | ERA5 label | 84 | 23 | 21 | 79% | 20% | 0.62 | 0.98 |
| 2019-2025 (901 days) | label rule (normal + 3) | mapped ERA5 vs the station threshold | 96 | 11 | 39 | 90% | 29% | 0.62 | 1.26 |
| 2019-2025 (901 days) | IMD rule (normal + 4.5) | ERA5 label | 45 | 5 | 60 | 90% | 57% | 0.38 | 2.10 |
| 2019-2025 (901 days) | IMD rule (normal + 4.5) | mapped ERA5 vs the station threshold | 45 | 5 | 21 | 90% | 32% | 0.61 | 1.32 |

### (iv) The station mapping

station = a + b x ERA5, least squares per calendar month on the usable station days of 1980-2018.

| Month | a | b | Days |
|---|---|---|---|
| 1 | -3.93 | 1.173 | 215 |
| 2 | -1.59 | 1.075 | 200 |
| 3 | -0.48 | 1.031 | 234 |
| 4 | -3.31 | 1.105 | 282 |
| 5 | -4.55 | 1.132 | 295 |
| 6 | -1.71 | 1.067 | 284 |
| 7 | +0.92 | 0.994 | 332 |
| 8 | +2.28 | 0.955 | 327 |
| 9 | +2.78 | 0.937 | 273 |
| 10 | -0.34 | 1.020 | 286 |
| 11 | -1.08 | 1.034 | 241 |
| 12 | -2.95 | 1.113 | 244 |

### (ii) Our test forecasts against the station

Target days 2019-01-01 to 2025-08-24 with a station value: 11295 rows (query date x lead), 21 year x season clusters. The networks are shown as the 10-seed ensemble mean and as the mean of the 10 seeds' RMSE (the test's primary score).

**All days**

| Forecast | Rows | RMSE vs ERA5 | RMSE vs station | RMSE station-mapped | Bias vs ERA5 | Bias vs station | Bias station-mapped |
|---|---|---|---|---|---|---|---|
| C3-hop (ensemble mean) | 11295 | 2.138 | 2.368 | 2.363 | +0.17 | +0.26 | +0.61 |
| C3-hop (mean of 10 seeds) | 11295 | 2.186 | 2.411 | 2.413 | +0.17 | +0.26 | +0.61 |
| Delhi-only LSTM (ensemble mean) | 11295 | 2.274 | 2.500 | 2.520 | +0.21 | +0.30 | +0.66 |
| Delhi-only LSTM (mean of 10 seeds) | 11295 | 2.314 | 2.536 | 2.562 | +0.21 | +0.30 | +0.66 |
| ridge_hw5 | 11295 | 2.272 | 2.500 | 2.548 | +0.35 | +0.43 | +0.78 |

**Heat season (15 Mar - 31 Jul)**

| Forecast | Rows | RMSE vs ERA5 | RMSE vs station | RMSE station-mapped | Bias vs ERA5 | Bias vs station | Bias station-mapped |
|---|---|---|---|---|---|---|---|
| C3-hop (ensemble mean) | 4505 | 2.410 | 2.625 | 2.693 | +0.23 | -0.08 | +0.60 |
| C3-hop (mean of 10 seeds) | 4505 | 2.457 | 2.668 | 2.741 | +0.23 | -0.08 | +0.60 |
| Delhi-only LSTM (ensemble mean) | 4505 | 2.661 | 2.880 | 2.991 | +0.36 | +0.05 | +0.75 |
| Delhi-only LSTM (mean of 10 seeds) | 4505 | 2.711 | 2.927 | 3.043 | +0.36 | +0.05 | +0.75 |
| ridge_hw5 | 4505 | 2.592 | 2.797 | 2.896 | +0.31 | +0.00 | +0.68 |

**Station heatwave days, label rule**

| Forecast | Rows | RMSE vs ERA5 | RMSE vs station | RMSE station-mapped | Bias vs ERA5 | Bias vs station | Bias station-mapped |
|---|---|---|---|---|---|---|---|
| C3-hop (ensemble mean) | 535 | 2.070 | 2.814 | 2.059 | -1.63 | -2.48 | -1.52 |
| C3-hop (mean of 10 seeds) | 535 | 2.122 | 2.851 | 2.120 | -1.63 | -2.48 | -1.52 |
| Delhi-only LSTM (ensemble mean) | 535 | 2.393 | 3.128 | 2.420 | -1.83 | -2.68 | -1.73 |
| Delhi-only LSTM (mean of 10 seeds) | 535 | 2.448 | 3.169 | 2.487 | -1.83 | -2.68 | -1.73 |
| ridge_hw5 | 535 | 2.442 | 3.184 | 2.460 | -1.95 | -2.80 | -1.86 |

**Station heatwave days, IMD rule**

| Forecast | Rows | RMSE vs ERA5 | RMSE vs station | RMSE station-mapped | Bias vs ERA5 | Bias vs station | Bias station-mapped |
|---|---|---|---|---|---|---|---|
| C3-hop (ensemble mean) | 250 | 2.317 | 3.093 | 2.279 | -1.99 | -2.81 | -1.83 |
| C3-hop (mean of 10 seeds) | 250 | 2.360 | 3.125 | 2.331 | -1.99 | -2.81 | -1.83 |
| Delhi-only LSTM (ensemble mean) | 250 | 2.635 | 3.426 | 2.658 | -2.24 | -3.06 | -2.09 |
| Delhi-only LSTM (mean of 10 seeds) | 250 | 2.673 | 3.455 | 2.705 | -2.24 | -3.06 | -2.09 |
| ridge_hw5 | 250 | 2.776 | 3.538 | 2.748 | -2.39 | -3.21 | -2.26 |

**ERA5 heatwave days (the test's extreme stratum)**

| Forecast | Rows | RMSE vs ERA5 | RMSE vs station | RMSE station-mapped | Bias vs ERA5 | Bias vs station | Bias station-mapped |
|---|---|---|---|---|---|---|---|
| C3-hop (ensemble mean) | 470 | 2.202 | 2.671 | 1.923 | -1.85 | -2.31 | -1.33 |
| C3-hop (mean of 10 seeds) | 470 | 2.249 | 2.710 | 1.989 | -1.85 | -2.31 | -1.33 |
| Delhi-only LSTM (ensemble mean) | 470 | 2.479 | 2.948 | 2.257 | -1.98 | -2.44 | -1.47 |
| Delhi-only LSTM (mean of 10 seeds) | 470 | 2.530 | 2.992 | 2.328 | -1.98 | -2.44 | -1.47 |
| ridge_hw5 | 470 | 2.537 | 3.023 | 2.307 | -2.12 | -2.58 | -1.63 |

**By lead, all days (ensemble means)**

| Forecast | Score | Day 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| C3-hop | RMSE vs station | 1.809 | 2.188 | 2.406 | 2.580 | 2.745 |
| C3-hop | Bias vs station | +0.10 | +0.09 | +0.25 | +0.33 | +0.52 |
| C3-hop | RMSE station-mapped vs station | 1.768 | 2.149 | 2.397 | 2.586 | 2.786 |
| C3-hop | Bias station-mapped vs station | +0.44 | +0.43 | +0.60 | +0.69 | +0.90 |
| Delhi-only LSTM | RMSE vs station | 1.941 | 2.341 | 2.570 | 2.720 | 2.829 |
| Delhi-only LSTM | Bias vs station | +0.14 | +0.26 | +0.30 | +0.36 | +0.42 |
| Delhi-only LSTM | RMSE station-mapped vs station | 1.926 | 2.353 | 2.590 | 2.756 | 2.866 |
| Delhi-only LSTM | Bias station-mapped vs station | +0.48 | +0.61 | +0.66 | +0.73 | +0.80 |
| ridge_hw5 | RMSE vs station | 1.833 | 2.316 | 2.573 | 2.766 | 2.872 |
| ridge_hw5 | Bias vs station | +0.19 | +0.34 | +0.46 | +0.56 | +0.60 |
| ridge_hw5 | RMSE station-mapped vs station | 1.884 | 2.365 | 2.622 | 2.817 | 2.920 |
| ridge_hw5 | Bias station-mapped vs station | +0.53 | +0.69 | +0.81 | +0.92 | +0.97 |

**By lead, station heatwave days, label rule (ensemble means)**

| Forecast | Score | Day 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| C3-hop | RMSE vs station | 1.946 | 2.687 | 2.908 | 3.143 | 3.203 |
| C3-hop | Bias vs station | -1.66 | -2.38 | -2.60 | -2.86 | -2.90 |
| C3-hop | RMSE station-mapped vs station | 1.226 | 1.911 | 2.133 | 2.358 | 2.436 |
| C3-hop | Bias station-mapped vs station | -0.63 | -1.41 | -1.65 | -1.94 | -1.97 |
| Delhi-only LSTM | RMSE vs station | 2.139 | 2.792 | 3.195 | 3.492 | 3.762 |
| Delhi-only LSTM | Bias vs station | -1.77 | -2.33 | -2.77 | -3.11 | -3.42 |
| Delhi-only LSTM | RMSE station-mapped vs station | 1.453 | 2.103 | 2.474 | 2.750 | 3.013 |
| Delhi-only LSTM | Bias station-mapped vs station | -0.75 | -1.36 | -1.83 | -2.19 | -2.53 |
| ridge_hw5 | RMSE vs station | 1.857 | 2.672 | 3.132 | 3.654 | 4.118 |
| ridge_hw5 | Bias vs station | -1.56 | -2.31 | -2.85 | -3.38 | -3.88 |
| ridge_hw5 | RMSE station-mapped vs station | 1.178 | 1.933 | 2.352 | 2.882 | 3.360 |
| ridge_hw5 | Bias station-mapped vs station | -0.52 | -1.33 | -1.92 | -2.50 | -3.04 |

**Paired tests** (delta = RMSE of the first minus RMSE of the second, same rows; negative = the first is better; paired cluster jackknife).

| Comparison | Stratum | Delta vs ERA5 [95% CI] | Delta vs station [95% CI] | Delta station-mapped [95% CI] | Clear change of order |
|---|---|---|---|---|---|
| C3-hop minus Delhi-only LSTM (10 seeds each) | All days | -0.128 [-0.214, -0.043] | -0.125 [-0.215, -0.036] | -0.149 [-0.256, -0.043] | no |
| C3-hop minus Delhi-only LSTM (10 seeds each) | Heat season (15 Mar - 31 Jul) | -0.254 [-0.369, -0.140], fragile (< 10 clusters) | -0.259 [-0.376, -0.142], fragile (< 10 clusters) | -0.302 [-0.445, -0.159], fragile (< 10 clusters) | no |
| C3-hop minus Delhi-only LSTM (10 seeds each) | Station heatwave days, label rule | -0.326 [-0.791, +0.138], fragile (< 10 clusters) | -0.318 [-0.785, +0.149], fragile (< 10 clusters) | -0.366 [-0.836, +0.103], fragile (< 10 clusters) | no |
| C3-hop ensemble minus ridge_hw5 | All days | -0.134 [-0.183, -0.086] | -0.132 [-0.190, -0.074] | -0.185 [-0.236, -0.133] | no |
| C3-hop ensemble minus ridge_hw5 | Heat season (15 Mar - 31 Jul) | -0.182 [-0.253, -0.111], fragile (< 10 clusters) | -0.171 [-0.260, -0.083], fragile (< 10 clusters) | -0.202 [-0.280, -0.125], fragile (< 10 clusters) | no |
| C3-hop ensemble minus ridge_hw5 | Station heatwave days, label rule | -0.372 [-0.607, -0.137], fragile (< 10 clusters) | -0.370 [-0.653, -0.088], fragile (< 10 clusters) | -0.401 [-0.694, -0.109], fragile (< 10 clusters) | no |

**Yes/no warnings** (in-season rows). R-point and R-prob use the forecast as issued, against the ERA5 label threshold. Station-mapped R-point uses the mapped forecast against the station's own threshold of that rule.

| Forecast | Events | Rule | Hits | Misses | False alarms | Hit rate | False-alarm ratio | ETS | Frequency bias |
|---|---|---|---|---|---|---|---|---|---|
| C3-hop (4505 rows) | station, label rule (normal + 3) (535) | R-point | 149 | 386 | 74 | 28% | 33% | 0.21 | 0.42 |
| C3-hop (4505 rows) | station, label rule (normal + 3) (535) | R-prob (p* = 0.37) | 170 | 365 | 90 | 32% | 35% | 0.23 | 0.49 |
| C3-hop (4505 rows) | station, label rule (normal + 3) (535) | station-mapped R-point | 236 | 299 | 167 | 44% | 41% | 0.29 | 0.75 |
| C3-hop (4505 rows) | station, IMD rule (normal + 4.5) (250) | R-point | 107 | 143 | 116 | 43% | 52% | 0.27 | 0.89 |
| C3-hop (4505 rows) | station, IMD rule (normal + 4.5) (250) | R-prob (p* = 0.37) | 113 | 137 | 147 | 45% | 57% | 0.26 | 1.04 |
| C3-hop (4505 rows) | station, IMD rule (normal + 4.5) (250) | station-mapped R-point | 54 | 196 | 37 | 22% | 41% | 0.17 | 0.36 |
| C3-hop (4505 rows) | ERA5, label rule (reference) (525) | R-point | 144 | 381 | 79 | 27% | 35% | 0.20 | 0.42 |
| C3-hop (4505 rows) | ERA5, label rule (reference) (525) | R-prob (p* = 0.37) | 165 | 360 | 95 | 31% | 37% | 0.23 | 0.50 |
| Delhi-only LSTM (4505 rows) | station, label rule (normal + 3) (535) | R-point | 149 | 386 | 165 | 28% | 53% | 0.17 | 0.59 |
| Delhi-only LSTM (4505 rows) | station, label rule (normal + 3) (535) | R-prob (p* = 0.24) | 295 | 240 | 384 | 55% | 57% | 0.26 | 1.27 |
| Delhi-only LSTM (4505 rows) | station, label rule (normal + 3) (535) | station-mapped R-point | 227 | 308 | 247 | 42% | 52% | 0.24 | 0.89 |
| Delhi-only LSTM (4505 rows) | station, IMD rule (normal + 4.5) (250) | R-point | 95 | 155 | 219 | 38% | 70% | 0.17 | 1.26 |
| Delhi-only LSTM (4505 rows) | station, IMD rule (normal + 4.5) (250) | R-prob (p* = 0.24) | 156 | 94 | 523 | 62% | 77% | 0.16 | 2.72 |
| Delhi-only LSTM (4505 rows) | station, IMD rule (normal + 4.5) (250) | station-mapped R-point | 59 | 191 | 64 | 24% | 52% | 0.17 | 0.49 |
| Delhi-only LSTM (4505 rows) | ERA5, label rule (reference) (525) | R-point | 158 | 367 | 156 | 30% | 50% | 0.19 | 0.60 |
| Delhi-only LSTM (4505 rows) | ERA5, label rule (reference) (525) | R-prob (p* = 0.24) | 295 | 230 | 384 | 56% | 57% | 0.26 | 1.29 |
| ridge_hw5 (4505 rows) | station, label rule (normal + 3) (535) | R-point | 121 | 414 | 80 | 23% | 40% | 0.16 | 0.38 |
| ridge_hw5 (4505 rows) | station, label rule (normal + 3) (535) | station-mapped R-point | 187 | 348 | 164 | 35% | 47% | 0.22 | 0.66 |
| ridge_hw5 (4505 rows) | station, IMD rule (normal + 4.5) (250) | R-point | 77 | 173 | 124 | 31% | 62% | 0.18 | 0.80 |
| ridge_hw5 (4505 rows) | station, IMD rule (normal + 4.5) (250) | station-mapped R-point | 49 | 201 | 34 | 20% | 41% | 0.16 | 0.33 |
| ridge_hw5 (4505 rows) | ERA5, label rule (reference) (525) | R-point | 121 | 404 | 80 | 23% | 40% | 0.17 | 0.38 |

**How much of the cold bias on station heatwave days is ERA5's own?** Forecast - station = (forecast - ERA5) + (ERA5 - station), means over the rows; ensemble means.

| Forecast | Station rule | Lead | Rows | Forecast - station | = forecast - ERA5 | + ERA5 - station | ERA5's share |
|---|---|---|---|---|---|---|---|
| C3-hop | label rule (normal + 3) | all | 535 | -2.48 | -1.63 | -0.85 | 34% |
| C3-hop | label rule (normal + 3) | 1 | 107 | -1.66 | -0.81 | -0.85 | 51% |
| C3-hop | label rule (normal + 3) | 2 | 107 | -2.38 | -1.53 | -0.85 | 36% |
| C3-hop | label rule (normal + 3) | 3 | 107 | -2.60 | -1.75 | -0.85 | 33% |
| C3-hop | label rule (normal + 3) | 4 | 107 | -2.86 | -2.02 | -0.85 | 30% |
| C3-hop | label rule (normal + 3) | 5 | 107 | -2.90 | -2.05 | -0.85 | 29% |
| C3-hop | IMD rule (normal + 4.5) | all | 250 | -2.81 | -1.99 | -0.82 | 29% |
| C3-hop | IMD rule (normal + 4.5) | 1 | 50 | -1.77 | -0.95 | -0.82 | 46% |
| C3-hop | IMD rule (normal + 4.5) | 2 | 50 | -2.66 | -1.84 | -0.82 | 31% |
| C3-hop | IMD rule (normal + 4.5) | 3 | 50 | -2.98 | -2.16 | -0.82 | 27% |
| C3-hop | IMD rule (normal + 4.5) | 4 | 50 | -3.33 | -2.51 | -0.82 | 25% |
| C3-hop | IMD rule (normal + 4.5) | 5 | 50 | -3.29 | -2.48 | -0.82 | 25% |
| Delhi-only LSTM | label rule (normal + 3) | all | 535 | -2.68 | -1.83 | -0.85 | 32% |
| Delhi-only LSTM | label rule (normal + 3) | 1 | 107 | -1.77 | -0.92 | -0.85 | 48% |
| Delhi-only LSTM | label rule (normal + 3) | 2 | 107 | -2.33 | -1.49 | -0.85 | 36% |
| Delhi-only LSTM | label rule (normal + 3) | 3 | 107 | -2.77 | -1.92 | -0.85 | 31% |
| Delhi-only LSTM | label rule (normal + 3) | 4 | 107 | -3.11 | -2.26 | -0.85 | 27% |
| Delhi-only LSTM | label rule (normal + 3) | 5 | 107 | -3.42 | -2.57 | -0.85 | 25% |
| Delhi-only LSTM | IMD rule (normal + 4.5) | all | 250 | -3.06 | -2.24 | -0.82 | 27% |
| Delhi-only LSTM | IMD rule (normal + 4.5) | 1 | 50 | -2.00 | -1.18 | -0.82 | 41% |
| Delhi-only LSTM | IMD rule (normal + 4.5) | 2 | 50 | -2.69 | -1.87 | -0.82 | 30% |
| Delhi-only LSTM | IMD rule (normal + 4.5) | 3 | 50 | -3.22 | -2.40 | -0.82 | 25% |
| Delhi-only LSTM | IMD rule (normal + 4.5) | 4 | 50 | -3.56 | -2.74 | -0.82 | 23% |
| Delhi-only LSTM | IMD rule (normal + 4.5) | 5 | 50 | -3.81 | -2.99 | -0.82 | 22% |
| ridge_hw5 | label rule (normal + 3) | all | 535 | -2.80 | -1.95 | -0.85 | 30% |
| ridge_hw5 | label rule (normal + 3) | 1 | 107 | -1.56 | -0.71 | -0.85 | 54% |
| ridge_hw5 | label rule (normal + 3) | 2 | 107 | -2.31 | -1.46 | -0.85 | 37% |
| ridge_hw5 | label rule (normal + 3) | 3 | 107 | -2.85 | -2.00 | -0.85 | 30% |
| ridge_hw5 | label rule (normal + 3) | 4 | 107 | -3.38 | -2.54 | -0.85 | 25% |
| ridge_hw5 | label rule (normal + 3) | 5 | 107 | -3.88 | -3.03 | -0.85 | 22% |
| ridge_hw5 | IMD rule (normal + 4.5) | all | 250 | -3.21 | -2.39 | -0.82 | 26% |
| ridge_hw5 | IMD rule (normal + 4.5) | 1 | 50 | -1.66 | -0.84 | -0.82 | 49% |
| ridge_hw5 | IMD rule (normal + 4.5) | 2 | 50 | -2.69 | -1.87 | -0.82 | 30% |
| ridge_hw5 | IMD rule (normal + 4.5) | 3 | 50 | -3.35 | -2.53 | -0.82 | 24% |
| ridge_hw5 | IMD rule (normal + 4.5) | 4 | 50 | -3.95 | -3.13 | -0.82 | 21% |
| ridge_hw5 | IMD rule (normal + 4.5) | 5 | 50 | -4.41 | -3.59 | -0.82 | 19% |

### (iii) Global models and GraphCast 2018 against the station

Each outside forecast is paired with C3-hop's ensemble mean on the rows both have and the station observed. The global models are scored on the test period (to 2025-08-24, where the station record ends); GraphCast 2018 on fold f4's 2018 windows with C3-hop's development forecasts (2018 lies inside the mapping's fitting years).

**All days**

| Forecast | Target days | Rows | Clusters | RMSE vs ERA5: it / C3-hop | RMSE vs station: it / C3-hop | Station-mapped: it / C3-hop | Bias vs station: it / C3-hop |
|---|---|---|---|---|---|---|---|
| AIFS | 2024-03-15 to 2025-03-10 | 1650 | 3 | 2.121 / 1.824 | 2.115 / 2.011 | 2.126 / 1.982 | -1.41 / -0.07 |
| AIFS Single (added after the first results) | 2025-02-19 to 2025-08-24 | 860 | 3 | 1.889 / 2.222 | 1.802 / 2.416 | 1.626 / 2.618 | -0.99 / +0.38 |
| GFS | 2024-01-20 to 2025-08-24 | 2700 | 6 | 3.583 / 1.942 | 3.779 / 2.145 | 4.325 / 2.206 | +2.86 / +0.13 |
| GFS, Tmax since 2021 | 2021-03-25 to 2025-08-24 | 7447 | 14 | 3.287 / 2.127 | 3.470 / 2.316 | 3.951 / 2.360 | +2.63 / +0.30 |
| IFS | 2024-02-05 to 2025-08-24 | 2620 | 6 | 1.767 / 1.949 | 1.958 / 2.133 | 1.928 / 2.197 | -0.53 / +0.10 |
| IFS HRES 9 km, 12 UTC run | 2024-03-15 to 2025-08-24 | 2430 | 5 | 1.667 / 1.973 | 1.873 / 2.166 | 1.842 / 2.227 | -0.41 / +0.08 |
| GraphCast 2018 | 2018-01-01 to 2018-12-31 | 1660 | 3 | 1.162 / 1.945 | 1.700 / 2.254 | 1.500 / 2.189 | -0.83 / -0.55 |

**Heat season (15 Mar - 31 Jul)**

| Forecast | Target days | Rows | Clusters | RMSE vs ERA5: it / C3-hop | RMSE vs station: it / C3-hop | Station-mapped: it / C3-hop | Bias vs station: it / C3-hop |
|---|---|---|---|---|---|---|---|
| AIFS | 2024-03-15 to 2025-03-10 | 605 | 1 | 1.739 / 1.904 | 1.994 / 2.136 | 1.640 / 2.055 | -1.32 / -0.62 |
| AIFS Single (added after the first results) | 2025-02-19 to 2025-08-24 | 645 | 1 | 2.035 / 2.361 | 1.873 / 2.585 | 1.706 / 2.811 | -1.02 / +0.50 |
| GFS | 2024-01-20 to 2025-08-24 | 1260 | 2 | 4.122 / 2.147 | 4.257 / 2.372 | 5.096 / 2.468 | +3.37 / -0.03 |
| GFS, Tmax since 2021 | 2021-03-25 to 2025-08-24 | 3137 | 5 | 3.939 / 2.428 | 3.994 / 2.596 | 4.791 / 2.713 | +3.21 / +0.13 |
| IFS | 2024-02-05 to 2025-08-24 | 1260 | 2 | 1.657 / 2.147 | 1.947 / 2.372 | 2.065 / 2.468 | -0.21 / -0.03 |
| IFS HRES 9 km, 12 UTC run | 2024-03-15 to 2025-08-24 | 1250 | 2 | 1.533 / 2.152 | 1.861 / 2.379 | 1.946 / 2.474 | -0.16 / -0.04 |
| GraphCast 2018 | 2018-01-01 to 2018-12-31 | 605 | 1 | 1.207 / 2.247 | 2.004 / 2.578 | 1.673 / 2.448 | -1.27 / -0.93 |

**Station heatwave days, label rule**

| Forecast | Target days | Rows | Clusters | RMSE vs ERA5: it / C3-hop | RMSE vs station: it / C3-hop | Station-mapped: it / C3-hop | Bias vs station: it / C3-hop |
|---|---|---|---|---|---|---|---|
| AIFS | 2024-03-15 to 2025-03-10 | 105 | 1 | 2.416 / 2.196 | 2.722 / 2.465 | 1.971 / 1.628 | -2.33 / -2.21 |
| AIFS Single (added after the first results) | 2025-02-19 to 2025-08-24 | 20 | 1 | 2.581 / 2.675 | 2.965 / 3.080 | 2.052 / 2.308 | -2.92 / -2.69 |
| GFS | 2024-01-20 to 2025-08-24 | 125 | 2 | 2.337 / 2.279 | 2.047 / 2.574 | 3.389 / 1.755 | +1.71 / -2.28 |
| GFS, Tmax since 2021 | 2021-03-25 to 2025-08-24 | 325 | 4 | 2.269 / 2.170 | 1.923 / 2.530 | 3.124 / 1.775 | +1.61 / -2.21 |
| IFS | 2024-02-05 to 2025-08-24 | 125 | 2 | 1.283 / 2.279 | 1.473 / 2.574 | 1.589 / 1.755 | -0.61 / -2.28 |
| IFS HRES 9 km, 12 UTC run | 2024-03-15 to 2025-08-24 | 125 | 2 | 1.376 / 2.279 | 1.686 / 2.574 | 1.274 / 1.755 | -1.20 / -2.28 |
| GraphCast 2018 | 2018-01-01 to 2018-12-31 | 80 | 1 | 0.947 / 1.906 | 2.721 / 3.597 | 2.010 / 2.822 | -2.37 / -3.44 |

**Station heatwave days, IMD rule**

| Forecast | Target days | Rows | Clusters | RMSE vs ERA5: it / C3-hop | RMSE vs station: it / C3-hop | Station-mapped: it / C3-hop | Bias vs station: it / C3-hop |
|---|---|---|---|---|---|---|---|
| AIFS | 2024-03-15 to 2025-03-10 | 70 | 1 | 2.482 / 2.437 | 2.718 / 2.622 | 1.988 / 1.762 | -2.30 / -2.36 |
| AIFS Single (added after the first results) | 2025-02-19 to 2025-08-24 | 0 | 0 | n/a / n/a | n/a / n/a | n/a / n/a | n/a / n/a |
| GFS | 2024-01-20 to 2025-08-24 | 70 | 1 | 2.046 / 2.437 | 1.932 / 2.622 | 3.301 / 1.762 | +1.48 / -2.36 |
| GFS, Tmax since 2021 | 2021-03-25 to 2025-08-24 | 160 | 3 | 2.184 / 2.438 | 1.916 / 2.752 | 3.137 / 1.974 | +1.56 / -2.46 |
| IFS | 2024-02-05 to 2025-08-24 | 70 | 1 | 0.701 / 2.437 | 0.871 / 2.622 | 1.479 / 1.762 | -0.18 / -2.36 |
| IFS HRES 9 km, 12 UTC run | 2024-03-15 to 2025-08-24 | 70 | 1 | 0.962 / 2.437 | 1.264 / 2.622 | 0.905 / 1.762 | -0.98 / -2.36 |
| GraphCast 2018 | 2018-01-01 to 2018-12-31 | 50 | 1 | 1.120 / 1.785 | 3.136 / 3.768 | 2.376 / 2.965 | -2.91 / -3.65 |

**ERA5 heatwave days (the test's extreme stratum)**

| Forecast | Target days | Rows | Clusters | RMSE vs ERA5: it / C3-hop | RMSE vs station: it / C3-hop | Station-mapped: it / C3-hop | Bias vs station: it / C3-hop |
|---|---|---|---|---|---|---|---|
| AIFS | 2024-03-15 to 2025-03-10 | 120 | 1 | 2.524 / 2.165 | 2.758 / 2.393 | 2.041 / 1.608 | -2.31 / -2.09 |
| AIFS Single (added after the first results) | 2025-02-19 to 2025-08-24 | 30 | 1 | 2.322 / 2.868 | 2.447 / 2.964 | 1.658 / 2.281 | -2.10 / -2.40 |
| GFS | 2024-01-20 to 2025-08-24 | 150 | 2 | 2.434 / 2.323 | 2.283 / 2.518 | 3.638 / 1.764 | +1.92 / -2.15 |
| GFS, Tmax since 2021 | 2021-03-25 to 2025-08-24 | 355 | 5 | 2.220 / 2.287 | 2.079 / 2.492 | 3.277 / 1.781 | +1.73 / -2.13 |
| IFS | 2024-02-05 to 2025-08-24 | 150 | 2 | 1.360 / 2.323 | 1.486 / 2.518 | 1.672 / 1.764 | -0.51 / -2.15 |
| IFS HRES 9 km, 12 UTC run | 2024-03-15 to 2025-08-24 | 150 | 2 | 1.445 / 2.323 | 1.657 / 2.518 | 1.316 / 1.764 | -1.11 / -2.15 |
| GraphCast 2018 | 2018-01-01 to 2018-12-31 | 20 | 1 | 1.049 / 1.919 | 2.994 / 3.993 | 2.079 / 3.076 | -2.80 / -3.89 |

**Paired tests** (delta = RMSE of the outside forecast minus RMSE of C3-hop; negative = the outside forecast is better).

| Comparison | Stratum | Delta vs ERA5 [95% CI] | Delta vs station [95% CI] | Delta station-mapped [95% CI] | Clear change of order |
|---|---|---|---|---|---|
| AIFS minus C3-hop | All days | +0.296 no CI (< 5 clusters), fragile (< 10 clusters) | +0.104 no CI (< 5 clusters), fragile (< 10 clusters) | +0.144 no CI (< 5 clusters), fragile (< 10 clusters) | no |
| AIFS minus C3-hop | Heat season (15 Mar - 31 Jul) | not testable | not testable | not testable | no |
| AIFS minus C3-hop | Station heatwave days, label rule | not testable | not testable | not testable | no |
| AIFS Single (added after the first results) minus C3-hop | All days | -0.333 no CI (< 5 clusters), fragile (< 10 clusters) | -0.614 no CI (< 5 clusters), fragile (< 10 clusters) | -0.992 no CI (< 5 clusters), fragile (< 10 clusters) | no |
| AIFS Single (added after the first results) minus C3-hop | Heat season (15 Mar - 31 Jul) | not testable | not testable | not testable | no |
| AIFS Single (added after the first results) minus C3-hop | Station heatwave days, label rule | not testable | not testable | not testable | no |
| GFS minus C3-hop | All days | +1.641 [+0.979, +2.303], fragile (< 10 clusters) | +1.634 [+1.069, +2.199], fragile (< 10 clusters) | +2.119 [+1.248, +2.990], fragile (< 10 clusters) | no |
| GFS minus C3-hop | Heat season (15 Mar - 31 Jul) | +1.975 no CI (< 5 clusters), fragile (< 10 clusters) | +1.886 no CI (< 5 clusters), fragile (< 10 clusters) | +2.627 no CI (< 5 clusters), fragile (< 10 clusters) | no |
| GFS minus C3-hop | Station heatwave days, label rule | +0.058 no CI (< 5 clusters), fragile (< 10 clusters) | -0.527 no CI (< 5 clusters), fragile (< 10 clusters) | +1.635 no CI (< 5 clusters), fragile (< 10 clusters) | no |
| GFS, Tmax since 2021 minus C3-hop | All days | +1.160 [+0.775, +1.545] | +1.154 [+0.791, +1.518] | +1.591 [+1.106, +2.076] | no |
| GFS, Tmax since 2021 minus C3-hop | Heat season (15 Mar - 31 Jul) | +1.511 [+0.925, +2.097], fragile (< 10 clusters) | +1.398 [+0.794, +2.002], fragile (< 10 clusters) | +2.079 [+1.388, +2.769], fragile (< 10 clusters) | no |
| GFS, Tmax since 2021 minus C3-hop | Station heatwave days, label rule | +0.098 no CI (< 5 clusters), fragile (< 10 clusters) | -0.607 no CI (< 5 clusters), fragile (< 10 clusters) | +1.348 no CI (< 5 clusters), fragile (< 10 clusters) | no |
| IFS minus C3-hop | All days | -0.183 [-0.742, +0.376], fragile (< 10 clusters) | -0.175 [-0.629, +0.280], fragile (< 10 clusters) | -0.268 [-0.857, +0.321], fragile (< 10 clusters) | no |
| IFS minus C3-hop | Heat season (15 Mar - 31 Jul) | -0.490 no CI (< 5 clusters), fragile (< 10 clusters) | -0.425 no CI (< 5 clusters), fragile (< 10 clusters) | -0.403 no CI (< 5 clusters), fragile (< 10 clusters) | no |
| IFS minus C3-hop | Station heatwave days, label rule | -0.997 no CI (< 5 clusters), fragile (< 10 clusters) | -1.101 no CI (< 5 clusters), fragile (< 10 clusters) | -0.166 no CI (< 5 clusters), fragile (< 10 clusters) | no |
| IFS HRES 9 km, 12 UTC run minus C3-hop | All days | -0.306 [-0.888, +0.276], fragile (< 10 clusters) | -0.293 [-0.735, +0.149], fragile (< 10 clusters) | -0.385 [-0.949, +0.178], fragile (< 10 clusters) | no |
| IFS HRES 9 km, 12 UTC run minus C3-hop | Heat season (15 Mar - 31 Jul) | -0.619 no CI (< 5 clusters), fragile (< 10 clusters) | -0.518 no CI (< 5 clusters), fragile (< 10 clusters) | -0.528 no CI (< 5 clusters), fragile (< 10 clusters) | no |
| IFS HRES 9 km, 12 UTC run minus C3-hop | Station heatwave days, label rule | -0.903 no CI (< 5 clusters), fragile (< 10 clusters) | -0.888 no CI (< 5 clusters), fragile (< 10 clusters) | -0.480 no CI (< 5 clusters), fragile (< 10 clusters) | no |
| GraphCast 2018 minus C3-hop | All days | -0.783 no CI (< 5 clusters), fragile (< 10 clusters) | -0.555 no CI (< 5 clusters), fragile (< 10 clusters) | -0.689 no CI (< 5 clusters), fragile (< 10 clusters) | no |
| GraphCast 2018 minus C3-hop | Heat season (15 Mar - 31 Jul) | not testable | not testable | not testable | no |
| GraphCast 2018 minus C3-hop | Station heatwave days, label rule | not testable | not testable | not testable | no |

**By lead, all days**

| Forecast | Score | Day 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| AIFS | RMSE vs station | 1.873 | 2.035 | 2.137 | 2.216 | 2.291 |
| C3-hop on AIFS's rows | RMSE vs station | 1.604 | 1.895 | 2.072 | 2.202 | 2.222 |
| AIFS Single (added after the first results) | RMSE vs station | 1.412 | 1.737 | 1.903 | 1.924 | 1.981 |
| C3-hop on AIFS Single's rows | RMSE vs station | 1.859 | 2.208 | 2.397 | 2.673 | 2.834 |
| GFS | RMSE vs station | 3.268 | 3.679 | 3.888 | 3.985 | 4.026 |
| C3-hop on GFS's rows | RMSE vs station | 1.686 | 1.992 | 2.173 | 2.354 | 2.438 |
| GFS, Tmax since 2021 | RMSE vs station | 3.087 | 3.333 | 3.530 | 3.630 | 3.733 |
| C3-hop on GFS, Tmax since 2021's rows | RMSE vs station | 1.732 | 2.113 | 2.353 | 2.547 | 2.709 |
| IFS | RMSE vs station | 1.650 | 1.774 | 1.930 | 2.085 | 2.289 |
| C3-hop on IFS's rows | RMSE vs station | 1.655 | 1.976 | 2.161 | 2.344 | 2.438 |
| IFS HRES 9 km, 12 UTC run | RMSE vs station | 1.549 | 1.727 | 1.849 | 1.978 | 2.202 |
| C3-hop on IFS HRES 9 km, 12 UTC run's rows | RMSE vs station | 1.688 | 2.021 | 2.198 | 2.376 | 2.464 |
| GraphCast 2018 | RMSE vs station | 1.554 | 1.679 | 1.677 | 1.749 | 1.828 |
| C3-hop on GraphCast 2018's rows | RMSE vs station | 1.805 | 2.090 | 2.300 | 2.457 | 2.542 |

**Yes/no warnings** (in-season rows; R-point only, no probabilities for these comparisons).

| Forecast | Events | Rule | Hits | Misses | False alarms | Hit rate | False-alarm ratio | ETS | Frequency bias |
|---|---|---|---|---|---|---|---|---|---|
| AIFS (605 rows) | station, label rule (normal + 3) (105) | R-point | 56 | 49 | 7 | 53% | 11% | 0.45 | 0.60 |
| AIFS (605 rows) | station, label rule (normal + 3) (105) | station-mapped R-point | 66 | 39 | 7 | 63% | 10% | 0.54 | 0.70 |
| AIFS (605 rows) | station, IMD rule (normal + 4.5) (70) | R-point | 46 | 24 | 17 | 66% | 27% | 0.49 | 0.90 |
| AIFS (605 rows) | station, IMD rule (normal + 4.5) (70) | station-mapped R-point | 31 | 39 | 5 | 44% | 14% | 0.38 | 0.51 |
| AIFS (605 rows) | ERA5, label rule (reference) (120) | R-point | 61 | 59 | 2 | 51% | 3% | 0.44 | 0.53 |
| C3-hop on AIFS's rows (605 rows) | station, label rule (normal + 3) (105) | R-point | 56 | 49 | 19 | 53% | 25% | 0.39 | 0.71 |
| C3-hop on AIFS's rows (605 rows) | station, label rule (normal + 3) (105) | station-mapped R-point | 71 | 34 | 30 | 68% | 30% | 0.46 | 0.96 |
| C3-hop on AIFS's rows (605 rows) | station, IMD rule (normal + 4.5) (70) | R-point | 49 | 21 | 26 | 70% | 35% | 0.46 | 1.07 |
| C3-hop on AIFS's rows (605 rows) | station, IMD rule (normal + 4.5) (70) | station-mapped R-point | 21 | 49 | 9 | 30% | 30% | 0.23 | 0.43 |
| C3-hop on AIFS's rows (605 rows) | ERA5, label rule (reference) (120) | R-point | 60 | 60 | 15 | 50% | 20% | 0.38 | 0.62 |
| AIFS Single (added after the first results) (645 rows) | station, label rule (normal + 3) (20) | R-point | 0 | 20 | 0 | 0% | n/a | 0.00 | 0.00 |
| AIFS Single (added after the first results) (645 rows) | station, label rule (normal + 3) (20) | station-mapped R-point | 0 | 20 | 3 | 0% | 100% | -0.00 | 0.15 |
| AIFS Single (added after the first results) (645 rows) | station, IMD rule (normal + 4.5) (0) | R-point | 0 | 0 | 0 | n/a | n/a | n/a | n/a |
| AIFS Single (added after the first results) (645 rows) | station, IMD rule (normal + 4.5) (0) | station-mapped R-point | 0 | 0 | 1 | n/a | 100% | 0.00 | n/a |
| AIFS Single (added after the first results) (645 rows) | ERA5, label rule (reference) (40) | R-point | 0 | 40 | 0 | 0% | n/a | 0.00 | 0.00 |
| C3-hop on AIFS Single's rows (645 rows) | station, label rule (normal + 3) (20) | R-point | 1 | 19 | 3 | 5% | 75% | 0.04 | 0.20 |
| C3-hop on AIFS Single's rows (645 rows) | station, label rule (normal + 3) (20) | station-mapped R-point | 4 | 16 | 25 | 20% | 86% | 0.07 | 1.45 |
| C3-hop on AIFS Single's rows (645 rows) | station, IMD rule (normal + 4.5) (0) | R-point | 0 | 0 | 4 | n/a | 100% | 0.00 | n/a |
| C3-hop on AIFS Single's rows (645 rows) | station, IMD rule (normal + 4.5) (0) | station-mapped R-point | 0 | 0 | 0 | n/a | n/a | n/a | n/a |
| C3-hop on AIFS Single's rows (645 rows) | ERA5, label rule (reference) (40) | R-point | 2 | 38 | 2 | 5% | 50% | 0.04 | 0.10 |
| GFS (1260 rows) | station, label rule (normal + 3) (125) | R-point | 125 | 0 | 391 | 100% | 76% | 0.16 | 4.13 |
| GFS (1260 rows) | station, label rule (normal + 3) (125) | station-mapped R-point | 125 | 0 | 470 | 100% | 79% | 0.12 | 4.76 |
| GFS (1260 rows) | station, IMD rule (normal + 4.5) (70) | R-point | 70 | 0 | 446 | 100% | 86% | 0.08 | 7.37 |
| GFS (1260 rows) | station, IMD rule (normal + 4.5) (70) | station-mapped R-point | 70 | 0 | 395 | 100% | 85% | 0.10 | 6.64 |
| GFS (1260 rows) | ERA5, label rule (reference) (160) | R-point | 160 | 0 | 356 | 100% | 69% | 0.21 | 3.23 |
| C3-hop on GFS's rows (1260 rows) | station, label rule (normal + 3) (125) | R-point | 57 | 68 | 22 | 46% | 28% | 0.35 | 0.63 |
| C3-hop on GFS's rows (1260 rows) | station, label rule (normal + 3) (125) | station-mapped R-point | 75 | 50 | 55 | 60% | 42% | 0.37 | 1.04 |
| C3-hop on GFS's rows (1260 rows) | station, IMD rule (normal + 4.5) (70) | R-point | 49 | 21 | 30 | 70% | 38% | 0.47 | 1.13 |
| C3-hop on GFS's rows (1260 rows) | station, IMD rule (normal + 4.5) (70) | station-mapped R-point | 21 | 49 | 9 | 30% | 30% | 0.25 | 0.43 |
| C3-hop on GFS's rows (1260 rows) | ERA5, label rule (reference) (160) | R-point | 62 | 98 | 17 | 39% | 22% | 0.31 | 0.49 |
| GFS, Tmax since 2021 (3137 rows) | station, label rule (normal + 3) (325) | R-point | 325 | 0 | 1003 | 100% | 76% | 0.16 | 4.09 |
| GFS, Tmax since 2021 (3137 rows) | station, label rule (normal + 3) (325) | station-mapped R-point | 325 | 0 | 1158 | 100% | 78% | 0.13 | 4.56 |
| GFS, Tmax since 2021 (3137 rows) | station, IMD rule (normal + 4.5) (160) | R-point | 160 | 0 | 1168 | 100% | 88% | 0.07 | 8.30 |
| GFS, Tmax since 2021 (3137 rows) | station, IMD rule (normal + 4.5) (160) | station-mapped R-point | 160 | 0 | 1010 | 100% | 86% | 0.09 | 7.31 |
| GFS, Tmax since 2021 (3137 rows) | ERA5, label rule (reference) (390) | R-point | 390 | 0 | 938 | 100% | 71% | 0.19 | 3.41 |
| C3-hop on GFS, Tmax since 2021's rows (3137 rows) | station, label rule (normal + 3) (325) | R-point | 108 | 217 | 64 | 33% | 37% | 0.24 | 0.53 |
| C3-hop on GFS, Tmax since 2021's rows (3137 rows) | station, label rule (normal + 3) (325) | station-mapped R-point | 165 | 160 | 142 | 51% | 46% | 0.31 | 0.94 |
| C3-hop on GFS, Tmax since 2021's rows (3137 rows) | station, IMD rule (normal + 4.5) (160) | R-point | 82 | 78 | 90 | 51% | 52% | 0.30 | 1.07 |
| C3-hop on GFS, Tmax since 2021's rows (3137 rows) | station, IMD rule (normal + 4.5) (160) | station-mapped R-point | 46 | 114 | 34 | 29% | 42% | 0.22 | 0.50 |
| C3-hop on GFS, Tmax since 2021's rows (3137 rows) | ERA5, label rule (reference) (390) | R-point | 115 | 275 | 57 | 29% | 33% | 0.22 | 0.44 |
| IFS (1260 rows) | station, label rule (normal + 3) (125) | R-point | 103 | 22 | 87 | 82% | 46% | 0.44 | 1.52 |
| IFS (1260 rows) | station, label rule (normal + 3) (125) | station-mapped R-point | 108 | 17 | 105 | 86% | 49% | 0.42 | 1.70 |
| IFS (1260 rows) | station, IMD rule (normal + 4.5) (70) | R-point | 70 | 0 | 120 | 100% | 63% | 0.33 | 2.71 |
| IFS (1260 rows) | station, IMD rule (normal + 4.5) (70) | station-mapped R-point | 69 | 1 | 76 | 99% | 52% | 0.44 | 2.07 |
| IFS (1260 rows) | ERA5, label rule (reference) (160) | R-point | 120 | 40 | 70 | 75% | 37% | 0.47 | 1.19 |
| C3-hop on IFS's rows (1260 rows) | station, label rule (normal + 3) (125) | R-point | 57 | 68 | 22 | 46% | 28% | 0.35 | 0.63 |
| C3-hop on IFS's rows (1260 rows) | station, label rule (normal + 3) (125) | station-mapped R-point | 75 | 50 | 55 | 60% | 42% | 0.37 | 1.04 |
| C3-hop on IFS's rows (1260 rows) | station, IMD rule (normal + 4.5) (70) | R-point | 49 | 21 | 30 | 70% | 38% | 0.47 | 1.13 |
| C3-hop on IFS's rows (1260 rows) | station, IMD rule (normal + 4.5) (70) | station-mapped R-point | 21 | 49 | 9 | 30% | 30% | 0.25 | 0.43 |
| C3-hop on IFS's rows (1260 rows) | ERA5, label rule (reference) (160) | R-point | 62 | 98 | 17 | 39% | 22% | 0.31 | 0.49 |
| IFS HRES 9 km, 12 UTC run (1250 rows) | station, label rule (normal + 3) (125) | R-point | 100 | 25 | 65 | 80% | 39% | 0.48 | 1.32 |
| IFS HRES 9 km, 12 UTC run (1250 rows) | station, label rule (normal + 3) (125) | station-mapped R-point | 108 | 17 | 86 | 86% | 44% | 0.46 | 1.55 |
| IFS HRES 9 km, 12 UTC run (1250 rows) | station, IMD rule (normal + 4.5) (70) | R-point | 68 | 2 | 97 | 97% | 59% | 0.37 | 2.36 |
| IFS HRES 9 km, 12 UTC run (1250 rows) | station, IMD rule (normal + 4.5) (70) | station-mapped R-point | 65 | 5 | 43 | 93% | 40% | 0.55 | 1.54 |
| IFS HRES 9 km, 12 UTC run (1250 rows) | ERA5, label rule (reference) (160) | R-point | 115 | 45 | 50 | 72% | 30% | 0.50 | 1.03 |
| C3-hop on IFS HRES 9 km, 12 UTC run's rows (1250 rows) | station, label rule (normal + 3) (125) | R-point | 57 | 68 | 22 | 46% | 28% | 0.35 | 0.63 |
| C3-hop on IFS HRES 9 km, 12 UTC run's rows (1250 rows) | station, label rule (normal + 3) (125) | station-mapped R-point | 75 | 50 | 55 | 60% | 42% | 0.37 | 1.04 |
| C3-hop on IFS HRES 9 km, 12 UTC run's rows (1250 rows) | station, IMD rule (normal + 4.5) (70) | R-point | 49 | 21 | 30 | 70% | 38% | 0.47 | 1.13 |
| C3-hop on IFS HRES 9 km, 12 UTC run's rows (1250 rows) | station, IMD rule (normal + 4.5) (70) | station-mapped R-point | 21 | 49 | 9 | 30% | 30% | 0.25 | 0.43 |
| C3-hop on IFS HRES 9 km, 12 UTC run's rows (1250 rows) | ERA5, label rule (reference) (160) | R-point | 62 | 98 | 17 | 39% | 22% | 0.31 | 0.49 |
| GraphCast 2018 (605 rows) | station, label rule (normal + 3) (80) | R-point | 28 | 52 | 4 | 35% | 12% | 0.30 | 0.40 |
| GraphCast 2018 (605 rows) | station, label rule (normal + 3) (80) | station-mapped R-point | 41 | 39 | 12 | 51% | 23% | 0.40 | 0.66 |
| GraphCast 2018 (605 rows) | station, IMD rule (normal + 4.5) (50) | R-point | 24 | 26 | 8 | 48% | 25% | 0.39 | 0.64 |
| GraphCast 2018 (605 rows) | station, IMD rule (normal + 4.5) (50) | station-mapped R-point | 8 | 42 | 5 | 16% | 38% | 0.13 | 0.26 |
| GraphCast 2018 (605 rows) | ERA5, label rule (reference) (30) | R-point | 20 | 10 | 12 | 67% | 38% | 0.46 | 1.07 |
| C3-hop on GraphCast 2018's rows (605 rows) | station, label rule (normal + 3) (80) | R-point | 1 | 79 | 3 | 1% | 75% | 0.01 | 0.05 |
| C3-hop on GraphCast 2018's rows (605 rows) | station, label rule (normal + 3) (80) | station-mapped R-point | 18 | 62 | 18 | 22% | 50% | 0.14 | 0.45 |
| C3-hop on GraphCast 2018's rows (605 rows) | station, IMD rule (normal + 4.5) (50) | R-point | 1 | 49 | 3 | 2% | 75% | 0.01 | 0.08 |
| C3-hop on GraphCast 2018's rows (605 rows) | station, IMD rule (normal + 4.5) (50) | station-mapped R-point | 0 | 50 | 0 | 0% | n/a | 0.00 | 0.00 |
| C3-hop on GraphCast 2018's rows (605 rows) | ERA5, label rule (reference) (30) | R-point | 0 | 30 | 4 | 0% | 100% | -0.01 | 0.13 |

**Cold bias on station heatwave days**

| Forecast | Station rule | Lead | Rows | Forecast - station | = forecast - ERA5 | + ERA5 - station | ERA5's share |
|---|---|---|---|---|---|---|---|
| AIFS | label rule (normal + 3) | all | 105 | -2.33 | -1.97 | -0.36 | 15% |
| AIFS | label rule (normal + 3) | 1 | 21 | -1.14 | -0.78 | -0.36 | 32% |
| AIFS | label rule (normal + 3) | 2 | 21 | -1.84 | -1.48 | -0.36 | 20% |
| AIFS | label rule (normal + 3) | 3 | 21 | -2.51 | -2.15 | -0.36 | 14% |
| AIFS | label rule (normal + 3) | 4 | 21 | -2.98 | -2.62 | -0.36 | 12% |
| AIFS | label rule (normal + 3) | 5 | 21 | -3.18 | -2.82 | -0.36 | 11% |
| AIFS | IMD rule (normal + 4.5) | all | 70 | -2.30 | -2.07 | -0.23 | 10% |
| AIFS | IMD rule (normal + 4.5) | 1 | 14 | -1.12 | -0.89 | -0.23 | 21% |
| AIFS | IMD rule (normal + 4.5) | 2 | 14 | -1.86 | -1.63 | -0.23 | 12% |
| AIFS | IMD rule (normal + 4.5) | 3 | 14 | -2.52 | -2.29 | -0.23 | 9% |
| AIFS | IMD rule (normal + 4.5) | 4 | 14 | -2.92 | -2.69 | -0.23 | 8% |
| AIFS | IMD rule (normal + 4.5) | 5 | 14 | -3.10 | -2.87 | -0.23 | 7% |
| C3-hop on AIFS's rows | label rule (normal + 3) | all | 105 | -2.21 | -1.85 | -0.36 | 16% |
| C3-hop on AIFS's rows | label rule (normal + 3) | 1 | 21 | -1.38 | -1.02 | -0.36 | 26% |
| C3-hop on AIFS's rows | label rule (normal + 3) | 2 | 21 | -2.15 | -1.79 | -0.36 | 17% |
| C3-hop on AIFS's rows | label rule (normal + 3) | 3 | 21 | -2.38 | -2.02 | -0.36 | 15% |
| C3-hop on AIFS's rows | label rule (normal + 3) | 4 | 21 | -2.65 | -2.29 | -0.36 | 14% |
| C3-hop on AIFS's rows | label rule (normal + 3) | 5 | 21 | -2.47 | -2.11 | -0.36 | 15% |
| C3-hop on AIFS's rows | IMD rule (normal + 4.5) | all | 70 | -2.36 | -2.13 | -0.23 | 10% |
| C3-hop on AIFS's rows | IMD rule (normal + 4.5) | 1 | 14 | -1.34 | -1.11 | -0.23 | 17% |
| C3-hop on AIFS's rows | IMD rule (normal + 4.5) | 2 | 14 | -2.23 | -2.00 | -0.23 | 10% |
| C3-hop on AIFS's rows | IMD rule (normal + 4.5) | 3 | 14 | -2.53 | -2.30 | -0.23 | 9% |
| C3-hop on AIFS's rows | IMD rule (normal + 4.5) | 4 | 14 | -2.98 | -2.75 | -0.23 | 8% |
| C3-hop on AIFS's rows | IMD rule (normal + 4.5) | 5 | 14 | -2.72 | -2.49 | -0.23 | 8% |
| AIFS Single (added after the first results) | label rule (normal + 3) | all | 20 | -2.92 | -2.54 | -0.38 | 13% |
| AIFS Single (added after the first results) | label rule (normal + 3) | 1 | 4 | -2.59 | -2.21 | -0.38 | 15% |
| AIFS Single (added after the first results) | label rule (normal + 3) | 2 | 4 | -2.98 | -2.60 | -0.38 | 13% |
| AIFS Single (added after the first results) | label rule (normal + 3) | 3 | 4 | -3.11 | -2.73 | -0.38 | 12% |
| AIFS Single (added after the first results) | label rule (normal + 3) | 4 | 4 | -3.13 | -2.74 | -0.38 | 12% |
| AIFS Single (added after the first results) | label rule (normal + 3) | 5 | 4 | -2.80 | -2.42 | -0.38 | 14% |
| AIFS Single (added after the first results) | IMD rule (normal + 4.5) | all | 0 | | | | |
| AIFS Single (added after the first results) | IMD rule (normal + 4.5) | 1 | 0 | | | | |
| AIFS Single (added after the first results) | IMD rule (normal + 4.5) | 2 | 0 | | | | |
| AIFS Single (added after the first results) | IMD rule (normal + 4.5) | 3 | 0 | | | | |
| AIFS Single (added after the first results) | IMD rule (normal + 4.5) | 4 | 0 | | | | |
| AIFS Single (added after the first results) | IMD rule (normal + 4.5) | 5 | 0 | | | | |
| C3-hop on AIFS Single's rows | label rule (normal + 3) | all | 20 | -2.69 | -2.31 | -0.38 | 14% |
| C3-hop on AIFS Single's rows | label rule (normal + 3) | 1 | 4 | -1.18 | -0.79 | -0.38 | 33% |
| C3-hop on AIFS Single's rows | label rule (normal + 3) | 2 | 4 | -2.12 | -1.74 | -0.38 | 18% |
| C3-hop on AIFS Single's rows | label rule (normal + 3) | 3 | 4 | -2.91 | -2.53 | -0.38 | 13% |
| C3-hop on AIFS Single's rows | label rule (normal + 3) | 4 | 4 | -3.58 | -3.20 | -0.38 | 11% |
| C3-hop on AIFS Single's rows | label rule (normal + 3) | 5 | 4 | -3.66 | -3.27 | -0.38 | 10% |
| C3-hop on AIFS Single's rows | IMD rule (normal + 4.5) | all | 0 | | | | |
| C3-hop on AIFS Single's rows | IMD rule (normal + 4.5) | 1 | 0 | | | | |
| C3-hop on AIFS Single's rows | IMD rule (normal + 4.5) | 2 | 0 | | | | |
| C3-hop on AIFS Single's rows | IMD rule (normal + 4.5) | 3 | 0 | | | | |
| C3-hop on AIFS Single's rows | IMD rule (normal + 4.5) | 4 | 0 | | | | |
| C3-hop on AIFS Single's rows | IMD rule (normal + 4.5) | 5 | 0 | | | | |
| GFS | label rule (normal + 3) | all | 125 | +1.71 | +2.07 | -0.36 | -21% |
| GFS | label rule (normal + 3) | 1 | 25 | +1.79 | +2.16 | -0.36 | -20% |
| GFS | label rule (normal + 3) | 2 | 25 | +2.02 | +2.38 | -0.36 | -18% |
| GFS | label rule (normal + 3) | 3 | 25 | +1.77 | +2.13 | -0.36 | -21% |
| GFS | label rule (normal + 3) | 4 | 25 | +1.45 | +1.81 | -0.36 | -25% |
| GFS | label rule (normal + 3) | 5 | 25 | +1.51 | +1.87 | -0.36 | -24% |
| GFS | IMD rule (normal + 4.5) | all | 70 | +1.48 | +1.71 | -0.23 | -16% |
| GFS | IMD rule (normal + 4.5) | 1 | 14 | +1.68 | +1.91 | -0.23 | -14% |
| GFS | IMD rule (normal + 4.5) | 2 | 14 | +1.90 | +2.13 | -0.23 | -12% |
| GFS | IMD rule (normal + 4.5) | 3 | 14 | +1.50 | +1.73 | -0.23 | -15% |
| GFS | IMD rule (normal + 4.5) | 4 | 14 | +1.06 | +1.30 | -0.23 | -22% |
| GFS | IMD rule (normal + 4.5) | 5 | 14 | +1.27 | +1.50 | -0.23 | -18% |
| C3-hop on GFS's rows | label rule (normal + 3) | all | 125 | -2.28 | -1.92 | -0.36 | 16% |
| C3-hop on GFS's rows | label rule (normal + 3) | 1 | 25 | -1.35 | -0.99 | -0.36 | 27% |
| C3-hop on GFS's rows | label rule (normal + 3) | 2 | 25 | -2.15 | -1.78 | -0.36 | 17% |
| C3-hop on GFS's rows | label rule (normal + 3) | 3 | 25 | -2.46 | -2.10 | -0.36 | 15% |
| C3-hop on GFS's rows | label rule (normal + 3) | 4 | 25 | -2.80 | -2.44 | -0.36 | 13% |
| C3-hop on GFS's rows | label rule (normal + 3) | 5 | 25 | -2.66 | -2.29 | -0.36 | 14% |
| C3-hop on GFS's rows | IMD rule (normal + 4.5) | all | 70 | -2.36 | -2.13 | -0.23 | 10% |
| C3-hop on GFS's rows | IMD rule (normal + 4.5) | 1 | 14 | -1.34 | -1.11 | -0.23 | 17% |
| C3-hop on GFS's rows | IMD rule (normal + 4.5) | 2 | 14 | -2.23 | -2.00 | -0.23 | 10% |
| C3-hop on GFS's rows | IMD rule (normal + 4.5) | 3 | 14 | -2.53 | -2.30 | -0.23 | 9% |
| C3-hop on GFS's rows | IMD rule (normal + 4.5) | 4 | 14 | -2.98 | -2.75 | -0.23 | 8% |
| C3-hop on GFS's rows | IMD rule (normal + 4.5) | 5 | 14 | -2.72 | -2.49 | -0.23 | 8% |
| GFS, Tmax since 2021 | label rule (normal + 3) | all | 325 | +1.61 | +2.02 | -0.40 | -25% |
| GFS, Tmax since 2021 | label rule (normal + 3) | 1 | 65 | +1.57 | +1.97 | -0.40 | -26% |
| GFS, Tmax since 2021 | label rule (normal + 3) | 2 | 65 | +1.72 | +2.12 | -0.40 | -23% |
| GFS, Tmax since 2021 | label rule (normal + 3) | 3 | 65 | +1.68 | +2.08 | -0.40 | -24% |
| GFS, Tmax since 2021 | label rule (normal + 3) | 4 | 65 | +1.54 | +1.94 | -0.40 | -26% |
| GFS, Tmax since 2021 | label rule (normal + 3) | 5 | 65 | +1.56 | +1.96 | -0.40 | -26% |
| GFS, Tmax since 2021 | IMD rule (normal + 4.5) | all | 160 | +1.56 | +1.91 | -0.36 | -23% |
| GFS, Tmax since 2021 | IMD rule (normal + 4.5) | 1 | 32 | +1.62 | +1.98 | -0.36 | -22% |
| GFS, Tmax since 2021 | IMD rule (normal + 4.5) | 2 | 32 | +1.83 | +2.18 | -0.36 | -20% |
| GFS, Tmax since 2021 | IMD rule (normal + 4.5) | 3 | 32 | +1.59 | +1.95 | -0.36 | -23% |
| GFS, Tmax since 2021 | IMD rule (normal + 4.5) | 4 | 32 | +1.34 | +1.70 | -0.36 | -27% |
| GFS, Tmax since 2021 | IMD rule (normal + 4.5) | 5 | 32 | +1.40 | +1.75 | -0.36 | -26% |
| C3-hop on GFS, Tmax since 2021's rows | label rule (normal + 3) | all | 325 | -2.21 | -1.81 | -0.40 | 18% |
| C3-hop on GFS, Tmax since 2021's rows | label rule (normal + 3) | 1 | 65 | -1.27 | -0.87 | -0.40 | 32% |
| C3-hop on GFS, Tmax since 2021's rows | label rule (normal + 3) | 2 | 65 | -1.99 | -1.59 | -0.40 | 20% |
| C3-hop on GFS, Tmax since 2021's rows | label rule (normal + 3) | 3 | 65 | -2.35 | -1.95 | -0.40 | 17% |
| C3-hop on GFS, Tmax since 2021's rows | label rule (normal + 3) | 4 | 65 | -2.70 | -2.30 | -0.40 | 15% |
| C3-hop on GFS, Tmax since 2021's rows | label rule (normal + 3) | 5 | 65 | -2.74 | -2.34 | -0.40 | 15% |
| C3-hop on GFS, Tmax since 2021's rows | IMD rule (normal + 4.5) | all | 160 | -2.46 | -2.10 | -0.36 | 15% |
| C3-hop on GFS, Tmax since 2021's rows | IMD rule (normal + 4.5) | 1 | 32 | -1.30 | -0.94 | -0.36 | 28% |
| C3-hop on GFS, Tmax since 2021's rows | IMD rule (normal + 4.5) | 2 | 32 | -2.16 | -1.80 | -0.36 | 17% |
| C3-hop on GFS, Tmax since 2021's rows | IMD rule (normal + 4.5) | 3 | 32 | -2.67 | -2.31 | -0.36 | 13% |
| C3-hop on GFS, Tmax since 2021's rows | IMD rule (normal + 4.5) | 4 | 32 | -3.09 | -2.73 | -0.36 | 12% |
| C3-hop on GFS, Tmax since 2021's rows | IMD rule (normal + 4.5) | 5 | 32 | -3.06 | -2.70 | -0.36 | 12% |
| IFS | label rule (normal + 3) | all | 125 | -0.61 | -0.25 | -0.36 | 60% |
| IFS | label rule (normal + 3) | 1 | 25 | -1.10 | -0.73 | -0.36 | 33% |
| IFS | label rule (normal + 3) | 2 | 25 | -0.95 | -0.58 | -0.36 | 38% |
| IFS | label rule (normal + 3) | 3 | 25 | -0.53 | -0.17 | -0.36 | 69% |
| IFS | label rule (normal + 3) | 4 | 25 | -0.37 | -0.00 | -0.36 | 99% |
| IFS | label rule (normal + 3) | 5 | 25 | -0.11 | +0.26 | -0.36 | 334% |
| IFS | IMD rule (normal + 4.5) | all | 70 | -0.18 | +0.05 | -0.23 | 127% |
| IFS | IMD rule (normal + 4.5) | 1 | 14 | -0.68 | -0.45 | -0.23 | 34% |
| IFS | IMD rule (normal + 4.5) | 2 | 14 | -0.48 | -0.25 | -0.23 | 48% |
| IFS | IMD rule (normal + 4.5) | 3 | 14 | -0.14 | +0.09 | -0.23 | 163% |
| IFS | IMD rule (normal + 4.5) | 4 | 14 | +0.01 | +0.24 | -0.23 | -1940% |
| IFS | IMD rule (normal + 4.5) | 5 | 14 | +0.38 | +0.62 | -0.23 | -60% |
| C3-hop on IFS's rows | label rule (normal + 3) | all | 125 | -2.28 | -1.92 | -0.36 | 16% |
| C3-hop on IFS's rows | label rule (normal + 3) | 1 | 25 | -1.35 | -0.99 | -0.36 | 27% |
| C3-hop on IFS's rows | label rule (normal + 3) | 2 | 25 | -2.15 | -1.78 | -0.36 | 17% |
| C3-hop on IFS's rows | label rule (normal + 3) | 3 | 25 | -2.46 | -2.10 | -0.36 | 15% |
| C3-hop on IFS's rows | label rule (normal + 3) | 4 | 25 | -2.80 | -2.44 | -0.36 | 13% |
| C3-hop on IFS's rows | label rule (normal + 3) | 5 | 25 | -2.66 | -2.29 | -0.36 | 14% |
| C3-hop on IFS's rows | IMD rule (normal + 4.5) | all | 70 | -2.36 | -2.13 | -0.23 | 10% |
| C3-hop on IFS's rows | IMD rule (normal + 4.5) | 1 | 14 | -1.34 | -1.11 | -0.23 | 17% |
| C3-hop on IFS's rows | IMD rule (normal + 4.5) | 2 | 14 | -2.23 | -2.00 | -0.23 | 10% |
| C3-hop on IFS's rows | IMD rule (normal + 4.5) | 3 | 14 | -2.53 | -2.30 | -0.23 | 9% |
| C3-hop on IFS's rows | IMD rule (normal + 4.5) | 4 | 14 | -2.98 | -2.75 | -0.23 | 8% |
| C3-hop on IFS's rows | IMD rule (normal + 4.5) | 5 | 14 | -2.72 | -2.49 | -0.23 | 8% |
| IFS HRES 9 km, 12 UTC run | label rule (normal + 3) | all | 125 | -1.20 | -0.83 | -0.36 | 30% |
| IFS HRES 9 km, 12 UTC run | label rule (normal + 3) | 1 | 25 | -1.33 | -0.96 | -0.36 | 27% |
| IFS HRES 9 km, 12 UTC run | label rule (normal + 3) | 2 | 25 | -1.31 | -0.95 | -0.36 | 28% |
| IFS HRES 9 km, 12 UTC run | label rule (normal + 3) | 3 | 25 | -1.23 | -0.86 | -0.36 | 30% |
| IFS HRES 9 km, 12 UTC run | label rule (normal + 3) | 4 | 25 | -1.23 | -0.86 | -0.36 | 30% |
| IFS HRES 9 km, 12 UTC run | label rule (normal + 3) | 5 | 25 | -0.88 | -0.52 | -0.36 | 41% |
| IFS HRES 9 km, 12 UTC run | IMD rule (normal + 4.5) | all | 70 | -0.98 | -0.75 | -0.23 | 24% |
| IFS HRES 9 km, 12 UTC run | IMD rule (normal + 4.5) | 1 | 14 | -1.09 | -0.86 | -0.23 | 21% |
| IFS HRES 9 km, 12 UTC run | IMD rule (normal + 4.5) | 2 | 14 | -1.07 | -0.84 | -0.23 | 22% |
| IFS HRES 9 km, 12 UTC run | IMD rule (normal + 4.5) | 3 | 14 | -1.05 | -0.82 | -0.23 | 22% |
| IFS HRES 9 km, 12 UTC run | IMD rule (normal + 4.5) | 4 | 14 | -0.99 | -0.76 | -0.23 | 23% |
| IFS HRES 9 km, 12 UTC run | IMD rule (normal + 4.5) | 5 | 14 | -0.69 | -0.46 | -0.23 | 33% |
| C3-hop on IFS HRES 9 km, 12 UTC run's rows | label rule (normal + 3) | all | 125 | -2.28 | -1.92 | -0.36 | 16% |
| C3-hop on IFS HRES 9 km, 12 UTC run's rows | label rule (normal + 3) | 1 | 25 | -1.35 | -0.99 | -0.36 | 27% |
| C3-hop on IFS HRES 9 km, 12 UTC run's rows | label rule (normal + 3) | 2 | 25 | -2.15 | -1.78 | -0.36 | 17% |
| C3-hop on IFS HRES 9 km, 12 UTC run's rows | label rule (normal + 3) | 3 | 25 | -2.46 | -2.10 | -0.36 | 15% |
| C3-hop on IFS HRES 9 km, 12 UTC run's rows | label rule (normal + 3) | 4 | 25 | -2.80 | -2.44 | -0.36 | 13% |
| C3-hop on IFS HRES 9 km, 12 UTC run's rows | label rule (normal + 3) | 5 | 25 | -2.66 | -2.29 | -0.36 | 14% |
| C3-hop on IFS HRES 9 km, 12 UTC run's rows | IMD rule (normal + 4.5) | all | 70 | -2.36 | -2.13 | -0.23 | 10% |
| C3-hop on IFS HRES 9 km, 12 UTC run's rows | IMD rule (normal + 4.5) | 1 | 14 | -1.34 | -1.11 | -0.23 | 17% |
| C3-hop on IFS HRES 9 km, 12 UTC run's rows | IMD rule (normal + 4.5) | 2 | 14 | -2.23 | -2.00 | -0.23 | 10% |
| C3-hop on IFS HRES 9 km, 12 UTC run's rows | IMD rule (normal + 4.5) | 3 | 14 | -2.53 | -2.30 | -0.23 | 9% |
| C3-hop on IFS HRES 9 km, 12 UTC run's rows | IMD rule (normal + 4.5) | 4 | 14 | -2.98 | -2.75 | -0.23 | 8% |
| C3-hop on IFS HRES 9 km, 12 UTC run's rows | IMD rule (normal + 4.5) | 5 | 14 | -2.72 | -2.49 | -0.23 | 8% |
| GraphCast 2018 | label rule (normal + 3) | all | 80 | -2.37 | -0.38 | -1.99 | 84% |
| GraphCast 2018 | label rule (normal + 3) | 1 | 16 | -2.22 | -0.23 | -1.99 | 90% |
| GraphCast 2018 | label rule (normal + 3) | 2 | 16 | -2.21 | -0.21 | -1.99 | 90% |
| GraphCast 2018 | label rule (normal + 3) | 3 | 16 | -2.43 | -0.44 | -1.99 | 82% |
| GraphCast 2018 | label rule (normal + 3) | 4 | 16 | -2.46 | -0.46 | -1.99 | 81% |
| GraphCast 2018 | label rule (normal + 3) | 5 | 16 | -2.53 | -0.54 | -1.99 | 79% |
| GraphCast 2018 | IMD rule (normal + 4.5) | all | 50 | -2.91 | -0.64 | -2.27 | 78% |
| GraphCast 2018 | IMD rule (normal + 4.5) | 1 | 10 | -2.57 | -0.30 | -2.27 | 88% |
| GraphCast 2018 | IMD rule (normal + 4.5) | 2 | 10 | -2.72 | -0.45 | -2.27 | 84% |
| GraphCast 2018 | IMD rule (normal + 4.5) | 3 | 10 | -3.03 | -0.76 | -2.27 | 75% |
| GraphCast 2018 | IMD rule (normal + 4.5) | 4 | 10 | -3.08 | -0.81 | -2.27 | 74% |
| GraphCast 2018 | IMD rule (normal + 4.5) | 5 | 10 | -3.14 | -0.87 | -2.27 | 72% |
| C3-hop on GraphCast 2018's rows | label rule (normal + 3) | all | 80 | -3.44 | -1.44 | -1.99 | 58% |
| C3-hop on GraphCast 2018's rows | label rule (normal + 3) | 1 | 16 | -2.74 | -0.74 | -1.99 | 73% |
| C3-hop on GraphCast 2018's rows | label rule (normal + 3) | 2 | 16 | -3.34 | -1.35 | -1.99 | 60% |
| C3-hop on GraphCast 2018's rows | label rule (normal + 3) | 3 | 16 | -3.64 | -1.65 | -1.99 | 55% |
| C3-hop on GraphCast 2018's rows | label rule (normal + 3) | 4 | 16 | -3.61 | -1.61 | -1.99 | 55% |
| C3-hop on GraphCast 2018's rows | label rule (normal + 3) | 5 | 16 | -3.85 | -1.85 | -1.99 | 52% |
| C3-hop on GraphCast 2018's rows | IMD rule (normal + 4.5) | all | 50 | -3.65 | -1.37 | -2.27 | 62% |
| C3-hop on GraphCast 2018's rows | IMD rule (normal + 4.5) | 1 | 10 | -3.23 | -0.96 | -2.27 | 70% |
| C3-hop on GraphCast 2018's rows | IMD rule (normal + 4.5) | 2 | 10 | -3.79 | -1.52 | -2.27 | 60% |
| C3-hop on GraphCast 2018's rows | IMD rule (normal + 4.5) | 3 | 10 | -3.97 | -1.69 | -2.27 | 57% |
| C3-hop on GraphCast 2018's rows | IMD rule (normal + 4.5) | 4 | 10 | -3.61 | -1.34 | -2.27 | 63% |
| C3-hop on GraphCast 2018's rows | IMD rule (normal + 4.5) | 5 | 10 | -3.62 | -1.35 | -2.27 | 63% |

## Notes

- Safdarjung, rebuilt: `followup/data/isd_daily_safdarjung.csv`, column `tmax`, 1980-01-02 to 2025-08-22, SHA-256 ecb6f60694226c2ca9b28c7f410c056099ed4ffc647f1a479694ea7f42bbb7fe (of the file as read).
- Palam, hourly maximum: `followup/data/isd_daily_palam.csv`, column `tmax_hourly`, 2002-06-24 to 2025-08-24, SHA-256 7a5d8bc678b1f15a5d12acfb91aecc193cccf5937bd7614e6048bc2e49206bff (of the file as read).
- Global-model values: `followup/data/open_meteo_runs_daily.parquet`, SHA-256 58ceaaaca724b4dbffbd79ea21717d9aa322efddc4ec5cec8b9d6e64660c4ccd.
- GraphCast 2018: included.
- Still to add, after P3-2 has run: GraphCast 2020 (P3-2a); the P3-2 corrector variants.
- WBGT is not part of this check (P3-3 is about Tmax). The humidity check is step 4 of P3-7.
- A single station is not the truth for a 75 km box either: its own siting and the city around it shape its readings. The check asks whether conclusions depend on which record is used.
