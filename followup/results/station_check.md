# The Safdarjung station as a second truth for Tmax (P3-3, post-test)

Pre-registered 2026-11-05 in `context/decisions_phase3.md` (P3-3), details fixed 2026-11-10 before any comparison was run. Descriptive: the test years were already used, so nothing here is confirmatory. Station: GHCN-Daily IN022021900 (Safdarjung), daily maximum, °C. ERA5: the 9-cell mean the models were trained and tested on. Bias = forecast - truth; 'ERA5 - station' is ERA5's own offset. Station-mapped = the forecast put through a per-month straight line fitted on 1980-2018 (station = a + b x ERA5).

## Read this first: the station's recent values are not dated to the day they occurred

Found after the registered results were seen (the check is the section after 'Station data'). Up to about 2008 a station value dated D is the maximum of day D. From 2011 it behaves like the larger of the maxima of D - 1 and D. From 2016 values are also missing on many days (36% of days in 2019-2025), mostly days warmer than the day before, so most remaining values are the previous day's maximum. Consequences:

- In 2019-2025, 63% of the station's values fall on a day cooler than the day before. On those days a forecast for day D is compared with the maximum of day D - 1.
- So the 2019-2025 station numbers below (the second period of (i), all of (ii), and (iii)) do **not** measure error against the station's true daily maximum. They are kept exactly as registered, for the record, and no conclusion is drawn from them.
- The 1.8 °C gap between ERA5 and the station in 2019-2025 is partly this dating effect. Matched to the larger of the two days, the gap is 0.9 °C. It was 0.4 °C in 1980-2008, matched to the same day.
- The station normal and the mapping are fitted on 1980-2018, which includes the affected years 2011-2018.

## Station data

| Period | Usable days | In the heat season | Heatwave days, label rule | Heatwave days, IMD rule | Last day |
|---|---|---|---|---|---|
| 1980-2018 | 13093 | 4956 | 533 | 199 | 2018-12-31 |
| 2019-2025 | 1544 | 593 | 112 | 46 | 2025-08-23 |

A usable day has a value and no quality flag. Gaps are not filled; a day without a station value is left out of every comparison. A station heatwave day is a single in-season day at or above min(45, max(40, normal + 3)) (label rule) or min(45, max(40, normal + 4.5)) (IMD rule), with the station's own 1980-2018 normal.

## Exploratory: are the station's dates right? (added after the results were seen)

Not registered. It uses the station and ERA5 only, no forecast; the blocks of years were read off its own per-year table (in the JSON). For the station value dated D: the mean and SD of ERA5 - station when ERA5 is taken on the same day D, on the day before, or as the larger of the two. The smallest SD shows which day the station value belongs to.

| Years | Days with a value | Same day: mean / SD | Day before: mean / SD | Larger of the two: mean / SD | Reporting days cooler than the day before (all days) | Consecutive values that are identical |
|---|---|---|---|---|---|---|
| 1980-2008 | 94% | -0.42 / 1.09 | -0.43 / 1.95 | +0.16 / 1.54 | 46% (46%) | 5% |
| 2009-2010 | 97% | -0.63 / 1.20 | -0.63 / 1.86 | -0.05 / 1.44 | 45% (45%) | 14% |
| 2011-2015 | 98% | -1.37 / 1.43 | -1.37 / 1.61 | -0.75 / 1.21 | 45% (45%) | 22% |
| 2016-2018 | 56% | -1.91 / 1.67 | -1.07 / 0.94 | -0.90 / 0.90 | 72% (48%) | 8% |
| 2019-2025 | 64% | -1.83 / 1.74 | -1.20 / 1.02 | -0.91 / 0.95 | 63% (46%) | 10% |

How to read it:

- If the dates are right, the same day has the smallest SD, about as many reporting days are cooler than the day before as in the calendar as a whole, and identical consecutive values are rare. That is the picture in 1980-2008.
- In 2011-2015 the larger of the two days fits best and identical consecutive values are several times more common: a hot day's maximum is repeated on the next, cooler day.
- From 2016 the value is present mainly on days cooler than the day before, and the day before fits far better than the same day.
- GHCN-Daily marks all of this station's values with source flag S (Global Summary of the Day). Its documentation warns that S values are derived from synoptic reports and may differ from true daily data.

## (i) ERA5 against the station

| Period | Days | ERA5 - station | RMSE | After the mapping: bias | RMSE |
|---|---|---|---|---|---|
| 1980-2018, all days | 13093 | -0.63 | 1.406 | -0.00 | 1.183 |
| 1980-2018, heat season | 4956 | -0.53 | 1.454 | +0.01 | 1.257 |
| 1980-2018, station heatwave days, label rule | 533 | -0.61 | 1.235 | -0.49 | 1.118 |
| 1980-2018, station heatwave days, IMD rule | 199 | -0.73 | 1.386 | -0.61 | 1.237 |
| 1980-2018, ERA5 heatwave days (mirror view) | 485 | -0.09 | 0.631 | -0.01 | 0.630 |
| 2019-2025, all days | 1544 | -1.83 | 2.526 | -1.20 | 2.032 |
| 2019-2025, heat season | 593 | -1.90 | 2.671 | -1.33 | 2.202 |
| 2019-2025, station heatwave days, label rule | 112 | -1.67 | 2.637 | -1.54 | 2.421 |
| 2019-2025, station heatwave days, IMD rule | 46 | -1.44 | 2.208 | -1.35 | 2.042 |
| 2019-2025, ERA5 heatwave days (mirror view) | 63 | -0.44 | 0.856 | -0.43 | 0.821 |

The mapping is fitted on 1980-2018, so its 1980-2018 rows are in-sample. Picking days by one record's heat makes the other record look colder on those days (selection), so the station-heatwave rows and the mirror row are read together.

**Do the heatwave labels agree?** In-season days with a station value. Events = station heatwave days; 'warnings' = ERA5 heatwave days.

| Period | Station rule | ERA5 side | Both | Station only | ERA5 only | Station days ERA5 also has | ERA5 days the station lacks | ETS | ERA5 days / station days |
|---|---|---|---|---|---|---|---|---|---|
| 1980-2018 (4956 days) | label rule (normal + 3) | ERA5 label | 398 | 135 | 87 | 75% | 18% | 0.61 | 0.91 |
| 1980-2018 (4956 days) | label rule (normal + 3) | mapped ERA5 vs the station threshold | 392 | 141 | 60 | 74% | 13% | 0.63 | 0.85 |
| 1980-2018 (4956 days) | IMD rule (normal + 4.5) | ERA5 label | 177 | 22 | 308 | 89% | 64% | 0.32 | 2.44 |
| 1980-2018 (4956 days) | IMD rule (normal + 4.5) | mapped ERA5 vs the station threshold | 122 | 77 | 30 | 61% | 20% | 0.52 | 0.76 |
| 2019-2025 (593 days) | label rule (normal + 3) | ERA5 label | 59 | 53 | 4 | 53% | 6% | 0.45 | 0.56 |
| 2019-2025 (593 days) | label rule (normal + 3) | mapped ERA5 vs the station threshold | 57 | 55 | 3 | 51% | 5% | 0.44 | 0.54 |
| 2019-2025 (593 days) | IMD rule (normal + 4.5) | ERA5 label | 32 | 14 | 31 | 70% | 49% | 0.38 | 1.37 |
| 2019-2025 (593 days) | IMD rule (normal + 4.5) | mapped ERA5 vs the station threshold | 24 | 22 | 1 | 52% | 4% | 0.49 | 0.54 |

## (iv) The station mapping

station = a + b x ERA5, least squares per calendar month on the usable station days of 1980-2018.

| Month | a | b | Days |
|---|---|---|---|
| 1 | +0.30 | 1.013 | 1113 |
| 2 | +1.57 | 0.965 | 1012 |
| 3 | +1.72 | 0.960 | 1116 |
| 4 | +2.91 | 0.928 | 1063 |
| 5 | +0.31 | 0.997 | 1105 |
| 6 | +4.21 | 0.902 | 1077 |
| 7 | +6.35 | 0.852 | 1102 |
| 8 | +5.31 | 0.877 | 1102 |
| 9 | +6.78 | 0.825 | 1081 |
| 10 | +3.66 | 0.902 | 1115 |
| 11 | +1.10 | 0.975 | 1098 |
| 12 | -0.37 | 1.038 | 1109 |

## (ii) Our test forecasts against the station

Target days 2019-01-02 to 2025-08-23 with a station value: 7714 rows (query date x lead), 21 year x season clusters. The networks are shown as the 10-seed ensemble mean and as the mean of the 10 seeds' RMSE (the test's primary score).

**All days**

| Forecast | Rows | RMSE vs ERA5 | RMSE vs station | RMSE station-mapped | Bias vs ERA5 | Bias vs station | Bias station-mapped |
|---|---|---|---|---|---|---|---|
| C3-hop (ensemble mean) | 7714 | 2.273 | 2.280 | 1.929 | +0.54 | -1.29 | -0.67 |
| C3-hop (mean of 10 seeds) | 7714 | 2.318 | 2.322 | 1.975 | +0.54 | -1.29 | -0.67 |
| Delhi-only LSTM (ensemble mean) | 7714 | 2.408 | 2.346 | 2.013 | +0.60 | -1.24 | -0.62 |
| Delhi-only LSTM (mean of 10 seeds) | 7714 | 2.444 | 2.382 | 2.051 | +0.60 | -1.24 | -0.62 |
| ridge_hw5 | 7714 | 2.410 | 2.134 | 1.877 | +0.77 | -1.06 | -0.47 |

**Heat season (15 Mar - 31 Jul)**

| Forecast | Rows | RMSE vs ERA5 | RMSE vs station | RMSE station-mapped | Bias vs ERA5 | Bias vs station | Bias station-mapped |
|---|---|---|---|---|---|---|---|
| C3-hop (ensemble mean) | 2965 | 2.588 | 2.382 | 2.084 | +0.70 | -1.20 | -0.67 |
| C3-hop (mean of 10 seeds) | 2965 | 2.632 | 2.428 | 2.130 | +0.70 | -1.20 | -0.67 |
| Delhi-only LSTM (ensemble mean) | 2965 | 2.850 | 2.517 | 2.259 | +0.86 | -1.03 | -0.52 |
| Delhi-only LSTM (mean of 10 seeds) | 2965 | 2.896 | 2.568 | 2.309 | +0.86 | -1.03 | -0.52 |
| ridge_hw5 | 2965 | 2.767 | 2.294 | 2.078 | +0.84 | -1.05 | -0.54 |

**Station heatwave days, label rule**

| Forecast | Rows | RMSE vs ERA5 | RMSE vs station | RMSE station-mapped | Bias vs ERA5 | Bias vs station | Bias station-mapped |
|---|---|---|---|---|---|---|---|
| C3-hop (ensemble mean) | 560 | 2.279 | 2.479 | 2.293 | -0.54 | -2.21 | -2.05 |
| C3-hop (mean of 10 seeds) | 560 | 2.331 | 2.522 | 2.332 | -0.54 | -2.21 | -2.05 |
| Delhi-only LSTM (ensemble mean) | 560 | 2.610 | 2.768 | 2.545 | -0.65 | -2.31 | -2.14 |
| Delhi-only LSTM (mean of 10 seeds) | 560 | 2.672 | 2.813 | 2.586 | -0.65 | -2.31 | -2.14 |
| ridge_hw5 | 560 | 2.455 | 2.619 | 2.421 | -0.62 | -2.29 | -2.12 |

**Station heatwave days, IMD rule**

| Forecast | Rows | RMSE vs ERA5 | RMSE vs station | RMSE station-mapped | Bias vs ERA5 | Bias vs station | Bias station-mapped |
|---|---|---|---|---|---|---|---|
| C3-hop (ensemble mean) | 230 | 2.111 | 2.647 | 2.469 | -0.93 | -2.37 | -2.22 |
| C3-hop (mean of 10 seeds) | 230 | 2.166 | 2.685 | 2.504 | -0.93 | -2.37 | -2.22 |
| Delhi-only LSTM (ensemble mean) | 230 | 2.367 | 2.966 | 2.744 | -1.13 | -2.57 | -2.40 |
| Delhi-only LSTM (mean of 10 seeds) | 230 | 2.419 | 2.998 | 2.773 | -1.13 | -2.57 | -2.40 |
| ridge_hw5 | 230 | 2.488 | 3.028 | 2.810 | -1.24 | -2.68 | -2.50 |

**ERA5 heatwave days (the test's extreme stratum)**

| Forecast | Rows | RMSE vs ERA5 | RMSE vs station | RMSE station-mapped | Bias vs ERA5 | Bias vs station | Bias station-mapped |
|---|---|---|---|---|---|---|---|
| C3-hop (ensemble mean) | 290 | 2.063 | 2.493 | 2.336 | -1.73 | -2.21 | -2.08 |
| C3-hop (mean of 10 seeds) | 290 | 2.114 | 2.534 | 2.374 | -1.73 | -2.21 | -2.08 |
| Delhi-only LSTM (ensemble mean) | 290 | 2.370 | 2.817 | 2.615 | -1.89 | -2.37 | -2.22 |
| Delhi-only LSTM (mean of 10 seeds) | 290 | 2.422 | 2.860 | 2.653 | -1.89 | -2.37 | -2.22 |
| ridge_hw5 | 290 | 2.378 | 2.791 | 2.606 | -1.96 | -2.44 | -2.28 |

**By lead, all days (ensemble means)**

| Forecast | Score | Day 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| C3-hop | RMSE vs station | 1.909 | 2.162 | 2.325 | 2.452 | 2.503 |
| C3-hop | Bias vs station | -1.44 | -1.42 | -1.30 | -1.23 | -1.06 |
| C3-hop | RMSE station-mapped vs station | 1.454 | 1.759 | 1.975 | 2.131 | 2.225 |
| C3-hop | Bias station-mapped vs station | -0.81 | -0.80 | -0.68 | -0.61 | -0.45 |
| Delhi-only LSTM | RMSE vs station | 1.860 | 2.130 | 2.407 | 2.578 | 2.662 |
| Delhi-only LSTM | Bias vs station | -1.33 | -1.21 | -1.23 | -1.22 | -1.18 |
| Delhi-only LSTM | RMSE station-mapped vs station | 1.426 | 1.780 | 2.079 | 2.267 | 2.368 |
| Delhi-only LSTM | Bias station-mapped vs station | -0.72 | -0.60 | -0.61 | -0.60 | -0.56 |
| ridge_hw5 | RMSE vs station | 1.664 | 1.890 | 2.182 | 2.345 | 2.486 |
| ridge_hw5 | Bias vs station | -1.25 | -1.07 | -1.03 | -0.97 | -0.99 |
| ridge_hw5 | RMSE station-mapped vs station | 1.244 | 1.598 | 1.933 | 2.137 | 2.286 |
| ridge_hw5 | Bias station-mapped vs station | -0.65 | -0.48 | -0.44 | -0.38 | -0.39 |

**By lead, station heatwave days, label rule (ensemble means)**

| Forecast | Score | Day 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| C3-hop | RMSE vs station | 1.773 | 2.286 | 2.587 | 2.815 | 2.784 |
| C3-hop | Bias vs station | -1.56 | -2.06 | -2.33 | -2.57 | -2.50 |
| C3-hop | RMSE station-mapped vs station | 1.629 | 2.116 | 2.393 | 2.604 | 2.579 |
| C3-hop | Bias station-mapped vs station | -1.44 | -1.91 | -2.16 | -2.38 | -2.32 |
| Delhi-only LSTM | RMSE vs station | 1.841 | 2.242 | 2.818 | 3.180 | 3.440 |
| Delhi-only LSTM | Bias vs station | -1.54 | -1.87 | -2.39 | -2.75 | -3.03 |
| Delhi-only LSTM | RMSE station-mapped vs station | 1.687 | 2.066 | 2.596 | 2.922 | 3.160 |
| Delhi-only LSTM | Bias station-mapped vs station | -1.42 | -1.73 | -2.21 | -2.54 | -2.80 |
| ridge_hw5 | RMSE vs station | 1.426 | 1.976 | 2.606 | 3.078 | 3.477 |
| ridge_hw5 | Bias vs station | -1.22 | -1.75 | -2.34 | -2.86 | -3.26 |
| ridge_hw5 | RMSE station-mapped vs station | 1.309 | 1.830 | 2.406 | 2.847 | 3.217 |
| ridge_hw5 | Bias station-mapped vs station | -1.13 | -1.62 | -2.17 | -2.64 | -3.02 |

**Paired tests** (delta = RMSE of the first minus RMSE of the second, same rows; negative = the first is better; paired cluster jackknife).

| Comparison | Stratum | Delta vs ERA5 [95% CI] | Delta vs station [95% CI] | Delta station-mapped [95% CI] | Clear change of order |
|---|---|---|---|---|---|
| C3-hop minus Delhi-only LSTM (10 seeds each) | All days | -0.126 [-0.233, -0.019] | -0.060 [-0.177, +0.057] | -0.076 [-0.165, +0.012] | no |
| C3-hop minus Delhi-only LSTM (10 seeds each) | Heat season (15 Mar - 31 Jul) | -0.264 [-0.439, -0.089], fragile (< 10 clusters) | -0.140 [-0.310, +0.030], fragile (< 10 clusters) | -0.179 [-0.324, -0.034], fragile (< 10 clusters) | no |
| C3-hop minus Delhi-only LSTM (10 seeds each) | Station heatwave days, label rule | -0.341 [-0.635, -0.047], fragile (< 10 clusters) | -0.291 [-0.720, +0.138], fragile (< 10 clusters) | -0.254 [-0.641, +0.133], fragile (< 10 clusters) | no |
| C3-hop ensemble minus ridge_hw5 | All days | -0.136 [-0.181, -0.092] | +0.146 [+0.070, +0.221] | +0.051 [-0.008, +0.110] | yes |
| C3-hop ensemble minus ridge_hw5 | Heat season (15 Mar - 31 Jul) | -0.179 [-0.250, -0.107], fragile (< 10 clusters) | +0.088 [-0.034, +0.210], fragile (< 10 clusters) | +0.006 [-0.078, +0.091], fragile (< 10 clusters) | no |
| C3-hop ensemble minus ridge_hw5 | Station heatwave days, label rule | -0.177 [-0.359, +0.006], fragile (< 10 clusters) | -0.139 [-0.464, +0.185], fragile (< 10 clusters) | -0.128 [-0.419, +0.163], fragile (< 10 clusters) | no |

**Yes/no warnings** (in-season rows). R-point and R-prob use the forecast as issued, against the ERA5 label threshold. Station-mapped R-point uses the mapped forecast against the station's own threshold of that rule.

| Forecast | Events | Rule | Hits | Misses | False alarms | Hit rate | False-alarm ratio | ETS | Frequency bias |
|---|---|---|---|---|---|---|---|---|---|
| C3-hop (2965 rows) | station, label rule (normal + 3) (560) | R-point | 165 | 395 | 32 | 29% | 16% | 0.23 | 0.35 |
| C3-hop (2965 rows) | station, label rule (normal + 3) (560) | R-prob (p* = 0.37) | 183 | 377 | 38 | 33% | 17% | 0.25 | 0.39 |
| C3-hop (2965 rows) | station, label rule (normal + 3) (560) | station-mapped R-point | 149 | 411 | 27 | 27% | 15% | 0.21 | 0.31 |
| C3-hop (2965 rows) | station, IMD rule (normal + 4.5) (230) | R-point | 126 | 104 | 71 | 55% | 36% | 0.39 | 0.86 |
| C3-hop (2965 rows) | station, IMD rule (normal + 4.5) (230) | R-prob (p* = 0.37) | 128 | 102 | 93 | 56% | 42% | 0.36 | 0.96 |
| C3-hop (2965 rows) | station, IMD rule (normal + 4.5) (230) | station-mapped R-point | 22 | 208 | 1 | 10% | 4% | 0.09 | 0.10 |
| C3-hop (2965 rows) | ERA5, label rule (reference) (315) | R-point | 114 | 201 | 83 | 36% | 42% | 0.25 | 0.63 |
| C3-hop (2965 rows) | ERA5, label rule (reference) (315) | R-prob (p* = 0.37) | 128 | 187 | 93 | 41% | 42% | 0.27 | 0.70 |
| Delhi-only LSTM (2965 rows) | station, label rule (normal + 3) (560) | R-point | 185 | 375 | 86 | 33% | 32% | 0.22 | 0.48 |
| Delhi-only LSTM (2965 rows) | station, label rule (normal + 3) (560) | R-prob (p* = 0.24) | 326 | 234 | 202 | 58% | 38% | 0.34 | 0.94 |
| Delhi-only LSTM (2965 rows) | station, label rule (normal + 3) (560) | station-mapped R-point | 171 | 389 | 77 | 31% | 31% | 0.21 | 0.44 |
| Delhi-only LSTM (2965 rows) | station, IMD rule (normal + 4.5) (230) | R-point | 121 | 109 | 150 | 53% | 55% | 0.28 | 1.18 |
| Delhi-only LSTM (2965 rows) | station, IMD rule (normal + 4.5) (230) | R-prob (p* = 0.24) | 168 | 62 | 360 | 73% | 68% | 0.23 | 2.30 |
| Delhi-only LSTM (2965 rows) | station, IMD rule (normal + 4.5) (230) | station-mapped R-point | 27 | 203 | 3 | 12% | 10% | 0.11 | 0.13 |
| Delhi-only LSTM (2965 rows) | ERA5, label rule (reference) (315) | R-point | 119 | 196 | 152 | 38% | 56% | 0.21 | 0.86 |
| Delhi-only LSTM (2965 rows) | ERA5, label rule (reference) (315) | R-prob (p* = 0.24) | 204 | 111 | 324 | 65% | 61% | 0.25 | 1.68 |
| ridge_hw5 (2965 rows) | station, label rule (normal + 3) (560) | R-point | 151 | 409 | 32 | 27% | 17% | 0.21 | 0.33 |
| ridge_hw5 (2965 rows) | station, label rule (normal + 3) (560) | station-mapped R-point | 130 | 430 | 24 | 23% | 16% | 0.18 | 0.28 |
| ridge_hw5 (2965 rows) | station, IMD rule (normal + 4.5) (230) | R-point | 96 | 134 | 87 | 42% | 48% | 0.27 | 0.80 |
| ridge_hw5 (2965 rows) | station, IMD rule (normal + 4.5) (230) | station-mapped R-point | 24 | 206 | 0 | 10% | 0% | 0.10 | 0.10 |
| ridge_hw5 (2965 rows) | ERA5, label rule (reference) (315) | R-point | 97 | 218 | 86 | 31% | 47% | 0.20 | 0.58 |

**How much of the cold bias on station heatwave days is ERA5's own?** Forecast - station = (forecast - ERA5) + (ERA5 - station), means over the rows; ensemble means.

| Forecast | Station rule | Lead | Rows | Forecast - station | = forecast - ERA5 | + ERA5 - station | ERA5's share |
|---|---|---|---|---|---|---|---|
| C3-hop | label rule (normal + 3) | all | 560 | -2.21 | -0.54 | -1.67 | 76% |
| C3-hop | label rule (normal + 3) | 1 | 112 | -1.56 | +0.11 | -1.67 | 107% |
| C3-hop | label rule (normal + 3) | 2 | 112 | -2.06 | -0.39 | -1.67 | 81% |
| C3-hop | label rule (normal + 3) | 3 | 112 | -2.33 | -0.67 | -1.67 | 71% |
| C3-hop | label rule (normal + 3) | 4 | 112 | -2.57 | -0.90 | -1.67 | 65% |
| C3-hop | label rule (normal + 3) | 5 | 112 | -2.50 | -0.84 | -1.67 | 67% |
| C3-hop | IMD rule (normal + 4.5) | all | 230 | -2.37 | -0.93 | -1.44 | 61% |
| C3-hop | IMD rule (normal + 4.5) | 1 | 46 | -1.49 | -0.05 | -1.44 | 96% |
| C3-hop | IMD rule (normal + 4.5) | 2 | 46 | -2.06 | -0.62 | -1.44 | 70% |
| C3-hop | IMD rule (normal + 4.5) | 3 | 46 | -2.43 | -0.99 | -1.44 | 59% |
| C3-hop | IMD rule (normal + 4.5) | 4 | 46 | -2.89 | -1.45 | -1.44 | 50% |
| C3-hop | IMD rule (normal + 4.5) | 5 | 46 | -2.98 | -1.54 | -1.44 | 48% |
| Delhi-only LSTM | label rule (normal + 3) | all | 560 | -2.31 | -0.65 | -1.67 | 72% |
| Delhi-only LSTM | label rule (normal + 3) | 1 | 112 | -1.54 | +0.13 | -1.67 | 109% |
| Delhi-only LSTM | label rule (normal + 3) | 2 | 112 | -1.87 | -0.20 | -1.67 | 89% |
| Delhi-only LSTM | label rule (normal + 3) | 3 | 112 | -2.39 | -0.72 | -1.67 | 70% |
| Delhi-only LSTM | label rule (normal + 3) | 4 | 112 | -2.75 | -1.08 | -1.67 | 61% |
| Delhi-only LSTM | label rule (normal + 3) | 5 | 112 | -3.03 | -1.36 | -1.67 | 55% |
| Delhi-only LSTM | IMD rule (normal + 4.5) | all | 230 | -2.57 | -1.13 | -1.44 | 56% |
| Delhi-only LSTM | IMD rule (normal + 4.5) | 1 | 46 | -1.63 | -0.19 | -1.44 | 88% |
| Delhi-only LSTM | IMD rule (normal + 4.5) | 2 | 46 | -2.02 | -0.59 | -1.44 | 71% |
| Delhi-only LSTM | IMD rule (normal + 4.5) | 3 | 46 | -2.64 | -1.20 | -1.44 | 54% |
| Delhi-only LSTM | IMD rule (normal + 4.5) | 4 | 46 | -3.10 | -1.67 | -1.44 | 46% |
| Delhi-only LSTM | IMD rule (normal + 4.5) | 5 | 46 | -3.44 | -2.00 | -1.44 | 42% |
| ridge_hw5 | label rule (normal + 3) | all | 560 | -2.29 | -0.62 | -1.67 | 73% |
| ridge_hw5 | label rule (normal + 3) | 1 | 112 | -1.22 | +0.45 | -1.67 | 137% |
| ridge_hw5 | label rule (normal + 3) | 2 | 112 | -1.75 | -0.08 | -1.67 | 95% |
| ridge_hw5 | label rule (normal + 3) | 3 | 112 | -2.34 | -0.68 | -1.67 | 71% |
| ridge_hw5 | label rule (normal + 3) | 4 | 112 | -2.86 | -1.19 | -1.67 | 58% |
| ridge_hw5 | label rule (normal + 3) | 5 | 112 | -3.26 | -1.59 | -1.67 | 51% |
| ridge_hw5 | IMD rule (normal + 4.5) | all | 230 | -2.68 | -1.24 | -1.44 | 54% |
| ridge_hw5 | IMD rule (normal + 4.5) | 1 | 46 | -1.24 | +0.19 | -1.44 | 116% |
| ridge_hw5 | IMD rule (normal + 4.5) | 2 | 46 | -1.94 | -0.50 | -1.44 | 74% |
| ridge_hw5 | IMD rule (normal + 4.5) | 3 | 46 | -2.70 | -1.26 | -1.44 | 53% |
| ridge_hw5 | IMD rule (normal + 4.5) | 4 | 46 | -3.50 | -2.07 | -1.44 | 41% |
| ridge_hw5 | IMD rule (normal + 4.5) | 5 | 46 | -4.02 | -2.58 | -1.44 | 36% |

## (iii) Global models and GraphCast 2018 against the station

Each outside forecast is paired with C3-hop's ensemble mean on the rows both have and the station observed. The global models are scored on the test period (to 2025-08-23, where the station file ends); GraphCast 2018 on fold f4's 2018 windows with C3-hop's development forecasts (2018 lies inside the mapping's fitting years).

**All days**

| Forecast | Target days | Rows | Clusters | RMSE vs ERA5: it / C3-hop | RMSE vs station: it / C3-hop | Station-mapped: it / C3-hop | Bias vs station: it / C3-hop |
|---|---|---|---|---|---|---|---|
| AIFS | 2024-03-15 to 2025-03-10 | 1140 | 3 | 2.200 / 1.876 | 3.977 / 2.480 | 3.401 / 2.002 | -3.51 / -1.87 |
| AIFS Single (added after the first results) | 2025-02-19 to 2025-08-23 | 555 | 3 | 1.878 / 2.272 | 3.481 / 2.309 | 2.867 / 1.864 | -2.96 / -1.42 |
| GFS | 2024-01-21 to 2025-08-23 | 1847 | 6 | 3.724 / 1.981 | 2.664 / 2.364 | 2.638 / 1.918 | +0.90 / -1.64 |
| GFS, Tmax since 2021 | 2021-03-25 to 2025-08-23 | 5458 | 14 | 3.439 / 2.224 | 2.535 / 2.278 | 2.559 / 1.919 | +0.90 / -1.31 |
| IFS | 2024-02-05 to 2025-08-23 | 1796 | 6 | 1.838 / 1.995 | 3.177 / 2.371 | 2.501 / 1.914 | -2.45 / -1.66 |
| IFS HRES 9 km, 12 UTC run | 2024-03-15 to 2025-08-23 | 1655 | 5 | 1.734 / 2.012 | 3.084 / 2.413 | 2.429 / 1.946 | -2.38 / -1.71 |
| GraphCast 2018 | 2018-01-01 to 2018-12-31 | 1108 | 3 | 1.182 / 1.977 | 2.433 / 2.098 | 1.950 / 1.757 | -1.85 / -1.34 |

**Heat season (15 Mar - 31 Jul)**

| Forecast | Target days | Rows | Clusters | RMSE vs ERA5: it / C3-hop | RMSE vs station: it / C3-hop | Station-mapped: it / C3-hop | Bias vs station: it / C3-hop |
|---|---|---|---|---|---|---|---|
| AIFS | 2024-03-15 to 2025-03-10 | 420 | 1 | 1.747 / 2.007 | 3.367 / 2.393 | 2.873 / 1.949 | -2.72 / -1.69 |
| AIFS Single (added after the first results) | 2025-02-19 to 2025-08-23 | 425 | 1 | 2.035 / 2.361 | 3.435 / 2.283 | 2.878 / 1.901 | -2.93 / -1.26 |
| GFS | 2024-01-21 to 2025-08-23 | 855 | 2 | 4.401 / 2.185 | 3.178 / 2.330 | 3.145 / 1.918 | +1.78 / -1.47 |
| GFS, Tmax since 2021 | 2021-03-25 to 2025-08-23 | 2261 | 5 | 4.201 / 2.530 | 3.071 / 2.388 | 3.088 / 2.075 | +1.76 / -1.20 |
| IFS | 2024-02-05 to 2025-08-23 | 855 | 2 | 1.879 / 2.185 | 2.768 / 2.330 | 2.165 / 1.918 | -1.77 / -1.47 |
| IFS HRES 9 km, 12 UTC run | 2024-03-15 to 2025-08-23 | 845 | 2 | 1.715 / 2.192 | 2.664 / 2.338 | 2.077 / 1.925 | -1.77 / -1.47 |
| GraphCast 2018 | 2018-01-01 to 2018-12-31 | 425 | 1 | 1.291 / 2.349 | 2.620 / 2.120 | 2.182 / 1.812 | -1.97 / -1.32 |

**Station heatwave days, label rule**

| Forecast | Target days | Rows | Clusters | RMSE vs ERA5: it / C3-hop | RMSE vs station: it / C3-hop | Station-mapped: it / C3-hop | Bias vs station: it / C3-hop |
|---|---|---|---|---|---|---|---|
| AIFS | 2024-03-15 to 2025-03-10 | 115 | 1 | 2.547 / 2.087 | 3.630 / 2.185 | 3.406 / 2.086 | -3.01 / -1.96 |
| AIFS Single (added after the first results) | 2025-02-19 to 2025-08-23 | 60 | 1 | 1.891 / 1.905 | 3.176 / 2.541 | 2.970 / 2.377 | -2.99 / -2.27 |
| GFS | 2024-01-21 to 2025-08-23 | 175 | 2 | 3.226 / 2.026 | 2.070 / 2.313 | 1.967 / 2.190 | +1.59 / -2.07 |
| GFS, Tmax since 2021 | 2021-03-25 to 2025-08-23 | 445 | 5 | 3.018 / 2.229 | 2.075 / 2.391 | 1.953 / 2.227 | +1.15 / -2.11 |
| IFS | 2024-02-05 to 2025-08-23 | 175 | 2 | 1.484 / 2.026 | 1.717 / 2.313 | 1.615 / 2.190 | -0.97 / -2.07 |
| IFS HRES 9 km, 12 UTC run | 2024-03-15 to 2025-08-23 | 175 | 2 | 1.494 / 2.026 | 1.906 / 2.313 | 1.797 / 2.190 | -1.46 / -2.07 |
| GraphCast 2018 | 2018-01-01 to 2018-12-31 | 80 | 1 | 1.241 / 2.077 | 2.414 / 2.455 | 2.172 / 2.147 | -1.97 / -2.30 |

**Station heatwave days, IMD rule**

| Forecast | Target days | Rows | Clusters | RMSE vs ERA5: it / C3-hop | RMSE vs station: it / C3-hop | Station-mapped: it / C3-hop | Bias vs station: it / C3-hop |
|---|---|---|---|---|---|---|---|
| AIFS | 2024-03-15 to 2025-03-10 | 70 | 1 | 2.767 / 2.321 | 3.746 / 2.452 | 3.536 / 2.357 | -3.09 / -2.25 |
| AIFS Single (added after the first results) | 2025-02-19 to 2025-08-23 | 15 | 1 | 1.698 / 1.215 | 3.076 / 1.995 | 2.888 / 1.882 | -2.83 / -1.89 |
| GFS | 2024-01-21 to 2025-08-23 | 85 | 2 | 2.680 / 2.168 | 1.724 / 2.378 | 1.621 / 2.281 | +1.24 / -2.19 |
| GFS, Tmax since 2021 | 2021-03-25 to 2025-08-23 | 185 | 4 | 2.575 / 2.153 | 1.675 / 2.610 | 1.570 / 2.457 | +1.06 / -2.31 |
| IFS | 2024-02-05 to 2025-08-23 | 85 | 2 | 1.106 / 2.168 | 1.183 / 2.378 | 1.144 / 2.281 | -0.63 / -2.19 |
| IFS HRES 9 km, 12 UTC run | 2024-03-15 to 2025-08-23 | 85 | 2 | 1.312 / 2.168 | 1.461 / 2.378 | 1.418 / 2.281 | -1.24 / -2.19 |
| GraphCast 2018 | 2018-01-01 to 2018-12-31 | 20 | 1 | 2.190 / 2.841 | 2.924 / 2.782 | 2.603 / 2.476 | -2.35 / -2.66 |

**ERA5 heatwave days (the test's extreme stratum)**

| Forecast | Target days | Rows | Clusters | RMSE vs ERA5: it / C3-hop | RMSE vs station: it / C3-hop | Station-mapped: it / C3-hop | Bias vs station: it / C3-hop |
|---|---|---|---|---|---|---|---|
| AIFS | 2024-03-15 to 2025-03-10 | 90 | 1 | 2.729 / 2.051 | 3.264 / 2.329 | 3.095 / 2.230 | -2.62 / -2.12 |
| AIFS Single (added after the first results) | 2025-02-19 to 2025-08-23 | 20 | 1 | 2.481 / 2.758 | 2.646 / 2.871 | 2.431 / 2.632 | -2.41 / -2.43 |
| GFS | 2024-01-21 to 2025-08-23 | 110 | 2 | 2.082 / 2.197 | 1.934 / 2.436 | 1.840 / 2.309 | +1.50 / -2.17 |
| GFS, Tmax since 2021 | 2021-03-25 to 2025-08-23 | 255 | 5 | 2.016 / 2.108 | 1.809 / 2.459 | 1.693 / 2.310 | +1.40 / -2.16 |
| IFS | 2024-02-05 to 2025-08-23 | 110 | 2 | 1.326 / 2.197 | 1.546 / 2.436 | 1.455 / 2.309 | -0.74 / -2.17 |
| IFS HRES 9 km, 12 UTC run | 2024-03-15 to 2025-08-23 | 110 | 2 | 1.499 / 2.197 | 1.829 / 2.436 | 1.731 / 2.309 | -1.38 / -2.17 |
| GraphCast 2018 | 2018-01-01 to 2018-12-31 | 20 | 1 | 1.049 / 1.919 | 1.554 / 2.540 | 1.342 / 2.219 | -1.37 / -2.46 |

**Paired tests** (delta = RMSE of the outside forecast minus RMSE of C3-hop; negative = the outside forecast is better).

| Comparison | Stratum | Delta vs ERA5 [95% CI] | Delta vs station [95% CI] | Delta station-mapped [95% CI] | Clear change of order |
|---|---|---|---|---|---|
| AIFS minus C3-hop | All days | +0.324 no CI (< 5 clusters), fragile (< 10 clusters) | +1.497 no CI (< 5 clusters), fragile (< 10 clusters) | +1.399 no CI (< 5 clusters), fragile (< 10 clusters) | no |
| AIFS minus C3-hop | Heat season (15 Mar - 31 Jul) | not testable | not testable | not testable | no |
| AIFS minus C3-hop | Station heatwave days, label rule | not testable | not testable | not testable | no |
| AIFS Single (added after the first results) minus C3-hop | All days | -0.394 no CI (< 5 clusters), fragile (< 10 clusters) | +1.172 no CI (< 5 clusters), fragile (< 10 clusters) | +1.002 no CI (< 5 clusters), fragile (< 10 clusters) | no |
| AIFS Single (added after the first results) minus C3-hop | Heat season (15 Mar - 31 Jul) | not testable | not testable | not testable | no |
| AIFS Single (added after the first results) minus C3-hop | Station heatwave days, label rule | not testable | not testable | not testable | no |
| GFS minus C3-hop | All days | +1.744 [+0.800, +2.688], fragile (< 10 clusters) | +0.300 [-0.617, +1.216], fragile (< 10 clusters) | +0.720 [-0.197, +1.637], fragile (< 10 clusters) | no |
| GFS minus C3-hop | Heat season (15 Mar - 31 Jul) | +2.216 no CI (< 5 clusters), fragile (< 10 clusters) | +0.848 no CI (< 5 clusters), fragile (< 10 clusters) | +1.227 no CI (< 5 clusters), fragile (< 10 clusters) | no |
| GFS minus C3-hop | Station heatwave days, label rule | +1.200 no CI (< 5 clusters), fragile (< 10 clusters) | -0.243 no CI (< 5 clusters), fragile (< 10 clusters) | -0.223 no CI (< 5 clusters), fragile (< 10 clusters) | no |
| GFS, Tmax since 2021 minus C3-hop | All days | +1.215 [+0.757, +1.673] | +0.257 [-0.085, +0.599] | +0.640 [+0.305, +0.974] | no |
| GFS, Tmax since 2021 minus C3-hop | Heat season (15 Mar - 31 Jul) | +1.671 [+1.056, +2.286], fragile (< 10 clusters) | +0.682 [+0.454, +0.911], fragile (< 10 clusters) | +1.013 [+0.730, +1.296], fragile (< 10 clusters) | no |
| GFS, Tmax since 2021 minus C3-hop | Station heatwave days, label rule | +0.789 [+0.223, +1.355], fragile (< 10 clusters) | -0.316 [-1.078, +0.447], fragile (< 10 clusters) | -0.274 [-0.941, +0.393], fragile (< 10 clusters) | no |
| IFS minus C3-hop | All days | -0.156 [-0.668, +0.355], fragile (< 10 clusters) | +0.806 [+0.301, +1.311], fragile (< 10 clusters) | +0.587 [+0.147, +1.028], fragile (< 10 clusters) | no |
| IFS minus C3-hop | Heat season (15 Mar - 31 Jul) | -0.307 no CI (< 5 clusters), fragile (< 10 clusters) | +0.438 no CI (< 5 clusters), fragile (< 10 clusters) | +0.247 no CI (< 5 clusters), fragile (< 10 clusters) | no |
| IFS minus C3-hop | Station heatwave days, label rule | -0.543 no CI (< 5 clusters), fragile (< 10 clusters) | -0.596 no CI (< 5 clusters), fragile (< 10 clusters) | -0.575 no CI (< 5 clusters), fragile (< 10 clusters) | no |
| IFS HRES 9 km, 12 UTC run minus C3-hop | All days | -0.278 [-0.726, +0.171], fragile (< 10 clusters) | +0.671 [+0.073, +1.268], fragile (< 10 clusters) | +0.483 [-0.056, +1.022], fragile (< 10 clusters) | no |
| IFS HRES 9 km, 12 UTC run minus C3-hop | Heat season (15 Mar - 31 Jul) | -0.477 no CI (< 5 clusters), fragile (< 10 clusters) | +0.326 no CI (< 5 clusters), fragile (< 10 clusters) | +0.152 no CI (< 5 clusters), fragile (< 10 clusters) | no |
| IFS HRES 9 km, 12 UTC run minus C3-hop | Station heatwave days, label rule | -0.532 no CI (< 5 clusters), fragile (< 10 clusters) | -0.408 no CI (< 5 clusters), fragile (< 10 clusters) | -0.393 no CI (< 5 clusters), fragile (< 10 clusters) | no |
| GraphCast 2018 minus C3-hop | All days | -0.795 no CI (< 5 clusters), fragile (< 10 clusters) | +0.336 no CI (< 5 clusters), fragile (< 10 clusters) | +0.193 no CI (< 5 clusters), fragile (< 10 clusters) | no |
| GraphCast 2018 minus C3-hop | Heat season (15 Mar - 31 Jul) | not testable | not testable | not testable | no |
| GraphCast 2018 minus C3-hop | Station heatwave days, label rule | not testable | not testable | not testable | no |

**By lead, all days**

| Forecast | Score | Day 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| AIFS | RMSE vs station | 3.875 | 3.999 | 4.031 | 3.986 | 3.994 |
| C3-hop on AIFS's rows | RMSE vs station | 2.123 | 2.435 | 2.542 | 2.626 | 2.641 |
| AIFS Single (added after the first results) | RMSE vs station | 3.343 | 3.533 | 3.593 | 3.525 | 3.407 |
| C3-hop on AIFS Single's rows | RMSE vs station | 2.014 | 2.235 | 2.344 | 2.425 | 2.507 |
| GFS | RMSE vs station | 2.132 | 2.474 | 2.738 | 2.890 | 2.996 |
| C3-hop on GFS's rows | RMSE vs station | 2.059 | 2.314 | 2.406 | 2.492 | 2.521 |
| GFS, Tmax since 2021 | RMSE vs station | 2.176 | 2.432 | 2.584 | 2.641 | 2.798 |
| C3-hop on GFS, Tmax since 2021's rows | RMSE vs station | 1.903 | 2.157 | 2.309 | 2.445 | 2.522 |
| IFS | RMSE vs station | 3.206 | 3.223 | 3.199 | 3.107 | 3.150 |
| C3-hop on IFS's rows | RMSE vs station | 2.066 | 2.319 | 2.412 | 2.502 | 2.532 |
| IFS HRES 9 km, 12 UTC run | RMSE vs station | 2.977 | 3.092 | 3.118 | 3.118 | 3.115 |
| C3-hop on IFS HRES 9 km, 12 UTC run's rows | RMSE vs station | 2.083 | 2.353 | 2.466 | 2.556 | 2.578 |
| GraphCast 2018 | RMSE vs station | 2.474 | 2.470 | 2.377 | 2.402 | 2.441 |
| C3-hop on GraphCast 2018's rows | RMSE vs station | 1.832 | 1.949 | 2.070 | 2.237 | 2.358 |

**Yes/no warnings** (in-season rows; R-point only, no probabilities for these comparisons).

| Forecast | Events | Rule | Hits | Misses | False alarms | Hit rate | False-alarm ratio | ETS | Frequency bias |
|---|---|---|---|---|---|---|---|---|---|
| AIFS (420 rows) | station, label rule (normal + 3) (115) | R-point | 47 | 68 | 0 | 41% | 0% | 0.33 | 0.41 |
| AIFS (420 rows) | station, label rule (normal + 3) (115) | station-mapped R-point | 44 | 71 | 0 | 38% | 0% | 0.31 | 0.38 |
| AIFS (420 rows) | station, IMD rule (normal + 4.5) (70) | R-point | 39 | 31 | 8 | 56% | 17% | 0.44 | 0.67 |
| AIFS (420 rows) | station, IMD rule (normal + 4.5) (70) | station-mapped R-point | 9 | 61 | 0 | 13% | 0% | 0.11 | 0.13 |
| AIFS (420 rows) | ERA5, label rule (reference) (90) | R-point | 47 | 43 | 0 | 52% | 0% | 0.46 | 0.52 |
| C3-hop on AIFS's rows (420 rows) | station, label rule (normal + 3) (115) | R-point | 67 | 48 | 8 | 58% | 11% | 0.45 | 0.65 |
| C3-hop on AIFS's rows (420 rows) | station, label rule (normal + 3) (115) | station-mapped R-point | 61 | 54 | 7 | 53% | 10% | 0.41 | 0.59 |
| C3-hop on AIFS's rows (420 rows) | station, IMD rule (normal + 4.5) (70) | R-point | 55 | 15 | 20 | 79% | 27% | 0.55 | 1.07 |
| C3-hop on AIFS's rows (420 rows) | station, IMD rule (normal + 4.5) (70) | station-mapped R-point | 5 | 65 | 0 | 7% | 0% | 0.06 | 0.07 |
| C3-hop on AIFS's rows (420 rows) | ERA5, label rule (reference) (90) | R-point | 55 | 35 | 20 | 61% | 27% | 0.41 | 0.83 |
| AIFS Single (added after the first results) (425 rows) | station, label rule (normal + 3) (60) | R-point | 0 | 60 | 0 | 0% | n/a | 0.00 | 0.00 |
| AIFS Single (added after the first results) (425 rows) | station, label rule (normal + 3) (60) | station-mapped R-point | 0 | 60 | 0 | 0% | n/a | 0.00 | 0.00 |
| AIFS Single (added after the first results) (425 rows) | station, IMD rule (normal + 4.5) (15) | R-point | 0 | 15 | 0 | 0% | n/a | 0.00 | 0.00 |
| AIFS Single (added after the first results) (425 rows) | station, IMD rule (normal + 4.5) (15) | station-mapped R-point | 0 | 15 | 0 | 0% | n/a | 0.00 | 0.00 |
| AIFS Single (added after the first results) (425 rows) | ERA5, label rule (reference) (20) | R-point | 0 | 20 | 0 | 0% | n/a | 0.00 | 0.00 |
| C3-hop on AIFS Single's rows (425 rows) | station, label rule (normal + 3) (60) | R-point | 3 | 57 | 1 | 5% | 25% | 0.04 | 0.07 |
| C3-hop on AIFS Single's rows (425 rows) | station, label rule (normal + 3) (60) | station-mapped R-point | 1 | 59 | 0 | 2% | 0% | 0.01 | 0.02 |
| C3-hop on AIFS Single's rows (425 rows) | station, IMD rule (normal + 4.5) (15) | R-point | 0 | 15 | 4 | 0% | 100% | -0.01 | 0.27 |
| C3-hop on AIFS Single's rows (425 rows) | station, IMD rule (normal + 4.5) (15) | station-mapped R-point | 0 | 15 | 0 | 0% | n/a | 0.00 | 0.00 |
| C3-hop on AIFS Single's rows (425 rows) | ERA5, label rule (reference) (20) | R-point | 2 | 18 | 2 | 10% | 50% | 0.08 | 0.20 |
| GFS (855 rows) | station, label rule (normal + 3) (175) | R-point | 175 | 0 | 203 | 100% | 54% | 0.32 | 2.16 |
| GFS (855 rows) | station, label rule (normal + 3) (175) | station-mapped R-point | 173 | 2 | 211 | 99% | 55% | 0.31 | 2.19 |
| GFS (855 rows) | station, IMD rule (normal + 4.5) (85) | R-point | 85 | 0 | 293 | 100% | 78% | 0.14 | 4.45 |
| GFS (855 rows) | station, IMD rule (normal + 4.5) (85) | station-mapped R-point | 84 | 1 | 206 | 99% | 71% | 0.21 | 3.41 |
| GFS (855 rows) | ERA5, label rule (reference) (110) | R-point | 110 | 0 | 268 | 100% | 71% | 0.19 | 3.44 |
| C3-hop on GFS's rows (855 rows) | station, label rule (normal + 3) (175) | R-point | 70 | 105 | 9 | 40% | 11% | 0.32 | 0.45 |
| C3-hop on GFS's rows (855 rows) | station, label rule (normal + 3) (175) | station-mapped R-point | 62 | 113 | 7 | 35% | 10% | 0.29 | 0.39 |
| C3-hop on GFS's rows (855 rows) | station, IMD rule (normal + 4.5) (85) | R-point | 55 | 30 | 24 | 65% | 30% | 0.47 | 0.93 |
| C3-hop on GFS's rows (855 rows) | station, IMD rule (normal + 4.5) (85) | station-mapped R-point | 5 | 80 | 0 | 6% | 0% | 0.05 | 0.06 |
| C3-hop on GFS's rows (855 rows) | ERA5, label rule (reference) (110) | R-point | 57 | 53 | 22 | 52% | 28% | 0.38 | 0.72 |
| GFS, Tmax since 2021 (2261 rows) | station, label rule (normal + 3) (445) | R-point | 419 | 26 | 597 | 94% | 59% | 0.26 | 2.28 |
| GFS, Tmax since 2021 (2261 rows) | station, label rule (normal + 3) (445) | station-mapped R-point | 417 | 28 | 613 | 94% | 60% | 0.25 | 2.31 |
| GFS, Tmax since 2021 (2261 rows) | station, IMD rule (normal + 4.5) (185) | R-point | 181 | 4 | 835 | 98% | 82% | 0.10 | 5.49 |
| GFS, Tmax since 2021 (2261 rows) | station, IMD rule (normal + 4.5) (185) | station-mapped R-point | 176 | 9 | 549 | 95% | 76% | 0.17 | 3.92 |
| GFS, Tmax since 2021 (2261 rows) | ERA5, label rule (reference) (280) | R-point | 280 | 0 | 736 | 100% | 72% | 0.17 | 3.63 |
| C3-hop on GFS, Tmax since 2021's rows (2261 rows) | station, label rule (normal + 3) (445) | R-point | 137 | 308 | 25 | 31% | 15% | 0.24 | 0.36 |
| C3-hop on GFS, Tmax since 2021's rows (2261 rows) | station, label rule (normal + 3) (445) | station-mapped R-point | 125 | 320 | 22 | 28% | 15% | 0.22 | 0.33 |
| C3-hop on GFS, Tmax since 2021's rows (2261 rows) | station, IMD rule (normal + 4.5) (185) | R-point | 106 | 79 | 56 | 57% | 35% | 0.41 | 0.88 |
| C3-hop on GFS, Tmax since 2021's rows (2261 rows) | station, IMD rule (normal + 4.5) (185) | station-mapped R-point | 22 | 163 | 1 | 12% | 4% | 0.11 | 0.12 |
| C3-hop on GFS, Tmax since 2021's rows (2261 rows) | ERA5, label rule (reference) (280) | R-point | 100 | 180 | 62 | 36% | 38% | 0.25 | 0.58 |
| IFS (855 rows) | station, label rule (normal + 3) (175) | R-point | 116 | 59 | 35 | 66% | 23% | 0.48 | 0.86 |
| IFS (855 rows) | station, label rule (normal + 3) (175) | station-mapped R-point | 113 | 62 | 33 | 65% | 23% | 0.47 | 0.83 |
| IFS (855 rows) | station, IMD rule (normal + 4.5) (85) | R-point | 74 | 11 | 77 | 87% | 51% | 0.40 | 1.78 |
| IFS (855 rows) | station, IMD rule (normal + 4.5) (85) | station-mapped R-point | 54 | 31 | 26 | 64% | 32% | 0.45 | 0.94 |
| IFS (855 rows) | ERA5, label rule (reference) (110) | R-point | 86 | 24 | 65 | 78% | 43% | 0.43 | 1.37 |
| C3-hop on IFS's rows (855 rows) | station, label rule (normal + 3) (175) | R-point | 70 | 105 | 9 | 40% | 11% | 0.32 | 0.45 |
| C3-hop on IFS's rows (855 rows) | station, label rule (normal + 3) (175) | station-mapped R-point | 62 | 113 | 7 | 35% | 10% | 0.29 | 0.39 |
| C3-hop on IFS's rows (855 rows) | station, IMD rule (normal + 4.5) (85) | R-point | 55 | 30 | 24 | 65% | 30% | 0.47 | 0.93 |
| C3-hop on IFS's rows (855 rows) | station, IMD rule (normal + 4.5) (85) | station-mapped R-point | 5 | 80 | 0 | 6% | 0% | 0.05 | 0.06 |
| C3-hop on IFS's rows (855 rows) | ERA5, label rule (reference) (110) | R-point | 57 | 53 | 22 | 52% | 28% | 0.38 | 0.72 |
| IFS HRES 9 km, 12 UTC run (845 rows) | station, label rule (normal + 3) (175) | R-point | 106 | 69 | 19 | 61% | 15% | 0.48 | 0.71 |
| IFS HRES 9 km, 12 UTC run (845 rows) | station, label rule (normal + 3) (175) | station-mapped R-point | 98 | 77 | 18 | 56% | 16% | 0.44 | 0.66 |
| IFS HRES 9 km, 12 UTC run (845 rows) | station, IMD rule (normal + 4.5) (85) | R-point | 70 | 15 | 55 | 82% | 44% | 0.45 | 1.47 |
| IFS HRES 9 km, 12 UTC run (845 rows) | station, IMD rule (normal + 4.5) (85) | station-mapped R-point | 39 | 46 | 11 | 46% | 22% | 0.37 | 0.59 |
| IFS HRES 9 km, 12 UTC run (845 rows) | ERA5, label rule (reference) (110) | R-point | 82 | 28 | 43 | 75% | 34% | 0.48 | 1.14 |
| C3-hop on IFS HRES 9 km, 12 UTC run's rows (845 rows) | station, label rule (normal + 3) (175) | R-point | 70 | 105 | 9 | 40% | 11% | 0.32 | 0.45 |
| C3-hop on IFS HRES 9 km, 12 UTC run's rows (845 rows) | station, label rule (normal + 3) (175) | station-mapped R-point | 62 | 113 | 7 | 35% | 10% | 0.28 | 0.39 |
| C3-hop on IFS HRES 9 km, 12 UTC run's rows (845 rows) | station, IMD rule (normal + 4.5) (85) | R-point | 55 | 30 | 24 | 65% | 30% | 0.47 | 0.93 |
| C3-hop on IFS HRES 9 km, 12 UTC run's rows (845 rows) | station, IMD rule (normal + 4.5) (85) | station-mapped R-point | 5 | 80 | 0 | 6% | 0% | 0.05 | 0.06 |
| C3-hop on IFS HRES 9 km, 12 UTC run's rows (845 rows) | ERA5, label rule (reference) (110) | R-point | 57 | 53 | 22 | 52% | 28% | 0.38 | 0.72 |
| GraphCast 2018 (425 rows) | station, label rule (normal + 3) (80) | R-point | 22 | 58 | 0 | 28% | 0% | 0.24 | 0.28 |
| GraphCast 2018 (425 rows) | station, label rule (normal + 3) (80) | station-mapped R-point | 22 | 58 | 0 | 28% | 0% | 0.24 | 0.28 |
| GraphCast 2018 (425 rows) | station, IMD rule (normal + 4.5) (20) | R-point | 10 | 10 | 12 | 50% | 55% | 0.29 | 1.10 |
| GraphCast 2018 (425 rows) | station, IMD rule (normal + 4.5) (20) | station-mapped R-point | 0 | 20 | 0 | 0% | n/a | 0.00 | 0.00 |
| GraphCast 2018 (425 rows) | ERA5, label rule (reference) (20) | R-point | 14 | 6 | 8 | 70% | 36% | 0.48 | 1.10 |
| C3-hop on GraphCast 2018's rows (425 rows) | station, label rule (normal + 3) (80) | R-point | 3 | 77 | 1 | 4% | 25% | 0.03 | 0.05 |
| C3-hop on GraphCast 2018's rows (425 rows) | station, label rule (normal + 3) (80) | station-mapped R-point | 3 | 77 | 1 | 4% | 25% | 0.03 | 0.05 |
| C3-hop on GraphCast 2018's rows (425 rows) | station, IMD rule (normal + 4.5) (20) | R-point | 2 | 18 | 2 | 10% | 50% | 0.08 | 0.20 |
| C3-hop on GraphCast 2018's rows (425 rows) | station, IMD rule (normal + 4.5) (20) | station-mapped R-point | 0 | 20 | 0 | 0% | n/a | 0.00 | 0.00 |
| C3-hop on GraphCast 2018's rows (425 rows) | ERA5, label rule (reference) (20) | R-point | 0 | 20 | 4 | 0% | 100% | -0.01 | 0.20 |

**Cold bias on station heatwave days**

| Forecast | Station rule | Lead | Rows | Forecast - station | = forecast - ERA5 | + ERA5 - station | ERA5's share |
|---|---|---|---|---|---|---|---|
| AIFS | label rule (normal + 3) | all | 115 | -3.01 | -1.99 | -1.01 | 34% |
| AIFS | label rule (normal + 3) | 1 | 23 | -1.86 | -0.85 | -1.01 | 55% |
| AIFS | label rule (normal + 3) | 2 | 23 | -2.50 | -1.49 | -1.01 | 41% |
| AIFS | label rule (normal + 3) | 3 | 23 | -3.25 | -2.23 | -1.01 | 31% |
| AIFS | label rule (normal + 3) | 4 | 23 | -3.61 | -2.59 | -1.01 | 28% |
| AIFS | label rule (normal + 3) | 5 | 23 | -3.82 | -2.81 | -1.01 | 27% |
| AIFS | IMD rule (normal + 4.5) | all | 70 | -3.09 | -2.27 | -0.82 | 27% |
| AIFS | IMD rule (normal + 4.5) | 1 | 14 | -1.94 | -1.12 | -0.82 | 42% |
| AIFS | IMD rule (normal + 4.5) | 2 | 14 | -2.59 | -1.77 | -0.82 | 32% |
| AIFS | IMD rule (normal + 4.5) | 3 | 14 | -3.31 | -2.49 | -0.82 | 25% |
| AIFS | IMD rule (normal + 4.5) | 4 | 14 | -3.70 | -2.88 | -0.82 | 22% |
| AIFS | IMD rule (normal + 4.5) | 5 | 14 | -3.92 | -3.10 | -0.82 | 21% |
| C3-hop on AIFS's rows | label rule (normal + 3) | all | 115 | -1.96 | -0.94 | -1.01 | 52% |
| C3-hop on AIFS's rows | label rule (normal + 3) | 1 | 23 | -1.41 | -0.40 | -1.01 | 72% |
| C3-hop on AIFS's rows | label rule (normal + 3) | 2 | 23 | -1.79 | -0.78 | -1.01 | 57% |
| C3-hop on AIFS's rows | label rule (normal + 3) | 3 | 23 | -2.06 | -1.04 | -1.01 | 49% |
| C3-hop on AIFS's rows | label rule (normal + 3) | 4 | 23 | -2.29 | -1.27 | -1.01 | 44% |
| C3-hop on AIFS's rows | label rule (normal + 3) | 5 | 23 | -2.23 | -1.21 | -1.01 | 46% |
| C3-hop on AIFS's rows | IMD rule (normal + 4.5) | all | 70 | -2.25 | -1.43 | -0.82 | 36% |
| C3-hop on AIFS's rows | IMD rule (normal + 4.5) | 1 | 14 | -1.45 | -0.63 | -0.82 | 57% |
| C3-hop on AIFS's rows | IMD rule (normal + 4.5) | 2 | 14 | -2.08 | -1.26 | -0.82 | 39% |
| C3-hop on AIFS's rows | IMD rule (normal + 4.5) | 3 | 14 | -2.38 | -1.56 | -0.82 | 34% |
| C3-hop on AIFS's rows | IMD rule (normal + 4.5) | 4 | 14 | -2.67 | -1.85 | -0.82 | 31% |
| C3-hop on AIFS's rows | IMD rule (normal + 4.5) | 5 | 14 | -2.69 | -1.87 | -0.82 | 31% |
| AIFS Single (added after the first results) | label rule (normal + 3) | all | 60 | -2.99 | -1.74 | -1.25 | 42% |
| AIFS Single (added after the first results) | label rule (normal + 3) | 1 | 12 | -2.71 | -1.46 | -1.25 | 46% |
| AIFS Single (added after the first results) | label rule (normal + 3) | 2 | 12 | -3.06 | -1.81 | -1.25 | 41% |
| AIFS Single (added after the first results) | label rule (normal + 3) | 3 | 12 | -3.15 | -1.90 | -1.25 | 40% |
| AIFS Single (added after the first results) | label rule (normal + 3) | 4 | 12 | -3.12 | -1.87 | -1.25 | 40% |
| AIFS Single (added after the first results) | label rule (normal + 3) | 5 | 12 | -2.91 | -1.66 | -1.25 | 43% |
| AIFS Single (added after the first results) | IMD rule (normal + 4.5) | all | 15 | -2.83 | -1.54 | -1.29 | 46% |
| AIFS Single (added after the first results) | IMD rule (normal + 4.5) | 1 | 3 | -2.57 | -1.28 | -1.29 | 50% |
| AIFS Single (added after the first results) | IMD rule (normal + 4.5) | 2 | 3 | -2.89 | -1.60 | -1.29 | 45% |
| AIFS Single (added after the first results) | IMD rule (normal + 4.5) | 3 | 3 | -3.02 | -1.73 | -1.29 | 43% |
| AIFS Single (added after the first results) | IMD rule (normal + 4.5) | 4 | 3 | -2.99 | -1.70 | -1.29 | 43% |
| AIFS Single (added after the first results) | IMD rule (normal + 4.5) | 5 | 3 | -2.67 | -1.37 | -1.29 | 48% |
| C3-hop on AIFS Single's rows | label rule (normal + 3) | all | 60 | -2.27 | -1.02 | -1.25 | 55% |
| C3-hop on AIFS Single's rows | label rule (normal + 3) | 1 | 12 | -1.16 | +0.10 | -1.25 | 108% |
| C3-hop on AIFS Single's rows | label rule (normal + 3) | 2 | 12 | -2.02 | -0.77 | -1.25 | 62% |
| C3-hop on AIFS Single's rows | label rule (normal + 3) | 3 | 12 | -2.66 | -1.41 | -1.25 | 47% |
| C3-hop on AIFS Single's rows | label rule (normal + 3) | 4 | 12 | -2.74 | -1.49 | -1.25 | 46% |
| C3-hop on AIFS Single's rows | label rule (normal + 3) | 5 | 12 | -2.80 | -1.54 | -1.25 | 45% |
| C3-hop on AIFS Single's rows | IMD rule (normal + 4.5) | all | 15 | -1.89 | -0.60 | -1.29 | 68% |
| C3-hop on AIFS Single's rows | IMD rule (normal + 4.5) | 1 | 3 | -0.98 | +0.31 | -1.29 | 132% |
| C3-hop on AIFS Single's rows | IMD rule (normal + 4.5) | 2 | 3 | -1.65 | -0.36 | -1.29 | 78% |
| C3-hop on AIFS Single's rows | IMD rule (normal + 4.5) | 3 | 3 | -2.10 | -0.80 | -1.29 | 62% |
| C3-hop on AIFS Single's rows | IMD rule (normal + 4.5) | 4 | 3 | -2.20 | -0.91 | -1.29 | 59% |
| C3-hop on AIFS Single's rows | IMD rule (normal + 4.5) | 5 | 3 | -2.52 | -1.23 | -1.29 | 51% |
| GFS | label rule (normal + 3) | all | 175 | +1.59 | +2.69 | -1.10 | -69% |
| GFS | label rule (normal + 3) | 1 | 35 | +1.43 | +2.53 | -1.10 | -77% |
| GFS | label rule (normal + 3) | 2 | 35 | +1.77 | +2.87 | -1.10 | -62% |
| GFS | label rule (normal + 3) | 3 | 35 | +1.66 | +2.76 | -1.10 | -66% |
| GFS | label rule (normal + 3) | 4 | 35 | +1.47 | +2.57 | -1.10 | -75% |
| GFS | label rule (normal + 3) | 5 | 35 | +1.63 | +2.73 | -1.10 | -67% |
| GFS | IMD rule (normal + 4.5) | all | 85 | +1.24 | +2.15 | -0.90 | -73% |
| GFS | IMD rule (normal + 4.5) | 1 | 17 | +1.21 | +2.11 | -0.90 | -75% |
| GFS | IMD rule (normal + 4.5) | 2 | 17 | +1.55 | +2.46 | -0.90 | -58% |
| GFS | IMD rule (normal + 4.5) | 3 | 17 | +1.34 | +2.24 | -0.90 | -68% |
| GFS | IMD rule (normal + 4.5) | 4 | 17 | +0.95 | +1.85 | -0.90 | -95% |
| GFS | IMD rule (normal + 4.5) | 5 | 17 | +1.17 | +2.07 | -0.90 | -78% |
| C3-hop on GFS's rows | label rule (normal + 3) | all | 175 | -2.07 | -0.97 | -1.10 | 53% |
| C3-hop on GFS's rows | label rule (normal + 3) | 1 | 35 | -1.33 | -0.23 | -1.10 | 83% |
| C3-hop on GFS's rows | label rule (normal + 3) | 2 | 35 | -1.87 | -0.78 | -1.10 | 59% |
| C3-hop on GFS's rows | label rule (normal + 3) | 3 | 35 | -2.26 | -1.17 | -1.10 | 48% |
| C3-hop on GFS's rows | label rule (normal + 3) | 4 | 35 | -2.44 | -1.35 | -1.10 | 45% |
| C3-hop on GFS's rows | label rule (normal + 3) | 5 | 35 | -2.42 | -1.33 | -1.10 | 45% |
| C3-hop on GFS's rows | IMD rule (normal + 4.5) | all | 85 | -2.19 | -1.29 | -0.90 | 41% |
| C3-hop on GFS's rows | IMD rule (normal + 4.5) | 1 | 17 | -1.37 | -0.46 | -0.90 | 66% |
| C3-hop on GFS's rows | IMD rule (normal + 4.5) | 2 | 17 | -2.00 | -1.10 | -0.90 | 45% |
| C3-hop on GFS's rows | IMD rule (normal + 4.5) | 3 | 17 | -2.33 | -1.43 | -0.90 | 39% |
| C3-hop on GFS's rows | IMD rule (normal + 4.5) | 4 | 17 | -2.58 | -1.68 | -0.90 | 35% |
| C3-hop on GFS's rows | IMD rule (normal + 4.5) | 5 | 17 | -2.66 | -1.75 | -0.90 | 34% |
| GFS, Tmax since 2021 | label rule (normal + 3) | all | 445 | +1.15 | +2.50 | -1.36 | -118% |
| GFS, Tmax since 2021 | label rule (normal + 3) | 1 | 89 | +0.94 | +2.30 | -1.36 | -144% |
| GFS, Tmax since 2021 | label rule (normal + 3) | 2 | 89 | +1.20 | +2.56 | -1.36 | -113% |
| GFS, Tmax since 2021 | label rule (normal + 3) | 3 | 89 | +1.14 | +2.50 | -1.36 | -119% |
| GFS, Tmax since 2021 | label rule (normal + 3) | 4 | 89 | +1.18 | +2.54 | -1.36 | -115% |
| GFS, Tmax since 2021 | label rule (normal + 3) | 5 | 89 | +1.26 | +2.62 | -1.36 | -107% |
| GFS, Tmax since 2021 | IMD rule (normal + 4.5) | all | 185 | +1.06 | +2.14 | -1.09 | -103% |
| GFS, Tmax since 2021 | IMD rule (normal + 4.5) | 1 | 37 | +0.96 | +2.05 | -1.09 | -113% |
| GFS, Tmax since 2021 | IMD rule (normal + 4.5) | 2 | 37 | +1.24 | +2.33 | -1.09 | -88% |
| GFS, Tmax since 2021 | IMD rule (normal + 4.5) | 3 | 37 | +1.13 | +2.21 | -1.09 | -96% |
| GFS, Tmax since 2021 | IMD rule (normal + 4.5) | 4 | 37 | +0.99 | +2.08 | -1.09 | -110% |
| GFS, Tmax since 2021 | IMD rule (normal + 4.5) | 5 | 37 | +0.97 | +2.05 | -1.09 | -113% |
| C3-hop on GFS, Tmax since 2021's rows | label rule (normal + 3) | all | 445 | -2.11 | -0.75 | -1.36 | 64% |
| C3-hop on GFS, Tmax since 2021's rows | label rule (normal + 3) | 1 | 89 | -1.44 | -0.09 | -1.36 | 94% |
| C3-hop on GFS, Tmax since 2021's rows | label rule (normal + 3) | 2 | 89 | -1.95 | -0.60 | -1.36 | 69% |
| C3-hop on GFS, Tmax since 2021's rows | label rule (normal + 3) | 3 | 89 | -2.23 | -0.87 | -1.36 | 61% |
| C3-hop on GFS, Tmax since 2021's rows | label rule (normal + 3) | 4 | 89 | -2.47 | -1.12 | -1.36 | 55% |
| C3-hop on GFS, Tmax since 2021's rows | label rule (normal + 3) | 5 | 89 | -2.44 | -1.08 | -1.36 | 56% |
| C3-hop on GFS, Tmax since 2021's rows | IMD rule (normal + 4.5) | all | 185 | -2.31 | -1.23 | -1.09 | 47% |
| C3-hop on GFS, Tmax since 2021's rows | IMD rule (normal + 4.5) | 1 | 37 | -1.37 | -0.29 | -1.09 | 79% |
| C3-hop on GFS, Tmax since 2021's rows | IMD rule (normal + 4.5) | 2 | 37 | -2.00 | -0.92 | -1.09 | 54% |
| C3-hop on GFS, Tmax since 2021's rows | IMD rule (normal + 4.5) | 3 | 37 | -2.41 | -1.32 | -1.09 | 45% |
| C3-hop on GFS, Tmax since 2021's rows | IMD rule (normal + 4.5) | 4 | 37 | -2.85 | -1.76 | -1.09 | 38% |
| C3-hop on GFS, Tmax since 2021's rows | IMD rule (normal + 4.5) | 5 | 37 | -2.93 | -1.85 | -1.09 | 37% |
| IFS | label rule (normal + 3) | all | 175 | -0.97 | +0.12 | -1.10 | 113% |
| IFS | label rule (normal + 3) | 1 | 35 | -1.65 | -0.55 | -1.10 | 67% |
| IFS | label rule (normal + 3) | 2 | 35 | -1.29 | -0.19 | -1.10 | 85% |
| IFS | label rule (normal + 3) | 3 | 35 | -0.82 | +0.28 | -1.10 | 134% |
| IFS | label rule (normal + 3) | 4 | 35 | -0.68 | +0.42 | -1.10 | 161% |
| IFS | label rule (normal + 3) | 5 | 35 | -0.43 | +0.66 | -1.10 | 253% |
| IFS | IMD rule (normal + 4.5) | all | 85 | -0.63 | +0.27 | -0.90 | 143% |
| IFS | IMD rule (normal + 4.5) | 1 | 17 | -1.25 | -0.35 | -0.90 | 72% |
| IFS | IMD rule (normal + 4.5) | 2 | 17 | -0.97 | -0.06 | -0.90 | 94% |
| IFS | IMD rule (normal + 4.5) | 3 | 17 | -0.46 | +0.44 | -0.90 | 195% |
| IFS | IMD rule (normal + 4.5) | 4 | 17 | -0.44 | +0.47 | -0.90 | 206% |
| IFS | IMD rule (normal + 4.5) | 5 | 17 | -0.04 | +0.86 | -0.90 | 2032% |
| C3-hop on IFS's rows | label rule (normal + 3) | all | 175 | -2.07 | -0.97 | -1.10 | 53% |
| C3-hop on IFS's rows | label rule (normal + 3) | 1 | 35 | -1.33 | -0.23 | -1.10 | 83% |
| C3-hop on IFS's rows | label rule (normal + 3) | 2 | 35 | -1.87 | -0.78 | -1.10 | 59% |
| C3-hop on IFS's rows | label rule (normal + 3) | 3 | 35 | -2.26 | -1.17 | -1.10 | 48% |
| C3-hop on IFS's rows | label rule (normal + 3) | 4 | 35 | -2.44 | -1.35 | -1.10 | 45% |
| C3-hop on IFS's rows | label rule (normal + 3) | 5 | 35 | -2.42 | -1.33 | -1.10 | 45% |
| C3-hop on IFS's rows | IMD rule (normal + 4.5) | all | 85 | -2.19 | -1.29 | -0.90 | 41% |
| C3-hop on IFS's rows | IMD rule (normal + 4.5) | 1 | 17 | -1.37 | -0.46 | -0.90 | 66% |
| C3-hop on IFS's rows | IMD rule (normal + 4.5) | 2 | 17 | -2.00 | -1.10 | -0.90 | 45% |
| C3-hop on IFS's rows | IMD rule (normal + 4.5) | 3 | 17 | -2.33 | -1.43 | -0.90 | 39% |
| C3-hop on IFS's rows | IMD rule (normal + 4.5) | 4 | 17 | -2.58 | -1.68 | -0.90 | 35% |
| C3-hop on IFS's rows | IMD rule (normal + 4.5) | 5 | 17 | -2.66 | -1.75 | -0.90 | 34% |
| IFS HRES 9 km, 12 UTC run | label rule (normal + 3) | all | 175 | -1.46 | -0.36 | -1.10 | 75% |
| IFS HRES 9 km, 12 UTC run | label rule (normal + 3) | 1 | 35 | -1.66 | -0.56 | -1.10 | 66% |
| IFS HRES 9 km, 12 UTC run | label rule (normal + 3) | 2 | 35 | -1.56 | -0.46 | -1.10 | 70% |
| IFS HRES 9 km, 12 UTC run | label rule (normal + 3) | 3 | 35 | -1.49 | -0.39 | -1.10 | 74% |
| IFS HRES 9 km, 12 UTC run | label rule (normal + 3) | 4 | 35 | -1.44 | -0.34 | -1.10 | 76% |
| IFS HRES 9 km, 12 UTC run | label rule (normal + 3) | 5 | 35 | -1.15 | -0.06 | -1.10 | 95% |
| IFS HRES 9 km, 12 UTC run | IMD rule (normal + 4.5) | all | 85 | -1.24 | -0.34 | -0.90 | 73% |
| IFS HRES 9 km, 12 UTC run | IMD rule (normal + 4.5) | 1 | 17 | -1.30 | -0.40 | -0.90 | 69% |
| IFS HRES 9 km, 12 UTC run | IMD rule (normal + 4.5) | 2 | 17 | -1.36 | -0.46 | -0.90 | 66% |
| IFS HRES 9 km, 12 UTC run | IMD rule (normal + 4.5) | 3 | 17 | -1.33 | -0.43 | -0.90 | 68% |
| IFS HRES 9 km, 12 UTC run | IMD rule (normal + 4.5) | 4 | 17 | -1.24 | -0.34 | -0.90 | 73% |
| IFS HRES 9 km, 12 UTC run | IMD rule (normal + 4.5) | 5 | 17 | -0.96 | -0.05 | -0.90 | 95% |
| C3-hop on IFS HRES 9 km, 12 UTC run's rows | label rule (normal + 3) | all | 175 | -2.07 | -0.97 | -1.10 | 53% |
| C3-hop on IFS HRES 9 km, 12 UTC run's rows | label rule (normal + 3) | 1 | 35 | -1.33 | -0.23 | -1.10 | 83% |
| C3-hop on IFS HRES 9 km, 12 UTC run's rows | label rule (normal + 3) | 2 | 35 | -1.87 | -0.78 | -1.10 | 59% |
| C3-hop on IFS HRES 9 km, 12 UTC run's rows | label rule (normal + 3) | 3 | 35 | -2.26 | -1.17 | -1.10 | 48% |
| C3-hop on IFS HRES 9 km, 12 UTC run's rows | label rule (normal + 3) | 4 | 35 | -2.44 | -1.35 | -1.10 | 45% |
| C3-hop on IFS HRES 9 km, 12 UTC run's rows | label rule (normal + 3) | 5 | 35 | -2.42 | -1.33 | -1.10 | 45% |
| C3-hop on IFS HRES 9 km, 12 UTC run's rows | IMD rule (normal + 4.5) | all | 85 | -2.19 | -1.29 | -0.90 | 41% |
| C3-hop on IFS HRES 9 km, 12 UTC run's rows | IMD rule (normal + 4.5) | 1 | 17 | -1.37 | -0.46 | -0.90 | 66% |
| C3-hop on IFS HRES 9 km, 12 UTC run's rows | IMD rule (normal + 4.5) | 2 | 17 | -2.00 | -1.10 | -0.90 | 45% |
| C3-hop on IFS HRES 9 km, 12 UTC run's rows | IMD rule (normal + 4.5) | 3 | 17 | -2.33 | -1.43 | -0.90 | 39% |
| C3-hop on IFS HRES 9 km, 12 UTC run's rows | IMD rule (normal + 4.5) | 4 | 17 | -2.58 | -1.68 | -0.90 | 35% |
| C3-hop on IFS HRES 9 km, 12 UTC run's rows | IMD rule (normal + 4.5) | 5 | 17 | -2.66 | -1.75 | -0.90 | 34% |
| GraphCast 2018 | label rule (normal + 3) | all | 80 | -1.97 | +0.26 | -2.24 | 113% |
| GraphCast 2018 | label rule (normal + 3) | 1 | 16 | -2.22 | +0.02 | -2.24 | 101% |
| GraphCast 2018 | label rule (normal + 3) | 2 | 16 | -1.91 | +0.33 | -2.24 | 117% |
| GraphCast 2018 | label rule (normal + 3) | 3 | 16 | -1.90 | +0.34 | -2.24 | 118% |
| GraphCast 2018 | label rule (normal + 3) | 4 | 16 | -1.98 | +0.26 | -2.24 | 113% |
| GraphCast 2018 | label rule (normal + 3) | 5 | 16 | -1.86 | +0.38 | -2.24 | 120% |
| GraphCast 2018 | IMD rule (normal + 4.5) | all | 20 | -2.35 | +0.98 | -3.33 | 142% |
| GraphCast 2018 | IMD rule (normal + 4.5) | 1 | 4 | -2.64 | +0.69 | -3.33 | 126% |
| GraphCast 2018 | IMD rule (normal + 4.5) | 2 | 4 | -2.23 | +1.10 | -3.33 | 150% |
| GraphCast 2018 | IMD rule (normal + 4.5) | 3 | 4 | -2.35 | +0.98 | -3.33 | 142% |
| GraphCast 2018 | IMD rule (normal + 4.5) | 4 | 4 | -2.30 | +1.03 | -3.33 | 145% |
| GraphCast 2018 | IMD rule (normal + 4.5) | 5 | 4 | -2.23 | +1.10 | -3.33 | 149% |
| C3-hop on GraphCast 2018's rows | label rule (normal + 3) | all | 80 | -2.30 | -0.06 | -2.24 | 97% |
| C3-hop on GraphCast 2018's rows | label rule (normal + 3) | 1 | 16 | -1.77 | +0.47 | -2.24 | 126% |
| C3-hop on GraphCast 2018's rows | label rule (normal + 3) | 2 | 16 | -2.11 | +0.12 | -2.24 | 106% |
| C3-hop on GraphCast 2018's rows | label rule (normal + 3) | 3 | 16 | -2.27 | -0.03 | -2.24 | 99% |
| C3-hop on GraphCast 2018's rows | label rule (normal + 3) | 4 | 16 | -2.63 | -0.39 | -2.24 | 85% |
| C3-hop on GraphCast 2018's rows | label rule (normal + 3) | 5 | 16 | -2.71 | -0.47 | -2.24 | 83% |
| C3-hop on GraphCast 2018's rows | IMD rule (normal + 4.5) | all | 20 | -2.66 | +0.67 | -3.33 | 125% |
| C3-hop on GraphCast 2018's rows | IMD rule (normal + 4.5) | 1 | 4 | -1.90 | +1.43 | -3.33 | 176% |
| C3-hop on GraphCast 2018's rows | IMD rule (normal + 4.5) | 2 | 4 | -2.67 | +0.67 | -3.33 | 125% |
| C3-hop on GraphCast 2018's rows | IMD rule (normal + 4.5) | 3 | 4 | -2.91 | +0.42 | -3.33 | 114% |
| C3-hop on GraphCast 2018's rows | IMD rule (normal + 4.5) | 4 | 4 | -3.06 | +0.27 | -3.33 | 109% |
| C3-hop on GraphCast 2018's rows | IMD rule (normal + 4.5) | 5 | 4 | -2.77 | +0.56 | -3.33 | 120% |

## Notes

- Station file: `sources/data/safdarjung_daily_tidy.csv` (not in the repository), SHA-256 a8fe98424ff2829c12601dbbe67de02fe07639cdf68a4ff9fdb4d78c24bbbe1b.
- Global-model values: `followup/data/open_meteo_runs_daily.parquet`, SHA-256 58ceaaaca724b4dbffbd79ea21717d9aa322efddc4ec5cec8b9d6e64660c4ccd.
- GraphCast 2018: included.
- Still to add, after P3-2 has run: GraphCast 2020 (P3-2a); the P3-2 corrector variants.
- WBGT is not part of this check: the station record has no humidity.
- A single station is not the truth for a 75 km box either: its own siting and the city around it shape its readings. The check asks whether conclusions depend on which record is used.
