# Station data cut at 2010: the registered numbers and the corrected ones

Proposed amendment A7 (`docs/PROPOSED_A7_station_years.md`). The study's code ran unchanged, with station Tmax files that stop at 2010-12-31 (rows dropped: safdarjung 5229, IN005010600 5285, IN020040900 5338). Registered = `results_v2/`, corrected = this folder. Full detail is in the JSON files here.

How to read it:
- "No CI" means fewer than 5 year clusters remain, so no decision is possible. It is not a rejection.
- H-A and A1 test on the alert days of 2007-2018. With station data only to the cut year, the test keeps only 2007-2010.
- The shares of unsafe or rest hours outside a window are computed over all the development years' alert days (for example 41 -> 33 clusters for light work), so they keep their CIs.

## H-A, Delhi window test (`h_a_delhi.json`)

| Work / status | Alert days | Best window minus W1 | Supported | Unsafe hours outside 12-4 pm |
|---|---|---|---|---|
| low/acclimatised | 1342 -> 1062 | +0.036 [+0.021, +0.051] -> +0.054 (no CI, 4 clusters) | True -> no test | 0.57 -> 0.57 |
| low/unacclimatised | 1342 -> 1062 | +0.014 [+0.003, +0.024] -> +0.030 (no CI, 4 clusters) | True -> no test | 0.60 -> 0.60 |
| moderate/acclimatised | 1342 -> 1062 | +0.003 [-0.000, +0.007] -> +0.009 (no CI, 4 clusters) | False -> no test | 0.63 -> 0.63 |
| moderate/unacclimatised | 1342 -> 1062 | +0.000 [-0.000, +0.001] -> +0.001 (no CI, 4 clusters) | False -> no test | 0.69 -> 0.68 |
| high/acclimatised | 1342 -> 1062 | +0.000 [-0.000, +0.001] -> +0.001 (no CI, 4 clusters) | False -> no test | 0.69 -> 0.68 |
| high/unacclimatised | 1342 -> 1062 | +0.000 [+0.000, +0.000] -> +0.000 (no CI, 4 clusters) | False -> no test | 0.72 -> 0.72 |
| very_high/acclimatised | 1342 -> 1062 | +0.000 [+0.000, +0.000] -> +0.000 (no CI, 4 clusters) | False -> no test | 0.70 -> 0.70 |
| very_high/unacclimatised | 1342 -> 1062 | +0.000 [+0.000, +0.000] -> +0.000 (no CI, 4 clusters) | False -> no test | 0.73 -> 0.73 |

Alert-population sensitivities:

| Population | Alert days | Best window minus W1 |
|---|---|---|
| S1 IMD (2 days, Safdarjung) | 146 -> 108 | +0.003 [-0.004, +0.009] -> +0.007 (no CI, 3 clusters) |
| S3 HAP departure >= +4 | 502 -> 388 | +0.015 [-0.010, +0.041] -> +0.036 (no CI, 4 clusters) |
| ERA5 9-cell Tmax proxy | 582 -> 582 | +0.007 [-0.003, +0.017] -> +0.007 [-0.003, +0.017] |

## A1, graded analysis (`graded_delhi.json`, first rows of h_a1)

| Work / level / WBGT | Alert days | Best window minus W1 | Rest needed outside 12-4 pm |
|---|---|---|---|
| light / TLV / wbgt | 1342 -> 1062 | +0.087 [+0.053, +0.120] -> +0.109 (no CI, 4 clusters) | 0.53 -> 0.54 |
| light / TLV / wbgt_shade | 1342 -> 1062 | +0.017 [-0.010, +0.045] -> +0.044 (no CI, 3 clusters) | 0.36 -> 0.36 |
| moderate / TLV / wbgt | 1342 -> 1062 | +0.025 [+0.014, +0.035] -> +0.039 (no CI, 4 clusters) | 0.59 -> 0.59 |
| moderate / TLV / wbgt_shade | 1342 -> 1062 | +0.002 [-0.007, +0.010] -> +0.011 (no CI, 4 clusters) | 0.54 -> 0.55 |
| heavy / TLV / wbgt | 1342 -> 1062 | +0.011 [+0.006, +0.017] -> +0.020 (no CI, 4 clusters) | 0.65 -> 0.65 |
| heavy / TLV / wbgt_shade | 1342 -> 1062 | +0.001 [-0.003, +0.004] -> +0.005 (no CI, 4 clusters) | 0.65 -> 0.65 |
| very_heavy / TLV / wbgt | 1342 -> 1062 | +0.006 [+0.004, +0.009] -> +0.010 (no CI, 4 clusters) | 0.69 -> 0.69 |
| very_heavy / TLV / wbgt_shade | 1342 -> 1062 | +0.001 [-0.001, +0.003] -> +0.003 (no CI, 4 clusters) | 0.69 -> 0.69 |
| light / AL / wbgt | 1342 -> 1062 | +0.016 [+0.008, +0.023] -> +0.028 (no CI, 4 clusters) | 0.60 -> 0.60 |
| light / AL / wbgt_shade | 1342 -> 1062 | +0.003 [-0.006, +0.011] -> +0.013 (no CI, 4 clusters) | 0.55 -> 0.55 |
| moderate / AL / wbgt | 1342 -> 1062 | +0.003 [+0.001, +0.005] -> +0.006 (no CI, 4 clusters) | 0.67 -> 0.67 |
| moderate / AL / wbgt_shade | 1342 -> 1062 | +0.000 [-0.002, +0.002] -> +0.003 (no CI, 4 clusters) | 0.64 -> 0.64 |
| heavy / AL / wbgt | 1342 -> 1062 | +0.001 [-0.000, +0.001] -> +0.002 (no CI, 4 clusters) | 0.70 -> 0.70 |
| heavy / AL / wbgt_shade | 1342 -> 1062 | +0.000 [-0.001, +0.002] -> +0.002 (no CI, 4 clusters) | 0.69 -> 0.69 |
| very_heavy / AL / wbgt | 1342 -> 1062 | +0.000 [-0.000, +0.000] -> +0.000 (no CI, 4 clusters) | 0.71 -> 0.71 |
| very_heavy / AL / wbgt_shade | 1342 -> 1062 | -0.001 [-0.002, +0.000] -> +0.000 (no CI, 4 clusters) | 0.70 -> 0.70 |

## A5, station checks (`station_checks.json`)

| City | Days | ERA5 minus station, in season (°C) |
|---|---|---|
| delhi | 7139 -> 5855 | -0.71 -> -0.49 |
| ahmedabad | 5508 -> 5229 | -1.28 -> -1.26 |
| chennai | 6900 -> 5601 | -1.54 -> -1.48 |

| City | Trigger | Station years | Seasons paid, station trigger | Same years, ERA5 trigger |
|---|---|---|---|---|
| delhi | MHT 45.27/1d | 34 -> 29 | 0.38 -> 0.31 | 0.22 -> 0.25 |
| delhi | MHT 46.53/1d | 34 -> 29 | 0.06 -> 0.07 | 0.03 -> 0.03 |
| delhi | MHT 47/1d | 34 -> 29 | 0.06 -> 0.07 | 0.00 -> 0.00 |
| ahmedabad | VimoSEWA 43.2/7d | 10 -> 10 | 0.80 -> 0.80 | 0.11 -> 0.11 |
| ahmedabad | SEWA 43.6/2c | 10 -> 10 | 0.90 -> 0.90 | 0.23 -> 0.23 |
| ahmedabad | SEWA 44.1/2c | 10 -> 10 | 0.60 -> 0.60 | 0.07 -> 0.07 |

## Section 18, alert gap (`alert_gap.json`)

| City | Alert days | Any-unsafe-hour days alerted | Rest-heavy days alerted |
|---|---|---|---|
| delhi | 146 -> 108 | 0.021 -> 0.016 | 0.051 -> 0.036 |
| ahmedabad | 575 -> 575 | 0.078 -> 0.078 | 0.223 -> 0.223 |
| chennai | 27 -> 20 | 0.003 -> 0.003 | 0.017 -> 0.012 |

Ahmedabad's alert gap uses the ERA5 rule (no station series), so it does not change.

