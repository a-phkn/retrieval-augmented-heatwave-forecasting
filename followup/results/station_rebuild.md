# P3-7: station records rebuilt from NOAA ISD

Pre-registered in `context/decisions_phase3.md` (P3-7, 2026-11-25). Daily files: `followup/data/isd_daily_<station>.csv` (IST days). Step 1 (the census) is in `followup/results/station_rebuild_census.json`.

## Days with each value

`Tmax used` is the series used from here on, after the deviation of 2026-11-25 (section at the end).

| Station | Years | Days | Tmax reported | Tmax hourly | Tmin reported | Tmin hourly | Dew point mean | Tmax used |
|---|---|---|---|---|---|---|---|---|
| safdarjung | 1980-2008 | 10593 | 10095 | 0 | 9927 | 0 | 0 | 10095 |
| safdarjung | 2009-2010 | 730 | 717 | 0 | 718 | 0 | 0 | 717 |
| safdarjung | 2011-2015 | 1826 | 1776 | 0 | 1786 | 0 | 0 | 1776 |
| safdarjung | 2016-2025 | 3524 | 245 | 1 | 3460 | 0 | 1 | 3437 |
| palam | 1980-2008 | 10593 | 35 | 1046 | 5 | 1766 | 1995 | 31 |
| palam | 2009-2010 | 730 | 1 | 339 | 1 | 619 | 709 | 1 |
| palam | 2011-2015 | 1826 | 24 | 806 | 23 | 1501 | 1735 | 0 |
| palam | 2016-2025 | 3524 | 2493 | 3283 | 3441 | 3419 | 3439 | 0 |
| ahmedabad | 1980-2008 | 10593 | 9754 | 7 | 9564 | 31 | 29 | 9109 |
| ahmedabad | 2009-2010 | 730 | 720 | 0 | 722 | 0 | 0 | 720 |
| ahmedabad | 2011-2015 | 1826 | 1806 | 24 | 1377 | 38 | 42 | 840 |
| ahmedabad | 2016-2025 | 3524 | 203 | 3038 | 3490 | 3402 | 3447 | 0 |
| chennai | 1980-2008 | 10593 | 9999 | 0 | 9839 | 3 | 0 | 9313 |
| chennai | 2009-2010 | 730 | 723 | 0 | 723 | 0 | 0 | 723 |
| chennai | 2011-2015 | 1826 | 1794 | 47 | 1723 | 43 | 47 | 840 |
| chennai | 2016-2025 | 3524 | 231 | 3388 | 3449 | 3464 | 3493 | 0 |

## Step 2(a): each station against itself

Reported Tmax against the station's own hourly Tmax, on days both exist. Passes if the lag-0 mean absolute difference is the smallest and the reported value is more than 0.5 °C below the hourly one on at most 5% of days.

| Station | Days | MAD lag -1 / 0 / +1 (°C) | Reported minus hourly (°C) | Share > 0.5 °C below | Passes |
|---|---|---|---|---|---|
| safdarjung | 0 | - | - | - | no day has both values |
| palam | 2371 | 1.70 / 0.70 / 1.65 | +0.50 | 7.2% | False |
| ahmedabad | 210 | 1.39 / 0.64 / 1.51 | -0.36 | 38.1% | False |
| chennai | 265 | 1.31 / 0.57 / 1.25 | +0.14 | 15.8% | False |

## Step 2(b): Safdarjung against GHCN, 1980-2008

Passes if the lag-0 mean absolute difference is the smallest, the mean difference is within ±0.3 °C and at least 80% of days agree within 0.5 °C. Tmin is descriptive.

| Series | Days | MAD lag -1 / 0 / +1 (°C) | Rebuilt minus GHCN (°C) | Within 0.5 °C | Passes |
|---|---|---|---|---|---|
| Tmax reported | 9971 | 1.45 / 0.00 / 1.44 | -0.00 | 99.9% | True |
| Tmin reported | 8758 | 1.23 / 0.00 / 1.37 | +0.00 | 100.0% | descriptive |

## Deviation of 2026-11-25 (A and B; post hoc, see the decision log)

| Check | Days | MAD lag -1 / 0 / +1 (°C) | Mean difference (°C) | Within 0.5 °C | Passes |
|---|---|---|---|---|---|
| (A) Safdarjung: next morning's 24-hour max vs the 12 UTC max | 940 | 1.52 / 0.02 / 1.51 | -0.01 | 99.5% | True |
| (B) palam: 12-hour groups vs own hourly Tmax | 0 | - | - | - | no day has both values |
| (B) ahmedabad: 12-hour groups vs own hourly Tmax | 6 | 0.97 / 0.87 / 1.66 | +0.87 | 0.0% below | True |
| (B) ahmedabad: 12-hour groups vs GHCN, 1980-2008 | 8343 | 1.08 / 0.00 / 1.08 | -0.00 | 99.9% | True |
| (B) chennai: 12-hour groups vs own hourly Tmax | 0 | - | - | - | no day has both values |
| (B) chennai: 12-hour groups vs GHCN, 1980-2008 | 8805 | 1.09 / 0.00 / 1.09 | -0.00 | 100.0% | True |

Safdarjung days filled from 2016 (A): 3192. Stations left out under (B): none.
