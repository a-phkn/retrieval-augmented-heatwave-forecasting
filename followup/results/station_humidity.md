# Delhi's ERA5 dew point against the stations, heat seasons 1980-2018 (P3-7 step 4)

Pre-registered 2026-11-25 (P3-7 step 4) and changed by the deviation of 2026-11-27, before any dew-point value was read (`context/decisions_phase3.md`). Descriptive: no pass rule and no prediction. Why: `evaluation_v2/premise_report.md` found ERA5's vapour pressure at Tmax stepping around 2000-01, and a station record shows whether such a step is in the air or in the reanalysis.

- Station: the mean of the readings at the 8 synoptic times of the IST day (02:30 to 23:30 IST, every 3 hours).
- ERA5: its own 2 m dew point at the same instants (the mean of the hourly values either side of each half hour), as the mean of the 9 cells and at the cell nearest the station.
- A day needs at least 7 of the 8 instants with both values, and both means use those instants. A heat season (15 Mar - 31 Jul) counts if at least 70 of its days have a value.

## Summary

| Station | Seasons that count | Change of ERA5 - station, 1990-1999 to 2002-2011: 9 cells [95% CI] | Nearest cell [95% CI] | Trend per decade (°C): station | ERA5, 9 cells | ERA5, nearest cell |
|---|---|---|---|---|---|---|
| Safdarjung | 38 (1980-2018) | -0.50 °C [-0.83, -0.20] (9 and 10 seasons) | -0.52 °C [-0.86, -0.20] (9 and 10 seasons) | +0.97 (1980-2018, 38 seasons) | +0.71 (1980-2018, 38 seasons) | +0.71 (1980-2018, 38 seasons) |
| Palam (Indira Gandhi International) | 15 (2004-2018) | n/a (no season in one of the decades) | n/a (no season in one of the decades) | +2.40 (2004-2018, 15 seasons) | +2.14 (2004-2018, 15 seasons) | +2.12 (2004-2018, 15 seasons) |

The change is the mean over the 2002-2011 seasons minus the mean over the 1990-1999 seasons. Its CI resamples the seasons within each decade (10,000 resamples, seed 0). 2000 and 2001, the years of the step, are in neither decade.

## Safdarjung

Raw files read: 39 (1980-2018). Days with a value: 12997, of which 4922 in the heat season. Nearest ERA5 cell: cell 5 (28.50N 77.25E).

- ERA5 - station, 9 cells: +0.01 °C in 1990-1999, -0.49 °C in 2002-2011; change -0.50 °C [-0.83, -0.20] (9 and 10 seasons).
- ERA5 - station, nearest cell: +0.00 °C in 1990-1999, -0.51 °C in 2002-2011; change -0.52 °C [-0.86, -0.20] (9 and 10 seasons).

| Season | Days | Station (°C) | ERA5, 9 cells | ERA5, nearest cell | ERA5 - station, 9 cells | ERA5 - station, nearest cell |
|---|---|---|---|---|---|---|
| 1980 | 137 | 15.79 | 16.39 | 16.24 | +0.59 | +0.44 |
| 1981 | 132 | 16.69 | 16.74 | 16.65 | +0.06 | -0.04 |
| 1982 | 133 | 16.64 | 17.18 | 17.18 | +0.54 | +0.54 |
| 1983 | 134 | 17.25 | 17.72 | 17.70 | +0.46 | +0.44 |
| 1984 | 138 | 17.24 | 17.09 | 17.02 | -0.15 | -0.22 |
| 1985 | 129 | 16.42 | 16.62 | 16.57 | +0.20 | +0.15 |
| 1986 | 137 | 17.42 | 17.00 | 17.00 | -0.42 | -0.42 |
| 1987 | 130 | 16.57 | 16.30 | 16.32 | -0.27 | -0.25 |
| 1988 | 125 | 16.92 | 16.43 | 16.30 | -0.50 | -0.62 |
| 1989 | 113 | 15.24 | 14.99 | 14.97 | -0.24 | -0.26 |
| 1990 | 102 | 18.08 | 18.08 | 18.10 | +0.01 | +0.03 |
| 1991 | 101 | 16.64 | 16.36 | 16.36 | -0.28 | -0.29 |
| 1992 (not counted: fewer than 70 days) | 15 | 12.11 | 11.76 | 11.80 | -0.34 | -0.31 |
| 1993 | 123 | 16.43 | 16.68 | 16.70 | +0.25 | +0.27 |
| 1994 | 118 | 16.73 | 16.96 | 17.01 | +0.23 | +0.27 |
| 1995 | 131 | 16.04 | 15.99 | 15.88 | -0.05 | -0.16 |
| 1996 | 121 | 16.86 | 17.63 | 17.58 | +0.77 | +0.72 |
| 1997 | 119 | 18.54 | 18.66 | 18.76 | +0.12 | +0.22 |
| 1998 | 119 | 18.47 | 18.17 | 18.12 | -0.31 | -0.36 |
| 1999 | 125 | 17.40 | 16.76 | 16.74 | -0.64 | -0.66 |
| 2000 | 125 | 19.70 | 18.66 | 18.65 | -1.04 | -1.05 |
| 2001 | 134 | 20.68 | 19.48 | 19.47 | -1.20 | -1.21 |
| 2002 | 118 | 17.59 | 16.88 | 16.83 | -0.72 | -0.76 |
| 2003 | 128 | 17.80 | 17.47 | 17.50 | -0.32 | -0.29 |
| 2004 | 134 | 18.58 | 18.45 | 18.35 | -0.13 | -0.23 |
| 2005 | 138 | 16.15 | 16.00 | 16.08 | -0.15 | -0.07 |
| 2006 | 139 | 18.65 | 18.52 | 18.54 | -0.13 | -0.11 |
| 2007 | 136 | 19.59 | 18.97 | 18.96 | -0.62 | -0.63 |
| 2008 | 139 | 19.53 | 18.97 | 18.98 | -0.55 | -0.54 |
| 2009 | 133 | 17.99 | 17.24 | 17.22 | -0.76 | -0.77 |
| 2010 | 129 | 18.30 | 17.12 | 17.04 | -1.18 | -1.26 |
| 2011 | 128 | 19.79 | 19.42 | 19.33 | -0.37 | -0.46 |
| 2012 | 136 | 17.56 | 16.86 | 16.85 | -0.70 | -0.71 |
| 2013 | 136 | 19.81 | 18.98 | 19.02 | -0.83 | -0.80 |
| 2014 | 136 | 19.84 | 18.41 | 18.48 | -1.43 | -1.36 |
| 2015 | 137 | 20.30 | 19.32 | 19.36 | -0.98 | -0.94 |
| 2016 | 138 | 20.48 | 20.27 | 20.17 | -0.21 | -0.31 |
| 2017 | 139 | 19.65 | 19.38 | 19.19 | -0.27 | -0.45 |
| 2018 | 137 | 19.68 | 19.60 | 19.36 | -0.08 | -0.32 |

## Palam (Indira Gandhi International)

Raw files read: 28 (1980-2018). Days with a value: 4619, of which 1768 in the heat season. Nearest ERA5 cell: cell 4 (28.50N 77.00E).


| Season | Days | Station (°C) | ERA5, 9 cells | ERA5, nearest cell | ERA5 - station, 9 cells | ERA5 - station, nearest cell |
|---|---|---|---|---|---|---|
| 2002 (not counted: fewer than 70 days) | 20 | 21.71 | 22.36 | 22.21 | +0.65 | +0.50 |
| 2003 (not counted: fewer than 70 days) | 47 | 22.35 | 22.91 | 22.71 | +0.56 | +0.36 |
| 2004 | 120 | 16.94 | 18.32 | 17.83 | +1.38 | +0.89 |
| 2005 | 117 | 15.51 | 16.42 | 16.45 | +0.91 | +0.94 |
| 2006 | 115 | 17.46 | 19.28 | 18.94 | +1.82 | +1.48 |
| 2007 | 113 | 19.92 | 19.29 | 18.84 | -0.63 | -1.08 |
| 2008 | 125 | 16.21 | 19.01 | 18.94 | +2.79 | +2.73 |
| 2009 | 104 | 16.71 | 17.91 | 17.58 | +1.20 | +0.87 |
| 2010 | 113 | 15.42 | 17.73 | 17.45 | +2.31 | +2.03 |
| 2011 | 119 | 18.06 | 19.82 | 19.61 | +1.76 | +1.55 |
| 2012 | 110 | 15.40 | 17.56 | 17.49 | +2.16 | +2.09 |
| 2013 | 113 | 16.96 | 19.31 | 19.22 | +2.35 | +2.26 |
| 2014 | 108 | 16.16 | 18.86 | 18.75 | +2.70 | +2.59 |
| 2015 | 111 | 16.95 | 19.55 | 19.35 | +2.60 | +2.40 |
| 2016 | 106 | 21.24 | 22.19 | 21.91 | +0.95 | +0.67 |
| 2017 | 120 | 20.39 | 20.45 | 20.00 | +0.06 | -0.40 |
| 2018 | 107 | 21.29 | 20.86 | 20.47 | -0.42 | -0.82 |

## Notes

- Only station and ERA5 values of 1980-2018 are read; the test years are not used.
- The 8 synoptic readings miss part of the daily cycle, so neither mean is a full-day mean. ERA5 is taken at the same instants, so the difference is like-for-like.
- Palam has no dew-point readings at these times before 2002, so its 1990s cannot be compared.
- A single station's dew point also reflects its surroundings, such as irrigation, green space and building around it. A change in ERA5 - station can come from the station's side as well as from ERA5's.
