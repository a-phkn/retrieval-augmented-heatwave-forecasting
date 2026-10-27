# Rule engine: how often each colour fires on 2007-2018 (descriptive checks)

Development years only; season Mar 15 - Jul 31; normals from each fold's training years. Written by `python -m advisory.colour_check` (decisions.md 2026-10-26 and 2026-10-27).

## A. Perfect forecast (the observed Tmax given as the forecast)

Season days by the colour of their heat-wave run (1668 days):

| Colour | Days | Share |
|---|---|---|
| GREEN | 1526 | 91.5% |
| YELLOW | 33 | 2.0% |
| ORANGE | 43 | 2.6% |
| RED | 66 | 4.0% |

Headline colour of the 5-day advisory, one per season day (1668 advisories):

| Year | GREEN | YELLOW | ORANGE | RED |
|---|---|---|---|---|
| 2007 | 122 | 12 | 5 | 0 |
| 2008 | 130 | 9 | 0 | 0 |
| 2009 | 115 | 3 | 11 | 10 |
| 2010 | 80 | 15 | 21 | 23 |
| 2011 | 139 | 0 | 0 | 0 |
| 2012 | 108 | 4 | 15 | 12 |
| 2013 | 129 | 2 | 3 | 5 |
| 2014 | 111 | 11 | 2 | 15 |
| 2015 | 125 | 8 | 6 | 0 |
| 2016 | 123 | 16 | 0 | 0 |
| 2017 | 121 | 11 | 1 | 6 |
| 2018 | 132 | 2 | 5 | 0 |
| **all** | **1435** (86%) | **93** (6%) | **69** (4%) | **71** (4%) |

## B. Real forecasts: C3-hop, 1668 season advisories

Forecast heatwave day under the probability rule: hot-day probability >= 0.37 (the advisory's rule); under the value rule: the forecast Tmax itself meets the label rule (the old rule). 'Perfect' = the colour the observed weather would have given for the same 5 days.

| Headline colour share | GREEN | YELLOW | ORANGE | RED |
|---|---|---|---|---|
| Perfect forecast | 86% | 6% | 4% | 4% |
| Probability rule | 87% | 3% | 5% | 5% |
| Value rule (old) | 89% | 3% | 4% | 4% |

| Rule | Perfect-forecast alerts (Yellow+) also raised | Forecast alerts not in the perfect forecast | Same colour |
|---|---|---|---|
| Probability rule | 68% | 27% (218 alerts vs 233) | 88% |
| Value rule (old) | 61% | 22% (183 alerts vs 233) | 89% |

Probability rule: number of advisories by colour

| Perfect forecast \ forecast | GREEN | YELLOW | ORANGE | RED |
|---|---|---|---|---|
| GREEN | 1376 | 25 | 22 | 12 |
| YELLOW | 51 | 17 | 23 | 2 |
| ORANGE | 16 | 10 | 30 | 13 |
| RED | 7 | 0 | 14 | 50 |

Value rule (old): number of advisories by colour

| Perfect forecast \ forecast | GREEN | YELLOW | ORANGE | RED |
|---|---|---|---|---|
| GREEN | 1395 | 19 | 11 | 10 |
| YELLOW | 59 | 17 | 16 | 1 |
| ORANGE | 22 | 8 | 26 | 13 |
| RED | 9 | 0 | 14 | 48 |

