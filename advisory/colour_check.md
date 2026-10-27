# Rule engine: how often each colour fires on observed 2007-2018 Tmax (descriptive check)

Perfect-forecast case: the engine is given the observed Tmax as its 'forecast'. Season Mar 15 - Jul 31; normals from each fold's training years. Written by `python -m advisory.colour_check` (decisions.md 2026-10-26).

## Season days by the colour of their heat-wave run (1668 days)

| Colour | Days | Share |
|---|---|---|
| GREEN | 1526 | 91.5% |
| YELLOW | 33 | 2.0% |
| ORANGE | 43 | 2.6% |
| RED | 66 | 4.0% |

## Headline colour of the 5-day advisory, one per season day (1668 advisories)

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

