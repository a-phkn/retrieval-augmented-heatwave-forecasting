# Proposed amendment A7 to `docs/PROTOCOL_V2.md`: station data limited to the correctly dated years

Proposed on 25 Nov 2026 by Sudarshan, on the branch `v2-station-fix` (made from `experiments-v2` at 5534094), for the
study's author to accept or reject. It was written before any corrected number was computed. It changes no file of
the study: the rerun uses the study's own code unchanged.

## Why

The GHCN-Daily station files the study reads are misdated or nearly empty after 2010. Source: P3-3 (11 Nov) and
P3-7 part B (25 Nov) in `context/decisions_phase3.md` on the branch `phase3-extras`.

- **Safdarjung** (`sources/data/safdarjung_daily_tidy.csv`, SHA-256 a8fe9842...):
  - from 2011, a day's value is often the higher of that day and the day before;
  - from 2016, about 40% of days are missing, and most of the rest are the previous day's maximum.
- **Chennai, Minambakkam** (IN020040900): the same two-day signature in 2011-2015. 32% of consecutive values are
  identical, against 7% in 1980-2008. Only 60% of days are present from 2016.
- **Ahmedabad** (IN005010600): 25% of days are present in 2011-2015, and 2% after.

## Change

Every analysis that reads station Tmax uses station values up to 2010-12-31 only. Later days count as days with no
station value, which the code already handles as missing.

The outputs affected:
- `h_a_delhi.json`: H-A, and its S1 and S3 alert populations, from Safdarjung;
- `graded_delhi.json`: A1, S2 alert days from Safdarjung;
- `station_checks.json`: A5;
- `alert_gap.json`: Section 18, post-test exploration.

Not affected: everything computed from ERA5 alone. That covers the city window tests on ERA5 alert days, H-C, E5,
the insurance audits on ERA5, and the test-period scoring.

## How it is run

`python -m v2_fix.station_safe_years --last-year 2010`
- The study's four runners run unchanged.
- Their station-file paths point to copies without the rows after 2010.
- Their outputs go to `results_v2_fix/station_to_2010/`, next to a comparison with the registered numbers.

`python -m v2_fix.station_safe_years --check` passes the full files through the same path. It must reproduce the
committed `results_v2` files, and it is run first.

**Note added 25 Nov, after the first check run:** byte-for-byte equality is too strict between machines.
- The study's own `v2.run_window_test`, run unchanged here, differs from the committed `h_a_delhi.json` in 9 CI bounds,
  in the 16th-17th significant digit (floating-point summation order).
- So the check compares values to one part in 10^9.
- `delhi_hourly_dev_shade.parquet` was not among the received files. It was rebuilt here with the study's own
  `python -m v2.build_shade_wbgt` from `delhi_hourly_dev.parquet`.

## Reporting

- The registered results stay as they are. The corrected numbers are reported beside them, as a correction for
  station data quality found after the results, not as a new test.
- If an H-A decision (supported or not) changes under the correction, both decisions are reported, and the text
  says which one rests on correctly dated station data.
- **Expected (written now):**
  - each analysis has fewer alert days;
  - the A5 ERA5-minus-station gaps shrink, for Delhi towards the 1980-2010 value of about -0.5 °C;
  - the H-A window result keeps its direction, since it also held on ERA5 alert days.
