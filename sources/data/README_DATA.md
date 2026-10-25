# Safdarjung daily data (GHCN-Daily)

- `SAFDARJUNG_GHCN_DAILY.csv`: unzipped original, GHCN-Daily long format, no header. Columns: station id, date (YYYYMMDD), element, value, measurement flag, quality flag, source flag, observation time. TMAX/TMIN/PRCP/TAVG values are in tenths of a unit (tenths of C, tenths of mm). Station IN022021900. The original file has no station-name metadata; the station identity as Safdarjung comes from the file name the user gave it, and is consistent with IMD's 2010 and 2015 annual maxima (45.4 C on 17 May 2010; 45.5 C on 25 and 26 May 2015).
- `safdarjung_daily_tidy.csv`: one row per day, header `date,tmax_c,tmin_c,prcp_mm,tmax_qflag,tmax_sflag`, values converted to C and mm. TMAX has 18,080 days (1944-01-01 to 2025-08-23); 4 TMAX values carry a quality flag (3 'O', 1 'I'). Gaps exist (for example 17-19 Apr 2010 are missing); do not assume a continuous series.
- `phase6_label_windows_ghcn.json`: Tmax summary for each Part C window. These are derived from GHCN-Daily and are not IMD-declared heat-wave dates.
- The original compressed file was moved to `sources/_to_delete/` (recoverable; delete it yourself once you are happy).
