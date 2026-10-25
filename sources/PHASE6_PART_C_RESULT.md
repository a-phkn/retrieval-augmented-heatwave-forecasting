# Phase 6 Part C result: are the 4 label-test heatwave dates real?

Status 5 Oct 2026. Short answer: **no period has IMD-declared Delhi dates.** 1998 is supported by an IMD sub-division spell, 2010 by an IMD region-level spell, and 2002 and 2015 only by derived station data. Full detail: `PHASE6_SOURCE_RESEARCH_REPORT.md` section 14. All PDFs are in `sources/pdfs/`; extracted text in `sources/txt/`.

## Per period

### 1998 (brief: ~24 May-1 Jun) — SUPPORTED at sub-division level, dates differ slightly
- Source A: Pai, Thapliyal, Kokate, "Decadal variation in the heat and cold waves over India during 1971-2000", MAUSAM 55(2), Apr 2004 (IMD Pune). Table 4, PDF p.8 (printed p.288), file `PAPER_HW_1971-2000.pdf`. Sub-division 13 (Haryana, Chandigarh & Delhi): heat-wave spell **22 May-3 Jun 1998**, 13 days.
- Source B: IMD monograph "Heat and Cold Waves in India", https://mausam.imd.gov.in/imd_latest/contents/Met_Monograph_Cold_Heat_Waves.pdf, Table 4.2 PDF p.61 (printed p.58): north-west India heat wave 21-28 May 1998, 43.4 C, 8 days; Table 5.1 PDF p.77: severe 15-28 May 1998.
- IMD-declared Delhi station dates: not given in any document.
- Peak Tmax, Safdarjung (GHCN-Daily, derived): **46.5 C on 28 May 1998**; 24 May-1 Jun all >= 43.4 C.

### 2002 (brief: ~10-20 May) — NOT SUPPORTED by IMD for Delhi
- IMD monograph Table 5.1 PDF p.77: north-west India severe spell **1-8 May 2002** (absent from Table 4.2). East coast 9-15 May 2002 (monograph PDF p.95; Pai et al. 2013, MAUSAM 64(4) p.588 says it was an Andhra Pradesh event).
- IMD-declared Delhi dates: none found.
- Peak Tmax, Safdarjung (GHCN, derived): **46.0 C on 18 May 2002**; every day 10-20 May >= 41 C; 1-8 May was cooler (max 43.0 C). So the brief window fits the station data but is not an IMD-listed Delhi spell.
- Recommendation: relabel as "derived from Safdarjung Tmax" or swap out.

### 2010 (brief: ~10-20 Apr) — SUPPORTED at region level, weak at Safdarjung
- IMD monograph Table 4.2 PDF p.61: north-west India heat wave **8-20 Apr 2010**, 43.1 C, 13 days; Table 5.1 PDF p.77: severe 5-11 and 13-19 Apr (also 12-17 May, 19-26 May). IMD Annual Climate Summary 2010 (file `IMD_CLIMATE_SUMMARY_2010.pdf`) PDF p.21: figure "heat wave 14-20 April"; PDF p.6: warmest pre-monsoon season since 1901.
- IMD-declared Delhi dates: none.
- Peak Tmax, Safdarjung (GHCN, derived): **42.9 C on 16 Apr 2010** (3 days missing in the window). Only moderate. IMD annual max for "New Delhi AP" was 45 C (17, 18, 22, 25, 26 May and 20 Jun), Annual Climate Summary 2010 PDF p.24, not April.

### 2015 (brief: ~22-26 May) — WEAK for Delhi
- IMD Annual Climate Summary 2015 (file `IMD_CLIMATE_SUMMARY_2015.pdf`) PDF p.5: severe and widespread heat wave over north/north-west, central, eastern and south-east India "during second fortnight of May". Table 2 PDF p.25: New Delhi AP annual max **45.5 C on 25 May**.
- IMD monograph Table 4.2 lists no 2015 north-west India event.
- IMD-declared Delhi dates: none.
- Peak Tmax, Safdarjung (GHCN, derived): **45.5 C on 25 and 26 May 2015**; 22-26 May all >= 43.6 C.

## Caveats to carry forward
- The IMD monograph and the Pai papers give regional/sub-division spells (90th-percentile or IMD 1988 criteria), not Delhi station declarations.
- "New Delhi AP" in IMD summaries is the airport station; no IMD document here is labelled Safdarjung.
- GHCN-Daily station IN022021900 is treated as Safdarjung (user-supplied file; matches IMD annual maxima for 2010 and 2015). Data and tidy CSV: `sources/data/`. Derived windows: `sources/data/phase6_label_windows_ghcn.json`. These are not IMD declarations and do not use IMD's departure-from-normal criterion.
- Do not describe the label test as validated against IMD for Delhi. Label each window "IMD region/sub-division spell" or "derived from Safdarjung Tmax".
