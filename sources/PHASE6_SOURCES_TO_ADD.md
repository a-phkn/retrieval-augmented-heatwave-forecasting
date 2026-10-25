# Phase 6: sources we could still add

Status as of 5 Oct 2026: source gathering is considered done for now (135 actions, 46 thresholds, all quotes verified). This list is what would improve the library if the documents can be found. Nothing here is required to start building.

Save PDFs in `sources\pdfs\` using the suggested filename, then ask for them to be read and verified the same way as the others. Pages that are not PDFs can be saved with Print to PDF.

## Higher value

| Source | Why it would help | Where to look | Suggested filename | Notes |
|---|---|---|---|---|
| CEA advisory "Summer Preparedness and Distribution System Reliability Measures" (July 2026) | Could replace the vague NDMA "re-schedule load shedding" line with specific DISCOM measures. Best chance to lift Part B. | cea.nic.in (Distribution Planning and Development circulars); search the exact title. Reported by Power Line, Business Standard, SolarQuarter, Power Peak Digest (news only, not citable). | `CEA_SUMMER_DISCOM_ADVISORY.pdf` | Not the same document as the El Nino advisory already in the library. Several searches found no PDF. |
| Delhi SLDC or Delhi Transco summer preparedness / contingency plan | Would give operating rules for the Delhi grid in heat. | delhisldc.org (Reports); Delhi Transco Ltd site | `DELHI_SLDC_<name>.pdf` | The 2023-24 and 2024-25 annual reports are already read: operational statistics only, no heatwave protocol. Earlier years (2007-08 onward) are unlikely to help. |
| Any Indian WBGT or heat-stress occupational limit (Ministry of Labour, DGFASLI, NIOH) | The library has no Indian WBGT value; only WHO/WMO Box 6 (OSHA-based) and NIOSH equations. | DGFASLI, NIOH Ahmedabad, ICMR-NIOH publications | `INDIA_WBGT_<source>.pdf` | A 2026 review says the OSH Code sets no actionable limits, so this may not exist. |

## Medium value

| Source | Why it would help | Where to look | Suggested filename | Notes |
|---|---|---|---|---|
| NCDC 2024 advisory for State Health Departments on the heat wave season | Same ground as the 2025 advisory; adds a year of history. | https://www.ncdc.mohfw.gov.in/wp-content/uploads/2024/03/Advisory-for-State-Health-Department-on-heat-wave-season-2024_NPCCHH.pdf | `NCDC_ADVISORY_STATES_2024.pdf` | The link showed ERR_BLOCKED_BY_CLIENT in the user's browser: try a private window with extensions off, or another browser. |
| ISO 7243:2017 full text | Official WBGT reference values (Annex A). The iTeh sample has front matter only. | iso.org/standard/67188.html (paywalled); a university or institutional library | `ISO_7243_2017.pdf` | Copyright: only short quotes would be usable. |
| Labour Ministry heatwave advisory letter (2026) | Primary letter behind the PIB release; may list more measures. | labour.gov.in; the PIB release (Release ID 2256186) may link it | `MOLE_ADVISORY_2026_LETTER.pdf` | A PIB summary and screenshot are already in the library, so this adds little. |

## Low value or optional

| Source | Why it would help | Notes |
|---|---|---|
| NPCCHH Report of Heat Related Activities 2024 (already in the folder) | Only one directive taken; the text layer is partly garbled. Re-read if more state-level detail is wanted. | https://heathealth.info/wp-content/uploads/Report-of-Heat-Related-Activities-2024_NPCCHH-compressed.pdf is a mirror that may extract more cleanly. |
| NDMA 2019 Guidelines, pages not yet cited | Roughly 50 of 62 scanned pages are unused. Each page needs OCR and a visual check, so the cost is high. | Already in the folder. |
| Delhi HAP 2025 Annexure 2 (communication plan, PDF pp.145-153) | Skimmed, not mined. Likely messaging guidance rather than actions. | Already in the folder. |
| Delhi HAP 2025 Annexure 4 (Do's and Don'ts, PDF pp.160-163) | Hindi infographics. Would need Hindi quotes and translation handling. | Already in the folder. |
| CEA power supply position reports (monthly, national) | Statistics only. Possible demand baseline for the forecasting side, not actions. | https://cea.nic.in/power-supply/?lang=en |

## Searched and not found (do not repeat without a new lead)

- DERC heatwave-specific order or circular (only the 2018-19 time-of-day tariff notice and general tariff orders exist).
- BSES Rajdhani summer plan (only BSES Yamuna demand-side slides, which are not heatwave-specific).
- Grid-India heatwave advisory (news only).
- Tata Power-DDL summer press release (12 Mar 2026): saved page is image-only and the site menu covers the text. Re-saving it with Reader View might make it quotable, but it describes DISCOM operations, not heatwave actions.

## Known issues to remember when building

- 123 of 135 actions have `tier_min` UNSPECIFIED (1 YELLOW, 7 ORANGE, 4 RED are the only named tiers). The rule engine will need a documented fallback.
- The 12-4 PM work-avoidance window is from a Delhi Labour Department circular, so it applies to workplaces, not the public.
- Source conflicts are listed in section 4 of the main report (including 24 C vs 25 C for AC use, and 108/102 vs 112 for ambulances).

## Part C (partly done)

The IMD monograph and the 2010, 2015 and 2016 climate summaries are now in the library (section 14 of the report). Still missing: IMD-declared Delhi dates and Safdarjung Tmax for 1998 and 2002, and Safdarjung (not airport) daily Tmax for all four years.

| Source | Why | Suggested filename |
|---|---|---|
| Peer-reviewed or IMD research papers on the 1998 and 2002 heat waves with Delhi data | Delhi dates and values for those years | PAPER_HW_<year>.pdf |
| Safdarjung daily Tmax (for example NOAA GHCN-Daily, or IMD data supply) | Derived labels if no declaration exists; not an official declaration | SAFDARJUNG_DAILY_TMAX.csv |
