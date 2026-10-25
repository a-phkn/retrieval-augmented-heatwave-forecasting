# Phase 6 source research: Delhi heatwave advisory action library

Updated 2026-10-05 (final totals: **135 actions, 58 thresholds**). Part A (health) is done; Part B (energy) is done to the extent public documents allow and is rated PARTIAL (section 12); Part C (heatwave dates) is partly confirmed (section 14): 1998 is supported at IMD sub-division level, 2010 at region level, 2002 and 2015 only by derived station data. Sections 1-8 below were written in earlier passes; where a count or status there differs from this paragraph, this paragraph and sections 9-15 win.

Every `verbatim_excerpt` in `phase6_candidate_actions.json` (135 entries) and `phase6_thresholds.json` (58 entries) was checked by script against the extracted text of the cited PDF page (whitespace, hyphens and quote marks ignored). Text-based PDFs matched exactly in the original pass; later passes were re-run through the same verifier with 0 hard failures (135 actions, 46 thresholds, re-run 5 Oct 2026; image-only documents and scanned pages are verified by eye and marked in the `verification` field). The NDMA 2019 Guidelines are a scanned PDF with no text layer, so I read them with OCR and then looked at the page images for every page I quote from; those entries are marked in the `verification` field. Nothing comes from memory, news or search snippets.

---

## 1. Source register

| doc_id | Title | Publisher | Year / version | URL | Accessible? | Scope | Notes |
|---|---|---|---|---|---|---|---|
| NDMA_HW_GUIDELINES_2019 | National Guidelines for Preparation of Action Plan – Prevention and Management of Heat Wave | NDMA, Ministry of Home Affairs | October 2019 (title page) | https://ndma.gov.in/sites/default/files/PDF/Guidelines/heatwaveguidelines.pdf (official; the copy from heathealth.info is byte-identical) | Y | health, and a Ministry of Power row | 62 pages, **all scanned**: no text layer. OCR'd; pages 40–47 and 52–57 are sideways or upside down. **Printed page = PDF page − 20.** Quotes were taken only from pages I viewed as images: PDF 35, 36, 37, 44, 48, 51 (printed 15, 16, 17, 24, 28, 31). |
| NCDC_ADVISORY_STATES_2025 | Advisory for State Health Departments for Summer 2025 | National Centre for Disease Control (NCDC), MoHFW | 2025 | https://ncdc.mohfw.gov.in/uploads/pdf/heat1.pdf | Y | health, and hospital power supply | 5 pages, text-readable. Cites the NDMA Guidelines 2019 at the official ndma.gov.in address. |
| NCDC_EMERGENCY_COOLING_2024 | Emergency Cooling for Severe Heat Related Illnesses | NCDC / NPCCHH, MoHFW | March 2024 (PDF creation date; the document itself carries no printed date) | https://ncdc.mohfw.gov.in/uploads/resource/1769333089_Emergency-Cooling-for-Severe-Heat-Related-Illnesses_March2024_NPCCHH.pdf | Y | health | 40 pages, text-readable. Cited by PDF page. |
| WHO_WMO_HEATWAVES_HEALTH_2015 | Heatwaves and Health: Guidance on Warning-System Development (WMO-No. 1142) | WMO and WHO | © 2015 (WHO's web page dates it 2016) | https://cdn.who.int/media/docs/default-source/climate-change/heat-waves-and-health---guidance-on-warning-system-development.pdf | Y | health, occupational heat | 114 pages, text-readable. Cited by PDF page (printed page = PDF page − 16). The Chapter 6 intervention examples are mostly from the USA/Europe, as the guide itself says (PDF p.64). |
| DELHI_HAP_2025 | Delhi Heat Action Plan 2025 | DDMA, GNCTD | 2025 | official DDMA site (you downloaded it; please add the exact URL) | Y | both | 163 pages; PDF page = printed page. Annexures (about pp.108–164) are mostly images and were **not read**. |
| DELHI_HAP_2024_25 | Delhi Heat Action Plan 2024-25 | DDMA, GNCTD | 2024-25 | https://ddma.delhi.gov.in/sites/default/files/ddma/generic_multiple_files/final_hap_12.04.2024.pdf | Y | both | 66 pages. Cross-check only; much of it is reproduced in the 2025 plan. |
| IMD_HW_FAQ | FAQ on Heat Wave | India Meteorological Department | Undated; cites data to 2023 | https://internal.imd.gov.in/section/nhac/dynamic/FAQ_heat_wave.pdf | Y | health | 16 pages, text-readable. |
| NDMA_HW_SLIDES (saved as `NDMA_HW_GUIDELINES.pdf`) | "Understanding of the Heat Wave" slides | NDMA | about 2018 | https://www.iipa.org.in/cms/public/uploads/152561610368684.pdf (third-party host) | Y | health | A slide deck, not the Guidelines. Superseded by NDMA_HW_GUIDELINES_2019 above; no quotes used. You can delete this file or keep it for reference. |

**Added in the third pass:** BEE_AC_24C_PIB (PIB release, 6 Jan 2020; image-only PDF, quotes checked by eye), NIOSH_2016_106_HEAT (US occupational WBGT criteria), ISO_7243_2017_SAMPLE (iTeh preview, 11 pages, front matter only, **no limit values**: Annex A is not in the sample), springer_paper and PMC_WORKSHOP_PROCEEDINGS (secondary, discussion only; used for one gap finding).

**Added in the fourth pass:** NPCCHH_NAP_HRI (National Action Plan on HRI, July 2021, 64 pp; printed page = PDF − 19; clinical workflows and hospital preparedness, no colour-coded tiers and no power content), NPCCHH_TRAINING_MANUAL_MO (32 pp, now mined: see section 11), BYPL_DSM (BSES Yamuna demand-side management slides, third-party host, not heatwave-specific), DERC_TOD_NOTICE (2018-19 time-of-day tariff proposals from the DISCOMs, not heatwave-related, no actions taken), TPDDL_SUMMER_PR (Tata Power-DDL press release of 12 Mar 2026; image-only and the saved page has the site menu covering the text, so nothing in it is quotable; visible numbers only: summer peak 2026-27 expected 2622 MW, 2025-26 2410 MW). NCDC_HEAT1.pdf is byte-identical to NCDC_ADVISORY_STATES_2025.pdf.

**Added in the fifth pass:** CEA_ELNINO_ADVISORY (2 pp, image-only; CEA Chief Electrical Inspectorate Division advisory to all DISCOM MDs and all electricity consumers on safe and judicious use of electricity under IMD's above-normal-temperature forecast; undated in the scan) and NEP_2022_32_FINAL_GAZETTE_CEA (National Electricity Plan 2022-32, 316 pp; searched, mentions air-conditioner demand growth and demand response in general but contains no heatwave actions; nothing taken).

**Added in the sixth pass:** PIB_LABOUR_ADVISORY_2026_SCREENSHOT (PIB release of 28 Apr 2026 on the Labour Ministry's nationwide heatwave advisory, supplied as a screenshot because the PDF would not download; image only), MOLE_ADVISORY_2025 (PIB release of 22 Apr 2025 on the Ministry's earlier letter to States/UTs; text layer, read but not mined separately since the 2026 release repeats its measures), NPCCHH_REPORT_2024 (91 pp, text layer partly garbled; one directive taken), DELHI_SLDC_ANNUAL_REPORT_2024-25 and 2023-24 (about 210 pp each; operational statistics, no heatwave actions: the 2024-25 report records a peak of 8,656 MW on 19.06.2024 at 15:06 with zero load shedding by any DISCOM at that moment, and lists a Delhi power-system contingency plan only as a task of a technical sub-committee).

**Still not obtained:** NCDC 2024 state advisory (user's browser blocks the link with ERR_BLOCKED_BY_CLIENT), the CEA advisory titled 'Summer Preparedness and Distribution System Reliability Measures' (known only from news reports of July 2026, not cited), full ISO 7243, and any Grid-India / DERC / BSES Rajdhani heatwave document.

---

## 2. Candidate action library

Machine-readable: `sources/phase6_candidate_actions.json`. Each entry carries `action_id`, `text`, `audience`, `tier_min`, `trigger`, `doc_id`, `page`, `verbatim_excerpt`, `confidence`, `notes`, `verification`, and (for NDMA) `printed_page`.

| Count by | Result |
|---|---|
| Total | 135 actions |
| Source | Delhi HAP 2025 49, NDMA 2019 17, NCDC advisory 2025 12, IMD FAQ 9, NPCCHH National Action Plan 9, WHO/WMO 8, NCDC emergency cooling 7, CEA El Nino advisory 7, NPCCHH Training Manual 7, Labour Ministry 2026 PIB 6, BEE 24 C PIB 2, NIOSH 1, NPCCHH report 2024 1 |
| Audience | health 47, public 41, municipal (incl. labour, schools, transport) 26, energy 21 |
| Tier | UNSPECIFIED 123, ORANGE 7, RED 4, YELLOW 1 |
| tier_min | YELLOW 1, ORANGE 7, RED 4, UNSPECIFIED 77 |
| Confidence | high 79, medium 8, low 2 |

**Where the tiers come from.** Tiers are set only where the source ties an action to a colour: the IMD/NDMA colour-code table (yellow, orange, red advice) and four Delhi HAP lines that name orange or red days (water tankers, public announcements, medical vans, medical camps). The other 77 actions say "during a heat alert", "when a heat wave is declared", or nothing about alert level. Mapping those onto your four tiers is a design decision for your team; the sources do not make it. WHO/WMO gives the principle for it: interventions "should be tied to the levels of warning used", with educational messages early (`WHO_TIE_INTERVENTIONS_TO_WARNING_LEVELS`).

**A useful finding on the colour table.** NDMA 2019 (printed p.15) gives the orange-alert advice in seven lettered items plus a sunstroke note, and defines the red alert as "Along with suggested action for Orange Alert, Extreme care needed for vulnerable people." So red means the orange advice plus extra care for the vulnerable. The IMD FAQ's red cell shows only the second half.

**Energy entries (11).** All `tier_min` UNSPECIFIED:

- *Delhi HAP (3):* priority power to hospitals and dispensaries, DISCOM care for hospital supply, AC at 25 °C awareness.
- *NDMA 2019 (3), Ministry of Power row, printed p.24:* advisory to states and utilities for uninterrupted supply, repair and maintenance "before and during the summer", and "Re-scheduling load shedding".
- *NCDC advisory 2025 (3):* "Coordinate with electricity distribution company/corporation for uninterrupted electricity supply to hospitals", bi-annual electrical load audits in ICUs, and energy conservation in health facilities.
- *WHO/WMO (2), low confidence:* utilities suspend electricity disconnections during a heatwave. These describe US and European practice, not Delhi or India; do not present them as Delhi actions.

---

## 3. Thresholds table

Machine-readable: `sources/phase6_thresholds.json` (58 rows after all passes; 46 plus twelve Part D rows on IMD's hot-and-humid wording, heat index launch and warning lead time). Highlights:

| Quantity | Threshold | Meaning | WBGT variant | doc_id | Page |
|---|---|---|---|---|---|
| Tmax, plains | ≥ 40 °C | Heat wave considered | n/a | IMD_HW_FAQ | 1 |
| Tmax departure | 4.5–6.4 °C / > 6.4 °C | Heat wave / severe | n/a | IMD_HW_FAQ | 1 |
| Actual Tmax | ≥ 45 / ≥ 47 °C | Heat wave / severe | n/a | IMD_HW_FAQ | 1 |
| IMD colour codes | yellow 2 days; orange severe 2 days or 4+ days; red severe > 2 days or > 6 days total | Persistence-based | n/a | IMD_HW_FAQ | 8 |
| Delhi HAP table | Yellow ≥ 40 °C; Orange ≥ 45 °C; Red ≥ 45 °C | Temperature-based (as printed) | n/a | DELHI_HAP_2025 | 10 |
| Work/rest, continuous work | 30.0 / 26.7 / 25.0 °C (light / moderate / heavy) | Permissible exposure | WBGT, OSHA-based, variant not stated | WHO_WMO_HEATWAVES_HEALTH_2015 | 71 |
| Work/rest, 75% work | 30.6 / 28.0 / 25.9 °C | Same | Same | WHO_WMO | 71 |
| Work/rest, 50% work | 31.4 / 29.4 / 27.9 °C | Same | Same | WHO_WMO | 71 |
| Work/rest, 25% work | 32.2 / 31.1 / 30.0 °C | Same | Same | WHO_WMO | 71 |
| Core body temperature for occupational limits | < 38 °C | Basis of ILO and similar standards | Body temperature | WHO_WMO | 67 |
| Fan use | dry-bulb > 35 °C | Hot air can make a person hotter | Dry-bulb | WHO_WMO | 65 |
| Evaporation | RH > 75% | Becomes ineffective | RH | NCDC_EMERGENCY_COOLING_2024 | 7 |
| Heat stroke | core ≥ 40 °C | Organ dysfunction | Core temperature | NCDC_EMERGENCY_COOLING_2024 | 7 |
| Cooling end point | core 38–39 °C | Stop active cooling | Core temperature | NCDC_EMERGENCY_COOLING_2024 | 10 |
| Percentile triggers | 90th, 95th, 99th of daily Tmax | NDMA 2019 Annexure 1 | n/a | NDMA_HW_GUIDELINES_2019 | 48 |
| Percentile triggers | 75th, 85th, 95th | NDMA recommendation as quoted by the Delhi HAP | WBGT listed only as a candidate variable | DELHI_HAP_2025 | 43 |

**On WBGT.** The only numeric WBGT limits found are WHO/WMO Box 6, which it says are adapted from OSHA (1999), a US source. The box does not say whether the values are for outdoor work in sun or indoor/shade, nor whether WBGT is measured or estimated. No Indian WBGT limit appears in any document read. NIOSH 2016-106 (p.125) gives RAL = 59.9 − 14.1 log10 M and REL = 56.7 − 11.5 log10 M (°C-WBGT, M in watts), US guidance only. A 2026 review (springer_paper p.11, secondary) says India's OSH Code 'fails to set actionable exposure limits'. The ISO 7243 sample contains no limit values. ISO 7243 appears only as a citation (1989 edition, WHO/WMO PDF p.103), so I cannot say what it specifies. Wet-bulb, WBGT and heat index are different quantities and nothing here lets you convert between them.

---

## 4. Conflicts and gaps

### Conflicts between sources

1. **IMD departure thresholds.** IMD FAQ: 4.5–6.4 °C heat wave, > 6.4 °C severe. Delhi HAP 2025 p.9 reproduces an older scheme (normal Tmax ≤ 40 °C: 5–6 / ≥ 7 °C; > 40 °C: 4–5 / ≥ 6 °C). Your brief says 6.5.
2. **Two definitions of the colours.** IMD/NDMA define colours by how long a heat wave lasts. The Delhi HAP table defines them by temperature (Orange and Red both ≥ 45 °C as printed, so they cannot be told apart). NDMA 2019 Annexure 1 (printed p.28) shows an Ahmedabad chart whose colour bands appear to start near 41, 43 and 45 °C; that is a chart reading, not a text quote.
3. **Percentile triggers.** NDMA 2019 Annexure 1 says 90th, 95th and 99th percentile. The Delhi HAP (p.43) says NDMA recommends 75th, 85th and 95th.
4. **Forecast lead time.** The IMD FAQ (p.6) says the 1600 hrs bulletin gives a five-day warning. The NCDC 2025 advisory (p.2) says "next four days". NDMA 2019 (printed p.14, OCR reading) says valid for the next four days. This matters for a 5-day forecasting model: confirm the current IMD lead time.
5. **Ice packs.** The Delhi HAP clinical step (p.65) says "Cool wet towels or ice packs to axillae, groin, and around neck". NCDC 2024 (PDF p.10) says "Evidence is lacking to support the use of ... ice packs placed strategically to the neck, axilla, and groin as primary cooling methods."
6. **Cool first or transport first.** The Delhi HAP (p.66) says that at 40 °C or above to "immediately transport to emergency department". NCDC 2024 and 2025 say "Cool First, Transport Second". NDMA 2019 (printed p.16) says to call 108/102 or take the victim to hospital immediately, and also lists cooling steps.
7. **Fans.** NDMA 2019 (printed p.31) and the IMD FAQ say "Use fans". WHO/WMO (PDF p.65) says that above a dry-bulb temperature of 35 °C hot air can make a person hotter; Delhi's Tmax regularly exceeds that.
8. **Work-hour windows.** The Delhi HAP gives 1–5 pm (p.81), "7:00am to 1:00pm, 4:00pm to 6:00pm" (p.56), schools closed 12–4 pm (p.58); the public is told to avoid 12–3 pm (IMD FAQ, NDMA, NCDC). WHO/WMO's UK-sourced advice uses 11 am–3 pm.
9. **Load shedding versus protecting hospitals.** The Delhi HAP tells utilities to keep power to hospitals (pp.54, 80) but lists "Scheduling load shedding" for the Power Department (p.59). That phrase appears to come from NDMA 2019's Ministry of Power row, "Re-scheduling load shedding" (printed p.24), which gives no criteria. Neither document says how to protect cooling centres, night shelters or homes.
10. **AC set-point.** The Delhi HAP asks for AC at 25 °C (p.53). The national 24 °C default is now verified: the BEE/PIB notice of 6 Jan 2020 was read (image-only PDF, checked visually), and the CEA El Nino advisory also says 24 °C (section 9 onward). The conflict with the HAP's 25 °C stands and is for the team to resolve.

### Gaps

- **Not obtained:** NCDC 2024 state advisory, NPCCHH 2024 activity report, Labour Ministry heatwave advisory for workers, CEA advisory to DISCOMs, full ISO 7243.
- **Not read:** Delhi HAP 2025 annexures; NDMA 2019 pages other than the six I checked (so the stakeholder table beyond the Ministry of Power row, and Annexure 3 treatment protocol, are unused); IMD's district-level colour-code documents.
- **Scanned pages:** none of the NDMA quotes is marked "UNVERIFIED – scanned document", because each one comes from a page I viewed. Any further NDMA quote would need the same check.

### Part B status

**Citable Delhi energy-grid heatwave actions found: PARTIAL (strengthened).** The Delhi HAP 2025 annexures supply Delhi-specific, official, verbatim energy actions: Annexure 1 row 20 (Electricity resilience programme, implementers CDRI, DDMA, Delhi Transco Limited) to analyse electricity-sector vulnerabilities to heat at community distribution level and to build a digital heat-risk map to plan for power demand, plus the Delhi Labour Department circular of 16/04/2025 (Annexure 3) with electrical-safety and switch-off instructions. Together with the CEA advisory, NDMA 2019 Ministry of Power row, three Delhi HAP lines on power to critical facilities, the NCDC 2025 hospital-supply line and the BEE 24 C default, there are 20+ citable energy items. It stays PARTIAL because the Delhi grid items are planned analyses or safety rules, not operating rules: nothing ties demand response, peak shaving or load shedding to an IMD alert colour, and no Delhi SLDC, Grid-India, DERC or BSES Rajdhani heatwave document exists in what was found.

---

## 5. Files

In `retrieval-augmented-heatwave-forecasting\sources\`:

- `PHASE6_SOURCE_RESEARCH_REPORT.md` (this report)
- `phase6_candidate_actions.json`, `phase6_thresholds.json`
- `txt\`, `txt_raw\` (extracted text); `ocr_ndma2019\`, `ocr_ndma2019_rot\` (OCR of the NDMA scan); `build\img\` (page images I checked)
- `build\` scripts: run `python verify_all.py` to re-check every quote

## 6. Suggested next steps

1. (Done in later passes.) Part B documents were searched and gathered; see sections 9-13 and `PHASE6_SOURCES_TO_ADD.md`. Current next step: Part C documents (section 14.2).
2. Add the NPCCHH National Action Plan on Heat Related Illnesses PDF to `sources\pdfs\` when you have it.
3. Decide how "during a heat alert" maps onto your four tiers; 123 of the 135 actions depend on that decision (only 12 name a tier: 1 YELLOW, 7 ORANGE, 4 RED).


## 7. Third-pass additions

Totals now: 92 actions, 41 thresholds, 0 hard failures. New actions: BEE_AC_DEFAULT_24C_MANDATORY, BEE_AC_NOTIFICATION_TEXT_24C (both energy, verified visually), NIOSH_MEDICAL_MONITORING_ABOVE_RAL. New thresholds: NIOSH RAL and REL equations, NIOSH 27.8 °C worked example, India WBGT-limit gap. Build files: `build/actions_def3.py`, `build/thr_def3.py`.

## 8. Fourth-pass additions

Totals now: 101 actions, 43 thresholds, 0 hard failures. Nine actions from the National Action Plan (PHC and emergency-department heatstroke workflow, hospital equipment list, two heat-alert items) and two thresholds (IMD declaration rule, clinical heatstroke definition). The NAP names no alert colours, so all nine are tier UNSPECIFIED. Two actions use short single-line fragments because the source table wraps text across lines; they are marked medium confidence. Build files: `build/actions_def4.py`, `build/thr_def4.py`. The training manual (32 pp) is extracted to `txt/` but not yet mined.

## 9. Fifth-pass additions

Totals now: 108 actions, 43 thresholds, 0 hard failures. Seven new energy/public actions from the CEA advisory, all `tier_min` UNSPECIFIED, verified visually (image-only PDF). New conflict: the CEA advisory and the BEE default say 24 °C, the Delhi HAP says 25 degree. Build file: `build/actions_def5.py`.

## 10. Sixth-pass additions

Totals now: 115 actions, 43 thresholds, 0 hard failures. Six Labour Ministry actions (reschedule working hours; drinking water, rest areas and cooling; emergency ice packs; factory and mine pace and two-person crews; ESIC/DGLW heat-stroke help desks; awareness at labour chowks) and one NCDC directive for clusters of more than 5 heat-linked deaths. All tier UNSPECIFIED. **Occupational WBGT gap confirmed:** the Ministry's 2025 and 2026 advisories give measures but no WBGT or temperature limits, so the only numeric WBGT limits in the library remain the US-derived WHO/WMO Box 6 and the NIOSH equations. Build file: `build/actions_def6.py`.

## 11. Seventh-pass additions (NPCCHH Training Manual for Medical Officers)

Totals now: 122 actions, 46 thresholds, 0 hard failures. Seven new actions (heat exhaustion protocol with the 104 °F decision point, heat stroke on-site and in-ambulance cooling, worker shift changes, public access to cool places) and three thresholds. The manual's section 7 on hospital preparedness repeats the National Action Plan and was not re-extracted.

New conflicts and inconsistencies found in the manual:

- **Emergency number:** the manual's first-aid table (p.11) says to call 112 for an ambulance; NDMA 2019 (printed p.16) and the National Action Plan say 108 and 102. The 112 quote could not be captured cleanly because the table interleaves columns, so it is not in the JSON.
- **Heat stroke temperature:** the manual states core temperature >= 40 C 'with 103 F or higher' on p.10, but 40 C is 104 F and its own algorithm (p.16) uses 104 F. Treat 103 F as a typo.
- **Transport vs cool first:** the manual's heat exhaustion step 4 says to start IV fluids and transport immediately at 104 F or above; NCDC emergency cooling (2024) says cool first, transport second.
- **Cooling floor:** the manual's target is 38.5 C or below within 2 hours; the National Action Plan says do not cool below 38 C.

## 12. Status of source gathering

Source gathering is complete enough to start building. What is solid: health actions (Delhi HAP, NDMA 2019, NCDC 2024-25, IMD, WHO/WMO, NPCCHH plan and manual, Labour Ministry advisories) and the official IMD criteria. What is thin and cannot be improved from public documents found: (1) almost every action is tier UNSPECIFIED, because sources rarely tie actions to colours (only 12 have a named tier: 1 YELLOW, 7 ORANGE, 4 RED); (2) no Indian WBGT limit exists in anything read; (3) energy actions are general and not Delhi-grid-specific, with no demand-response or load-shedding rule tied to alerts (Part B: PARTIAL); (4) unobtained: the CEA advisory 'Summer Preparedness and Distribution System Reliability Measures', the NCDC 2024 state advisory, full ISO 7243, most NDMA 2019 pages beyond those cited, and the Delhi HAP 2025 communication plan (Annexure 2), which was skimmed, not mined.

## 13. Eighth-pass additions (Delhi HAP 2025 annexures)

Totals now: 135 actions, 46 thresholds, 0 hard failures. Annexures read: Annexure 1 (district implementation template, PDF pp.112-143, searched for energy rows), Annexure 3 (Labour Department circular, PDF pp.155-158, scanned images, mined), Annexure 4 (Do's and Don'ts, PDF pp.160-163, Hindi infographics, not quoted; the first-aid graphic on p.162 refers to calling 108). Annexure 2 (communication plan, pp.145-153) was skimmed, not mined. Thirteen new actions: eleven from the Labour circular (including 'avoid peak hours (12 - 4 PM)', the only clock window for Delhi found, and electrical items such as RCD above 2 kW and a safety officer above 250 kW) and two from Annexure 1 row 20. Scanned pages inside a text PDF are verified by eye; the verifier now handles page-level image checks. Build file: `build/actions_def8.py`.

Added caution: the 12-4 PM window comes from a Delhi Labour Department circular, so it applies to workplaces, not the general public; the HAP's own public advice should be checked before generalising it.

## 14. Part C: are the four label-test heatwave dates confirmed against IMD?

Documents read (all in `sources\pdfs`, text extracted to `sources\txt`): IMD monograph 'Heat and Cold Waves in India' (202 pp; PDF page = printed page + 3), IMD Annual Climate Summary 2010 (25 pp) and 2015 (27 pp). Nothing here is from news or search snippets. **Important limits:** (a) none of these documents lists IMD-*declared* heat-wave days for Delhi; the monograph gives regional spells for north-west India (NWI) computed from the 90th-percentile criterion, not the IMD declaration rule in section 3. (b) The Climate Summaries give each station's *annual* extreme Tmax with its dates, not a daily series. (c) The station is named 'New Delhi AP' (the airport station). It is not labelled Safdarjung anywhere, so no Safdarjung Tmax has been confirmed. I believe 'AP' here is Palam, not Safdarjung, but the documents do not say so.

| Brief's period | What the IMD documents say | Verdict |
|---|---|---|
| ~24 May-1 Jun 1998 | Monograph Table 4.2 (PDF p.61): NWI heat wave 21-28 May 1998, mean intensity 43.4 C, 8 days. Table 5.1 (PDF p.77): severe heat wave over NWI 15-28 May 1998. Table 5.2 (PDF p.91) lists 15-28 May 1998 among 13 severe NWI events. | **Partly supported, dates differ.** The IMD spell is 21-28 May (heat wave) / 15-28 May (severe), not 24 May-1 Jun. It ends 28 May, so 29 May-1 Jun is not supported. No Delhi or Safdarjung value. (A Red Cross bulletin of 11 Jun 1998 reportedly gave Delhi 46.2 C on 26 May 1998; not IMD, not read directly.) |
| ~10-20 May 2002 | Monograph Table 5.1 (PDF p.77): severe heat wave over NWI **1-8 May 2002**. Table 4.2 has **no** 2002 NWI event (it jumps from 1999 to 2004). A separate east-coast list (PDF p.95, from Ratnam et al.) has 9-15 May 2002, which is not Delhi. | **Not supported as stated.** The only NWI spell is 1-8 May, and the two NWI tables disagree on whether 2002 qualifies. 10-20 May is not an IMD NWI heat wave. No Delhi value. |
| ~10-20 Apr 2010 | Monograph Table 4.2 (PDF p.61): NWI heat wave **8-20 Apr 2010**, 43.1 C, 13 days. Table 5.1 (PDF p.77): severe spells 5-11 Apr and 13-19 Apr 2010 (also 12-17 May and 19-26 May). Climate Summary 2010 PDF p.21 has a figure captioned 'maximum temperature anomaly during the heat wave 14-20 April'. | **Supported at region level.** The 10-20 Apr window sits inside 8-20 Apr. No Delhi-specific declared dates, no April Safdarjung value (the station table gives annual maxima only). |
| ~22-26 May 2015 | Climate Summary 2015 PDF p.5: severe and widespread heat wave over north/north-west, central, eastern and south-east India 'during second fortnight of May'. Table 2 (PDF p.25): New Delhi AP annual max **45.5 C on 25 May 2015**. Monograph Table 4.2 has **no 2015 NWI event** (2014 then 2016). | **Weakly supported for Delhi.** Delhi's hottest day of 2015 (45.5 C at the airport station) falls on 25 May, inside the window, and the season summary describes a widespread late-May heat wave. But IMD's NWI list has no 2015 event, the main 2015 event was in Andhra Pradesh and Telangana, and no Delhi declaration is given. |

Also found: Climate Summary 2010 PDF p.24, Table 2: New Delhi AP annual max 45 C, on 17, 18, 22, 25, 26 May and 20 Jun 2010. The 2010 summary (PDF p.6) says heat wave conditions prevailed over northern, north-western and central India almost throughout the season, making it the warmest pre-monsoon season since 1901.

**What this means for the label test.** Only 2010 matches the brief's window at region level. 1998 is shifted (21-28 May, not 24 May-1 Jun), 2002 is wrong (1-8 May, not 10-20 May), and 2015 has a Delhi peak on 25 May without an IMD NWI event. Do not describe the label test as validated against IMD for Delhi. Either re-centre the windows on the IMD spells above and label them 'north-west India regional spell, 90th-percentile criterion', or derive Delhi heat-wave days from Safdarjung daily Tmax with the IMD criteria in section 3 and label them 'derived, not IMD-declared'.

### 14.1 Two IMD papers added (PDFs in `sources\pdfs`)

- **Pai, Thapliyal and Kokate, MAUSAM 55(2), April 2004, 'Decadal variation in the heat and cold waves over India during 1971-2000'** (IMD, Pune). Table 4 (PDF p.8, printed p.288) lists heat-wave spells of 10 or more days by IMD sub-division. Sub-division 13 is 'Haryana, Chandigarh & Delhi'. It lists a heat-wave spell for sub-division 13 of **22 May to 3 Jun 1998, 13 days** (28 sub-divisions affected in total). **This is the first IMD-sourced spell that covers Delhi's sub-division for 1998, and the brief's 24 May-1 Jun sits inside it.** It is a sub-division spell, not a Delhi-station declaration. The paper stops at 2000, so it says nothing about 2002, 2010 or 2015.
- **Pai, Nair and Ramanathan, MAUSAM 64(4), October 2013, 'Long term climatology and trends of heat waves over India during the recent 50 years (1961-2010)'** (IMD). It describes the May 2002 heat wave as an Andhra Pradesh event, 9-15 May 2002 (more than 1000 deaths), and says about 200 more people died in other parts of the country in the same month. It names no Delhi dates for 2002 or 2010 (it mentions a May 2010 heat wave that killed more than 250 in north India, no dates). It does not support a Delhi-specific 2002 spell.

### 14.2 Safdarjung daily Tmax from GHCN-Daily (derived, not IMD-declared)

The GHCN-Daily file (station IN022021900) was unzipped to `sources\data\` and converted to `safdarjung_daily_tidy.csv` (see `data\README_DATA.md`). The file carries no station name; its identity as Safdarjung is from the file name you gave it, and it agrees with IMD's annual maxima for 2010 (45.4 C on 17 May; IMD lists 45 C for New Delhi AP on that date) and 2015 (45.5 C on 25 and 26 May; IMD lists 45.5 C on 25 May). Data run 1944-01-01 to 2025-08-23 for Tmax, with gaps.

| Window | Days with data | Max Tmax | Days >= 40 C | Days >= 45 C |
|---|---|---|---|---|
| 1998 brief, 24 May-1 Jun | 9 of 9 | 46.5 C on 28 May | 9 | 4 |
| 1998 IMD NWI spell, 21-28 May | 8 of 8 | 46.5 C on 28 May | 8 | 4 |
| 2002 brief, 10-20 May | 11 of 11 | 46.0 C on 18 May | 11 | 1 |
| 2002 IMD NWI spell, 1-8 May | 8 of 8 | 43.0 C on 8 May | 5 | 0 |
| 2010 brief, 10-20 Apr | 8 of 11 | 42.9 C on 16 Apr | 7 | 0 |
| 2010 IMD NWI spell, 8-20 Apr | 10 of 13 | 42.9 C on 16 Apr | 8 | 0 |
| 2015 brief, 22-26 May | 5 of 5 | 45.5 C on 25 May | 5 | 2 |

What the data add: 1998 and 2015 are clearly hot at Safdarjung in the brief's windows. For **2002**, Delhi's heat was in 9-18 May (every day 10-20 May is at least 41 C), not 1-8 May, which is when IMD's north-west India severe spell is listed. So the brief's 2002 window fits the station data but is not an IMD-listed Delhi spell; it overlaps the east-coast (Andhra Pradesh) event dates. For **2010**, April stays at or below 42.9 C with three missing days, so it is a moderate April spell at Safdarjung, and IMD's NWI spell may reflect other stations. These are labels derived from observations. They do not use IMD's departure criteria (normals were not available) and are not IMD declarations.

**Updated verdict.** 1998: supported at sub-division level (22 May-3 Jun, Table 4) plus Safdarjung data. 2002: not IMD-supported for Delhi; supported only by station data. 2010: supported at region level, weaker at Safdarjung. 2015: station data support it, and IMD's summary describes a widespread late-May heat wave, but no IMD Delhi or NWI spell is listed. None has IMD-declared Delhi station dates. Label the test windows 'IMD region/sub-division spell' or 'derived from Safdarjung Tmax' as appropriate.

Machine-readable summary: `phase6_label_dates_check.json`.

## 15. Design decisions and caveats for the build (not source findings)

Each item says what the sources show and what the advisory must do. None of these is resolved by the sources; they need a team decision, written down.

1. **Tier mapping is a team decision.** 123 of 135 actions have no alert level; only 12 name a colour (1 YELLOW, 7 ORANGE, 4 RED). Assigning the rest to GREEN/YELLOW/ORANGE/RED is a design choice. Document the rule used (for example 'UNSPECIFIED actions are allowed from YELLOW upward') and mark it as the team's, not the sources'.
2. **Two colour systems conflict.** IMD/NDMA colours depend on how many days a heat wave lasts and its severity; the Delhi HAP colours depend on temperature (Yellow >= 40 C; Orange and Red both >= 45 C, so those two cannot be told apart by temperature alone). The advisory must state which system it follows and must not blend them.
3. **Forecast lead time.** The IMD FAQ describes 5-day warnings; NCDC and NDMA say 4. The 5-day model should not be described as matching an official 5-day product without that caveat.
4. **No Indian WBGT limit** exists in anything read. The only numbers are US-derived (OSHA via WHO/WMO Box 6, and NIOSH RAL/REL equations). The advisory must not attach official-looking WBGT thresholds.
5. **Energy actions are PARTIAL.** They are general national guidance (NDMA, CEA, BEE, Labour); no Delhi grid rule is tied to alert levels, and no SLDC, DERC or DISCOM heatwave document was found.
6. **Medical conflicts between official documents must not be resolved silently:** cool first (NCDC emergency cooling) vs transport first (NPCCHH Training Manual); ice packs; fans above 35 C (NDMA/IMD say use fans, WHO/WMO warns); ambulance 108/102 (NDMA, National Action Plan, HAP infographic) vs 112 (Training Manual). The advisory should show the attributed alternatives or defer to a clinician-reviewed choice.
7. **Part C dates are only partly confirmed (section 14).** 1998 is supported at IMD sub-division level, 2010 at region level; 2002 and 2015 rest on derived GHCN Safdarjung data, not IMD declarations. Do not describe the label test as validated against IMD for Delhi.

## 16. Part D: humid heat and IMD warning lead time

Full result in `sources/PHASE6_PART_D_RESULT.md`. Summary: (D2) IMD has an official 'Hot & Humid Weather' definition (IMD FAQ p.2: station Tmax 3 C above normal together with above-normal relative humidity; no RH value, no colour category) and launched an experimental Heat Index in 2023 (PIB release, read from screenshots and verified visually), but no heat-index thresholds or colour categories were found. (D3) The lead-time conflict is resolved: a 26 Jul 2023 PIB release (screenshots, verified visually) says IMD's colour-coded warnings including heat wave are issued daily for the next five days, and for the next seven days from July 2023, so the current figure is seven days; NDMA 2019 (printed p.14, OCR match and page image) and NCDC 2025 repeat the 2017 four-day heat-warning system, and the undated IMD FAQ says five. (D1) No dated Delhi humid-heat event before 2019 was found in official or peer-reviewed documents; an IMD-authored paper (Sci Rep 2022, p.2) says Delhi has lower heat stress than coastal metros because of lower humidity, which is context, not an event list. 58 thresholds now, 0 hard failures.

