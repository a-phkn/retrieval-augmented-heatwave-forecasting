# Phase 6, Part E result: alert colours and which action belongs to which colour

Date: 7 Oct 2026. Files changed: `phase6_candidate_actions.json` (new fields `tier_min`, `tier_source`, `trigger_type`), `phase6_thresholds.json` (58 to 71 rows), `build/tiers_def.py`, `build/thr_def8.py`, `build/verify_all.py`. New document: `pdfs/AHMEDABAD_HAP_2018.pdf` (file title says "Heat Action Plan 2019: Easy Read Version"). Verifier: 135 actions, 71 thresholds, 0 hard failures.

## Headline

1. **No Delhi document gives a colour for any action beyond the 12 already known.** HAP 2025 has no department-by-colour table; its three annexures (district template, communication plan, labour circular) name no colours.
2. **The only action-by-colour table found is Ahmedabad's** (HAP 2019 update, PDF pp.14-22, 15 departments). It gave a colour to **30** of the 123 uncoloured actions. IMD's own FAQ table (p.8) gave **2** more. **91 stay UNSPECIFIED.**
3. Delhi's heat-alert actions on HAP 2025 pp.79-81 are close copies of the bullets in Ahmedabad's Red Alert list. That is why the Red matches are so clean, and it is also why Delhi's own text says only "during a heat alert".
4. **The Ahmedabad colours are precedent, not Delhi policy**, and Ahmedabad's colour bands (Yellow 41.1-43, Orange 43.1-44.9, Red ≥ 45 °C) are not Delhi's. Every such row carries `tier_basis: ahmedabad_precedent`.

## E1. Which colour system defines Delhi's heat alerts?

**Q1. Which system do Delhi's authorities use?** The documents do not say. I found no sentence like "the HAP is activated on IMD orange alert". The closest are these two:
- HAP 2025 p.79: the Municipal Corporation "Activates the citywide heat alert and response mechanism based, on the Department of Meteorology's weather forecast". No colour is named. Same wording in HAP 2024-25 (printed p.57).
- HAP 2025 p.79 also says to monitor "the heat alert level based on the weather temperature severity forecast (see section Heat Alert Severity)". **No section with that title exists** in the plan; I searched the whole text.
The plan prints **three colour schemes**: IMD's duration-based table (copied from the IMD FAQ), the p.10 temperature table, and a departure-from-normal table on p.45. The p.10 table is captioned "source: NDMA".

**Q2. What separates Red from Orange in the Delhi HAP table?** Nothing. I looked at HAP 2025 p.10 and HAP 2024-25 p.11 as page images. Both print Orange "Heat alert day >= 45°C" and Red "Extreme heat alert day >= 45°C". There is no second condition (days, heat index, night minimum, humidity). It looks like a printing duplication, but the plans do not say so. The 2024-25 table is identical. (In HAP 2024-25 it is Table 5 on printed p.11.)

**Q3. IMD Red condition (i).** Confirmed against the page image of IMD FAQ p.8: "Severe heat wave persists for more than 2 days." Condition (ii) is "Total number of heat/severe heat wave days exceeding 6 days."

**Q4. GREEN.** Defined in two places, with no actions of its own: HAP 2025 p.10, "Green (No Action) / Normal Day / Maximum temperatures are near normal", and IMD FAQ p.8, "Comfortable temperature. No cautionary action required." with suggested actions "Nil". The p.45 table calls the bottom level "White (Normal)" instead, "Below Normal Maximum Temp."

**p.45 (new, image only):** "Identification of Color Signals for high Alerts": Red "Normal Maximum Temp increase 6° C to more"; Orange "Normal Maximum Temp increase 4° C to 5° C"; Yellow "Nearby Normal Maximum Temp." (labelled "Heat-wave Warning"); White "Below Normal Maximum Temp." Its Yellow label conflicts with IMD's Yellow, which means heat wave conditions on 2 days. Recorded as 4 new threshold rows. Whether HAP 2024-25 has the same page was not checked.

## E2. Which action belongs to which colour?

Searched, in the brief's order:
1. Delhi HAP 2025 and 2024-25 (full text, all annexures and tables, plus the p.10/p.11 and p.45 page images): colour appears only in the 12 existing actions (tankers and announcements on orange/red days, medical camps and extra vans on red).
2. DDMA: the DDMA site file `hap_delhi_21.4.25_3.pdf` you saved is **byte-identical in size to `DELHI_HAP_2025.pdf`** (5,498,949 bytes), so it adds nothing. No separate DDMA SOP or alert-level action plan found. A Delhi "Heat Wave Action Plan 2026" is reported in the news (Tribune, and others) but I could not find its official PDF; the Tribune text you pasted names no alert colours.
3. NDMA 2019 guidelines: colour appears only in the existing NDMA rows (pp.34-37).
4. MoHFW/NCDC/NPCCHH: `NCDC_ADVISORY_STATES_2025`, `NCDC_EMERGENCY_COOLING_2024` and `NPCCHH_NAP_HRI` assign nothing by colour. The NCDC 2023 state advisory says to send advisories "as per the colour coded recommendations in NDMA guidelines" (second-hand summary only; your download link was broken, so not used).
5. Energy (CEA, BSES/TPDDL, SLDC, BEE): nothing ties a grid action to a heat colour. The only energy colours come from Ahmedabad's Electricity Board row.
6. Fallback: Ahmedabad HAP (this is the 2019 update; the file is called 2018 in your folder).

### Counts for the 123 uncoloured actions

| Source of the colour | Actions |
|---|---|
| Delhi source | 0 |
| National source (IMD FAQ p.8) | 2 |
| Ahmedabad precedent | 30 |
| None, still UNSPECIFIED | 91 |

Of the 32 newly coloured: 24 RED, 4 ORANGE, 4 YELLOW. The 12 earlier actions keep their colour and now also have `tier_source`.

### Summary table: the 32 newly coloured actions

| action_id | colour found | source (doc, page) | exact quote | same / equivalent | trigger_type |
|---|---|---|---|---|---|
| ENERGY_DISCOM_AC_25C_AWARENESS | YELLOW | AHMEDABAD_HAP_2018, PDF p.18 | "Create awareness among people on energy conservation." | equivalent action (general energy-conservation awareness, no AC setting named) | standing |
| HEALTH_SURVEILLANCE_HEAT_CASES | YELLOW | AHMEDABAD_HAP_2018, PDF p.16 | "Hospitals should be instructed to update their admissions and emergency case records to track heat-related morbidity and mortality." | equivalent action | standing |
| NDMA_ENERGY_RESCHEDULE_LOAD_SHEDDING | YELLOW | AHMEDABAD_HAP_2018, PDF p.18 | "Power shedding should be cut down/reduced during severe heat (frequency and timing)." | equivalent action (Ahmedabad Red also says no power cuts) | standing |
| HEALTH_WEEKLY_IMPACT_REPORT | ORANGE | AHMEDABAD_HAP_2018, PDF p.16 | "Produce weekly reports of the public health impact for AMC Nodal Officer during a heat alert." | same action (listed at both Orange and Red; lowest colour used) | forecast |
| NAP_HOSP_AMBULANCE_ICE_PACKS_COLD_WATER | ORANGE | AHMEDABAD_HAP_2018, PDF p.17 | "Keep sufficient stocks of reusable soft plastic ice packs for the 108 citywide emergency ambulances." | equivalent action (ice packs only; no cold water) | standing |
| SCHOOL_RESCHEDULE_TIMINGS | ORANGE | AHMEDABAD_HAP_2018, PDF p.20 | "Restrict the school timings, if necessary" | equivalent action | forecast |
| COMMS_ALERT_TO_HOSPITALS_PRACTITIONERS | RED | AHMEDABAD_HAP_2018, PDF p.15 | "Develop an SMS alert system to send direct messages to private practitioners in addition to the medical professionals at public hospitals and UHCs." | same action | forecast |
| COMMS_ALERT_VIA_WHATSAPP_SMS | RED | AHMEDABAD_HAP_2018, PDF p.15 | "Circulate warnings via text alerts or WhatsApp mobile messages" | equivalent action | forecast |
| DELHI_LABOUR_DRINKING_WATER_WORKPLACE | RED | AHMEDABAD_HAP_2018, PDF p.17 | "Provide sufficient portable drinking water." | equivalent action | forecast |
| DELHI_LABOUR_EMERGENCY_KIT_CONSTRUCTION | RED | AHMEDABAD_HAP_2018, PDF p.17 | "Pilot project to provide emergency ice packs and heat-illness prevention materials to traffic police, BRTS transit staff and construction workers." | equivalent action | forecast |
| DELHI_LABOUR_SHIFT_AVOID_12_4PM | RED | AHMEDABAD_HAP_2018, PDF p.17 | "Change working hours of labourers." | equivalent action | forecast |
| ENERGY_DISCOM_HOSPITAL_SUPPLY_CARE | RED | AHMEDABAD_HAP_2018, PDF p.14 | "Communicate the local utility protocol to prioritize maintaining power to critical facilities (such as hospitals and UHCs)." | equivalent action | standing |
| ENERGY_POWER_PRIORITY_CRITICAL_FACILITIES | RED | AHMEDABAD_HAP_2018, PDF p.14 | "Communicate the local utility protocol to prioritize maintaining power to critical facilities (such as hospitals and UHCs)." | same action | forecast |
| HEALTH_108_SMS_VULNERABLE_AREAS | RED | AHMEDABAD_HAP_2018, PDF p.17 | "Disseminate SMS text messages to warn local residents during a heat alert." | same action | forecast |
| HEALTH_EMS_ICE_PACKS_IV | RED | AHMEDABAD_HAP_2018, PDF p.17 | "Ensure adequate supply of ice packs and IV fluids." | same action | standing |
| HEALTH_HOSPITAL_SURGE_STAFF | RED | AHMEDABAD_HAP_2018, PDF p.16 | "Increase staffing at hospitals and UHCs to attend to the influx of patients during a heat alert, if feasible." | same action | forecast |
| MAN_WORKERS_CHANGE_SHIFT_TIME_HOURS | RED | AHMEDABAD_HAP_2018, PDF p.17 | "Change working hours of labourers." | equivalent action | forecast |
| MOLE_EMERGENCY_ICE_PACKS_HEAT_ILLNESS_MATERIAL | RED | AHMEDABAD_HAP_2018, PDF p.17 | "Pilot project to provide emergency ice packs and heat-illness prevention materials to traffic police, BRTS transit staff and construction workers." | equivalent action | forecast |
| MOLE_RESCHEDULE_WORKING_HOURS | RED | AHMEDABAD_HAP_2018, PDF p.17 | "Change working hours of labourers." | equivalent action | forecast |
| MUNI_COOLING_CENTRES_ACTIVATE | RED | AHMEDABAD_HAP_2018, PDF p.14 | "Activate “cooling centers,” such as temples, public buildings, malls, during a heat alert." | equivalent action (Delhi text also lists night shelters) | forecast |
| MUNI_DRINKING_WATER_DISTRIBUTION | RED | AHMEDABAD_HAP_2018, PDF p.14 | "Increase efforts to distribute fresh drinking water to the public." | same action | forecast |
| MUNI_SHADE_ACCESS_VULNERABLE | RED | AHMEDABAD_HAP_2018, PDF p.14 | "Expand access to shaded areas for outdoor workers, slum communities, and other vulnerable populations." | same action | forecast |
| PUBLIC_CHECK_VULNERABLE_NEIGHBOURS | RED | AHMEDABAD_HAP_2018, PDF p.18 | "Check on vulnerable neighbours, particularly during a heat alert." | same action | forecast |
| SCHOOL_NO_OPERATION_12_4 | RED | AHMEDABAD_HAP_2018, PDF p.20 | "Ensure that all school and colleges should be closed during heatwave days." | equivalent action (Ahmedabad closes schools for the whole day; Delhi 12-4 pm) | forecast |
| SCHOOL_PREMISES_AS_SHELTER | RED | AHMEDABAD_HAP_2018, PDF p.20 | "If school is not functioning, permit use of school premises as shelter during day time." | same action | forecast |
| TRANSPORT_SHADE_WATER_ORS_STANDS | RED | AHMEDABAD_HAP_2018, PDF p.19 | "Ensure availability of shade / shelters, drinking water, ORS packets etc., in bus stands, auto stands etc." | same action | standing |
| WORK_EMERGENCY_ICE_PACKS_PROTECTION | RED | AHMEDABAD_HAP_2018, PDF p.17 | "Pilot project to provide emergency ice packs and heat-illness prevention materials to traffic police, BRTS transit staff and construction workers." | same action | forecast |
| WORK_RESCHEDULE_PEAK_HOURS_ADVISORY | RED | AHMEDABAD_HAP_2018, PDF p.17 | "Change working hours of labourers." | equivalent action | forecast |
| WORK_SHIFT_AWAY_1_5PM | RED | AHMEDABAD_HAP_2018, PDF p.17 | "Encourage employers to shift outdoor workers’ schedules away from peak afternoon hours (1pm – 5pm) during a heat alert." | same action | forecast |
| WORK_WATER_AT_SITES | RED | AHMEDABAD_HAP_2018, PDF p.17 | "Provide sufficient portable drinking water." | equivalent action | standing |
| PUBLIC_AVOID_SUN_12_3 | YELLOW | IMD_HW_FAQ, PDF p.8 | "Avoid heat exposure." | equivalent action (IMD: general; no clock window) | forecast |
| PUBLIC_DRINK_WATER | ORANGE | IMD_HW_FAQ, PDF p.8 | "Drink sufficient water- even if not thirsty." | same action | forecast |

Ahmedabad pages are PDF pages (printed page = PDF page - 1). `tier_min` is the **lowest colour at which the source lists the action**; where Ahmedabad lists it at more than one colour I used the lowest.

### Unmatched actions (91)
Left UNSPECIFIED because no source assigns them to a colour, or the closest Ahmedabad line was not clearly the same action. Examples I deliberately did not map: HEALTH_HOSPITAL_READINESS (Ahmedabad's "Keep emergency ward ready" is not under a colour), MUNI_NIGHT_SHELTERS_OPEN_ALL_DAY and MUNI_WATER_ACCESS_RELIGIOUS_TRANSIT (in Ahmedabad's Phase 2 text, not its colour table), SCHOOL_NO_OPEN_AIR_CLASS (Ahmedabad's "avoid physical activities" is a different action), and the clinical and energy-efficiency actions.

## Trigger type (all 135 actions)

forecast: 59, standing: 46, observation: 30. Of the 91 still UNSPECIFIED: forecast 24, standing 38, observation 29.

- `observation` = clinical or reporting steps taken when a case, a body temperature, a cluster of deaths or a measured exposure already exists (e.g. CLINICAL_40C_IV_TRANSPORT, NCDC_CLUSTER_DEATHS_DISTRICT_TASK_FORCE_MEETING). A forecast cannot trigger these.
- `standing` = season-long or one-time preparation and general habits (hospital equipment lists, energy-efficiency advice, the BEE AC rule). These do not need a colour.
- `forecast` = taken because an alert, warning or forecast says heat is coming.
These labels are my reading of each action's own trigger text and are not quoted from a source; please treat them as a team judgement to review. The Delhi labour circular rows are marked `forecast` because their stated trigger is IMD's seasonal above-normal forecast.

## Things the team should know

1. Using Ahmedabad colours in the advisory means borrowing another city's policy. The colours there are driven by Ahmedabad's temperature bands, and three of the mapped actions are only partial matches (school closure whole-day vs 12-4 pm, cooling centres without night shelters, ice packs without cold water). The `match` column says which.
2. Delhi's own colour definition is unresolved (two tables with Orange = Red, plus the p.45 scheme). The advisory needs a team decision on which table defines GREEN/YELLOW/ORANGE/RED before any colour-to-action mapping can be turned on.
3. Two actions got a national (IMD) colour: PUBLIC_DRINK_WATER at ORANGE ("Drink sufficient water- even if not thirsty.") and PUBLIC_AVOID_SUN_12_3 at YELLOW ("Avoid heat exposure."). The second is a looser match (IMD gives no 12-3 pm window).
4. IMD FAQ p.8 also lists colour-linked actions not in the library yet: Yellow "Wear lightweight, light-coloured, loose, cotton clothes" and "Cover your head: Use a cloth, hat or umbrella"; Orange "Avoid dehydration" and homemade drinks (lassi, torani, lemon water, buttermilk). Add them as new actions if wanted.
5. Still missing, optional: the official Delhi Heat Wave Action Plan 2026 PDF, a later Ahmedabad HAP (2023 or later) to see if its colour table changed, and the NCDC 2023 state advisory.
