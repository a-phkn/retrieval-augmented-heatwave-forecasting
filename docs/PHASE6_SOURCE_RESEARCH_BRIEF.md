# Research brief: official sources for a Delhi heatwave advisory system

> **How to use this file:** paste the whole file into Claude (chat or Cowork) with web search enabled. Everything below the line is the instruction for the research assistant.

---

## Your task

You are helping a student research team build a **heatwave advisory generator for Delhi, India**. A forecasting model predicts the next 5 days of heat stress (air temperature, Wet-Bulb Globe Temperature (WBGT), Heat Index). A rule engine then turns the forecast into a risk tier (GREEN / YELLOW / ORANGE / RED), and an LLM writes an advisory.

**The safety rule:** the LLM may only recommend an action that exists in a human-verified *action library*. Every library entry must be backed by a **word-for-word quote from an official, publicly accessible document**. Your job is to find those documents and extract candidate actions with their exact quotes.

**Do not invent, paraphrase into a quote, or "reconstruct" any text.** If you can't open a document and read the passage yourself, say so and leave the quote empty. A missing source is a useful result; a fabricated one is harmful, because this feeds a public-health system.

---

## Part A — Health sources (priority 1, needed in any case)

Find the **current official versions** (state the year/version) of:

1. **IMD (India Meteorological Department)**
   - Heatwave definition and criteria for plains stations: the ≥ 40 °C threshold, departure from normal (4.5 °C / 6.5 °C), and the absolute ≥ 45 °C / ≥ 47 °C criteria.
   - The colour-coded warning system (green/yellow/orange/red) and the impacts and suggested actions per colour.
2. **NDMA (National Disaster Management Authority)**
   - *Guidelines for Preparation of Action Plan – Prevention and Management of Heat Wave* (latest revision).
   - Any NDMA heatwave do's-and-don'ts material.
3. **Delhi government / DDMA**
   - Delhi Heat Action Plan (latest year available). Note any actions tied to specific alert levels: cooling centres, water points, hospital preparedness, work-hour changes, school timings.
4. **Ministry of Health & Family Welfare / NCDC**
   - The National Action Plan on Heat Related Illnesses (NPCCHH programme), and heat advisories for hospitals and the public.
5. **WHO / WMO**
   - *Heatwaves and Health: Guidance on Warning-System Development* (WMO/WHO 2015).
   - Relevant WHO heat-health action planning guidance.
6. **Occupational heat (WBGT thresholds)**
   - Any official Indian guidance on WBGT-based work/rest limits (e.g. labour ministry, BIS).
   - The international reference, ISO 7243: describe what it specifies; quote only if publicly accessible.
   - Note **which WBGT variant** each threshold refers to (outdoor in sun, indoor/shade, measured globe temperature vs estimated).

## Part B — Energy-grid sources (priority 2; this decides whether energy actions are allowed at all)

We need to know whether **public, citable** documents exist that recommend specific actions for electricity demand during heatwaves in Delhi (or India-wide). Look for:

1. **DERC (Delhi Electricity Regulatory Commission)** — demand-response regulations, orders or programmes.
2. **Delhi DISCOMs** — BSES Rajdhani, BSES Yamuna, Tata Power-DDL: demand-response / peak-load programmes, summer peak advisories, consumer appeals during heatwaves.
3. **Delhi SLDC (State Load Dispatch Centre)** and **Grid-India (formerly POSOCO)** — peak-demand advisories and summer preparedness reports.
4. **BEE (Bureau of Energy Efficiency) / Ministry of Power** — the AC default-temperature (24 °C) notification. Report exactly what it mandates (who, which appliances, from when) and whether it covers commercial buildings or only room-AC factory defaults.
5. **CEA (Central Electricity Authority)** and **Ministry of Power** summer/heatwave advisories, e.g. directions to utilities on peak preparedness.
6. Any **Delhi Heat Action Plan section on power supply** (e.g. uninterrupted supply to hospitals and cooling centres). This is the most likely bridge between health and energy actions.

For Part B, state clearly at the end: **"Citable Delhi energy-grid heatwave actions found: YES / PARTIAL / NO"**, with one sentence of justification.

---

## Acceptance criteria for a source

A source counts only if **all** of these hold:
- It is published by a government body, regulator, utility, or UN agency (WHO/WMO). No news articles, blogs or consultancies as primary sources; news may be used only to *locate* an official document.
- It is publicly accessible: a working URL, preferably a PDF from an official domain (`.gov.in`, `.nic.in`, `who.int`, `wmo.int`, `derc.gov.in`, DISCOM sites, etc.).
- You actually opened it and read the quoted passage.
- You can give the page or section number for every quote.

---

## Output format (strict)

### 1. Source register (one row per document)

| doc_id | Title | Publisher | Year / version | URL | Accessible? (Y/N) | Scope (health / energy / both) | Notes |
|---|---|---|---|---|---|---|---|

Use short, stable `doc_id`s, e.g. `IMD_HW_CRITERIA_2024`, `NDMA_HAP_GUIDELINES_2019`, `DELHI_HAP_2025`, `BEE_AC_DEFAULT_2020`.

### 2. Candidate action library (one entry per action), as JSON

```json
[
  {
    "action_id": "HEALTH_COOLING_CENTRES_ACTIVATE",
    "text": "Short imperative action as stated in the source",
    "audience": "health | energy | municipal | public",
    "tier_min": "YELLOW | ORANGE | RED",
    "trigger": "Plain-language trigger condition stated in the source (e.g. 'IMD orange alert issued')",
    "doc_id": "DELHI_HAP_2025",
    "section": "Section / page reference",
    "verbatim_excerpt": "Exact quoted text from the document, copied character-for-character",
    "url": "Direct link",
    "confidence": "high | medium | low",
    "notes": "Ambiguities, e.g. tier not stated explicitly in source"
  }
]
```

Rules for this list:
- `verbatim_excerpt` must be copied exactly. Keep it short (≤ 3 sentences), enough to support the action.
- Set `tier_min` **only if the source links the action to an alert level**. Otherwise write `"UNSPECIFIED"` and explain in `notes`. Do not guess.
- Aim for 15–40 health actions covering public, municipal, hospital/health-system and outdoor-worker audiences. Include energy actions only if they meet the acceptance criteria.

### 3. Thresholds table

| Quantity (Tmax / departure / WBGT / Heat Index) | Threshold | Meaning (e.g. heatwave, severe, work-rest limit) | WBGT variant (if WBGT) | doc_id | section |
|---|---|---|---|---|---|

### 4. Conflicts and gaps

- List any **conflicting guidance** between sources (e.g. different thresholds, or health priority vs load-shedding).
- List what you **could not find or could not access**, and where you looked.
- Give your Part B verdict: YES / PARTIAL / NO.

---

## Things to be careful about

- Delhi heat action plans are revised yearly; always prefer the most recent and note the year.
- IMD criteria are defined for **station** observations. Just record them; don't adapt them.
- Wet-bulb temperature ≠ WBGT ≠ Heat Index. Record exactly which quantity each threshold uses.
- If a document is only an image/scan and you can't read the text reliably, mark the quote as `"UNVERIFIED – scanned document"`.
- Don't include any action you can't trace to a document, even if it seems sensible.

---

## Part C — Verify historical Delhi heatwave dates (used as label acceptance tests)

The forecasting model's heatwave labels are checked against these Delhi / north-India
heatwaves. They were chosen from memory and need confirming against IMD records (e.g. IMD
annual climate summaries, heat-wave reports, press releases) or peer-reviewed papers:

| Period to verify | What to confirm |
|---|---|
| ~24 May – 1 Jun 1998 | Heatwave over Delhi / north India; dates and peak Tmax at Safdarjung |
| ~10 – 20 May 2002 | Heatwave over Delhi; dates and peak Tmax |
| ~10 – 20 Apr 2010 | Exceptionally hot April in Delhi; IMD heat-wave days |
| ~22 – 26 May 2015 | Heatwave over Delhi (the 2015 Indian heatwave); Delhi dates and peak Tmax |

For each: the official source (title, URL, page), IMD-declared heat-wave dates for Delhi if
given, and peak Tmax. If a period is NOT supported by an official source, say so.

---

## Part D — Humid heat in Delhi, IMD's humid-heat wording, and the warning lead time

Context: the forecasting team found that Delhi's **humid heatwaves** (high wet-bulb / WBGT, mostly
June–August, monsoon) are largely **different events** from its **dry heatwaves** (high Tmax,
April–June). The model now forecasts a physical WBGT, and the advisory will add a clearly labelled
"humid heat stress" note alongside the official (Tmax-based) heatwave tiers. We need sources for the
three items below. Same rules as Parts A–C: official or peer-reviewed sources only, exact quotes,
page numbers, and "not found" is a valid answer.

### D1. Documented humid-heat events in Delhi, 1980–2018 (to check our humid-heat label)
Find episodes when Delhi had dangerous **hot-and-humid** conditions (not just high Tmax), from
official or peer-reviewed sources. Examples of what counts:
- IMD bulletins, press releases or annual climate summaries that mention "hot and humid", "warm and
  humid" or "sultry" conditions over Delhi, with dates;
- peer-reviewed studies of heat-related deaths or hospital admissions in Delhi during the monsoon
  (e.g. monsoon breaks), with dates;
- Delhi government or NDMA reports of humid-heat health impacts.

**Only events before 1 January 2019** (later years are a locked test period). Do **not** start from
any list of dates we give you: search independently, so the check stays unbiased.

For each event, give: dates, the source (title, publisher, year, URL, page), the exact quote, and the
evidence level (IMD statement / peer-reviewed / government report / station data only). If none are
found, say so and list where you looked.

### D2. Does IMD define or warn about humid heat?
- Does IMD (or NDMA) have any **official criterion, index or warning category for hot-and-humid
  conditions**: heat index, apparent temperature, wet-bulb, or "warm and humid" warnings?
  Quote the definition and thresholds if they exist, with document and page.
- If IMD issues an operational **heat index** product (some reports mention an experimental
  product from around 2024), find the official description: what it measures, its thresholds, and its
  colour categories.
- We need this to word the advisory's humid-heat note so it is consistent with, and never seems to
  contradict, IMD.

### D3. Current IMD heat-wave warning lead time
Your report found a conflict: the IMD FAQ says the 1600 hrs bulletin gives a **five-day** warning,
while NCDC 2025 and NDMA 2019 say **four days**. Find IMD's **current** official statement of how many
days ahead its heat-wave warnings (and district-level colour-coded warnings) are issued. Quote it with
the document, its date and the page.

### D4. Housekeeping in the existing report (only if quick)
- Section 1 still says "89 entries" / "37 entries"; the final totals are 135 actions / 46 thresholds.
- Section 4 conflict 10 calls the AC 24 °C rule "unverified", but the BEE notice was added later.
Please update these so the report is internally consistent.

Return: a short markdown file `sources/PHASE6_PART_D_RESULT.md` with D1–D4, plus any new PDFs in
`sources/pdfs/` and quotes added to the JSON files in the same verified format as before.
