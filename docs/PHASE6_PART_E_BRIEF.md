# Phase 6 research brief, Part E: alert colours and which action belongs to which colour

**For:** Cowork (source research). **From:** the forecasting team. **Date:** 2026-10-06.
**Return:** `sources/PHASE6_PART_E_RESULT.md`, plus updates to `sources/phase6_candidate_actions.json` and
`sources/phase6_thresholds.json` in the same verified format as Parts A–D. Save any new documents in
`sources/pdfs/` (use Print to PDF for web pages).

## Why we need this

The advisory works in three steps: forecast → **alert colour** (GREEN / YELLOW / ORANGE / RED) → the
actions allowed at that colour. Your action library has 135 verified actions. The sources name a colour for
only 12 of them (1 YELLOW, 7 ORANGE, 4 RED); the other **123 are `tier_min: UNSPECIFIED`**. Without
colours, the advisory can't decide which actions to recommend on a given day. We must not invent the
colours ourselves, so we need them from official documents.

## Ground rules (same as Parts A–D)

- Official sources first: IMD, NDMA, Delhi Government / DDMA (Delhi Disaster Management Authority), MoHFW /
  NCDC / NPCCHH, CEA / BEE / Delhi DISCOMs, or peer-reviewed papers.
- Every claim gets an **exact quote, document, page number and date**.
- **"Not found" is a valid answer.** Say where you looked. Don't fill gaps with guesses.
- Only change `tier_min` in the JSON when a source states the colour. Put the quote in the row and add a
  field `tier_source` (doc_id, page). Leave everything else `UNSPECIFIED`.
- Energy actions are now **in scope** (team decision 2026-10-06), so they need colours too.

---

## E1. Which colour system defines Delhi's heat alerts?

We have two systems that disagree:

| System | Yellow | Orange | Red |
|---|---|---|---|
| **IMD / NDMA**, based on how long the heat lasts (`IMD_HW_FAQ`, p.8) | Heat wave conditions at isolated pockets persist on 2 days | (i) Severe heat wave persists for 2 days, or (ii) heat wave persists for 4 days or more | (i) Severe heat wave persists for more than 2 days, or (ii) total heat / severe heat wave days exceed 6 |
| **Delhi Heat Action Plan 2025**, based on temperature (`DELHI_HAP_2025`, p.10, "Heat Alert Thresholds for Delhi City (source: NDMA)") | Tmax ≥ 40 °C | Tmax ≥ 45 °C | **also ≥ 45 °C as printed** |

Questions:

1. **Which system do Delhi's authorities actually use** to issue and act on heat alerts: IMD's colour
   warnings, the HAP temperature table, or both? Look for a sentence in the Delhi HAP 2025 (or DDMA / Delhi
   Government orders) saying which alert triggers the plan's actions. A quote such as "the HAP is activated
   on IMD orange alert" is exactly what we need.
2. **What separates RED from ORANGE in the Delhi HAP table?** Please look at page 10 of the HAP 2025 PDF
   itself (the page image, not only the text layer). Is there a second condition, such as a number of
   consecutive days, a heat index, a night (minimum) temperature or humidity? Or is it a printing error?
   Also check the **Delhi HAP 2024-25** (`DELHI_HAP_2024_25`) for the same table: does it differ?
3. **IMD Red, condition (i):** your notes say its words are mixed with neighbouring cells in the PDF text.
   Please confirm "Severe heat wave persists for more than 2 days" against the page image and quote it.
4. **GREEN:** do any of the documents define a GREEN / "no alert" / "normal day" level with its own
   actions, such as seasonal preparedness? Quote it if so.

## E2. Which action belongs to which colour?

The best source is an **official table or list that assigns actions or department responsibilities to
alert colours**: "at Yellow do X, at Orange do Y, at Red do Z". Please search, in this order:

1. **Delhi HAP 2025 and 2024-25:** a department-wise responsibility matrix or an annexure by alert level.
   Check the end of the plan and every table, not only the text.
2. **Delhi DDMA:** a heat-wave SOP, a heat action plan order or an alert-level action plan, if separate from
   the HAP.
3. **NDMA 2019 Heat Wave Guidelines** (`NDMA_HW_GUIDELINES_2019`): annexures with sample colour-coded action
   plans or department checklists.
4. **MoHFW / NCDC / NPCCHH** (`NPCCHH_NAP_HRI`, `NCDC_ADVISORY_STATES_2025`): hospital or health-department
   actions by alert level, e.g. "on Orange alert, activate heat-stroke rooms".
5. **Energy:** CEA (`CEA_ELNINO_ADVISORY`), BSES / TPDDL (Delhi DISCOMs) and Delhi SLDC documents. Do any
   tie demand-side or grid actions to a heat alert or alert colour?
6. **Fallback precedent (only if Delhi has nothing):** the **Ahmedabad Heat Action Plan** (2019 or later),
   which most Indian HAPs copied and which has an action-by-colour table. Label anything from it clearly as
   "precedent from another city, not Delhi policy".

For **each of the 123 UNSPECIFIED actions** (`action_id` in `phase6_candidate_actions.json`):

- if a source assigns it (or a clearly equivalent action) to a colour, set `tier_min`, add the quote and
  `tier_source`;
- if not, leave it `UNSPECIFIED`.

In the result file, give a summary table:

| action_id | colour found | source (doc, page) | exact quote | "same action" or "equivalent action"? |
|---|---|---|---|---|

and a count: how many of the 123 got a colour from a Delhi source, from a national source, from the
Ahmedabad precedent, or none.

**Helpful extra (quick):** for each action, mark whether its trigger is

- **forecast-based**: the action is taken *because an alert or forecast says heat is coming*
  (e.g. "make public announcements a day before"), or
- **observation-based**: the action is taken *when something has already happened* (a patient with heat
  stroke, a body temperature ≥ 40 °C, more than 5 heat deaths in a day), or
- **standing / seasonal**: always on during the heat season, or a one-time preparation
  (e.g. "keep emergency wards ready", "equipment list before the season").

A column `trigger_type` with `forecast` / `observation` / `standing` is enough. It helps the team decide
how the advisory should handle the second and third kinds.

## Return checklist

- [ ] `sources/PHASE6_PART_E_RESULT.md` with E1 (questions 1–4) and E2 (summary table + counts)
- [ ] `phase6_candidate_actions.json`: `tier_min`, `tier_source`, `trigger_type` filled where found
- [ ] `phase6_thresholds.json`: any new colour definitions (with quotes)
- [ ] new PDFs saved in `sources/pdfs/`
- [ ] verifier re-run: 0 hard failures
