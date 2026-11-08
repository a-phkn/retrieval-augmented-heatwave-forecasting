# Demo page: brief

One web page that shows how the Delhi heat advisory works: the 5-day forecast, the alert colours, the advisory for
each day and the "If this happens" section. It is for the project presentation and the paper. It must open in a
browser with nothing to install.

## What the page shows

1. **A forecast date to pick:** one of the 52 evaluation cases (13 per alert colour) in `advisory/eval/cases.json`.
2. **A 5-day chart:**
   - the forecast maximum temperature (Tmax) against the normal for each day;
   - the 7 observed days before it;
   - each day's chance of heat-wave conditions.
3. **The alert colours:** each forecast day's colour (Green, Yellow, Orange, Red) and the highest one. Show this label
   word for word next to them: "IMD-style, adjusted for area-average data; not an official IMD warning".
4. **Click a day to see its advisory:**
   - the day's sentence;
   - the highlighted actions (at most 8), each with its quote and source (document and page);
   - "More actions" folded underneath.
   - Some actions carry an extra label (their colour comes from Ahmedabad's plan); show it with the action.
5. **The humid-heat note**, when there is one. It comes from our WBGT, so it never uses IMD's colours or words.
6. **Heat-season preparedness** actions.
7. **"If this happens"**, kept separate: the public actions, then the steps for health professionals behind their
   disclaimer.

## Show the rule-based advisory

The blind rating found that the language-model version adds advice no source backs (`advisory/eval/rating_results.md`),
so the page shows the rule-based advisory. Never add your own wording to an advisory: every action shown is the
library's own text and quote.

## Where the data comes from

- `advisory/eval/cases.json`: each case has `query_date`, `headline_colour`, `observed` (7 days: `date`, `tmax`,
  `normal`) and `forecast` (5 days: `date`, `tmax`, `normal`, `p_hot`, `wbgt`).
- `advisory/action_library.json`: 133 actions, each with `action_id`, `text`, `audience`, `section` and `min_colour`,
  and a `source` giving `doc_id`, `page` and the quoted `text`.
- The advisory for one case (Python, from the repo root, with the project's `.venv`):

  ```python
  from advisory.cases import assessment_of, load_cases
  from advisory.library import load_library
  from advisory.verifier import template_advisory

  doc, library = load_cases(), load_library()
  case = doc["cases"][0]
  a = assessment_of(case, doc["wbgt_threshold"])
  adv = template_advisory(a, library)
  ```

  - `adv` holds `label`, `headline_colour`, `summary`, `days` (`date`, `colour`, `p_hot`, `text`),
    `forecast_actions` (highlighted ids, with any extra `label`), `preparedness`, `if_this_happens` (`public`,
    `professionals`) and `humid_heat_note`.
  - "More actions" are `advisory.rules.forecast_actions(adv["headline_colour"], library)` minus the highlighted
    ones.
  - The professionals' disclaimer is `advisory.library.CLINICAL_DISCLAIMER`.
- Suggested set-up:
  - `demo/build_data.py` writes `demo/data.json`: all 52 cases, each with its advisory and the text and quote of
    every action it uses.
  - A static `demo/index.html` (plus CSS and JavaScript) reads it. Commit `data.json` too, so the page works offline.
- For layout ideas, open `advisory/eval/rating_pack.html` in a browser. It shows the same advisories as a plain list.

## Rules for this branch

- Work only on `demo-page`. Put everything in `demo/`, as new files; don't edit existing ones.
- Don't change anything in `training/`, `models/`, `pipeline/`, `retrieval/`, `evaluation/` or `configs/`: that
  breaks the test lock, and CI fails.
- Never commit `.env` files or API keys. Ask before adding a Python package; the page itself needs none.
- Plain HTML, CSS and JavaScript, so it opens straight from the file. A chart library from a CDN is fine; so is a
  hand-drawn SVG.
- Never show colour alone: write the colour's name too. Check it is readable at phone width.
- When it works, open a pull request into `main`. It is merged after review.

## Done when

- All 52 cases load, and each one's highest colour matches `headline_colour` in `cases.json`.
- Every action shows its quote and source, and the not-an-IMD-warning label is visible.
- It opens offline in Chrome and Firefox, at desktop and phone width.
