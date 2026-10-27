# Every model in this project, explained in plain English

Written 2026-10-20, after the final test. It covers:

- every model and building block we built or compared with, under its code name;
- what each one uses and how it differs from the others;
- how to read the numbers;
- how GraphCast compares;
- why retrieval behaves the way it does.

The test numbers come from `evaluation_test/test_results.md`; development numbers come from `evaluation_v2/`.

---

## 1. How to read the numbers

### The main score: RMSE, in °C

Every model makes a forecast for each of the next 5 days. For each forecast day we take:

> **error = forecast − what really happened**

**RMSE** (root mean square error) is the square root of the average of the squared errors:

> RMSE = √( average of error² )

- **It is in °C**, the same unit as the temperature. An RMSE of 2.26 means the forecasts are off by about 2.3 °C on a
  typical day.
- **Lower is better.** 0 would be a perfect forecast.
- **It punishes big misses more than small ones,** because errors are squared before averaging. Missing by 4 °C
  counts four times as much as missing by 2 °C. That suits heat forecasting, where a big miss on a dangerous day
  matters most.
- **Each neural model is trained 10 times** with different random starting points (10 "seeds"). The number we report
  is the average RMSE of those 10 copies, so one lucky or unlucky training run cannot decide a comparison.
- Example: C3-hop's WBGT RMSE of 2.26 vs the Delhi-only LSTM's 2.39 means C3-hop's WBGT forecasts are about 0.13 °C
  closer to the truth on a typical day, roughly 5% less error.

### "All days" and "heatwave days"

- **All days:** every forecast day in the period, scored together.
- **Heatwave days:** only the days that **actually turned out** to be heatwave days (the "extreme" stratum). They are
  the days that matter most, and the hardest to forecast.
  - **Tmax heatwave day:** between 15 March and 31 July, and either (Tmax ≥ 40 °C and at least 3 °C above normal for
    that date) or Tmax ≥ 45 °C, and part of a run of hot days (an episode).
    - "Normal" is the 1980-onwards average for that day of the year, from training years only.
    - The 3 °C threshold, instead of IMD's 4.5 °C, is used because our temperature is the average over a box of 9
      ERA5 grid cells around Delhi, which is smoother than a single weather station.
  - **WBGT heatwave day:** between 15 March and 30 September, and daily maximum WBGT at or above the 95th percentile of
    the training years (about 36.2 °C), and part of an episode.
- **A warning about heatwave-day RMSE:** it can be "improved" just by forecasting hotter all the time. That is called
  the forecaster's dilemma. So we always report the bias (average error, too warm or too cold) next to it, and the
  warning score below.

### Other scores used in places

| Score | What it measures | Better is |
|---|---|---|
| **Bias** | The average error. Negative = the forecast runs too cold. | closer to 0 |
| **Brier score** | How good the "chance of a heatwave day" warnings are: the average of (forecast probability − what happened)², where "what happened" is 1 = heatwave, 0 = not. | lower |
| **Brier skill** | How much better the warning is than just using the usual chance for that time of year. 0 = no better; 1 = perfect. | higher |
| **CRPS** | How good a *range* forecast is (e.g. "likely 34, 1-in-10 chance above 37"), in °C. For a single number it equals the plain error. | lower |
| **Coverage** | How often the truth falls inside the forecast range. For an 80% range, ideal is 80%. | closer to the nominal % |

### What "confirmed" means

- A difference counts as **confirmed** only if its 95% confidence interval stays entirely on one side of zero.
- The intervals come from a **paired cluster-jackknife**. Both models are scored on exactly the same days. The test
  then leaves out one year-and-season block at a time and checks how much the difference moves, because neighbouring
  days are not independent.
- For the four main test claims (H1–H4), we also applied a **Holm correction**, because testing several claims at once
  raises the chance of a lucky "win".

### Which years

| Period | Used for |
|---|---|
| 1980 up to each validation block | Training: everything before each fold's block (1980–2006 for the first fold, 1980–2015 for the last, 1980–2018 for the test). The last 2 of those years are held back to decide when to stop training, so the models are fitted on the rest (the test models on 1980–2016). |
| **2007–2018, development** | Four rolling blocks of 3 years. Every model choice was made here. |
| **2019-01-01 to 2026-09-06, the test** | Locked until the protocol was frozen (git tag `test-lock`), then used **once**. |

---

## 2. What the models see

- **What we forecast:** for Delhi, the next 5 days of either
  - **Tmax**, the daily maximum air temperature, or
  - **WBGT**, the wet-bulb globe temperature: a heat-stress index combining temperature, humidity, sun and wind. We
    compute it physically with the Liljegren method.

  Each is the average over a box of 9 ERA5 cells (about 75 km across) around Delhi. Tmax and WBGT are two separate
  model "families". Each model is trained for one of them.
- **Delhi inputs:** the last **14 days** of Delhi's daily weather: Tmax, Tmin, mean temperature, humidity, wind,
  pressure, sunshine, day-of-year, years since 1980, and the target's own recent values and climatology. That is 13–14
  numbers per day.
- **Upstream inputs** (only the "upstream" models): the same 14 days at **27 points** on a 2° grid over north-west
  India and Pakistan, 24–32°N × 68–78°E. That is 12 numbers per point per day: temperatures, dew point, humidity,
  sunshine, soil moisture, pressure, wind. Hot, dry air reaches Delhi from that direction.
- **Data source:** ERA5 reanalysis (the European weather service's reconstruction of past weather), daily, from 1980.

### How to decode the code names

| Part of the name | Meaning |
|---|---|
| `A1` | the original (v1) Delhi-only LSTM: Tmax, old heatwave labels, hot days weighted ×20 |
| `A1prime` (A1′) | the same LSTM with the new (v2) heatwave labels |
| `A2` / `A2r` | WBGT models using the Australian Bureau of Meteorology's simple WBGT formula (raw / anomaly target) |
| `A2L` / `A2Lr` | WBGT models using the physical (Liljegren) WBGT; `r` = the model predicts the anomaly (difference from normal) |
| `_hw5` | hot days count **5×** in training (the "hot weight"). The pre-registered sweep rule picked 1; we fixed 5 instead, as a disclosed deviation (`context/decisions.md`), because weight 1 gave up the heatwave-day skill |
| `_dp` | the model predicts a correction to "damped persistence" (an early experiment) |
| `_tuned` | settings (size, learning rate, dropout) picked in the tuning round |
| `_C2`, `_C3`, `_C3hop`, `_C4` … | the upstream graph models (section 5) |
| `_U1` | the upstream LSTM (section 5) |
| `_R0`, `_Rg`, `_Rz` … | retrieval versions (section 7) |

**The two "controls"** (the Delhi-only LSTMs every new model is compared with):

- **`A1prime_hw5`** for Tmax;
- **`A2Lr_hw5`** for WBGT.

---

## 3. The simple reference forecasts (no learning)

| Name | What it forecasts | Why it is there |
|---|---|---|
| **Climatology** ("the usual for this date") | The long-term average for that day of the year (training years only). | Any useful model must beat "it'll be normal". |
| **Persistence** ("same as yesterday") | Every one of the next 5 days equals the last observed day. | Hard to beat at day 1, poor later. |
| **Damped persistence** | Yesterday's departure from normal, shrinking back towards normal day by day, at a rate fitted on training data. | A smarter version of persistence. |

---

## 4. The Delhi-only LSTM family (the controls)

**What an LSTM is:** a neural network that reads a sequence one day at a time, keeping a memory of what it has seen.
After the 14 days it outputs the 5 forecast days.

- **Our LSTM:** 2 layers, 64 memory units, dropout 0.2. It sees **Delhi's inputs only**.
- **Training:** the same recipe for every neural model here:
  - Adam optimiser;
  - up to 100 passes over the data;
  - stop when the score on the last 2 training years stops improving (patience 10).

| Code | What changed vs the one before | Status |
|---|---|---|
| `A1`, `A1_repro` | the original v1 model, reproduced bit for bit | legacy reference |
| `A1prime` | new v2 heatwave labels | step on the ladder |
| `A1prime_hw5` | hot days weighted 5× instead of 20× (20× made it forecast too warm on normal days) | **Tmax control** |
| `A2`, `A2r` | WBGT target (BoM formula) | superseded by the physical WBGT |
| `A2L`, `A2Lr` | physical WBGT target (`A2Lr` predicts the anomaly) | step on the ladder |
| `A2Lr_hw5` | the same with hot weight 5 | **WBGT control** |

**`PH_lstm`, the physics head (dropped).** This was the WBGT control's LSTM with a different ending. Instead of
predicting WBGT directly, it predicted temperature, humidity and the other ingredients, then computed WBGT from them
with the exact physical formula inside the model. It was dropped by its pre-set rule: it was worse than the direct
models (+0.04 °C WBGT, +0.13 °C Tmax).

---

## 5. The upstream models: C3-hop and its relatives

All of these see Delhi **plus the 27 upstream points**. They differ in **how information gets from the upstream
points to Delhi's forecast**. That difference is what the comparisons test.

### The graph network idea

- Delhi and the 27 points are **nodes**; points less than 370 km apart are **neighbours**. With the 2° grid, that means
  each point's neighbours are the points next to it.
- Every node has a small memory (a GRU, a simpler cousin of the LSTM). It reads that node's own weather day by day.
- With **edges** (connections), each node also receives **messages** from its neighbours' memories every day.
- At the end of the 14 days, the forecast is computed from what the model "reads out". In most versions that is
  Delhi's memory; in some, more.

### Each relative, precisely

| Code | Connections between points | How upstream information reaches Delhi's forecast | Weakness or point of the test |
|---|---|---|---|
| **`U1`** (upstream LSTM) | **None, no graph at all.** The 27 points' daily numbers are simply appended to Delhi's, giving one long list of ~340 numbers per day, fed to the ordinary LSTM. | The LSTM has to work out on its own which of the 340 numbers belong to which place. | Tests "is more data enough, without any spatial organisation?" |
| **`C2`** ("graph without connections") | **None.** Each point has its own memory but never talks to the others during the 14 days. | At the end: Delhi's memory **plus the plain average of all 27 upstream memories** (the "pool"). Every point's latest day reaches the forecast, but with no notion of distance or direction. | Tests whether the connections add anything beyond simply including the region. |
| **`C3`** (original graph) | Fixed connections between neighbours, weighted by distance. | Messages move **one neighbour per day**, using the *previous* day's memory; the forecast reads **Delhi's memory only**. | **Flaw:** a point 4 hops away needs 4+ days to reach Delhi, so its **most recent day never reaches the forecast**, and that last day is exactly where the useful upstream signal is. |
| `C3_tuned` | as C3 | as C3, with tuned size and learning rate | same flaw |
| **`C3pool`** (C3 + pooled readout) | as C3 | As C3, **plus**, at the end, the plain average of all upstream memories, like C2. The latest upstream day reaches the forecast through that average. | Fixes C3's flaw, but the final step ignores distance: a point 100 km away counts the same as one 1,000 km away. |
| **`C3hop` (OUR MODEL)** | C3's neighbour layout, but grouped into **rings by exact distance**: ring 1 = direct neighbours, ring 2 = neighbours of neighbours, up to ring 4. Every upstream point is within 4 rings of Delhi. | **(1)** Each day, every node hears from **all** points within 4 rings at once: the average of each ring, with its own learned weight per ring. **(2)** At the end, the latest day of every ring is passed into Delhi, again with a separate weight per ring; the forecast reads Delhi's memory plus that ring summary. | Fixes C3's flaw **and** keeps distance structure: the model can learn that 600–1,000 km away matters differently from 200 km away. Method: "shortest-path message passing" (Abboud et al. 2022), inside a DCRNN-style graph recurrence (Li et al. 2018). |
| `C4` | Connections that **change daily with the wind**: strong when the wind blows from one point towards the other, weak otherwise. | As C3 (one hop per day, Delhi readout) | Same flaw as C3; was no better than C3. |
| `C4a` | C4 plus extra connections the model learns itself (only between neighbours) | as C4 | Same flaw; no better. |

**Sizes:**

- C3-hop: ~76,000 trainable numbers (parameters) for Tmax, with memory size 64; ~21,000 for WBGT, with memory size 32.
- The Delhi-only LSTM: ~56,000.

C3-hop is our own PyTorch code, not a ready-made library model.

### What the comparisons showed

- **Including the region is what helps.** U1, C2, C3-pool and C3-hop all beat the Delhi-only LSTM.
- **C3-hop fixed C3's flaw:** about 0.08–0.10 °C better than C3 in development.
- **C3-hop is statistically tied with C2 and C3-pool,** in development and on the test. So we can say "seeing the
  upstream region makes forecasts clearly better", but **not** "the graph's connections or distance rings add skill on
  top". The model is still a genuine graph network, and the occlusion maps in section 9 show what it learned.
- **U1 ties C3-hop on WBGT but is clearly worse on Tmax all days** (test: +0.10 °C). Giving an LSTM all the numbers
  without organising them spatially is not as good.

---

## 6. The straight-line formulas (ridge regression)

| Code | What it is |
|---|---|
| `ridge` | A fixed weighted sum. For each forecast day: weight × Delhi's last departure from normal + weights × the 27 upstream points' Tmax departures over the last 3 days. Fitted by ordinary least squares with a small penalty against huge weights. It sees the same 27 upstream points as the graph, but only their Tmax departures (no humidity or other variables), and can only combine them linearly. |
| `ridge_hw5` | The same formula, fitted with hot days counted 5×, like our neural networks. The comparison pre-registered for H3: the closest simple model, but **not like-for-like**. It has fewer inputs, and about 96% of C3-hop's heatwave-day advantage over it is a smaller cold bias. |

The plain ridge plays safe: it forecasts close to normal, so it is good on ordinary days and very cold on heatwaves.

---

## 7. Retrieval (the "RAG" part) and every variant we tried

**What retrieval means here:** for today's 14-day window, search all past windows from the training years only. Never
the same heatwave, and never anything less than 19 days before. Pick the **5 most similar** past situations and give
the model **what actually happened in the 5 days after each one**.

The network uses **attention**: it scores how relevant each of the 5 looks and takes a weighted average of their
outcomes. In practice, the weights stayed almost equal, so the model was effectively averaging them.

### What "similar" was based on, rung by rung

| Code | "Similar" means… | Random control (same model, random past days instead) | Result |
|---|---|---|---|
| `RA-v1` | 17 summary numbers of Delhi's 14 days (the original v1 model) | — | not clearly better than the plain LSTM |
| `R0` | the same 17 Delhi numbers, recomputed per fold | `R0rand` | no gain (rule G3 failed) |
| `R1` | R0, but only past days within ±30 days of today's date in the year | `R1rand` | no gain |
| **`Rg`** | the **regional** pattern: departures from normal at the 27 upstream points + Delhi over the last 3 days (+ dew point for WBGT) | `R0rand` | **WBGT −2% on all days**, beats random; no heatwave-day gain |
| `R2` | Rg, but the 5 are picked to be *different from each other* (maximal marginal relevance) | `R0rand` | ≈ Rg; failed |
| `R3` | Rg, using only the last 15 years when recent weather looks unstable (a "drift gate") | — | screened only: added nothing |
| `R4` | Rg, with old analogues' outcomes shifted up by the warming since then | — | screened only: added nothing |
| **`Rz`** | the **upper-air pattern**: the 500 hPa height map (weather systems about 5.5 km up) over South Asia, 10–45°N × 40–100°E | `R0rand` | on C3-hop, WBGT −1% on all days in development; failed (no heatwave gain); **did not replicate on 2019–2021** |
| `Rh` | the **humid-air pattern** at 12 extra points east and south-west of Delhi | `R0rand` | screened only: no gain |
| AnEn | no network: just the average of the 5 analogues' outcomes (the classic "analog ensemble") | — | reference only |

### Retrieval on our graph model ("G-R*")

| Code | What it is | Result |
|---|---|---|
| `C3hop_Rg` | C3-hop, unchanged, plus Rg's 5 analogues fed in at Delhi's final readout (through attention) | no gain, development or test |
| `C3hop_rand` | the same, with random past days (the control) | — |
| `C3hop_Rz` | the same with Rz's upper-air analogues | development: −1% on all days, but failed the rule; test 2019–2021: no gain |

**Why Rg helped the LSTM but not C3-hop:** Rg's analogues carry regional information. The Delhi-only LSTM can't see
the region, so it gains; C3-hop already reads the region directly, so the analogues have nothing new to add.

### Other retrieval uses we tried

- **Analog error correction:** find past *forecasts* like today's and subtract how wrong the model was on them. It made
  heatwave days **much worse** (+0.7 to +0.9 °C). See section 11.
- **Retrieval for warnings:** "of the 20 most similar past situations, how many became heatwaves?", blended into the
  warning probability. No improvement.

---

## 8. The test results (2019-01-01 to 2026-09-06)

**The metric:** RMSE in °C, the average size of the forecast error, so **lower is better** (section 1). It is the mean
over 10 training seeds, over all forecast days 1–5. "All days" = every forecast day; "heatwave days" = only the days
that actually were heatwave days.

| Model | WBGT RMSE, all days | WBGT RMSE, heatwave days | Tmax RMSE, all days | Tmax RMSE, heatwave days |
|---|---|---|---|---|
| **C3-hop (ours)** | **2.26** | **3.02** | **2.23** | **2.31** |
| Delhi-only LSTM (control) | 2.39 | 3.10 | 2.36 | 2.55 |
| Ridge, fair (hot days ×5) | 2.30 | 3.74 | 2.32 | 2.53 |
| Ridge, plain | 2.29 | 4.15 | 2.28 | 2.89 |
| Climatology ("usual for this date") | 2.57 | 4.75 | 2.87 | 4.45 |
| Persistence ("same as yesterday") | 3.04 | 3.78 | 2.77 | 2.56 |
| C2 (no connections, pooled) | 2.25 | 3.12 | 2.23 | 2.33 |
| C3-pool | 2.25 | 3.10 | 2.23 | 2.28 |
| U1 (LSTM with all 28 points) | 2.27 | 3.10 | 2.34 | 2.29 |
| LSTM + Rg retrieval | 2.34 | 3.28 | 2.34 | 2.65 |
| C3-hop + Rg retrieval | 2.26 | 3.10 | 2.23 | 2.43 |

**Bias on heatwave days** (average error; negative = too cold): C3-hop −2.66 °C for WBGT and −1.92 °C for Tmax. All
models run cold on real heatwave days. Section 11 explains why.

### The four pre-registered claims (WBGT, Holm-corrected)

| Claim | Difference in RMSE (°C), 95% interval | Confirmed? |
|---|---|---|
| H1: C3-hop beats the Delhi-only LSTM, all days | −0.127 (−0.180 to −0.075) | **yes** |
| H2: … on heatwave days | −0.081 (−0.261 to +0.100) | no. The test could detect about 0.25 °C (the pre-test estimate was ~0.45), so the development gain (−0.26) did not reproduce |
| H3: C3-hop beats the hot-day-weighted ridge on heatwave days | −0.718 (−0.895 to −0.540) | **yes**, mostly through a smaller cold bias |
| H4: LSTM + Rg beats the Delhi-only LSTM, all days | −0.042 (−0.080 to −0.004) | no (raw p 0.033, Holm-adjusted 0.066) |

**Warnings:** the Tmax heatwave warning has Brier skill +0.28 (Delhi-only LSTM +0.23). For WBGT it is +0.07 (weak).
As a yes/no warning, the averaged C3-hop forecast catches 27% of Tmax heatwave days (37% of its warnings are false
alarms) but only 9% of WBGT heatwave days (68% false alarms) (`evaluation_post_test/test_period_extras.md`).
That rule ("warn when the averaged forecast reaches the threshold") is the main reason. Counting every day that meets
the label rule (6% caught under the plain rule), warning instead when the hot-day probability passes a level chosen
on 2007–2018 catches about 47% of WBGT heat-stress days, but about 4 in 5
of those warnings are false alarms (ETS 0.04 → 0.09). Under the same tuned rule, C3-hop is no better a yes/no warner
than the LSTM or the ridge models (post-test, `evaluation_post_test/warning_rules.md`).

**Range forecasts:** the 80% range contains the truth 77–78% of the time on all days, but only 29% (WBGT) and 54%
(Tmax) of the time on heatwave days: there the forecasts are too cold and the ranges too narrow.

---

## 9. What the graph learned (occlusion maps)

We switched off one input at a time, replacing it with its long-term average, in all 80 trained C3-hop models
(10 seeds × 4 folds) on the development years. Then we measured how much worse the forecast got (`evaluation/graph_explain.py`).

- **No single upstream point matters on ordinary days.** Each one changes the RMSE by less than 0.02 °C, because its
  neighbours carry the same information.
- **Whole rings matter a lot.** For Tmax heatwave days, switching off the farthest rings (600–1,000 km) costs over
  1.3 °C.
- **On heatwave days, the far-western desert points (Thar, Pakistan; 68–72°E) matter most.** That is where Delhi's
  hot, dry air comes from.

See `evaluation_v2/figures/graph_importance.png`.

---

## 10. GraphCast: the outside yardstick, and how it was better

**What GraphCast is:**

- Google DeepMind's global AI weather model (Lam et al. 2023, *Science*), itself a graph neural network.
- It was trained on ERA5 from 1979 to 2017, and forecasts the **entire global atmosphere** (the surface and dozens of
  levels up to the stratosphere) in 6-hour steps, up to 10 days ahead.
- We did not run it. We used its published 2018 forecasts from the WeatherBench 2 archive.

**How we compared:**

- **Tmax only:** GraphCast has no WBGT.
- **2018 only:** the year WeatherBench 2 publishes; it sits inside our development block.
- **Same windows:** the 361 forecast windows with query dates in 2018.
- **Same box:** GraphCast's temperature averaged over the same Delhi box.
- **No information advantage in time:** GraphCast was started at 17:30 IST the evening before the first forecast day,
  so it knew nothing our models didn't, in time.
- **Daily maximum:** GraphCast only gives values every 6 hours, which misses the afternoon peak. We converted those to
  a daily maximum with a correction fitted on our own 1980–2015 hourly data. That conversion alone costs about
  **0.67 °C** even for a perfect 6-hourly forecast (the "floor").

**The result** (Tmax RMSE in °C, lower is better):

| | All days | Day 1 | Day 2 | Day 3 | Day 4 | Day 5 | Heatwave days* |
|---|---|---|---|---|---|---|---|
| **GraphCast** | **1.15** | **0.91** | **1.11** | **1.16** | **1.23** | **1.32** | 1.05 |
| C3-hop (ours) | 1.98 | 1.35 | 1.76 | 2.05 | 2.24 | 2.32 | 2.01 |
| Delhi-only LSTM | 2.11 | 1.54 | 1.95 | 2.21 | 2.33 | 2.38 | 2.40 |
| Perfect 6-hourly forecast (floor) | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 | 0.69 |

\* Only 20 heatwave forecast-days in 2018, too few to rank models on; shown for completeness.

**How it was better:**

- About **40% lower error** on all days.
- **The gap grows with lead time:** 0.44 °C worse for us on day 1, 1.00 °C worse on day 5. GraphCast sees the weather
  systems that will arrive in 3–5 days; our models see 28 points of daily surface weather.
- Its bias was small (−0.17 °C overall, −0.71 °C on heatwave days); ours on heatwave days is −1.8 °C.

**Why that's expected:** GraphCast has the whole atmosphere at fine resolution, every 6 hours. We use daily data at
28 points.

**Our contribution is not to beat it.** It is:

- a cheap, transparent local model that runs on a laptop;
- WBGT heat stress, which GraphCast does not produce;
- ranges and warnings that are calibrated on ordinary days (on heatwave days they are too cold and too narrow;
  section 8);
- a heat-advisory system built on top.

Report: `evaluation_v2/graphcast_yardstick.md`.

---

## 10b. Pretrained general-purpose forecasters (Chronos-2, TimesFM 3.0): our model beats both

**What they are:**

- **Chronos-2** (Amazon, 2025) and **TimesFM 3.0** (Google) are "time-series foundation models": large networks
  pretrained on millions of time series of every kind (sales, traffic, energy, weather…).
- They are meant to forecast *any* series without extra training ("zero-shot").
- They are what many people would try first today.

**Why we chose these two:**

- **They are what people actually reach for first.**
  - When someone today wants a quick AI forecast of a series like Delhi's temperature, the usual first step is to
    download one of these and run it with no training.
  - They are by far the most-used pretrained forecasters on Hugging Face, the main public model hub. Downloads in
    the last month (Hugging Face's count, checked 2026-10-21):
    - Chronos-2: about **15.6 million**;
    - TimesFM 3.0 + 2.5: about **2.6 million**;
    - Salesforce's Moirai 2.0 (small), the next most popular alternative: about 0.26 million.
  - So these two answer the question a mentor or reviewer is most likely to ask: *"why build your own model
    instead of just using Chronos or TimesFM?"*
- **They come from two different, major teams** (Amazon and Google) with different designs. A result that holds
  against both is not about one model's quirks.
- **We used their newest versions,** Chronos-2 and TimesFM 3.0, not older or smaller ones. Beating last year's small
  versions would be weak evidence; we wanted the strongest opponents available.
- **They are free, open and run locally on a laptop CPU.** No data leaves the machine, and anyone can reproduce the
  comparison.
- **Why not others:**
  - **TimeGPT** (Nixtla) is a paid online service, and our data would have to be sent to it.
  - **Moirai** and **Lag-Llama** are much less used; adding them would mean more comparisons without changing the
    conclusion.
  - **Weather foundation models** (Pangu-Weather, FuXi, Aurora, GraphCast) are a different kind of model: they need
    the whole global atmosphere as input and are in GraphCast's class, which we already know is better than ours
    (section 10). Chronos and TimesFM, like our models, need no global atmosphere, only a local data series, so they
    are the natural ready-made alternative for this project. It is not a like-for-like contest: we gave them one
    series each (Chronos-2 could also take extra input series), while our models see many variables.

**How we compared** (after the main test, as a labelled extra; rules written down first):

- Each model got only the target's own last 1,024 days (about 2.8 years) of Delhi's daily Tmax or WBGT.
- It forecast the next 5 days for every 2019–2026 test window, using its median as the forecast.
- It was scored on exactly the same days, with the same heatwave labels and the same statistical test as our models.
- Download: 1.8 GB of weights, kept inside the project.

**Result (test years 2019–2026, RMSE in °C, lower is better):**

| Model | WBGT, all days | WBGT, heatwave days | Tmax, all days | Tmax, heatwave days |
|---|---|---|---|---|
| **C3-hop (ours)** | **2.26** | **3.02** | **2.23** | **2.31** |
| Delhi-only LSTM (ours, control) | 2.39 | 3.10 | 2.36 | 2.55 |
| Chronos-2 (Amazon) | 2.39 | 3.82 | 2.41 | 3.10 |
| TimesFM 3.0 (Google) | 2.40 | 3.83 | 2.41 | 3.05 |

- **C3-hop beats both on all days** (by 0.13–0.18 °C) **and on heatwave days** (by 0.75–0.81 °C), for WBGT and Tmax.
  Every one of these differences is statistically clear (p ≤ 0.002).
- **Even our Delhi-only LSTM** ties them on ordinary days and beats them on heatwave days: clearly for WBGT
  (−0.72 °C, p ≤ 0.002), less clearly for Tmax (−0.55 °C, p 0.039, vs Chronos-2; −0.51 °C, p 0.051, vs TimesFM).
- **Why:** they only see the history of one number and extrapolate its patterns. Our models also see the other
  weather variables, the day-of-year climatology, and (C3-hop) the upstream region the heat comes from. On heatwave
  days the foundation models are about 3.5 °C too cold for WBGT, more than ours (2.7 °C).
- **Caveat:** their pretraining data may include weather series from these years, which could only have helped them.

Report: `evaluation_post_test/foundation_baselines.md`.

---

## 11. Why retrieval helps ordinary days a little but not heatwave days

**What retrieval does:** it finds the 5 past situations that looked most like today and hands the model what
happened next in each. The model mostly **averages** those 5 outcomes; its attention stayed almost even across them.

**On ordinary days, averaging helps:**

- The 5 past situations carry a hint of the wider **regional** weather, which a Delhi-only model cannot see.
- Averaging 5 outcomes cancels their individual quirks.
- Result: about **−0.04 °C** on all days, the same in development (−0.044) and on the test (−0.042). On the test
  this was **not confirmed** after correcting for the four claims (H4, Holm-adjusted p 0.066), and about half of it
  is simply a cooler average forecast: the Delhi-only LSTM runs slightly warm, and retrieval cools it.

**On heatwave days, the same averaging hurts:**

1. **Similar-looking days usually did not become extreme.** A heatwave is rare. When today looks like the start of
   one, most past days that looked like this turned out merely warm. So the average of their outcomes is **cooler**
   than what really happens on a heatwave day. This is "regression to the mean".
2. **The model is already too cold on heatwave days** (WBGT about 2.7 °C low), and retrieval pulls it colder still.
   On the test, the cold bias on heatwave days got worse with retrieval: WBGT −2.75 → −2.94 °C, Tmax −2.02 → −2.20 °C.
3. **There are too few past heatwaves, and the model can't single out the right one.** Only a few dozen heatwave
   episodes exist in the training years. The model weighs all 5 analogues almost equally, so the ordinary ones dilute
   any truly matching extreme one.
4. **Likely but not proven: the analogues come from a cooler climate.** They are 14–17 years old on average, and humid
   heat (WBGT) has been rising, about +0.32 °C per decade over 1980–2018 (95% range +0.16 to +0.48). Our
   warming-correction variant (R4) did not
   help in screening, though, so this is not demonstrated.

**The same mechanism explains the analog-correction failure.** When C3-hop *forecasts* a hot day, it is usually too
*warm* (mostly false alarms). Corrections learned from similar past forecasts therefore cool it down, and that makes
the days that really were extreme even colder.

**Honesty point:** in development, Rg slightly *helped* heatwave days (−0.16 °C, by shrinking the cold bias) but
never beat its own random-retrieval control there; on the test it *hurt* them (+0.18 °C, p 0.04, one exploratory
comparison among many, uncorrected). The heatwave-day effect is small and unstable either way. Only the ordinary-day
gain was consistent in size, and even that was not confirmed on the test.

**In one sentence:** retrieval averages past situations, which smooths ordinary forecasts, but drags heatwave
forecasts towards "typical warm" exactly when they need to be more extreme. Only information the model doesn't
already have can fix heatwave days, and none we tried (the upper-air pattern, humid air, past heatwave starts) did.

---

## 12. Where the code lives

| Module | File |
|---|---|
| Delhi-only LSTM | `models/lstm.py` |
| Retrieval LSTM (v2) | `models/retrieval_lstm_v2.py` |
| Graph network (C2, C3, C3-pool, C4, C4a) | `models/dstgnn.py` |
| C3-hop (ring averaging) | `models/dstgnn_multihop.py` |
| C3-hop + retrieval (G-R*) | `models/dstgnn_retrieval.py` |
| Physics head | `models/physics_head.py`, `training/physics_train.py` |
| Training (every model, from a config) | `training/train_unified.py`, `configs/*.json` |
| Rolling folds and the test fold | `training/folds.py`, `training/test_lock.py` |
| Upstream inputs and graph edges | `training/graph_data.py`, `pipeline/graph.py` |
| Retrieval (all modes) | `retrieval/fold_retrieval.py` |
| Heatwave labels | `pipeline/labels_v2.py` |
| Physical WBGT | `pipeline/wbgt_liljegren.py` |
| Statistics (cluster-jackknife) | `evaluation/stats.py`, `evaluation/compare_v2.py` |
| Ridge baselines | `evaluation/ridge_baselines.py` |
| GraphCast yardstick | `evaluation/graphcast_yardstick.py` |
| Range forecasts, warnings, occlusion | `evaluation/range_forecasts.py`, `evaluation/calibration_check.py`, `evaluation/graph_explain.py` |
| The test run and its scoring | `training/test_run.py`, `evaluation/test_evaluation.py` |
