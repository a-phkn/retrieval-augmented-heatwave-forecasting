# Project Guide: Retrieval-Augmented Heatwave Forecasting for Delhi

> **Living document.** Read this first. It explains the whole project in plain language:
> what we build, how every part works, what the results are so far, and what comes next.
> It is updated whenever a result, a design decision or the plan changes. See the change
> log at the end. Short forms are explained in [`GLOSSARY.md`](GLOSSARY.md); the full
> plan is in [`PLAN_REVIEW_v5.md`](PLAN_REVIEW_v5.md).
>
> **Last updated:** 2026-10-06 (Week 3: retrieval runs started, DSTGNN skeleton)

---

## 1. The project in one paragraph

We forecast **heat stress in Delhi for the next 5 days**, using the previous 14 days of
weather. The twist is **retrieval**: before forecasting, the model looks up similar
situations from the past ("analogues": 14-day stretches of weather that looked like
today) and sees what happened after them. The question is whether this helps, especially
on heatwave days. The project then grows in two directions:
- a better forecaster, using a graph model over upstream regions and/or physics built
  into the model;
- an **advisory generator**, which turns the forecast into a risk level and a short
  advisory that may only recommend actions found in official documents (IMD, NDMA, WHO).

**Output:** a research paper plus a capstone demo.

**Team:** 4 people, 8 weeks. There is one role per person:

| Role | Owns |
|---|---|
| **P1** | Data and labels |
| **P2** | Forecasting models |
| **P3** | Retrieval |
| **P4** | Advisory and demo |

---

## 2. The data

| What | Details |
|---|---|
| Source | **ERA5** reanalysis (ECMWF's reconstruction of past weather), downloaded through the free **Open-Meteo** archive API. |
| Area | **9 grid cells** on a 0.25° grid over Delhi (28.25–28.75 °N, 77.00–77.50 °E). Each cell is about 25 km wide. |
| Time | Hourly, **1980-01-01 to 2026-09-06**, local time (IST). 17,051 days. |
| v1 variables (already downloaded) | Hourly temperature, relative humidity, wind speed and surface pressure. Daily values (max/min/mean T, mean RH, ...) are computed from these. |
| v2 variables (downloaded and verified, 2026-10-05) | Dew point, wind direction, solar radiation (total, direct, diffuse), cloud cover and soil moisture. See §8.2 for why. |
| Upstream points (download ready to run) | **Daily** values at 27 points on a 2° grid over north-west India and Pakistan (24–32 °N, 68–78 °E; 3 mountain points above 1,000 m dropped): max/min/mean temperature, dew point, humidity, wind speed and direction, sunshine, soil moisture, pressure. These are the nodes of the regional graph model (§9). Script: `pipeline/download_era5_upstream.py`. |
| Daily table | `datasets/all_daily.parquet` (v1) and `datasets_v2/all_daily_v2.parquet` (v2, adds heat-stress measures). Values are the **average over the 9 cells**. |

**Training examples ("windows").** Each example has:
- **input:** 14 consecutive days;
- **target:** the next 5 days.

The date of the first forecast day is the **query date**. There are 17,025 windows,
listed in a frozen index (`splits/window_index_v1.parquet`), so every model uses the
exact same examples.

**Splits.**

| Split | Years | Windows |
|---|---|---|
| Train | 1980–2015 | 13,131 |
| Validation (val) | 2016–2018 | 1,092 |
| Test | 2019 onwards | 2,802 |

The **test split is locked**: nobody looks at it until the very end (Week 7). That is the
only way the final numbers are honest.

---

## 3. What we forecast and how we score it

**v1 target:** daily **maximum temperature (Tmax)**, for each of the next 5 days
("lead 1" to "lead 5").

**Score: RMSE** (root mean squared error, in °C). It is roughly the typical size of a
forecast error, and lower is better.

**Strata.** Because heatwaves are rare, every forecast day is also put in one of three
groups ("strata"), and RMSE is reported for each group:

| Stratum | v1 rule (Tmax anomaly measured in standard deviations, σ) | Val forecast-days |
|---|---|---|
| normal | anomaly ≤ 1.0 σ | 4,575 |
| unusual | 1.0–1.5 σ, or > 1.5 σ but not part of a heatwave episode | 731 |
| extreme | > 1.5 σ **and** part of an episode (≥ 3-day span) | 154 |

- "Anomaly" means how far Tmax is above the normal for that calendar day. "Normal" is
  the climatology, explained in §4.
- In v1, a **hot day** is anomaly > 1.5 σ, at any time of year, so unusually warm winter
  days count too. v2 changes this (§8.3).

**Weighted loss (hot_weight = 20).** During training, errors on hot days count **20×**
more than errors on other days. This pushes the model to get heatwaves right. The
side-effect is that the model learns to **forecast too warm in general** (see §6).

---

## 4. The simple reference forecasts (baselines)

A fancy model is only worth something if it beats simple rules. None of these is a public
dataset or a public model. **We compute all of them ourselves** from our own training
data, in a few lines of code.

| Baseline | How it forecasts | Why it matters |
|---|---|---|
| **Persistence** | "Tomorrow and the next 4 days = today's Tmax." | The most naive forecast. Good at lead 1, bad at lead 5. |
| **Climatology** | "Each day = the normal Tmax for that calendar date." The normal (mean and spread) for each day of the year is computed over the **training years only (1980–2015)**, using all dates within ±7 days of that calendar day. Example: the normal for 20 May is the average Tmax of 13–27 May over 1980–2015. | Hard to beat at long leads. Weather forgets the past after a few days, so the seasonal normal becomes the best guess. |
| **Damped persistence** | Take today's anomaly (in σ), shrink it a bit more for each lead, and add it back to the normal. The shrink factors φ were fitted on training data: **0.81, 0.64, 0.51, 0.43, 0.37** for leads 1–5. Example: today is 3 °C above normal, so lead 1 is about 2.4 °C above normal and lead 5 about 1.1 °C above. | Combines the best of the two above: it starts like persistence and fades to climatology. **It is currently the strongest baseline**, and our credibility floor. |
| **AnEn** (analogue ensemble) | Find the 5 most similar past windows (the same retrieval as the RA model, §5.2) and average what happened after them. No training. | Tests whether retrieval carries information *without* any neural network. |

---

## 5. The two v1 models

Both were built in v1, then **retrained locally** (the original files were lost) and
**frozen** (fingerprinted so they can't change unnoticed). Each was trained with 5 random
seeds; results are the average over seeds.

### 5.1 Baseline LSTM (A1): `models/lstm.py`, `training/train_lstm.py`

An **LSTM** is a neural network that reads a sequence one step at a time and keeps a
memory of what it has seen.
- **Input:** 14 days × 13 features. The 13 features are 10 weather and calendar values
  (Tmax, Tmin, Tmean, humidity, wind, pressure, solar radiation, two day-of-year
  signals, years since 1980), plus the Tmax normal, its spread and today's anomaly. All
  are rescaled to a common scale ("z-normalised").
- **Network:** 2 LSTM layers with 64 units and 20% dropout, then a small head
  (64 → 32 → 5).
- **Output:** 5 numbers, Tmax for days 1–5.
- **Training:** Adam optimiser, learning rate 0.001, batches of 64, up to 100 passes over
  the data. It stops when the validation loss hasn't improved for 10 passes. Loss is
  weighted MSE with hot_weight 20.

### 5.2 Retrieval-augmented LSTM (RA-v1): `models/retrieval_lstm.py`

The same LSTM, plus a look-up step.

1. **Describe every window by 17 summary numbers** (`retrieval/features.py`): average,
   last-day and trend of the anomaly; number of hot days; Tmax level and trend; humidity
   level and trend; wind; pressure level and trend; and so on.
2. **Search** (`retrieval/query.py`): a **FAISS** index (a fast similarity search
   library) finds the past windows whose summaries are closest to today's. Three rules
   decide which past windows may be used:
   - (a) no future information: the analogue must be completely finished before today's
     forecast starts;
   - (b) a validation query may only use training-period analogues;
   - (c) no analogue from the same ongoing heatwave.

   Two more rules keep the analogues varied: at most 2 per heatwave episode, and at least
   10 days apart. The model gets the top **K = 5**.
3. **Fuse:** the same LSTM encodes today's window and each analogue. An **attention**
   layer weighs the analogues by how similar they look to the model, and passes along
   **what actually happened after each analogue**, i.e. its real 5-day outcome. The head
   then forecasts from today's encoding plus this "context".

**The idea:** when a heatwave is building, the model can see "the last 5 times it looked
like this, the next days went to 44–46 °C", and use that, instead of shrinking towards
average values as neural networks tend to do.

---

## 6. Results so far (validation 2016–2018, v1 labels)

RMSE in °C (lower is better). Neural models are averaged over 5 seeds.

| Model | All days | Normal | Unusual | Extreme |
|---|---|---|---|---|
| Persistence | 2.615 | 2.614 | 2.705 | 2.179 |
| Climatology | 2.344 | **1.940** | 3.486 | 5.031 |
| **Damped persistence** | **2.134** | 1.992 | 2.660 | 3.162 |
| AnEn (retrieval, no network) | 2.327 | 2.159 | 2.832 | 3.930 |
| **LSTM (A1)** | 2.401 | 2.524 | **1.682** | 1.285 |
| **LSTM + retrieval (RA-v1)** | 2.420 | 2.543 | 1.713 | **1.201** |

RMSE by lead day (all days):

| Model | Lead 1 | Lead 2 | Lead 3 | Lead 4 | Lead 5 |
|---|---|---|---|---|---|
| Damped persistence | **1.58** | **2.05** | **2.26** | 2.34 | 2.34 |
| Climatology | 2.35 | 2.35 | 2.35 | 2.34 | 2.34 |
| LSTM (A1) | 1.71 | 2.17 | 2.51 | 2.67 | 2.79 |
| LSTM + retrieval (RA-v1) | 1.73 | 2.23 | 2.54 | 2.69 | 2.76 |

### What this means, in plain words

1. **The LSTM is great on hot days, poor on normal days.** On extreme days its error is
   about 1.3 °C, against 3.2 °C for damped persistence and 5.0 °C for climatology. But on
   normal days it forecasts on average **1.1 °C too warm**, because hot_weight = 20 taught
   it to lean warm. Over all days it is **worse than damped persistence** (2.40 vs 2.13,
   a clear difference, p = 0.003) and no better than climatology.
2. **Careful: judging a model only on observed-extreme days rewards a warm bias.** A model
   that always forecasts hot looks great on hot days. This is known as the "forecaster's
   dilemma" (Lerch et al. 2017). So we also require models to beat the simple baselines
   over **all** days.
3. **Retrieval did not clearly help yet.**
   - RA-v1 minus A1: all days **+0.02 °C** (95% range −0.11 to +0.15), extreme days
     **−0.09 °C** (95% range −0.58 to +0.41).
   - Both ranges include zero, so the result is **inconclusive**.
   - The attention spreads almost evenly over the 5 analogues, so RA-v1 basically averages
     them.
4. **The analogues do carry information.** AnEn (pure retrieval) beats random past
   windows easily, and beats climatology on unusual and extreme days. Its skill is mostly
   in leads 1–2.
5. **Why "inconclusive" is expected:** the validation years contain only 9 season
   clusters (5 for extreme days). With that little data, only very large improvements
   (> 50% on extreme days) could be detected. This is why v2 uses **rolling folds** (§8.4).

How we test differences: a **paired cluster-jackknife t-test** (`evaluation/stats.py`). It
treats each year-and-season block as one unit, because neighbouring days are not
independent. Full table: `evaluation_v2/g0_val_comparison.md`.

---

## 7. Other things we found (Week 1)

- **Humid heat is rising in Delhi; dry heat (Tmax) is not detectably rising.** These are
  heat-season trends over 1980–2015, per decade:

  | Measure | Trend per decade | 95% range |
  |---|---|---|
  | Tmax | +0.14 °C | −0.19 to +0.48, not significant |
  | Wet-bulb temperature | +0.35 °C | significant |
  | BoM WBGT | +0.38 °C | significant |
  | Heat Index | +0.47 °C | significant |

  This is why we move to a heat-stress target, and why Tmax works as a natural
  "control". Caveat: ERA5 humidity may have a data artefact around 2000–01, so we still
  need to cross-check against weather-station data.
- **The BoM WBGT formula is an index, not a true WBGT.** It uses only temperature and
  humidity. Against the physical outdoor WBGT (built in Week 3, §8.2) it reads **2–3 °C
  too high in the humid monsoon months (Jun–Sep)** and about right otherwise. (An earlier
  rough estimate of "~6 °C" compared it with a *shade* WBGT; corrected 2026-10-05.) So we
  never apply official absolute WBGT thresholds to it.
- **Retrieval ranking is weak:** within the top 20 analogues, the more similar ones are
  barely better than the less similar ones.
- **Seeds matter:** different random seeds of the same model differ by about 0.2 °C on
  extreme days. So decision runs use **10 seeds**.

Reports: `evaluation_v2/premise_report.md`, `evaluation_v2/mde.md`.

---

## 8. v2: what changes and why

### 8.1 Why a new target (WBGT)

**WBGT** (Wet-Bulb Globe Temperature) measures heat stress on the human body. It combines
temperature, humidity, sun and wind. Tmax alone misses humid heat, and humid heat is what
is rising (§7).

### 8.2 Which WBGT needs which data (v1 vs v2 download)

There are two ways to compute WBGT, and they need different data:

| Version | Needs | Data source | Status |
|---|---|---|---|
| **BoM approximation** (`wbgt_bom_max`) | Temperature + humidity only | **v1 raw files** (already have them) | ✅ Built in Week 2. This is the target of A2/A2r. |
| **Liljegren (physical) WBGT** (`wbgt_lj_max`) | Temperature, humidity, **solar radiation**, **wind**, pressure | **v2 download** (radiation) + v1 files | ✅ Built in Week 3 (`pipeline/wbgt_liljegren.py`, `datasets_v2/wbgt_liljegren_daily.parquet`) |

**How the physical WBGT is computed, and who we credit:**
- **WBGT = 0.7 × natural wet-bulb + 0.2 × black-globe + 0.1 × air temperature.** The wet-bulb
  and globe temperatures come from heat-balance equations for a wet wick and a black globe
  in sun and wind (Liljegren et al. 2008).
- **Code:** ported from **Liljegren's original C program** (Argonne National Laboratory,
  open-source licence; the notice is kept in the file).
- **Method for reanalysis data:** follows **Kong & Huber (2022), *Earth's Future***, and their
  reference implementation **PyWBGT**. We use their approach for:
  - hourly-mean radiation (average sun angle over the sunlit part of each hour);
  - the direct-sunlight fraction taken from the reanalysis;
  - converting 10 m wind to 2 m;
  - solving the equations with a bracketed root finder.

  We did not copy their code (it is licensed CC BY-NC-SA). We used it as an independent
  reference to check ours.
- **Checks** (`tests/test_wbgt_liljegren.py`):
  - matches Liljegren's original algorithm within its own 0.02 K tolerance;
  - matches Kong & Huber's implementation to within **0.007 °C** on 3,000 real Delhi hours;
  - correct sun positions at the solstices;
  - basic physics: more sun → higher WBGT, more wind in sun → lower WBGT.

**What it showed (1980–2018):**
- The BoM index is 2–3 °C too high in Jun–Sep and about right in other months.
- It tracks the physical WBGT only partly (correlation 0.74, Apr–Sep).
- The **humid-heat trend holds with physical WBGT: +0.32 °C/decade** [+0.16, +0.48], while
  Tmax shows no detectable trend.

The v2 download also brings **wind direction** (for the upstream/graph model) and **soil
moisture** (dry soil amplifies heatwaves), which are physics inputs for later models.

**Is comparing v2 models with the v1 LSTM unfair?** No. v2 models are never compared with
the frozen v1 LSTM. Every v2 model is compared with an LSTM **retrained on exactly the
same data, target, labels and folds** (A1′, A2, A2r below). The frozen v1 models stay as
a historical reference.

### 8.3 New heatwave labels (labels v2), `pipeline/labels_v2.py`

- A **hot day** must fall in the heat season (15 Mar – 31 Jul) and meet one of:
  - Tmax ≥ 40 °C **and** anomaly ≥ 3 °C; or
  - Tmax ≥ 45 °C.
- An anomaly of 4.5 °C or more is marked **severe**, which is IMD's heatwave departure.
- **Why 3 °C and not IMD's 4.5 °C?** ERA5 averages over 25 km cells, which smooths station
  peaks. With 4.5 °C there would be only 1 heatwave episode in the validation years, too
  few to learn or test anything. The rule was chosen by a pre-agreed minimum-sample rule
  (≥ 25 episodes over the fold years; we get 30).
- **Episode:** hot days at most 2 days apart join one episode, and an episode must span
  ≥ 2 days.
- **Labels are always defined from Tmax,** even when the model predicts WBGT, because
  official warnings are Tmax-based.
- **Acceptance test:** the labels must flag 4 documented pre-2019 heatwaves, each with at
  least 2 "extreme" days. The windows come from the sources, never from the labels:

  | Event | Window | Evidence |
  |---|---|---|
  | 1998 | 22 May – 3 Jun | IMD sub-division spell (Haryana, Chandigarh & Delhi), Pai et al. 2004 |
  | 2002 | 10 – 20 May | Safdarjung station only (≥ 40 °C every day, peak 46.0 °C) |
  | 2010 | 8 – 20 Apr | IMD north-west India heat wave (IMD monograph) |
  | 2015 | 22 – 26 May | IMD 2015 summary (severe heat, second half of May) + Safdarjung station |

  IMD records heatwaves by region, not by city, so **none of these is an IMD-declared Delhi
  heatwave**. Say "checked against IMD regional/sub-division spells and Safdarjung station
  records", never "validated against IMD for Delhi". Details: `sources/PHASE6_PART_C_RESULT.md`.

**WBGT heatwave label** (humid heat):
- **Rule:** a day in **15 Mar – 30 Sep** whose daily max WBGT is at or above a percentile of
  in-season values from the fold's training years.
- **Why a longer season:** humid heat peaks in the Jul–Sep monsoon.
- **Choosing the percentile:** the highest of 99 / 98 / 97.5 / 95 / 92.5 / 90 that gives
  ≥ 25 episodes, the same rule as for Tmax. Result (the label in use, physical WBGT since
  2026-10-05): **95th percentile ≈ 36.2 °C → 29 episodes**, spread over May–Sep (sun and
  humidity) (`configs/wbgt_label.json`, `evaluation/select_wbgt_label.py`).
- **Key finding: dry and humid heatwaves are mostly different events.** Tmax heatwaves fall in
  Apr–Jun; BoM-WBGT heatwaves in Jun–Aug. Only 27 days are hot under both (1980–2018). A
  Tmax-only warning system would miss most humid heatwaves.
- **Sensitivity:** with the BoM index the same rule picks the 97.5th percentile (≈ 37.7,
  28 episodes). Its days overlap the physical-WBGT label days only partly (54 shared).

### 8.4 Rolling folds, `training/folds.py`

Instead of one validation block (2016–18), we use **four**, and each fold trains only on
the years before its block:

| Fold | Trains on | Validates on |
|---|---|---|
| f1 | 1980–2006 | 2007–2009 |
| f2 | 1980–2009 | 2010–2012 |
| f3 | 1980–2012 | 2013–2015 |
| f4 | 1980–2015 | 2016–2018 (= the v1 split) |

- That gives 12 validation years instead of 3, so 4× more evidence.
- Everything that depends on "training years" is **recomputed per fold**: the normal
  (climatology), anomalies, labels and scaling. So no fold can peek at its validation
  years. This is checked by a test that changes all later data and confirms that nothing
  in training changes.
- **Early stopping** (choosing when to stop training) now uses the **last 2 training
  years**, never the validation block that is scored. v1 stopped on the scored block,
  which flatters the model.

### 8.5 The v2 control chain: one change at a time

| Run | What it is | Change from parent |
|---|---|---|
| A1 | Frozen v1 LSTM (Tmax, v1 labels) | — |
| A1′ | LSTM on Tmax with **labels v2** | labels, early stopping, 2 fewer fitting years (see note) |
| A2 | LSTM predicting **BoM WBGT** | target only |
| A2r | A2 predicting the **anomaly** (WBGT minus its normal) | target form only |

Note: A1′ vs A1 mixes three effects. The A1′ → A2 → A2r chain is clean, because all three
use the same early-stopping rule.

- Configs are in `configs/*.json`; the trainer is `training/train_unified.py`.
- Configured as A1, the new trainer reproduces the frozen A1 **bit-for-bit**, so the new
  machinery is proven not to change results by itself.
- **A2r is expected to fix the warm bias.** By predicting the departure from normal, the
  model falls back to the normal when it has no signal.

**Week 3 results** (gate G2; out-of-fold 2007–2018, 10 seeds, `evaluation_v2/week3_controls.md`).
RMSE in °C; Δ is model minus baseline, so negative = model better.

| Model | All days | Δ vs climatology | Δ vs damped persistence | Extreme days | Δ vs damped persistence |
|---|---|---|---|---|---|
| A1′ (Tmax) | 2.29 | −0.30 ✅ | **+0.11 ❌** | 1.58 | −1.51 ✅ |
| A2 (WBGT) | 1.55 | −0.32 ✅ | −0.04 (n.s.) | 1.28 | −0.47 ✅ |
| A2r (WBGT anomaly) | 1.54 | −0.33 ✅ | **−0.06 ✅** (p = 0.02) | 1.34 | −0.41 ✅ |

- **Averaging the 10 seeds** (an ensemble) improves all three: A2 −0.074 and A2r −0.077 vs damped
  persistence (both significant); A1′ +0.070 (still worse).
- **A1′ over-forecasts heatwaves** (the forecaster's dilemma). It forecasts a heatwave on about
  1,200 forecast-days per seed, against about 730 real ones, with a +1.7 °C bias on those days.
- **Next:** try to bring A1′ (and the others) above the floor with a pre-set candidate list:
  - seed ensemble;
  - `hot_weight` re-sweep;
  - Tmax anomaly target;
  - learning a correction on top of damped persistence.

  The control model is chosen after these runs.

**Decision 2026-10-05: the WBGT models now forecast the *physical* WBGT and use the WBGT label.**
- The new chain, still one change per step:

  | Run | Change from parent |
  |---|---|
  | A1′ | — |
  | A2L_t | target → physical WBGT (still the Tmax label) |
  | A2L | label → WBGT label |
  | A2Lr | target → anomaly |

  The BoM-index runs A2/A2r stay as historical results.
- WBGT models are also scored on the Tmax label, so the two kinds of heat stay comparable.
- **Two conditions:**
  - Official alert tiers stay Tmax-based. Humid heat is a separate, clearly labelled heat-stress
    note, never an official "heatwave" declaration.
  - The final model must output both Tmax and WBGT (the physics head does this: see §9).
- **Pre-registered improvement runs** (15 runs, all finished 2026-10-06):
  - Tmax anomaly target (A1prime_r);
  - learning a correction to damped persistence (A1prime_dp, A2L_dp);
  - `hot_weight` 1 / 5 / 10 for A1′, A2L and A2Lr.
- **How the control was to be chosen** (rule fixed *before* the runs, `context/decisions.md`):
  among runs not significantly worse than damped persistence, take the lowest all-days error;
  near-ties go to the simpler run.
- The 10-seed average stays a reported extra, as decided on 2026-10-04.

**Improvement results** (`evaluation_v2/week3_controls.md`). The one lever that matters is
`hot_weight` (how much extra weight hot days get in training). The other two ideas (anomaly
target for Tmax, correction to damped persistence) did not help.

| Run | hot_weight | All days | Δ vs damped persistence | Extreme days | Δ vs damped persistence | Hot days it forecasts (per seed) |
|---|---|---|---|---|---|---|
| A1′ (Tmax) | 1 | 2.14 | −0.04 ✅ | 3.05 | −0.03 (no gain) | 115 (real: 730) |
| **A1′ (Tmax)** | **5** | **2.18** | **−0.01 (tie) ✅** | **2.15** | **−0.93** | 512 |
| A1′ (Tmax) | 10 | 2.23 | +0.04 (borderline) ✅ | 1.83 | −1.25 | 885 |
| A1′ (Tmax) | 20 | 2.29 | +0.11 ❌ | 1.58 | −1.51 | 1,198 |
| A2Lr (WBGT) | 1 | 2.29 | −0.08 ✅ | 4.17 | −0.31 | **0** |
| **A2Lr (WBGT)** | **5** | **2.37** | **+0.00 (tie) ✅** | **3.27** | **−1.20** | 54 |
| A2Lr (WBGT) | 10 | 2.52 | +0.15 ❌ | 2.70 | −1.77 | 378 |
| A2Lr (WBGT) | 20 | 2.76 | +0.39 ❌ | 2.27 | −2.21 | 1,615 |

This is a trade-off: less weight on hot days gives a better all-days score but worse heatwave
forecasts.

- **Raw RMSE across targets is not comparable.** Physical WBGT is harder to forecast than the BoM
  index because it includes sun and wind: even climatology scores 2.50 against 1.87. A2r (BoM)
  and A2Lr_hw1 (physical) both beat damped persistence by about 3.5 %.
- **Decision 2026-10-06: the controls are `A1prime_hw5` (Tmax) and `A2Lr_hw5` (WBGT).** This
  departs from the pre-registered rule, which picks hot_weight 1. That model brings no heatwave
  gain for Tmax, and for WBGT it **never forecasts a hot day**, so it is useless for heat warnings.
  - hot_weight is fixed at 5; the rule picks everything else unchanged.
  - The decision was made on development data only; the test years are still locked.
  - The paper must report the change and show the rule's own choice alongside.
  - hw5 ties damped persistence on all days and beats it by about 1 °C on extreme days.
  - Lesson: later selection rules (e.g. for retrieval) must score both all days and heat days.

---

## 9. What comes next

| Week | P1: data and labels | P2: models | P3: retrieval | P4: advisory and demo |
|---|---|---|---|---|
| 3 | Upstream-region download; check that upstream heat leads Delhi | Run A1′, A2, A2r (10 seeds × 4 folds) vs damped persistence; hot_weight re-sweep; DSTGNN skeleton + unit tests | R0 / R0-rand / R1 on the folds | Action-library schema from the official sources |
| 4 | Liljegren WBGT from the v2 download; check the BoM index against it | DSTGNN controls; physics head | R2 (diversity), R3 (drift gate), R4 (climate-shift correction) | Rule engine (forecast → risk tier) |
| 5 | Dry vs humid heatwave analysis | **Choose the final model ("BB\*")**; attach the best retrieval step | Analogue-age analysis | Verifier (checks every action against the library) + tests |
| 6 | Pre-registration document | Uncertainty bands (conformal); **test lock** | Retrieval mechanism figures | Constrained LLM advisory on mock forecasts; adversarial tests |
| 7 | Case studies 2019, 2022, 2024 | **One test-set run (10 seeds)** | Retrieval results on test | Advisory on real forecasts; human evaluation |
| 8 | Paper: data and labels | Paper: models and results | Paper: retrieval | Demo + reproducibility pack |

**The final model has three parts, used together:**
- **Backbone (reads the inputs): the regional DSTGNN** (graph neural network). Its nodes
  are 27 points over north-west India and Pakistan, plus Delhi. Delhi's heatwaves are often
  hot dry air blown in from the north-west over 1–3 days, so a graph that follows the wind
  can see them coming.
- **Physics head (the output layer, on top of the backbone; design fixed 2026-10-07):**
  - For each forecast day and each of Delhi's 9 grid cells, the model predicts the weather
    "ingredients" at that cell's hottest hour: temperature, humidity, pressure, wind, sunshine,
    the share of direct sun, and the sun's angle.
  - It then computes that cell's WBGT with the **exact** physical (Liljegren) formula, built into
    the model, and averages the 9 cells. That is exactly how the WBGT target is defined, so
    perfect ingredients give the perfect answer.
  - It also outputs **Tmax**: per cell, the temperature at the hottest hour plus a gap that can't
    be negative (the day's maximum is never below any hour's), averaged over the cells. So Tmax
    and WBGT are always physically consistent, as the 2026-10-05 condition requires.
  - **Trained on:** WBGT and Tmax errors, equally, plus a small penalty (weight 0.1) for
    ingredients that differ from the real ones, so the ingredients stay meaningful (e.g. "this
    heat stress is humidity-driven").
  - **Why exact, not a learned imitation:** the imitation was 0.40 °C off on hot-day peaks, more
    than any gain we are chasing.
  - **Stays unless clearly worse:** kept unless, on all days, it is more than 0.05 °C worse than
    separate direct models (for WBGT or for Tmax). Otherwise: two separate models, disclosed.
- **Retrieval:** whichever retrieval step earns its place. So far G3 selected none; Rg (regional
  matching) helped WBGT on all days and R2 (varied analogues) passed its screen; G-R* tests them on
  the chosen backbone.

The backbone is decided by G-D3 at the end of Week 5 ("BB\*"): the graph, only if it is not
worse than both the plain LSTM and U1 (the LSTM given the same upstream data) by more than
0.05 °C; otherwise the plain LSTM, or U1 if the data, not the graph, is what helps.

**DSTGNN skeleton (built 2026-10-06; `models/dstgnn.py`, `pipeline/graph.py`):**
- **Nodes:** Delhi (node 0) plus the 27 upstream points.
- **Edges:** each day a node sends information only to its **neighbours** (points within
  370 km: the 8 surrounding points, for Delhi too), and only to those
  **downwind** of it. The edge is stronger when the wind is stronger and the nodes are
  closer, and it uses the previous day's wind, the wind that actually carried the air.
  Optionally, the network also learns extra edges of its own, but only between the same
  neighbours, so it can't invent a 1,000 km jump in one day. (An independent review caught
  that the first version allowed exactly that.)
- **Through time:** a graph GRU steps through the 14 input days, and each day a node takes
  in its neighbours' state from the day before. So information moves one node (about
  220 km) per day, about the speed of a typical pre-monsoon north-westerly wind.
- **The forecast** is read from Delhi's final state. Upstream data can reach Delhi *only*
  along the edges, which is what lets us measure what the graph adds.
- **Gate G-D1 passed** (17 tests; synthetic inputs on the real 28-node graph):
  - output shapes are right and gradients reach every part of the model;
  - day t never sees later days;
  - no edges means upstream data has no effect;
  - information moves exactly one node per day. On the real graph, heat at the farthest
    point (24 N 68 E, 5 hops away) reaches Delhi only if it happened at least 5 days
    earlier;
  - edges point downwind, use the previous day's wind and stay between neighbours;
  - reordering the upstream nodes changes nothing.
- **Gate G-D0 passed (2026-10-07): upstream heat does help.** The upstream download is
  complete and verified: 27 points × 47 years, every day present, no missing values.
  - **The check:** a simple linear forecast using Delhi's own recent heat *plus* the 27 points'
    heat on the last 3 days, compared with damped persistence on 2007–2018, fitted fold by fold.
  - **The result:** it cut Delhi's Tmax error over leads 1–3 from 2.009 to 1.898 °C (−0.11 °C,
    95% CI −0.14 to −0.08). That's bigger than any LSTM improvement so far. On heatwave days
    the gain was −0.23 °C.
  - **Where the signal comes from:** most from the west and south-west (Thar desert, Kutch,
    500–1000 km away), least from points next to Delhi. Map:
    `evaluation_v2/figures/gd0_upstream_map.png`.
  - **How it compares** (Tmax, 2007–2018, all 5 leads). The linear upstream forecast (2.104 °C)
    beats every Delhi-only LSTM on all days, our control included (2.175; 0.07 °C better, a
    real difference). On heatwave days, the LSTMs trained with extra weight on hot days win by far
    (control 2.153 vs linear 2.908). The same holds for WBGT. Nothing yet does both; that is the
    graph network's job. It hasn't been trained yet: the G-D0 forecast is a linear formula, not
    the graph.
  - **What it doesn't show:** a delay that grows with distance. Almost every point helps most
    with yesterday's value, which looks like the large-scale heat pattern rather than air seen
    moving point to point. Whether the graph structure itself adds anything is tested in G-D3,
    against an LSTM given the same upstream data.
- **G-D3 result (2026-10-08): U1 for now, and the edges hurt.** All 10 graph runs are done
  (`evaluation_v2/graph_gates.md`). Error on all days, °C (lower is better):

  | Model | Tmax | WBGT |
  |---|---|---|
  | Plain LSTM (control) | 2.175 | 2.373 |
  | U1 (LSTM + flattened upstream data) | 2.199 | 2.270 |
  | C2 (graph code, **no edges**: the forecast reads every upstream point directly) | **2.051** | **2.254** |
  | C3 (fixed geographic edges) | 2.147 | 2.346 |
  | C4 / C4a (wind edges / + learned edges) | 2.142 / 2.138 | 2.338 / 2.334 |

  - Every graph model trains cleanly (G-D2) and beats the plain LSTM.
  - The chosen graph, C3, is not worse than U1 for Tmax but is 0.08 °C worse for WBGT, past the
    0.05 margin, so the rule we agreed says **U1**.
  - Adding edges makes the model about 0.09 °C *worse* than C2 in both families. Our best guess:
    with edges, the forecast reads only Delhi's node, so upstream information has to squeeze
    through Delhi's neighbours one hop a day and gets diluted at every hop ("over-squashing").
    C2 avoids this by reading every upstream point directly.
- **The tuning round (decided 2026-10-08, running now).** As agreed, one fair tuning round
  runs before any fallback. Tuning is picked only on each fold's last 2 training years, never the
  2007–2018 years we report:
  - tuned **C3** (the agreed candidate) and tuned **U1**, so the comparison stays fair;
  - one extra arm, **C3-pool**: C3's edges plus C2's direct view of the upstream points. It was
    added *after* seeing the result, which is disclosed; the locked 2019+ test years would confirm
    it if it is chosen.
  - Order of choice: tuned C3 if it passes, else tuned C3-pool, else the agreed fallback.
  - Check: the first tuning combination equals the original C3 and reproduced its result exactly.

**Retrieval ladder (R0 → R4):**
- **R0** is the RA-v1 design.
- **R0-rand** is the same model fed random past windows: the "is retrieval itself
  helping?" control.
- **R1** adds time alignment, **R2** diversity, **R3** a drift gate, and **R4** a
  correction for analogues from an older, cooler climate.

The best step is then attached to the final model.

**Retrieval on the folds (built 2026-10-06; result 2026-10-07, below):**
- **Fold-aware:** analogues are searched only among each fold's own training windows, with
  features and normalisation from those years (`retrieval/fold_retrieval.py`). v1's
  retrieval used 1980–2015 statistics, which would have leaked folds f1–f3's validation
  years.
- **Checked against v1:** on fold f4 with v1 labels, it returns *exactly* v1's analogues
  for all 1,092 validation days.
  - It also exposed a small v1 limitation: for 881 training days in 1980–88, v1 found
    fewer than 5 analogues. v1 screened only the 500 most similar windows, so early windows
    with few eligible candidates came up short. The new code checks all candidates.
- **Runs:** R0, R0-rand and R1 on each control (`A1prime_hw5`, `A2Lr_hw5`), 10 seeds × 4
  folds, plus the non-neural analogue ensemble (AnEn) as a reference.
- **Model:** `models/retrieval_lstm_v2.py`, the RA-v1 design. When a day has no past
  analogue at all, it now adds nothing; RA-v1 added a small learned offset instead.
- **Rule fixed before the results** (gate G3, `context/decisions.md`): a rung counts only
  if all three hold:
  1. it is not worse than its control on all days;
  2. it is significantly better on extreme days;
  3. it is **significantly** better than its own random control on extreme days.
- **Tightened after an independent review, before any result was seen:**
  - condition 3 must be significant, not a 0.001 °C difference;
  - R1 is compared with **R1-rand** (random past windows from the same time of year), so
    R1 can't win just by matching the season. R1-rand is two extra runs, added after the
    current queue;
  - the "forecasting warmer" check is reported but is not a pass/fail condition, because
    no threshold for it was fixed in advance.

**G3 result (2026-10-07): retrieval does not help yet. No rung passes, for Tmax or WBGT.**

"Δ" is the change in extreme-day error vs the plain model (°C); negative = better. The range
in brackets is the 95% interval; if it crosses 0, the change is not significant.

| | R0 vs plain model | R1 vs plain model | R0 vs its random version | R1 vs its random version |
|---|---|---|---|---|
| Tmax | +0.07 [−0.05, +0.19] | +0.02 | +0.07 [−0.10, +0.24] | +0.02 [−0.16, +0.19] |
| WBGT | −0.06 [−0.14, +0.01] | −0.07 [−0.13, +0.00] | −0.02 [−0.11, +0.07] | −0.03 [−0.09, +0.03] |

- **Tmax:** retrieval adds nothing; it is slightly worse on extreme days.
- **WBGT:** every retrieval model, *including the random ones* (−0.04), is a little better on
  extreme days, but not significantly. The real rungs are no better than their random
  versions, so the gain does not come from the past days being looked up. It comes from the
  larger model, and partly from forecasting warmer: on WBGT the retrieval models forecast
  about 75 hot days per seed instead of 54, with a larger warm bias.
- **All days:** every retrieval model is about 0.015 °C worse. For WBGT R1 this is just
  significant (p = 0.045), so R1 also fails condition 1.
- **How small a gain could we have seen?** About 0.24 °C on Tmax and 0.08–0.12 °C on WBGT
  (extreme days, 80% power). A smaller benefit is not ruled out.
- **The analogue ensemble** (average what followed the most similar past windows, no network)
  is far worse than the plain model (+1.1 °C Tmax, +1.5 °C WBGT on extreme days). So the most
  similar past windows say little about the next 5 days on their own.
- **Checked by an independent reviewer:** every number reproduced to 4 decimals from the
  saved forecasts; R1-rand verified as a fair control; no leakage found. Its two reporting
  fixes (p values shown, extra caveats) are applied.
- **What happens next:** nothing is attached to the final model yet. Week 4 tries the
  remaining rungs (R2 varied analogues, R3 drift gate, R4 correction for older, cooler
  analogues). Each is judged by the same rule against its own random version. Mechanism checks
  (analogue age, redundancy, season mismatch) will show *why* retrieval does or does not
  help. A well-explained "retrieval adds no information here" is still a publishable result
  (plan v5, risk 5).
- Full tables: `evaluation_v2/week3_controls.md`; decision record: `context/decisions.md`.

**Can tuning fix it? Checked 2026-10-07: no, the information isn't there.**
- **The check:** without training anything, does knowing what happened after the retrieved past days improve a
  simple forecast that already uses the query's own 14 days? It was scored on each fold's last 2 training years,
  so no validation data was touched, and the reading rule was written down first.
- **Result:** the improvement is about 0.000 °C for both rungs, both families.
- **Why:** the retrieved days' futures *do* track the real outcome (correlation 0.69 for Tmax on day 1), but
  only because those days were picked for looking like the query's recent weather. What followed them is
  what the query's own recent weather already predicts. The retrieval repeats information the model already
  has; random past days carry none at all.
- **So:** tuning how analogues are fed into the network (K, input form, attention) would not help. Retrieval can
  only add something if it matches on information the query *lacks*, such as the regional upstream heat pattern
  that G-D0 showed is useful, or if R2–R4 change what is retrieved.
- Report: `evaluation_v2/retrieval_information_check.md`.

**Rg: matching on the regional pattern (added 2026-10-07, after G3; disclosed as such)**
- **What it is:** instead of past days that looked like Delhi, Rg retrieves past days whose *regional* pattern
  matched the last 3 days (heat at the 27 upstream points over north-west India and Pakistan, plus dew point for
  WBGT). The rule was written down first, and a training-years screen showed this information is new to the model.
- **WBGT result:** about 2% lower error on all days (−0.044 °C, p = 0.003), in all 4 validation blocks, and
  clearly better than random past days on all days (−0.059 °C, p < 0.001). This is the first retrieval variant
  that measurably helps.
- **Why G3 still says "none":** on extreme days Rg beat the plain model (−0.16 °C) but not random past days
  (−0.12 °C, p = 0.11; the test can only see about 0.21 °C). The rule is not loosened after the fact.
- **Careful reading of the extreme-day gain:** it comes entirely from forecasting *less far below* the true value
  on those days (bias −3.0 → −2.8 °C). Rg also forecasts many more hot days (147 vs 54 per seed), mostly false
  alarms.
- **Tmax:** no detectable change (−0.003 °C on all days). Possibly because the Tmax model gets the analogues'
  outcomes in raw °C, or because regional matches are often from a different season. Untested ideas, not findings.
- **What it means:** the model never sees upstream data, so Rg seems to pass regional information to a Delhi-only
  model indirectly. Whether it still helps a model that sees upstream data directly (the graph) is the G-R* test.
- Independently reviewed (no leakage; numbers reproduced). Report: `evaluation_v2/week3_controls.md`.

**R2–R4 on top of Rg (screened 2026-10-07, training years only, rules written first)**
- **R2, varied analogues (MMR):** picks similar past days that are also *different from each other*. It adds a
  little on top of Rg (about −0.01 °C for both Tmax and WBGT), and the 5 analogues are less alike
  (similarity 0.79 → 0.70). **Passes: will be trained** after the graph runs (its code would change the hash
  the graph runs depend on). Code added 2026-10-08; queued after the graph tuning round.
- **R3, drift gate:** when the recent weather series looks unstable, use only analogues from the last 15
  years. The gate does switch on (27% of days for Tmax, 12% for WBGT), but it adds nothing over Rg. Not trained.
- **R4, warming correction:** shifts old analogues' outcomes by the warming trend since then. Adds nothing over
  Rg. Not trained.
- Report: `evaluation_v2/retrieval_ladder_screen.md`.

**Why retrieval behaves as it does (mechanism metrics, 2026-10-07; descriptive)**
- **The network treats its 5 analogues as equal.** Its attention is almost perfectly even in every rung, and
  it doesn't favour the most similar analogue. It effectively averages them.
- **R0/R1 pick near-duplicates** (mean similarity 0.88 to each other), so the average adds little beyond one of
  them. R2's diversity targets exactly this.
- **Rg's analogues are more varied but often off-season** (about 2 months apart on average; only 35–41% within a
  month of the same time of year). This may be why Rg did not help Tmax.
- **Rg's analogue outcomes track the real outcome best on ordinary days**, but on WBGT extreme days they don't
  track it at all. Its extreme-day gain came from a smaller cold bias, not from foreseeing the extremes.
- Analogues are 14–17 years old on average in every rung, so correcting for warming (R4) had little to fix.
- Report: `evaluation_v2/retrieval_mechanisms.md`.

### 9.1 How RAG helps the DSTGNN, and how we prove it (design; results pending)

**What each part contributes:**

| Part | What it sees | What it contributes |
|---|---|---|
| **DSTGNN** (graph model) | The last 14 days of weather at Delhi and 27 upstream points (Rajasthan, Thar, Punjab, Pakistan) | **What is happening now, and where heat is coming from.** Graph edges follow the wind, so it can see hot air 1–3 days before it reaches Delhi. |
| **RAG** (analogue retrieval) | Every past 14-day period since 1980 (training years only) | **What happened the last times things looked like this:** the most similar past situations and their *real* next 5 days. |

**Why combine them:** neural networks learn from averages, so they under-forecast rare
extremes, which is exactly where they've seen few examples. RAG supplies evidence ("the 5 most
similar past situations went on to 44–46 °C"), and the DSTGNN decides how much to trust it
(attention), given what it sees upstream.

**How they connect (model "G-R\*"):**
1. The DSTGNN encodes the current regional situation.
2. The retriever returns K similar past situations. It only searches the past, never the same
   heatwave, and never validation or test years.
3. Attention combines their real outcomes with the DSTGNN's encoding.
4. The output head forecasts the 5 days.

**How we know RAG is really helping:** "lower error with RAG" is not enough. Each
alternative explanation gets its own control:

| Alternative explanation | How we rule it out |
|---|---|
| "It's just a bigger model" | **Random-retrieval control (R0-rand):** the identical model fed *random* past days. Real retrieval must beat random retrieval, which proves the gain comes from the analogues' *information*, not the extra machinery. **This is the key test.** |
| "Luck with seeds or years" | **10 seeds × 4 rolling folds** (12 validation years) and the cluster-jackknife test. "Helps" means the 95% confidence interval excludes zero. |
| "It only looks good on heatwave days by forecasting hot all the time" | It must **not get worse over all days** (the damped-persistence floor), and it is also scored on the days it *forecasts* a heatwave (forecaster's dilemma). |

**Two designs that make the case:**
- **2×2 experiment:** {plain LSTM, DSTGNN} × {without RAG, with RAG}. It shows whether RAG helps
  either backbone, and whether the graph and RAG provide *different* or *overlapping* information.
  If the upstream graph already sees the heatwave coming, RAG's gain may shrink on the DSTGNN, and
  that is a finding too.
- **Mechanism evidence:**
  - attention should concentrate on useful analogues, not spread evenly;
  - gains should be larger when the analogues' futures resemble what really happened;
  - a Tmax-vs-WBGT contrast tests whether old analogues go stale as humid heat rises.

**Fairness rules:**
- "With RAG" and "without RAG" differ only in retrieval: same data, folds, seeds and training recipe.
- Pre-registered as primary hypothesis **H-A**, and checked once on the locked 2019+ test years.
- Every result is reported, including a null result.

**Where we stand (2026-10-05):**
- The analogues carry information: averaging them alone beats climatology on hot days, mostly
  at leads 1–2.
- The first RAG model (RA-v1) did **not** clearly beat the plain LSTM: −0.09 °C on extreme
  days, 95% range −0.58 to +0.41.
- Its attention was nearly uniform. The retrieval ladder (R1–R4) is meant to fix that.

**One-line answer:** we compare the identical DSTGNN with real retrieval, with no retrieval
and with random retrieval, over 10 seeds and 12 validation years with confidence intervals.
RAG counts as helping only if real retrieval beats both.

*To be added when G-R\* is built:* exactly which spatio-temporal parts of the graph model RAG
helps (see the change log).

**Advisory (Phase 6):**
- forecast → rule engine → risk tier (GREEN / YELLOW / ORANGE / RED) → LLM writes the
  advisory;
- every recommended action must come from a verified library of **word-for-word quotes
  from official documents**;
- **health and energy actions** (energy included by team decision 2026-10-06, since official
  energy sources were found: CEA, BEE and Delhi power-company notices);
- which alert colour unlocks which action is being sourced by Cowork (Part E,
  `docs/PHASE6_PART_E_BRIEF.md`); only 12 of 135 actions have a colour from the sources so far;
- **"If this happens"** (decision 2026-10-06): actions triggered by something that has already
  happened (a heat-stroke patient, heat deaths) are kept in their own clearly labelled section,
  never mixed with the forecast actions. Clinical steps sit behind a "for health professionals"
  disclaimer;
- **front end** (decision 2026-10-06): one interactive web page (Week 8) with the 5-day chart,
  the alert colours, a click-a-day advisory and the "If this happens" section. Nothing to install;
  it could be upgraded to React later.

**What IMD says about humid heat and warning lead time** (Cowork research Part D, checked
against the saved documents; details in `sources/PHASE6_PART_D_RESULT.md`):
- **IMD's own humid-heat term** is "Hot & Humid Weather": a station's maximum temperature 3 °C
  above normal *together with* above-normal relative humidity (IMD heat-wave FAQ, p.2). It has
  no humidity value and no colour category. IMD launched an *experimental* heat index in 2023
  (PIB, 26 Jul 2023) but has published no thresholds for it.
  → Our humid-heat note must say it comes from **our** physical WBGT and never present a WBGT
  or heat-index threshold as IMD's.
- **Lead time:** since July 2023, IMD issues colour-coded warnings, including heat-wave
  warnings, daily **for the next seven days**. This replaced five days (IMD FAQ), and before
  that four days for the 2017 heat-warning system (NDMA 2019, NCDC 2025). Our 5-day forecast is
  inside IMD's horizon; it adds detail, it does not go beyond it.
- **No documented Delhi humid-heat events before 2019** were found in IMD, government or
  peer-reviewed sources. So the WBGT label cannot be checked against official events; this is a
  stated limitation. (A 2022 paper by IMD authors notes Delhi's heat stress is lower than
  Chennai's because Delhi is less humid.)

---

## 10. Ground rules (why the results can be trusted)

1. **Test lock:** nobody evaluates on 2019+ until Week 7. Even the tests in the code never
   read 2019+ data.
2. **Frozen v1:** data, window index and v1 models are fingerprinted
   (`data/MANIFEST.json`, git tags `v1-frozen`, `v1-frozen-ra`).
   `scripts/make_manifest.py --check` must say "Manifest OK".
3. **One change per run:** every run differs from its parent in one thing, so any gain
   has one cause.
4. **Beat the simple baselines first:** every model must beat damped persistence over all
   days, not just on heatwave days.
5. **10 seeds and proper statistics** for every decision, with cluster-based tests and
   rolling folds.
6. **Every run is logged** in `registry/runs.csv`: config, data and code fingerprints, git
   commit, seeds, threads, and skill against climatology and persistence on the same
   target.
7. **Automatic checks on every push (CI)** (`.github/workflows/ci.yml`). GitHub runs:
   - the full test suite and the frozen-v1 check, on Windows with the exact pinned
     package versions;
   - a scan of the whole git history for leaked secrets (gitleaks);
   - a check of the pinned packages for known vulnerabilities (pip-audit).

   A red ✗ next to a commit on GitHub means something broke: fix it before training on
   that commit. CI only reads the repo; it never changes code, data or models.

---

## 11. Where things are and how to run them

| Folder | What's in it |
|---|---|
| `data/`, `datasets/`, `datasets_v2/` | Raw ERA5 (v1), daily tables (v1, v2) |
| `splits/` | Frozen window index |
| `pipeline/` | v2 data: heat-stress formulas, climatology, labels v2, v2 downloader |
| `retrieval/` | Feature vectors, FAISS index, analogue search |
| `models/` | LSTM, retrieval LSTM; `frozen/` holds the v1 checkpoints |
| `training/` | v1 training scripts (unchanged), `folds.py`, `train_unified.py` (v2) |
| `evaluation/`, `evaluation_v2/` | Statistics, baselines, comparison reports |
| `configs/` | One JSON file per v2 run |
| `tests/` | 145 automated checks (`.venv\Scripts\python.exe -m pytest -q`) |
| `context/` | Running notes: progress, decisions, run commands |
| `docs/` | This guide, the glossary, the plan, the reproducibility log |
| `.github/workflows/` | CI: automatic tests and security checks on every push |

Setup and commands: `context/environment.md` and `context/RUN_COMMANDS.md`. Always use
the project's virtual environment (`.venv`); nothing is installed globally.

---

## 12. Open items

- v2 download complete and verified; Liljegren WBGT built and checked (§8.2).
- ~~Decide whether the WBGT models should forecast the *physical* WBGT~~ Done (2026-10-05): yes.
- ~~Choose the G2 control models~~ Done (2026-10-06): A1prime_hw5 and A2Lr_hw5 (§8.5).
- Advisory: map the 123 actions with no alert level onto the colour tiers (team decision pending).
- Upstream download: run `pipeline/download_era5_upstream.py` (about 3–4 days of free API quota), then the G-D0 check (does upstream heat lead Delhi?).
- ~~Confirm the 4 acceptance heatwave dates~~ Done (2026-10-05): 3 have IMD regional or
  sub-division support and 1 (2002) is station-only; none is an IMD Delhi declaration (§8.3).
- Find official advisory sources (research brief Parts A–B, `docs/PHASE6_SOURCE_RESEARCH_BRIEF.md`).
- Station humidity cross-check for the humid-heat trend.

---

## Change log

| Date | Change |
|---|---|
| 2026-10-04 | First version: v1 results, Week-1 findings, v2 data/labels/folds/trainer (Week 2). |
| 2026-10-05 | Added CI (tests, frozen-v1 check, secret scan, dependency audit), ground rule 7. |
| 2026-10-05 | v2 download verified; upstream downloader (27 NW India / Pakistan points) added. |
| 2026-10-05 | Acceptance heatwaves sourced (Part C): windows updated, evidence level stated per event. |
| 2026-10-05 | §9.1: how RAG helps the DSTGNN and how we prove it (design, controls, current status). |
| 2026-10-05 | WBGT models switched to physical WBGT + WBGT label (with conditions); 15 pre-registered improvement runs started. |
| 2026-10-05 | Physical (Liljegren) WBGT built, crediting Liljegren/Argonne and Kong & Huber; BoM claim corrected (2–3 °C, not 6); WBGT label (97.5th pct); Week-3 G2 results. |
| 2026-10-06 | 15 improvement runs done; hot_weight is the lever; controls A1prime_hw5 / A2Lr_hw5 (disclosed deviation from the rule); §8.3 corrected: the label in use is physical WBGT, 95th pct; CI made robust to runner CPU differences. |
| 2026-10-06 | Part D (IMD humid-heat wording, 7-day lead time) in §9; fold-aware retrieval R0/R0-rand/R1 built and pre-registered (G3 rule), runs started; DSTGNN skeleton passes G-D1. |
| 2026-10-07 | R1-rand added and run; G3: no retrieval rung helps yet (Tmax or WBGT), independently reviewed; Week 4 continues with R2–R4 and mechanism checks. |
| 2026-10-07 | Retrieval information check: analogues add no information beyond the query's own inputs, so tuning the network's use of them won't help. |
| 2026-10-07 | Rg (regional-pattern retrieval) added after G3: WBGT −2% error on all days and beats random analogues; extreme-day condition missed (p = 0.11), so G3 still none; reviewed. |
| 2026-10-07 | Physics head chosen (exact formula, per cell at each cell's peak hour); graph runs C2–C4a + U1 started (G-D4 passed); R2–R4 screened: R2 passes, R3/R4 add nothing over Rg. |
| 2026-10-08 | Graph runs done: G-D3 says U1 (graph not worse than U1 for Tmax, 0.08 °C worse for WBGT); edges hurt, the edgeless C2 is best. One tuning round started (C3, U1, and the post-result arm C3-pool); then PH_lstm and R2. |
