# Project Guide: Retrieval-Augmented Heatwave Forecasting for Delhi

> **Living document.** Read this first. It explains the whole project in plain language:
> what we build, how every part works, what the results are so far, and what comes next.
> It is updated whenever a result, a design decision or the plan changes. See the change
> log at the end. Short forms are explained in [`GLOSSARY.md`](GLOSSARY.md); the full
> plan is in [`PLAN_REVIEW_v5.md`](PLAN_REVIEW_v5.md).
>
> **Last updated:** 2026-10-04 (end of Week 2)

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
| v2 variables (download in progress) | Dew point, wind direction, solar radiation (total, direct, diffuse), cloud cover and soil moisture. See §8.2 for why. |
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
  humidity, and reads about 6 °C higher than a shade WBGT in July. So we never apply
  official absolute WBGT thresholds to it.
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
| **Liljegren (physical) WBGT** | Temperature, humidity, **solar radiation**, **wind**, pressure | **v2 download** (radiation, dew point, ...) | ⏳ After the v2 download finishes (Weeks 3–4). Used to **check and calibrate** the BoM index, and possibly as the final target. |

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
- **Acceptance test:** the labels must flag known pre-2019 heatwaves (May–Jun 1998, May
  2002, Apr 2010, May 2015). The dates are being checked against IMD records.

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
  model falls back to the normal when it has no signal. First 1-seed check on f4 (not a
  result yet): A2r beat WBGT climatology by 26%.

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

**Final model options:**
- **Regional DSTGNN** (graph neural network): the nodes are ~20–30 points over
  north-west India and Pakistan, plus Delhi. Delhi's heatwaves are often hot dry air
  blown in from the north-west over 1–3 days, so a graph that follows the wind can see
  them coming.
- **Physics head:** the model predicts temperature and humidity, then computes WBGT with
  the physical formula inside the model, so the outputs are always physically consistent.

If the DSTGNN doesn't beat a plain LSTM given the same upstream data, we fall back to the
physics-guided LSTM.

**Retrieval ladder (R0 → R4):**
- **R0** is the RA-v1 design.
- **R0-rand** is the same model fed random past windows: the "is retrieval itself
  helping?" control.
- **R1** adds time alignment, **R2** diversity, **R3** a drift gate, and **R4** a
  correction for analogues from an older, cooler climate.

The best step is then attached to the final model.

**Advisory (Phase 6):**
- forecast → rule engine → risk tier (GREEN / YELLOW / ORANGE / RED) → LLM writes the
  advisory;
- every recommended action must come from a verified library of **word-for-word quotes
  from official documents**;
- **health actions only**, unless official energy-grid sources are found.

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

Setup and commands: `context/environment.md` and `context/RUN_COMMANDS.md`. Always use
the project's virtual environment (`.venv`); nothing is installed globally.

---

## 12. Open items

- v2 download (radiation etc.) running; then build Liljegren WBGT and check BoM against it.
- Confirm the 4 acceptance heatwave dates against IMD records (research brief Part C).
- Find official advisory sources (research brief Parts A–B, `docs/PHASE6_SOURCE_RESEARCH_BRIEF.md`).
- Station humidity cross-check for the humid-heat trend.

---

## Change log

| Date | Change |
|---|---|
| 2026-10-04 | First version: v1 results, Week-1 findings, v2 data/labels/folds/trainer (Week 2). |
