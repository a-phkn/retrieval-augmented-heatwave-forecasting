# Glossary

Short forms and terms used in this repository, in `docs/PLAN_REVIEW_v5.md`, and in the
`context/` and `evaluation_v2/` reports. Grouped by topic.

## Versions, models and runs

Every run changes exactly **one** thing from its "parent" run, so any improvement can be
attributed to that one change.

| Term | Meaning |
|---|---|
| **v1 / v2** | v1 = the project as originally built (Tmax target, original heatwave labels, original data); frozen and tagged `v1-frozen` / `v1-frozen-ra`. v2 = the new version (WBGT target, new labels, extra variables). |
| **A0** | Reference forecasts with no learning or minimal fitting: persistence, climatology, damped persistence. |
| **A1** | The original baseline LSTM (predicts Tmax, weighted loss with hot_weight = 20). Frozen in `models/frozen/lstm_tmax_v1/`. |
| **A1′** | The same LSTM recipe retrained with the **new heatwave labels** (only the labels change). |
| **A2** | The same LSTM recipe retrained to predict **WBGT** instead of Tmax. The control for all v2 models. |
| **A2r** | A2 predicting the **anomaly** (departure from the seasonal normal) instead of the raw value. |
| **RA-v1 (R0)** | The original retrieval-augmented LSTM (LSTM + attention over 5 retrieved past analogues). Frozen in `models/frozen/ra_lstm_v1/`. |
| **R1–R4** | The "retrieval ladder", one improvement per step: R1 calendar (time) alignment, R2 diversity (MMR), R3 drift gate, R4 climate-shift correction of old analogues. |
| **C1–C4** | DSTGNN controls: C1 plain LSTM; C2 one shared LSTM per grid node, no graph; C3 fixed geographic graph; C4 full DSTGNN. C4 vs C2 = what the graph itself adds. |
| **G-R\*** | The best retrieval step (from R1–R4) added to the DSTGNN. |
| **BB\*** | "Best backbone": the architecture finally chosen (DSTGNN, or the physics-guided LSTM if the DSTGNN fails its gates). |
| **H1** | Optional extra classification head that predicts the yes/no heatwave flag directly. |
| **Physics head (2a)** | Model predicts temperature and humidity, then computes WBGT/Heat Index with the physical formulas inside the model. |

## Baselines

| Term | Meaning |
|---|---|
| **Persistence** | "Each of the next 5 days = today's maximum temperature." |
| **Climatology** | "Each day = the long-term (train-years) average for that calendar date." |
| **Damped persistence** | "Normal for that date + a shrinking fraction of today's departure from normal" (fractions 0.81, 0.64, 0.51, 0.43, 0.37 for days 1–5, fitted on training years). Currently the strongest model on val. |
| **AnEn** (analogue ensemble) | Average of what happened after the 5 most similar past situations (the same 5 the retrieval model uses). No neural network. |
| **AnEn-random** | Same, but with 5 random past situations. A control. |

## Plan checkpoints

| Term | Meaning |
|---|---|
| **G0–G6** | Weekly gates: conditions that must hold before moving on. G0 (Week 1) = honest LSTM-vs-retrieval comparison exists; G5 = test-set lock. |
| **G-D0–G-D4** | DSTGNN feasibility gates: spatial signal exists (D0), code correct (D1), trains stably (D2), not worse than the LSTM (D3), affordable compute (D4). |
| **Test lock** | The 2019–2026 test data is used exactly once, at the end, after all choices are written down in `docs/PREREGISTRATION.md`. Prevents unconscious tuning to the test set. |
| **Credibility floor** | A minimum standard a model must meet (e.g. not worse than climatology / damped persistence overall) before its heatwave-day gains count. |

## Statistics and evaluation

| Term | Meaning |
|---|---|
| **RMSE** | Root-mean-square error: typical forecast error in deg C, big misses weighted more. Lower is better. |
| **MAE** | Mean absolute error: average size of the error in deg C. |
| **Stratum (normal / unusual / extreme)** | Forecast days grouped by how hot they actually were; "extreme-stratum RMSE" = error on heatwave days. |
| **Forecast-conditioned stratum** | Days the model *predicted* to be extreme (the complement to judging only days that turned out extreme). |
| **Seed** | The random starting point of training. Same model, different seed -> slightly different results, so models are trained with several seeds and averaged. |
| **Cluster (G)** | A block of days treated as one unit in the statistics: one season (Jan–Mar 14, Mar 15–Jul 31, Aug–Dec) of one year. G = number of blocks. Overlapping forecasts are not independent, so blocks are counted, not days. |
| **CI** | Confidence interval: the range in which the true difference plausibly lies. If it includes 0 the result is inconclusive. |
| **p-value** | How surprising the result would be if there were truly no difference; below 0.05 is called significant. |
| **FPR** | False-positive rate: how often a test reports "significant" when there is no real difference (should be about 5%). |
| **Power** | Probability that a test detects a real improvement of a given size. |
| **MDE** | Minimum detectable effect: the smallest improvement that can be detected with 80% power given our data. |
| **Paired cluster (jackknife) t-test** | The main significance test (`evaluation/stats.py`): compares two models on the same days, recomputing the result with one block left out at a time to measure uncertainty honestly. |
| **DM test** | Diebold–Mariano test of equal forecast accuracy; used as a secondary check. |
| **Rolling-origin folds / OOF** | Train on all years before a block, evaluate on the block (2007–09, 2010–12, 2013–15, 2016–18). OOF = out-of-fold predictions; gives about 12 years of honest evaluation instead of 3. |
| **Forecaster's dilemma** | Judging only on days that turned out extreme rewards models that always predict hot (Lerch et al. 2017). |
| **Retention (future-similarity)** | Whether a more similar past window also has a future closer to the truth (from the SARAF paper). |
| **Seed-noise floor** | The part of the MDE caused by training randomness alone; reduced by more seeds, not more data blocks. |

## Heat measures

| Term | Meaning |
|---|---|
| **Tmax** | Daily maximum air temperature. |
| **WBGT** | Wet-Bulb Globe Temperature: heat-stress index combining temperature, humidity, sun and wind. |
| **BoM WBGT approximation (Tier-A)** | Formula from Australia's Bureau of Meteorology using only temperature and humidity; a temperature–humidity index that runs about 6 deg C above a shade WBGT in Delhi's humid season. Columns `wbgt_bom_*`. |
| **Liljegren WBGT (Tier-B)** | Physics-based WBGT that uses radiation and wind (needs the v2 download). |
| **Tw** | Wet-bulb temperature (Stull 2011 formula): lowest temperature reachable by evaporative cooling; a clean humid-heat measure. |
| **HI** | Heat Index ("feels-like" temperature, US National Weather Service formula). |
| **Anomaly / standardised anomaly (z)** | Departure from the seasonal normal (deg C) / that departure divided by the normal day-to-day spread. |
| **Labels v1 / v2** | Heatwave-day definitions. v1 = year-round relative anomaly (original). v2 = Mar 15–Jul 31 season, Tmax >= 40 C and anomaly >= 3 C (IMD 4.5 C as "severe"). |

## Data, retrieval and architecture

| Term | Meaning |
|---|---|
| **ERA5 / ERA5T** | ECMWF global weather reanalysis (the data source, via Open-Meteo) / its preliminary recent months, which can be revised. |
| **ERA5-Land** | Finer (0.1 deg) land version of ERA5; dropped from the plan because it adds little independent signal for Delhi. |
| **N9 / N42** | 9 grid cells (current data, 0.25 deg apart) / 42 finer cells (dropped). |
| **Regional upstream graph** | Plan v5 DSTGNN design: nodes across NW India/Pakistan, where Delhi's hot air comes from 1–3 days earlier. |
| **DSTGNN** | Dynamic spatio-temporal graph neural network: passes information between map locations. |
| **FAISS** | Library used for fast similarity search over past windows. |
| **Analogue** | A past 14-day weather window similar to the current one; its following 5 days are the "analogue outcome". |
| **SARAF** | Stationarity-aware retrieval-augmented forecasting (Zhou et al. 2026); source of the time-alignment, diversity and stationarity ideas. |
| **MMR** | Maximal marginal relevance: choosing analogues that are relevant and not near-duplicates of each other. |
| **ADF / KPSS** | Statistical tests for whether a series is stationary (stable over time). |
| **Sen slope / Mann–Kendall** | Robust trend estimate (per decade) and its significance test. |
| **M1 / M2** | Retrieval database modes: M1 strict (only train/val years, primary); M2 walk-forward (any past day, operational only). |
| **Manifest** | `data/MANIFEST.json`: fingerprints (hashes) of every frozen file; `tests/test_manifest.py` fails if one changes. |

## Remote sensing and Phase 6

| Term | Meaning |
|---|---|
| **NDVI / NDBI** | Satellite vegetation index / built-up-area index (from Landsat). |
| **LST** | Land surface temperature (from MODIS satellites). |
| **HVI** | Heat-vulnerability index combining the above (optional ward map for advisories). |
| **GEE** | Google Earth Engine, the platform for those satellite data. |
| **UHI** | Urban heat island. |
| **IMD / NDMA / HAP** | India Meteorological Department / National Disaster Management Authority / Heat Action Plan. |
| **DERC / DISCOM / BEE** | Delhi Electricity Regulatory Commission / electricity distribution company / Bureau of Energy Efficiency. |
| **Rule engine / verifier** | Phase 6 safety layers: the rule engine sets the risk tier deterministically; the verifier rejects any advisory action without an approved source. |
