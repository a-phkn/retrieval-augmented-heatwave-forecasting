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
| **A1′** | The same LSTM recipe retrained with the **new heatwave labels** on the rolling folds (also uses the new early-stopping rule, so it differs from A1 in labels, early stopping and 2 fewer fitting years). |
| **A2** | The same LSTM recipe retrained to predict the **BoM WBGT index** instead of Tmax. Kept as a historical result since the switch to physical WBGT (2026-10-05). |
| **A2r** | A2 predicting the **anomaly** (departure from the seasonal normal) instead of the raw value. Historical, like A2. |
| **A2L_t / A2L / A2Lr** | The physical-WBGT chain ("L" = Liljegren), one change per step from A1′: A2L_t switches the target to physical WBGT (still the Tmax label); A2L switches to the WBGT label; A2Lr predicts the anomaly. |
| **A1prime** | How A1′ is spelled in file and config names (`configs/A1prime.json`). |
| **Run-name suffixes** | `_r` = anomaly target; `_dp` = dp_residual target; `_hw1` / `_hw5` / `_hw10` = hot_weight 1 / 5 / 10 (no suffix = 20). Example: `A2Lr_hw5` = A2Lr with hot_weight 5. |
| **A1_repro** | Config that re-runs v1's A1 recipe in the new trainer; it must reproduce the frozen A1 bit-for-bit. |
| **Control (G2 control)** | The plain-LSTM model each new idea (retrieval, DSTGNN) is compared against. Chosen 2026-10-06: **`A1prime_hw5`** (Tmax) and **`A2Lr_hw5`** (WBGT), a disclosed deviation from the pre-registered rule (`context/decisions.md`). |
| **LSTM** | Long short-term memory network: a neural network that reads a sequence (here the last 14 days) and remembers what matters. Our base model. |
| **RAG** | Retrieval-augmented generation (here: forecasting): the model looks up similar past situations (analogues) and uses what happened after them. |
| **GNN** | Graph neural network: a network whose inputs are points on a map (nodes) connected by links (edges) along which information is passed. |
| **RA-v1 (R0)** | The original retrieval-augmented LSTM (LSTM + attention over 5 retrieved past analogues). Frozen in `models/frozen/ra_lstm_v1/`. |
| **R1–R4** | The "retrieval ladder", one improvement per step: R1 calendar (time) alignment, R2 diversity (MMR), R3 drift gate, R4 climate-shift correction of old analogues. |
| **R0 / R0-rand / R1 run names** | `A1prime_hw5_R0`, `A1prime_hw5_R0rand`, `A1prime_hw5_R1` (and the same for `A2Lr_hw5`): the control plus retrieval. R0 = most similar analogues; R0-rand = random eligible analogues (the control for retrieval itself); R1 = R0 limited to analogues within ±30 days of the same time of year. |
| **Fold-aware retrieval** | `retrieval/fold_retrieval.py`: analogues searched only among a fold's own training windows, with features and normalisation from its training years, so validation years never leak into retrieval. |
| **C1–C4** | DSTGNN controls: C1 plain LSTM (Delhi only); C2 the same recurrent network on every node but no edges (Delhi's state plus the average upstream state); C3 fixed geographic edges; C4 full DSTGNN (daily wind-gated edges, optionally plus adaptive ones). The original plan read C4 vs C2 as "what the graph itself adds"; that rule was superseded when C3-hop became the final model, and C2 vs C3-hop showed no graph-structure gain. |
| **G-R\*** | The test of whether the best retrieval step (Rg and/or R2–R4) still helps once it is added to the chosen backbone (the graph, or U1). |
| **U1** | The flattened-upstream control: the plain LSTM given the same 27 upstream points' daily data as the graph, laid side by side in one input row. If the graph is not better than U1, the gain comes from the data, not the graph. |
| **C4a** | C4 plus learned "adaptive" connections between neighbouring points (reported next to C4). |
| **C3-pool** | C3's fixed edges plus C2's way of reading the forecast: Delhi's state *and* the average of all upstream points' states. Added 2026-10-08 after the G-D3 result (disclosed) to test whether the edges only hurt because of how the forecast was read out. |
| **Over-squashing** | A known weakness of graph networks (Alon & Yahav, ICLR 2021): information from far-away nodes must pass through a few neighbours and gets squeezed and diluted on the way. Our first guess for why adding edges made the graph worse; the real cause turned out to be C3's "last day never arrives" flaw (see C3-hop). |
| **C3-hop** (final model) | Our graph network (`models/dstgnn_multihop.py`). Points are grouped into **rings** by exact distance from each node (ring 1 = neighbours … ring 4); every day each node hears from all 4 rings at once (one weight per ring), and at the end the latest day of every ring is passed into Delhi. Fixes C3's flaw (an upstream point's most recent day never reached the forecast) and keeps distance structure. Shortest-path message passing, Abboud et al. 2022. Chosen 2026-10-12; confirmed on the test 2026-10-19. |
| **Ring / hop** | Hop = one step between neighbouring points (~220 km); ring k = all points exactly k hops away. All 27 upstream points are within 4 hops of Delhi. |
| **Tuning round** | One agreed search over 8 settings (size, learning rate, dropout) per model, chosen only on each fold's last 2 training years, never on the reported years; run once before any fallback, never repeated. |
| **Ridge baseline** | A plain linear regression (with a small penalty on large weights) from Delhi's and the upstream points' recent heat to the next 5 days. No neural network; the yardstick every model must beat to claim it learned something a linear formula can't. |
| **Forecaster's dilemma** | Scoring only on days that turned out extreme rewards a forecaster who always forecasts warmer (Lerch et al. 2017). That's why extreme-day results come with their bias split out. |
| **Recalibration** | Correcting a model's forecasts with a straight line (actual ≈ a + b × forecast) fitted on its own training years. If two models' gap vanishes after it, the gap was bias, not skill. |
| **Brier score** | The error of a probability forecast for a yes/no event (here: "will it be a hot day?"): the average of (forecast probability − outcome)². **Brier skill** compares it with always forecasting the usual frequency: above 0 = useful. |
| **Selection-free years** | 2007, 2010, 2013 and 2016–18: reported years never used for early stopping or tuning, so a cleaner check of the backbone tests. |
| **Dropout** | Randomly switching off part of the network during training so it can't lean on any one piece; a standard guard against overfitting. Off when forecasting. |
| **Non-inferiority margin** | How much worse a model may be and still count as "not worse": for G-D3, the upper end of the 95% range of its error difference must stay below +0.05 °C. |
| **BB\*** | "Best backbone": the architecture finally chosen. **C3-hop** (user decision 2026-10-12). |
| **H1–H4 (test hypotheses)** | The four claims tested once on 2019–2026 (`docs/CONFIRMATORY_PROTOCOL.md`): H1 C3-hop beats the Delhi-only LSTM on WBGT all days (confirmed); H2 … on WBGT heatwave days (not confirmed); H3 C3-hop beats ridge_hw5 on WBGT heatwave days (confirmed); H4 LSTM + Rg beats the Delhi-only LSTM on WBGT all days (not confirmed after Holm). |
| **H1 (old plan v5 meaning)** | An optional extra classification head predicting the yes/no heatwave flag directly. Not built; not the same as the test hypothesis H1. |
| **Physics head (2a)** | The model predicts the weather "ingredients" (temperature, humidity, wind, sunshine) and computes WBGT from them with the exact physical (Liljegren) formula inside the model. Chosen design (2026-10-07): exact formula (A), applied separately in each of Delhi's 9 cells at that cell's own hottest hour, then averaged (D3). Built and run 2026-10-12, then **dropped** by its pre-registered keep rule; the final models predict WBGT directly. |

## Baselines

| Term | Meaning |
|---|---|
| **Persistence** | "Each of the next 5 days = today's maximum temperature." |
| **Climatology** | "Each day = the long-term (train-years) average for that calendar date." |
| **Damped persistence** | "Normal for that date + a shrinking fraction of today's departure from normal" (fractions 0.81, 0.64, 0.51, 0.43, 0.37 for days 1–5, fitted on training years). The strongest simple baseline, and the credibility floor. |
| **φ (phi)** | The shrink fractions of damped persistence, one per lead day; refitted on each fold's training years. |
| **AnEn** (analogue ensemble) | Average of what happened after the 5 most similar past situations (the same 5 the retrieval model uses). No neural network. |
| **AnEn-random** | Same, but with 5 random past situations. A control. |

## Plan checkpoints

| Term | Meaning |
|---|---|
| **G0–G6** | Weekly gates: conditions that must hold before moving on. G0 (Week 1) = honest LSTM-vs-retrieval comparison exists; G5 = test-set lock. |
| **G-D0–G-D4** | DSTGNN feasibility gates: spatial signal exists (D0), code correct (D1), trains stably (D2), not worse than the LSTM (D3), affordable compute (D4). |
| **Test lock** | The 2019–2026 test data is used exactly once, at the end, after every choice is written down in `docs/CONFIRMATORY_PROTOCOL.md` and frozen as the git tag `test-lock`. `training/test_lock.py` refuses to read 2019+ data unless HEAD contains that tag and the code under training/ models/ pipeline/ retrieval/ evaluation/ configs/ is unchanged since it. Used once, 2026-10-19. Prevents unconscious tuning to the test set. |
| **Test fold** | The fold used for the one test run: fitted on 1980–2016, with 2017–2018 held back to decide when to stop training; forecasts 2019-01-01 to 2026-09-06. `test2021` = the same ending 2021, for the Rz side-check. Both are guarded by the test lock and kept out of the development fold list. |
| **Pre-registration** | Writing a rule (what is run, what counts as success) into `context/decisions.md` **before** the result exists, so the result can't shape the rule. |
| **Post-test exploration** | Any analysis run after the one test evaluation (e.g. the foundation-model comparison). Labelled as such; it cannot change the test's conclusions. Code lives in `post_test/`, outside the locked directories. |
| **Screen** | A cheap check on training years only, with a linear model and no network training, of whether an idea carries information beyond a baseline (e.g. the retrieval information check, the Rz screen, the heatwave-information screen S2). Only ideas that pass get a full training run. |
| **Credibility floor** | A minimum standard a model must meet before its heatwave-day gains count: **not significantly worse than damped persistence on all days** (decision 2026-10-04). |
| **G2** | The Week-3 gate: choose the control models. |
| **G3** | The gate that picks the best retrieval rung. Pre-registered rule (tightened before results): a rung counts only if it is not worse than its control on all days, significantly better on extreme days, and significantly better than its own random control (R0-rand for R0, R1-rand for R1) on extreme days. |
| **R1-rand** | Random eligible analogues from within ±30 days of the same time of year: R1's fair random control. In configs its mode is `time_rand`. |
| **Rg** | Regional-pattern retrieval: past days matched on the heat (and, for WBGT, humidity) pattern at the 27 upstream points over the last 3 days, not on Delhi alone. Its random control is R0-rand (same pool). Added after G3. |
| **Rz** | Upper-air retrieval: past days matched on the 500 hPa height map over South Asia (10–45°N, 40–100°E), the weather systems ~5.5 km up. Helped C3-hop on WBGT all days in development (−0.023 °C) but failed G3, and did not replicate on 2019–2021. |
| **Rh** | Humid-air retrieval: past days matched on humidity and temperature at 12 extra points east and south-west of Delhi. Screened only (S2); no gain. |
| **S2 (heatwave-information screen)** | Screen of four ways to bring new information to WBGT heatwave days: humid-air points as input, Rh retrieval, matching past heatwave starts, the upper-air pattern as input. None passed. |
| **Analog error correction (AnEn-corr)** | Correcting a forecast by the average error the model made on the 20 most similar past forecasts. Made heatwave days much worse (regression to the mean). |
| **Regression to the mean** | Extreme values tend to be followed, or matched, by less extreme ones. Here: past situations that looked like today usually turned out milder than a real heatwave, so averaging them pulls heatwave forecasts towards "typical warm". |
| **Tie margin** | In the control rule, two runs within 0.02 °C all-days RMSE count as tied, and the simpler one (fewer changes) wins. |

## Statistics and evaluation

| Term | Meaning |
|---|---|
| **RMSE** | Root-mean-square error: typical forecast error in deg C, big misses weighted more. Lower is better. |
| **MAE** | Mean absolute error: average size of the error in deg C. |
| **Stratum (normal / unusual / extreme)** | Forecast days grouped by how hot they actually were; "extreme-stratum RMSE" = error on heatwave days. |
| **Forecast-conditioned stratum** | Days the model *predicted* to be extreme (the complement to judging only days that turned out extreme). |
| **Seed** | The random starting point of training. Same model, different seed -> slightly different results, so models are trained with several seeds and averaged. |
| **Cluster (G)** | A block of days treated as one unit in the statistics: one season (Jan–Mar 14, Mar 15–Jul 31, Aug–Dec) of one year. G = number of blocks. Overlapping forecasts are not independent, so blocks are counted, not days. |
| **CI** | Confidence interval: the range in which the true difference plausibly lies. If it includes 0 the result is inconclusive. (Not to be confused with CI = continuous integration, see "Engineering"; the meaning is clear from context.) |
| **n.s.** | Not significant: the CI includes 0, so the difference may be chance. |
| **Δ (delta)** | A difference, always "model minus reference". Negative = the model's error is smaller, so the model is better. |
| **p-value** | How surprising the result would be if there were truly no difference; below 0.05 is called significant. |
| **FPR** | False-positive rate: how often a test reports "significant" when there is no real difference (should be about 5%). |
| **Power** | Probability that a test detects a real improvement of a given size. |
| **MDE** | Minimum detectable effect: the smallest improvement that can be detected with 80% power given our data. |
| **80% power** | A real effect of the MDE size would be detected (p < 0.05) 4 times out of 5. A non-significant result therefore rules out effects of about that size, not smaller ones. |
| **Paired cluster (jackknife) t-test** | The main significance test (`evaluation/stats.py`): compares two models on the same days, recomputing the result with one block left out at a time to measure uncertainty honestly. |
| **DM test** | Diebold–Mariano test of equal forecast accuracy; used as a secondary check. |
| **Rolling-origin folds / OOF** | Train on all years before a block, evaluate on the block (2007–09, 2010–12, 2013–15, 2016–18). OOF = out-of-fold predictions; gives about 12 years of honest evaluation instead of 3. |
| **Forecaster's dilemma** | Judging only on days that turned out extreme rewards models that always predict hot (Lerch et al. 2017). |
| **Retention (future-similarity)** | Whether a more similar past window also has a future closer to the truth (from the SARAF paper). |
| **Seed-noise floor** | The part of the MDE caused by training randomness alone; reduced by more seeds, not more data blocks. |
| **10-seed ensemble** | The average of the 10 seeds' forecasts. Reported as an extra only; the primary score is the average of the 10 per-seed RMSEs (decision 2026-10-04). |
| **Holm correction** | When several claims are tested together, each p-value is adjusted upwards (step by step, smallest first) so the chance of any false "win" stays at 5%. Used for H1–H4. |
| **Bias** | The average error (forecast − truth). Negative = the forecast runs too cold. On real heatwave days all our models are too cold. |
| **Range forecast** | A forecast given as a distribution, e.g. "likely 34 °C, 1-in-10 chance above 37 °C" (10th / 50th / 90th percentiles; `evaluation/range_forecasts.py`). |
| **CRPS** | Continuous ranked probability score: the error of a range forecast in °C (lower is better); equals the plain error for a single-number forecast. **CRPSS** = its skill compared with climatology. |
| **Coverage** | How often the truth falls inside a forecast range; an 80% range should contain it about 80% of the time. |
| **Hot-day probability / warning** | The forecast chance that a day will be a heatwave day, from the forecast and its usual error; scored with the Brier score. |
| **Occlusion** | Switching one input off (setting it to its long-term average) and measuring how much the forecast gets worse, to see what a trained model relies on (`evaluation/graph_explain.py`). |

## Training settings

| Term | Meaning |
|---|---|
| **hot_weight** | How many times more a hot-day error counts in the training loss (1 = no extra weight). Higher = better on heatwave days but warmer bias on other days. v1 used 20; the controls use 5. |
| **Target form (raw / anomaly / dp_residual)** | What the network is asked to predict: the value itself (raw); its departure from the seasonal normal (anomaly); or a correction to the damped-persistence forecast (dp_residual). Predictions are always converted back to °C before scoring. |
| **Early stopping (inner_2y / val_block)** | Training stops when the error on held-out data stops improving. `inner_2y` holds out the last 2 *training* years of each fold (used by all v2 runs); `val_block` uses the validation block itself (v1 behaviour, kept only for A1_repro). |
| **Registry** | `registry/runs.csv`: one row per run and fold, with settings, scores and fingerprints (hashes) of the data and code used. |
| **Bit-for-bit** | Reproduced exactly, down to the last binary digit, not just approximately. Holds only on the same CPU type. |

## Heat measures

| Term | Meaning |
|---|---|
| **Tmax** | Daily maximum air temperature. |
| **WBGT** | Wet-Bulb Globe Temperature: heat-stress index combining temperature, humidity, sun and wind. |
| **BoM WBGT approximation (Tier-A)** | Formula from Australia's Bureau of Meteorology using only temperature and humidity; a temperature–humidity index (no sun or wind). Against the physical WBGT it reads 2–3 °C too high in Jun–Sep and about right otherwise (an earlier '~6 °C' figure compared it with a shade WBGT). Columns `wbgt_bom_*`. |
| **Liljegren WBGT (physical WBGT)** | The validated physical outdoor WBGT: 0.7 × natural wet-bulb + 0.2 × black-globe + 0.1 × air temperature, from heat-balance equations using sun and wind (Liljegren et al. 2008). Ported from Liljegren's Argonne C code; adapted to reanalysis following Kong & Huber (2022) and checked against their PyWBGT. Columns `wbgt_lj_*` (`datasets_v2/wbgt_liljegren_daily.parquet`). |
| **Natural wet-bulb (Tnwb) / black-globe (Tg) temperature** | The two physical components of WBGT: the temperature of a wet wick exposed to sun and wind, and of a black sphere in the sun. |
| **WBGT heatwave label** | Humid-heat label: day in 15 Mar – 30 Sep with daily max **physical** WBGT ≥ its 95th in-season percentile (training years; ≈ 36.2 °C, 29 episodes); percentile chosen by the ≥ 25-episode rule (`configs/wbgt_label.json`). With the BoM index the same rule gives the 97.5th percentile (sensitivity only). |
| **Tier-A / Tier-B** | Plan v5's names for the two WBGT versions: Tier-A = BoM approximation (temperature and humidity only); Tier-B = Liljegren physical WBGT (also sun and wind; built from the v2 download). |
| **Ta** | Air temperature (the third WBGT component). |
| **GHI** | Global horizontal irradiance: total sunshine reaching flat ground (W/m²); an input to physical WBGT. |
| **ISO 7243** | The international standard defining WBGT and its work-rest limits. We don't apply its absolute limits to our reanalysis WBGT. |
| **Tw** | Wet-bulb temperature (Stull 2011 formula): lowest temperature reachable by evaporative cooling; a clean humid-heat measure. |
| **HI** | Heat Index ("feels-like" temperature, US National Weather Service formula). |
| **Anomaly / standardised anomaly (z)** | Departure from the seasonal normal (deg C) / that departure divided by the normal day-to-day spread. |
| **Labels v1 / v2** | Heatwave-day definitions. v1 = year-round relative anomaly (original). v2 = Mar 15–Jul 31 season, Tmax >= 40 C and anomaly >= 3 C (IMD 4.5 C as "severe"). |

## Data, retrieval and architecture

| Term | Meaning |
|---|---|
| **ERA5 / ERA5T** | ECMWF global weather reanalysis (the data source, via Open-Meteo) / its preliminary recent months, which can be revised. |
| **Open-Meteo / API** | The free web service we download ERA5 from / application programming interface: the address a program calls to fetch data. It limits calls per hour (HTTP 429 = "too many requests, wait"). |
| **IST** | Indian Standard Time (UTC + 5:30); all our timestamps use it. |
| **parquet / JSON** | File formats: parquet = compact table files (our datasets and predictions); JSON = text files (configs, raw downloads). |
| **NW** | North-west (the upstream region: Rajasthan, Pakistan). |
| **CPU / GPU** | Computer processor / graphics processor (faster for big neural networks). Our models train on the CPU. |
| **LLM** | Large language model; used only in Phase 6 to word advisories, under strict checks. |
| **P1–P4** | The four team roles: P1 data and labels, P2 models, P3 retrieval, P4 advisory and demo. |
| **ERA5-Land** | Finer (0.1 deg) land version of ERA5; dropped from the plan because it adds little independent signal for Delhi. |
| **N9 / N42** | 9 grid cells (current data, 0.25 deg apart) / 42 finer cells (dropped). |
| **Regional upstream graph** | Plan v5 DSTGNN design: nodes across NW India/Pakistan, where Delhi's hot air comes from 1–3 days earlier. |
| **DSTGNN** | Dynamic spatio-temporal graph neural network: passes information between map locations. Ours (`models/dstgnn.py`) has Delhi as node 0 and the 27 upstream points as the other nodes. |
| **Edge (advective / geographic)** | A link along which a node passes information to another. Geographic edges depend only on distance (fixed). Advective edges change daily: a node only sends to nodes **downwind** of it, more strongly with stronger wind (`pipeline/graph.py`). |
| **Adaptive adjacency** | Extra edges the network learns by itself (Graph WaveNet, Wu et al. 2019), for links the wind rule misses. |
| **Graph GRU (DCRNN-style)** | The recurrent unit that steps through the 14 input days, receiving each neighbour's previous-day state; information moves one node (~220 km) per day. GRU = gated recurrent unit, a simpler cousin of the LSTM. DCRNN: Li et al. 2018. |
| **FAISS** | Library used for fast similarity search over past windows. |
| **Analogue** | A past 14-day weather window similar to the current one; its following 5 days are the "analogue outcome". |
| **SARAF** | Stationarity-aware retrieval-augmented forecasting (Zhou et al. 2026); source of the time-alignment, diversity and stationarity ideas. |
| **500 hPa height (z500)** | The height (≈ 5.5 km) at which air pressure is 500 hPa; its map shows the large weather systems (high-pressure ridges that drive heatwaves). From ERA5 via WeatherBench 2; used by Rz. |
| **GraphCast** | Google DeepMind's global AI weather model (Lam et al. 2023), a graph network forecasting the whole atmosphere. Used as an outside yardstick on 2018 Tmax: ~40% lower error than our models. |
| **WeatherBench 2** | Public archive and benchmark of weather forecasts and ERA5 data (Rasp et al. 2024); source of the GraphCast forecasts and the 500 hPa data. |
| **Foundation model (time series)** | A large network pretrained on millions of time series to forecast any series without extra training. We compared **Chronos-2** (Amazon) and **TimesFM 3.0** (Google) after the test; C3-hop beat both. |
| **Zero-shot** | Using a pretrained model on new data with no further training. |
| **MMR** | Maximal marginal relevance: choosing analogues that are relevant and not near-duplicates of each other. |
| **ADF / KPSS** | Statistical tests for whether a series is stationary (stable over time). |
| **Sen slope / Mann–Kendall** | Robust trend estimate (per decade) and its significance test. |
| **M1 / M2** | Retrieval database modes: M1 strict (only train/val years, primary); M2 walk-forward (any past day, operational only). |
| **Manifest** | `data/MANIFEST.json`: fingerprints (hashes) of every frozen file; `tests/test_manifest.py` fails if one changes. |

## Engineering and CI

| Term | Meaning |
|---|---|
| **CI** (continuous integration) | Automatic checks GitHub runs after every push (`.github/workflows/ci.yml`): all tests, the frozen-file check, a secret scan and a dependency audit. |
| **GitHub Actions / runner** | GitHub's CI service / the rented machine a CI job runs on. Runners vary in CPU type, so float results can differ in the last digits. |
| **gitleaks** | Tool that scans the whole git history for leaked secrets (passwords, API keys). |
| **pip-audit** | Tool that checks our pinned Python packages against databases of known security holes. |
| **Lockfile** | `requirements.lock.txt`: the exact version of every installed package, so everyone (and CI) gets the same environment. |
| **SHA / SHA-256 (hash, fingerprint)** | A short code computed from a file's contents; any change gives a different code. Used to pin CI actions and tools to exact versions, and in the manifest and registry. |
| **venv** | `.venv`: the project's private Python installation, so nothing is installed system-wide. |

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
| **Rule engine / verifier** | Phase 6 safety layers (`advisory/rules.py`, `advisory/verifier.py`): the rule engine sets each day's alert colour from the forecast by fixed rules (never the LLM); the verifier rejects any advisory whose colours, actions, quotes, numbers or labels don't match the rule engine and the action library. |
| **Action library** | `advisory/action_library.json`: the 133 actions the advisory may recommend, each with a word-for-word official quote and page, its section (forecast / preparedness / "If this happens") and its minimum colour. Built from `sources/phase6_candidate_actions.json` by `advisory/library.py`. |
| **Heat-wave run** | Consecutive heat-wave days (observed and forecast together). The rule engine gives every day of a run one colour from its length and its severe days, following IMD's colour table. An observed day is a heat-wave day by the label rule; a forecast day when its hot-day probability is at least 37% (decision 2026-10-27). |
| **Warning rules (R-point, R-prob, R-shift)** | Ways to turn a forecast into a yes/no warning: the forecast reaches the threshold (R-point); the hot-day probability passes a level (R-prob); the forecast comes within a margin of the threshold (R-shift). Levels and margins chosen on 2007–2018 (`post_test/warning_rules.py`). |
| **ETS (equitable threat score)** | A yes/no warning score balancing hits against false alarms: 0 = no better than chance, 1 = perfect. |
| **Preparedness actions** | Standing heat-season measures (e.g. stock supplies before summer) that no forecast triggers; shown all heat season whatever the colour (decision 2026-10-26). |
| **Template advisory** | An advisory built from the rules alone, with fixed sentences and no LLM. It passes the verifier; it is the reference the LLM is compared with and the fallback when an LLM advisory is rejected. |
| **IMD-style alert mapping (B + D)** | How our forecast becomes an alert colour (decided 2026-10-22): IMD's heat-wave rules with the departure threshold adjusted for area-average data (≥ 3 °C heat wave, ≥ 4.5 °C severe; or ≥ 45 °C), IMD's persistence rules for the colours, plus the hot-day probability shown next to each day. Labelled "not an official IMD warning". |
| **tier_basis** | Where an action's alert colour comes from: the source itself; `delhi_any_alert` (Delhi's plan says "during a heat alert", so shown at Yellow and up); or `ahmedabad_precedent` (Ahmedabad's plan; not Delhi policy). |
