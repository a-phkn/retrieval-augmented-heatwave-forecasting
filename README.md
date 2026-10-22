# Retrieval-augmented heatwave forecasting for Delhi

Five-day forecasts of daily maximum temperature (Tmax) and heat stress (WBGT, the wet-bulb globe temperature) for
Delhi, from ERA5 reanalysis data, with a heat-advisory system built on top. This is a capstone project: a team of four
(roles P1–P4).

The final forecaster is **C3-hop**, our own graph neural network. It reads the last 14 days at Delhi and at 27
points upwind over north-west India and Pakistan, where Delhi's hot air comes from. We also tested whether
**analogue retrieval** (finding similar past situations and using what happened next) improves the forecasts, and
documented in detail when it helps and when it doesn't.

**Status (2026-10-22):**
- Forecasting research complete: the protocol was frozen and the test run once.
- The heat advisory (Phase 6) is in progress.

## Main results

All results are on the locked test years, **2019-01-01 to 2026-09-06**. That period was used once, after the
evaluation rules were frozen as the git tag `test-lock`.

The numbers are **RMSE in °C**: the typical size of the forecast error, so lower is better. Ours are averaged over
10 training runs. "Heatwave days" = days that actually were heatwave days.

| Model | WBGT, all days | WBGT, heatwave days | Tmax, all days | Tmax, heatwave days |
|---|---|---|---|---|
| **C3-hop (ours, final)** | **2.26** | **3.02** | **2.23** | **2.31** |
| Delhi-only LSTM (standard deep-learning baseline) | 2.39 | 3.10 | 2.36 | 2.55 |
| Ridge regression on the same regional data (hot days weighted) | 2.30 | 3.74 | 2.32 | 2.53 |
| "Usual for this date" (climatology) | 2.57 | 4.75 | 2.87 | 4.45 |
| "Same as yesterday" (persistence) | 3.04 | 3.78 | 2.77 | 2.56 |
| Chronos-2 (Amazon), zero-shot* | 2.39 | 3.82 | 2.41 | 3.10 |
| TimesFM 3.0 (Google), zero-shot* | 2.40 | 3.83 | 2.41 | 3.05 |

\* Pretrained time-series foundation models, compared after the test as a labelled extra.

**The pre-registered claims (WBGT):**

| Claim | Result |
|---|---|
| C3-hop beats the Delhi-only LSTM on all days | **confirmed**: −0.13 °C |
| … and on heatwave days | not confirmed: −0.08 °C, below what the test period could detect |
| C3-hop beats the ridge model on heatwave days | **confirmed**: −0.72 °C |
| Analogue retrieval improves the Delhi-only LSTM on all days | not confirmed: −0.04 °C, Holm-adjusted p = 0.066 |

**What this adds up to:**

- **Seeing the upstream region is what helps.** The graph without its connections does about as well, so we make no
  claim that the graph structure itself adds skill.
- **Retrieval** helps ordinary days a little when it brings information the model lacks, but never helped heatwave
  days. The reason is explained in `docs/MODELS_EXPLAINED.md` §11.
- **Global AI weather models are better.** GraphCast had about 40% lower Tmax error in 2018, because it sees the whole
  atmosphere. Our contribution is a cheap, transparent local model with heat stress and calibrated warnings.

## Where to read more

| Document | What it is |
|---|---|
| `docs/PROJECT_GUIDE.md` | the plain-language guide to the whole project and every result |
| `docs/MODELS_EXPLAINED.md` | every model and code name, the metrics, GraphCast, the foundation models, and why retrieval behaves as it does |
| `docs/CONFIRMATORY_PROTOCOL.md` | the frozen test protocol (hypotheses, power, safeguards) |
| `docs/GLOSSARY.md` | the terms used |
| `context/decisions.md` | the dated decision log, with each rule written down before it was run |
| `evaluation_test/test_results.md` | the full test report |

## Setup (Windows, Python 3.13)

```bash
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.venv\Scripts\python.exe -m pytest tests -q
```

- Install into the project venv, never globally.
- `requirements.txt` lists the direct dependencies; `requirements.lock.txt` is the exact environment.
- The test suite takes about 6 minutes. A few tests skip themselves when the large raw data is not present.

## Data

- **Committed (enough to rerun everything except the raw rebuilds):**
  - the daily datasets in `datasets_v2/`;
  - the 500 hPa extracts and the GraphCast extract in `data/external/`;
  - all forecasts in `predictions_v2/` (development) and `predictions_test/` (test).
- **Not committed (large; regenerable):**
  - **Raw ERA5:** daily and hourly ERA5 from Open-Meteo (non-commercial use) is fetched by
    `pipeline/download_era5_v2.py`, `download_era5_upstream.py` and `download_era5_humid.py`.
  - **500 hPa heights:** read from WeatherBench 2 (`pipeline/download_era5_z500.py`).
  - **Model checkpoints:** `models/v2/`, `models/test/`.

## How to run things

| Task | Command |
|---|---|
| Train one model on the four development folds | `python -m training.train_unified --config configs/A2Lr_hw5_C3hop.json` |
| Compare models (development, 2007–2018) | `python -m evaluation.compare_v2` |
| The test run (only at the `test-lock` tag) | `git checkout test-lock`, then `python -m training.test_run --stage all` and `python -m evaluation.test_evaluation` |
| Foundation-model comparison | `python -m post_test.foundation_baselines` |

`training/test_lock.py` refuses to read 2019+ data unless the code is exactly the code frozen in `test-lock`.

## Data sources and methods credited

- **Weather data:** ERA5 reanalysis (Copernicus / ECMWF), via Open-Meteo. ERA5 500 hPa heights and GraphCast
  forecasts are from WeatherBench 2 (Rasp et al. 2024).
- **Physical WBGT:** Liljegren et al. (2008), following the implementation of Kong & Huber (2022).
- **Graph methods:** the graph recurrence follows DCRNN (Li et al. 2018); the ring message passing follows Abboud et
  al. (2022).
- **Advisory sources:** IMD, NDMA, the Delhi Heat Action Plan and other official documents (catalogued for Phase 6).
