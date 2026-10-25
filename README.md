# Retrieval-augmented heatwave forecasting for Delhi

Five-day forecasts of daily maximum temperature (Tmax) and heat stress (WBGT, the wet-bulb globe temperature) for
Delhi, from ERA5 reanalysis data, with a heat-advisory system built on top. This is a capstone project: a team of four
(roles P1–P4).

The final forecaster is **C3-hop**, our own graph neural network. It reads the last 14 days at Delhi and at 27
points upwind over north-west India and Pakistan, where Delhi's hot air comes from. We also tested whether
**analogue retrieval** (finding similar past situations and using what happened next) improves the forecasts, and
documented in detail when it helps and when it doesn't.

**Status (2026-10-25):**
- Forecasting research complete: the protocol was frozen and the test run once.
- The heat advisory (Phase 6) is in progress.

## Main results

All results are on the locked test years, **2019-01-01 to 2026-09-06**. That period was scored once, after the
evaluation rules were frozen as the git tag `test-lock`.¹

The numbers are **RMSE in °C**: the typical size of the forecast error, so lower is better. Ours are averaged over
10 training runs. "Heatwave days" = days that actually were heatwave days.

| Model | WBGT, all days | WBGT, heatwave days | Tmax, all days | Tmax, heatwave days |
|---|---|---|---|---|
| C3-hop (ours, final) | 2.26 | 3.02 | 2.23 | 2.31 |
| Delhi-only LSTM (standard deep-learning baseline) | 2.39 | 3.10 | 2.36 | 2.55 |
| Ridge regression on the upstream points' recent Tmax (hot days weighted)² | 2.30 | 3.74 | 2.32 | 2.53 |
| Plain ridge regression, same inputs² | 2.29 | 4.15 | 2.28 | 2.89 |
| "Usual for this date" (climatology) | 2.57 | 4.75 | 2.87 | 4.45 |
| "Same as yesterday" (persistence) | 3.04 | 3.78 | 2.77 | 2.56 |
| Chronos-2 (Amazon), zero-shot³ | 2.39 | 3.82 | 2.41 | 3.10 |
| TimesFM 3.0 (Google), zero-shot³ | 2.40 | 3.83 | 2.41 | 3.05 |

Close relatives of C3-hop score the same within noise: the version without graph connections (C2) and the one that
pools the region (C3-pool) are 0.01 °C lower on all days, and none of these differences is significant. The
averaged ("ensemble") forecast of the 10 runs is a little better again (C3-hop: 2.23 WBGT, 2.19 Tmax on all days;
`evaluation_post_test/test_period_extras.md`).

¹ Two early sanity scripts from the v1 data stage (`check_heatwaves.py`, `retrieval/query.py`) looked at
  2019/2022 temperatures and one 2022 date before the freeze. No model score was seen. See `docs/KNOWN_DEVIATIONS.md`.
² The ridge models see Delhi's last value plus the 27 upstream points' **Tmax** anomalies over 3 days, not the
  humidity or the other variables C3-hop gets, so they are a simpler model with fewer inputs.
³ Pretrained time-series foundation models, compared after the test as a labelled extra. They saw only the target's
  own past (one variable), so this shows what a ready-made model gives, not a like-for-like contest.

**The pre-registered claims (WBGT):**

| Claim | Result |
|---|---|
| H1: C3-hop beats the Delhi-only LSTM on all days | **confirmed**: −0.127 °C [−0.180, −0.075] |
| H2: … and on heatwave days | not confirmed: −0.081 °C [−0.261, +0.100] |
| H3: C3-hop beats the hot-day-weighted ridge on heatwave days | **confirmed**: −0.718 °C [−0.895, −0.540] |
| H4: analogue retrieval improves the Delhi-only LSTM on all days | not confirmed: −0.042 °C [−0.080, −0.004], Holm-adjusted p = 0.066 |

**How to read them:**

- **Seeing the upstream region is what helps.** C3-hop beats the Delhi-only LSTM on all days (H1). The graph without
  its connections does as well, so we make no claim that the graph structure itself adds skill.
- **C3-hop ties the plain ridge on all days** (−0.04 °C, not significant, both families). Its advantage over the
  ridge models is on heatwave days (H3), and about 96% of that gap is a smaller cold bias: every model forecasts
  real heatwave days too cold, the ridge more so.
- **H2 is a real null, not only a small sample.** The test could detect a difference of about 0.25 °C; the
  development result (−0.26 °C) would have been detected. The test difference was a third of that, so development
  probably overstated the heatwave-day gain.
- **H1 compares a tuned model with an untuned control.** C3-hop's settings were tuned in development; the Delhi-only
  LSTM's were not. The untuned graph relative C2 gets the same gain, which suggests tuning is not the explanation.
- **Retrieval did not deliver.** H4 was not confirmed after correcting for four tests, and about half of its small
  all-day gain is simply a cooler average forecast. In development, regional retrieval helped heatwave days a little
  (−0.16 °C) but never beat its own random-retrieval control; on the test it made them worse (+0.18, exploratory).
  `docs/MODELS_EXPLAINED.md` §11 explains why.
- **Ranges and warnings work on ordinary days, not on heatwave days.** The 80% forecast range holds the truth 77–78%
  of the time over all days, but only 29% (WBGT) and 54% (Tmax) of the time on heatwave days, where forecasts are too
  cold and too narrow. As a yes/no WBGT heatwave warning, C3-hop catches about 9% of heatwave days, and about two in
  three of its warnings are false alarms (Tmax: 27% caught, 37% false alarms).
- **Global AI weather models are better.** GraphCast had about 40% lower Tmax error in 2018, because it sees the whole
  atmosphere. Our contribution is a cheap, transparent local model, a careful test of retrieval, and heat-stress
  forecasts.

All truth is ERA5 reanalysis averaged over 9 grid cells, not a weather station; the last ~3 months were preliminary
(ERA5T) when downloaded. Removing those months changes nothing (H1 −0.134, H3 −0.700, H4 −0.044).

## Where to read more

| Document | What it is |
|---|---|
| `docs/PROJECT_GUIDE.md` | the plain-language guide to the whole project and every result |
| `docs/MODELS_EXPLAINED.md` | every model and code name, the metrics, GraphCast, the foundation models, and why retrieval behaves as it does |
| `docs/CONFIRMATORY_PROTOCOL.md` | the frozen test protocol (hypotheses, power, safeguards) |
| `docs/KNOWN_DEVIATIONS.md` | known issues in the frozen code and protocol, disclosed rather than changed |
| `docs/GLOSSARY.md` | the terms used |
| `context/decisions.md` | the dated decision log, with each rule written down before it was run |
| `evaluation_test/test_results.md` | the full test report |
| `evaluation_post_test/` | post-test extras: foundation models, ensemble and warning scores, bias split |

## Setup (Windows, Python 3.13)

In PowerShell, from the repo root:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.venv\Scripts\python.exe -m pytest tests -q
```

- Install into the project venv, never globally.
- `requirements.txt` lists the direct dependencies; `requirements.lock.txt` is the exact core environment.
- The foundation-model comparison needs extra packages (`requirements-post-test.lock.txt`, installed the same way)
  and downloads about 1.8 GB of model weights into `.hf_cache/` in the project, pinned to fixed versions.
- The test suite takes about 6 minutes (`-m "not slow"` skips the two slowest tests). Some tests need the retrieval
  index, which is gitignored: build it first with `python -m retrieval.build_index` and
  `python -m retrieval.precompute_analogues`, or those tests skip. Tests that need the raw downloads skip without them.
- The exact-retraining test (`test_reproduces_frozen_a1_seed0_bit_for_bit`) only passes on the CPU type the model was
  frozen on; float results differ in the last digits between CPU types.

## Data

- **Committed:**
  - the daily datasets in `datasets_v2/`;
  - the 500 hPa extracts and the GraphCast extract in `data/external/`;
  - all development forecasts in `predictions_v2/` and all test forecasts in `predictions_test/`;
  - our Phase 6 source catalogues and results in `sources/` (the official PDFs themselves are not committed; the
    report lists where to download them).
- **Not committed (large; regenerable):**
  - **Raw ERA5:** from Open-Meteo (non-commercial use), fetched by `pipeline/download_era5_v2.py`,
    `download_era5_upstream.py` and `download_era5_humid.py`. The final ~3 months are revised by ECMWF after first
    release, so a new download can differ slightly there.
  - **500 hPa heights:** read from WeatherBench 2 (`pipeline/download_era5_z500.py`).
  - **Model checkpoints:** `models/v2/` (development) and `models/test/`.
  - **Inner-block forecasts:** `predictions_v2/inner/`.
- **Consequence:** the comparison reports (`evaluation.compare_v2`, the test report) rerun from a fresh clone. The
  calibration, range-forecast, analog-correction and warning-retrieval reports need `predictions_v2/inner/`, and the
  graph explanation needs `models/v2/`. Both come from retraining the development models (10 seeds × 4 folds per
  model, several hours each on a CPU).

## How to run things

Use the venv interpreter (`.venv\Scripts\python.exe`) for every command.

| Task | Command |
|---|---|
| Train one model on the four development folds | `python -m training.train_unified --config configs/A2Lr_hw5_C3hop.json` |
| Compare models (development, 2007–2018) | `python -m evaluation.compare_v2` |
| Re-score the test from the committed forecasts | `python -m evaluation.test_evaluation` (should reproduce `evaluation_test/test_results.md` exactly) |
| Check the test outputs are untouched | `python -m scripts.make_test_manifest --check` |
| Foundation-model comparison | `python -m post_test.foundation_baselines` |
| Post-test ensemble and warning scores | `python -m post_test.test_period_extras` |

**Reading the test period needs a full clone with tags** (`git clone` gets them; CI fetches them explicitly).
`training/test_lock.py` refuses to read 2019+ data unless the tag `test-lock` is in the history and the code under
`training/ models/ pipeline/ retrieval/ evaluation/ configs/` is exactly the code frozen there. Later commits that only
touch other folders (docs, results, `post_test/`) are fine.

**Retraining the test models** is possible but is not the test: `python -m training.test_run --stage all` skips every
model whose test forecasts exist, so it only retrains after you move `predictions_test/` aside. It takes about 9 hours
on a CPU, and the results match the committed ones exactly only on the same CPU type.

## Repository map

| Folder or file | What it is |
|---|---|
| `pipeline/`, `training/`, `models/`, `retrieval/`, `evaluation/`, `configs/` | the v2 code, frozen at `test-lock` |
| `post_test/` | analyses run after the test (not frozen) |
| `datasets_v2/`, `predictions_v2/`, `evaluation_v2/` | v2 development data, forecasts and reports |
| `predictions_test/`, `evaluation_test/`, `registry/` | the one test run, its report, and the run registry |
| `docs/`, `context/` | the guides, protocol and decision log |
| `sources/` | Phase 6 (heat advisory) source research and catalogues |
| **Legacy v1** (first version, kept frozen as the reference): `download_era5.py`, `process_weather.py`, `prepare_datasets.py`, `check_heatwaves.py`, `Step5_Colab_Minimal.ipynb`, `config/`, `datasets/`, `events/`, `splits/`, `predictions/`, `predictions_v1/`, `archive_v1/`, `models/frozen/` | fingerprinted in `data/MANIFEST.json`; not used by v2 except as a check |

## Data sources and methods credited

- **Weather data:** ERA5 reanalysis (Copernicus / ECMWF), via Open-Meteo. ERA5 500 hPa heights and GraphCast
  forecasts are from WeatherBench 2 (Rasp et al. 2024).
- **Physical WBGT:** Liljegren et al. (2008), following the implementation of Kong & Huber (2022).
- **Graph methods:** the graph recurrence follows DCRNN (Li et al. 2018); the ring message passing follows Abboud et
  al. (2022).
- **Advisory sources:** IMD, NDMA, the Delhi Heat Action Plan and other official documents (catalogued for Phase 6).
