# Retrieval study (branch `retrieval-improvements`)

Why our retrieval (RAG) did not help, and what could. It follows the mentor's feedback of 2026-11-09. Every step was
pre-registered in [`context/decisions_retrieval.md`](../context/decisions_retrieval.md) before it ran. Plain-language
summary: [`docs/PROJECT_GUIDE.md`](../docs/PROJECT_GUIDE.md) §9.2. Development years (2007-2018) only; nothing here
reads the 2019-2026 test years or changes a frozen file.

**What it found** (details in each step's report):
- Retrieval helps only when it brings information the model's inputs lack. Our models already get the season and the
  region, so Delhi's own past adds nothing (step 1, both families).
- RAFT works in its own kind of setting: on Delhi's hourly temperature with a 12-hour window it cuts the error by 7%
  (step 1e). On our daily task it adds nothing.
- Only upper-air retrieval (matching the 500 hPa pattern) helps C3-hop: about 1% on ordinary WBGT days (step 2).
- Given directly, the same upper-air pattern helps more. C3-hop's all-days error falls by about 4% (WBGT) and 3%
  (Tmax), and the gain needs the right day's upper air, not just more inputs (the shuffled-year control). This is not
  retrieval (step 4).
- Retrieval keyed on ECMWF's own forecast (an analogue ensemble, Tmax 2016-2018) beats C3-hop by 26%, but only ties
  ECMWF's corrected forecast: the forecast carries the information (step 5).

| File | What it does |
|---|---|
| `raft.py` | RAFT (Han et al. 2025, arXiv 2505.04163) as published: Pearson retrieval on pooled windows (periods 1, 2, 4), softmax(rho / 0.1) weights, changes (not levels) as values. |
| `testbed.py` | The information ladder (B0 own history, B1 + season, B2 + all Delhi inputs, B3 + region) x the retrieval arms (none, RAFT, R0, random controls), as ridge models; RAFT's add-on mode on the saved network forecasts. |
| `diagnostics.py` | Does a similar past mean a similar future? Oracle analogues; the "cold days" checks. |
| `hourly.py` | The positive control: RAFT on Delhi's hourly temperature with 12- and 96-hour windows. |
| `screens.py` | Step 2: retrieval types that bring information C3-hop lacks (Rz, the 500 hPa pattern; LW, season-to-date) and retrieval for ranges (KR). |
| `step1.py`, `step2.py` | The runners and reports of steps 1 and 2. |
| `z500_direct.py` | Step 4 (not retrieval): C3-hop with the 500 hPa pattern as a direct input (10 principal-component scores per input day), its shuffled-year control (each year's scores taken from another training year), and the report. |
| `hres_anen.py` | Step 5: an analogue ensemble keyed on ECMWF's HRES forecast (Delle Monache et al. 2013), compared with C3-hop and with corrected HRES. |
| `data/` | Step 5's extract: HRES 2 m temperature over the Delhi box, runs from 2016-2018 only (WeatherBench 2). |
| `predictions/` | Step 4's forecasts, 10 seeds x 4 folds per run. Its checkpoints (`checkpoints/`) are not committed. |

## How to rerun it

Run from the repo root, inside the project's virtual environment.

**Steps 1 and 2.** Each family takes 15-20 minutes; the hourly control takes about 20 minutes and needs the raw hourly
ERA5 files in `data/raw/era5_monthly/`.

```bash
python -m retrieval_study.step1 --family Tmax
```

```bash
python -m retrieval_study.step1 --family WBGT
```

```bash
python -m retrieval_study.step1 --report
```

```bash
python -m retrieval_study.hourly
```

```bash
python -m retrieval_study.step2 --family Tmax
```

```bash
python -m retrieval_study.step2 --family WBGT
```

```bash
python -m retrieval_study.step2 --report
```

**Step 3** was not run: the end-to-end C3-hop + Rz run it calls for already existed from October
(`predictions_v2/A2Lr_hw5_C3hop_Rz`; see the RS-2 DEVIATION in the decision log).

**Step 4** trains C3-hop (10 seeds x 4 folds) once per command, 2-4 hours each on our CPU. A fold whose forecasts
already exist is skipped, so an interrupted run resumes where it stopped. `--control` runs the shuffled-year control.

```bash
python -m retrieval_study.z500_direct --family WBGT
```

```bash
python -m retrieval_study.z500_direct --family Tmax
```

```bash
python -m retrieval_study.z500_direct --family WBGT --control
```

```bash
python -m retrieval_study.z500_direct --family Tmax --control
```

```bash
python -m retrieval_study.z500_direct --report
```

**Step 5** reads the committed HRES extract; it downloads from WeatherBench 2 only if the extract is missing.

```bash
python -m retrieval_study.hres_anen
```

**What a fresh clone can rerun.** Steps 4 and 5 rerun from committed files. Step 1's RAFT add-on and step 2's add-on
screens and ranges also read the networks' saved inner-block forecasts (`predictions_v2/inner/`), which are not
committed; they come from retraining the development models (see the main README's Data section).

Results: `results/step1.md`, `results/step1e.md`, `results/step2.md`, `results/step4.md` and `results/step5.md` (and
the matching `.json` files).
Tests: `tests/test_retrieval_study_*.py` (offline, plus one check against fold f4's arrays).
