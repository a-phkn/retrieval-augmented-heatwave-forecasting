# Retrieval study (branch `retrieval-improvements`)

Why our retrieval (RAG) did not help, and what could. It follows the mentor's feedback of 2026-11-09. Every step was
pre-registered in [`context/decisions_retrieval.md`](../context/decisions_retrieval.md) before it ran. Plain-language
summary: [`docs/PROJECT_GUIDE.md`](../docs/PROJECT_GUIDE.md) §9.2. Development years (2007-2018) only; nothing here
reads the 2019-2026 test years or changes a frozen file.

| File | What it does |
|---|---|
| `raft.py` | RAFT (Han et al. 2025, arXiv 2505.04163) as published: Pearson retrieval on pooled windows (periods 1, 2, 4), softmax(rho / 0.1) weights, changes (not levels) as values. |
| `testbed.py` | The information ladder (B0 own history, B1 + season, B2 + all Delhi inputs, B3 + region) x the retrieval arms (none, RAFT, R0, random controls), as ridge models; RAFT's add-on mode on the saved network forecasts. |
| `diagnostics.py` | Does a similar past mean a similar future? Oracle analogues; the "cold days" checks. |
| `hourly.py` | The positive control: RAFT on Delhi's hourly temperature with 12- and 96-hour windows. |
| `screens.py` | Step 2: retrieval types that bring information C3-hop lacks (Rz, the 500 hPa pattern; LW, season-to-date) and retrieval for ranges (KR). |
| `step1.py`, `step2.py` | The runners and reports. |

Run from the repo root, inside the project's virtual environment. Each family takes 15-20 minutes; the hourly
control takes about 20 minutes and needs the raw hourly ERA5 files in `data/raw/era5_monthly/`.

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

Results: `results/step1.md`, `results/step1e.md`, `results/step2.md` (and the matching `.json` files).
Tests: `tests/test_retrieval_study_*.py` (offline, plus one check against fold f4's arrays).
