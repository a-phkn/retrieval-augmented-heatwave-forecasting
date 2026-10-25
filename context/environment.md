# Environment

## Python and isolation
- **Python 3.13** (Windows), in a project-local virtual environment at `.venv/`
  (gitignored). Nothing is installed into the system Python.
- Torch is the **CPU-only** build (PyPI Windows wheels); no CUDA locally.
  GPU work (DSTGNN sweeps, LLM) runs on Colab.
- `.venv/` is ~1.3 GB. Deleting that folder removes the whole environment.

## Setup (first time, from repo root)
```powershell
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.venv\Scripts\python.exe -m retrieval.build_index            # gitignored index needed by some tests
.venv\Scripts\python.exe -m retrieval.precompute_analogues
.venv\Scripts\python.exe -m pytest -q                        # about 400 tests, ~6 min; a few skip without raw data
```
Optional, for the post-test foundation-model comparison only:
`.venv\Scripts\python.exe -m pip install -r requirements-post-test.lock.txt` (weights, ~1.8 GB, go to `.hf_cache/`).
Call the venv interpreter directly (`.venv\Scripts\python.exe ...`) or activate it
first (`.venv\Scripts\Activate.ps1`). Never run `pip install` without one of these.

## Dependency files
- `requirements.txt`: direct dependencies, pinned to the verified versions.
  Add new packages here (and only after agreeing on them).
- `requirements.lock.txt`: the exact core environment (everything the code, the tests and CI need).
- `requirements-post-test.txt` / `requirements-post-test.lock.txt`: Chronos-2, TimesFM 3.0 and their
  dependencies, for `post_test/foundation_baselines.py` only.
- Regenerate after an upgrade: `pip freeze`, then put the packages only the post-test models need into the
  post-test lock (split on 2026-10-25 from the installed metadata: 43 core, 25 post-test packages).

Phase-specific packages (e.g. the Phase 6 LLM stack) are added when their phase starts, after agreeing on them.

## Run commands (always from repo root, as modules)
Cross-imports between `training/`, `models/`, `retrieval/` require `-m`:
```powershell
.venv\Scripts\python.exe -m retrieval.build_index           # ~10 s; regenerates gitignored index files
.venv\Scripts\python.exe -m retrieval.precompute_analogues  # top-20 analogue table
.venv\Scripts\python.exe -m training.train_lstm             # 5-seed LSTM (overwrites v1 outputs -- back up first)
.venv\Scripts\python.exe -m training.evaluate_lstm
```

## Colab
Do **not** install `requirements.txt` on Colab (it ships its own CUDA torch). Use the
bundle pattern in `Step5_Colab_Minimal.ipynb` and `pip install` only missing packages
(e.g. `faiss-cpu`). Commands are in `RUN_COMMANDS.md`.

## Data provenance
Raw ERA5 is pulled via Open-Meteo by `download_era5.py`. Raw JSON (`data/raw/`) is
gitignored (275 MB) and was restored locally on 2026-10-04 from the original download
(verified: it reproduces `weather_daily.parquet` exactly; folder fingerprint in
`data/MANIFEST.json`). `data/raw/legacy_full_range/` holds older single-request
downloads for cells 1–6 and is not used by any code.
The current downloader uses `date.today()`, so the end date drifts on re-download;
the v2 downloader fixes `END_DATE = 2026-09-06` (see `docs/PLAN_REVIEW_v5.md`).

## Verified state (2026-10-04; see the README for the current state)
- venv created; `pytest` → 135 passed at the time (after freeze + stats work; 9 before).
- `retrieval.build_index` rebuilt the index; `feature_normalization_stats.json`
  reproduced byte-identically (no git diff).

## Repo hygiene note
Extracting the project from a zip can convert line endings (CRLF/LF) and show a
spurious `git diff` on many files. This is cosmetic, not content drift.
