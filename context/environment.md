# Environment

## Python
Python 3.12 used in the verification/dev sandbox this session.

## Dependencies (as introduced so far)
- `pandas`, `pyarrow` — data I/O (already required by P1's scripts).
- `pytest` — leakage tests.
- `torch` (CPU build) — models/training, introduced in Step 3.
- `faiss-cpu` — NOT YET introduced, needed when P3 (retrieval) starts.

No `requirements.txt` exists yet in the repo — worth adding once the
dependency set stabilizes past this session (flagging, not adding
speculatively).

## Local run commands (Step 3)
```bash
pip install pandas pyarrow torch --index-url https://download.pytorch.org/whl/cpu
# Run from repo root, as modules (cross-imports between training/ and models/
# require this -- `python training/train_lstm.py` directly will fail with
# ModuleNotFoundError since the script's own dir gets added to sys.path
# instead of repo root):
python -m training.data          # rebuilds/verifies window arrays + normalization stats
python -m training.baselines     # persistence + climatology sanity metrics
python -m training.train_lstm    # 5-seed LSTM training loop
python -m training.evaluate_lstm # extended metrics from the saved checkpoints (no retraining)
```
`training/` and `models/` both have `__init__.py` so they import as packages.

## Colab
Not needed yet — Step 3 training is classified LOCAL (see `build_plan.md`).
If a future step needs Colab (e.g. large ablation sweeps), note the exact
notebook cell commands here when that happens, including how to mount/copy
the `datasets/` parquet files since Colab won't have the local filesystem.

## Data provenance
Raw ERA5 pulled via Open-Meteo by `download_era5.py`. Raw JSON is gitignored;
only processed parquet files are versioned. Re-running `download_era5.py`
will pull new data up to whatever "present" resolves to at run time — expect
`weather_daily.parquet`'s end date to drift on re-download, this is expected
per the script's own truncate-to-latest-complete-date logic.

## Repo hygiene note
Extracting the project zip converts line endings (CRLF vs. LF) on already-
committed files, which shows up as a spurious `git diff` on nearly every
tracked file. This is cosmetic (equal insertions/deletions per file,
confirmed by diff), not real content drift. Not fixed as part of this
session; mention if starting fresh from a zip again.
