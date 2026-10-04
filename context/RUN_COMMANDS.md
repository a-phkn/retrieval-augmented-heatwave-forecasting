# Run Commands

Every command below is run from the repo root. Python packages that
cross-import each other (`training/`, `models/`, `retrieval/`) must be run
with `python -m`, not `python path/to/file.py` directly — the latter fails
with `ModuleNotFoundError` because it adds the script's own directory to
`sys.path` instead of the repo root. See `environment.md` for why.

## Setup (once)

```bash
pip install -r requirements.txt
```

If you want the smaller CPU-only PyTorch build instead of the default
(which pulls a CUDA build even with no GPU):
```bash
pip install pandas pyarrow pytest faiss-cpu
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Step 0 — sanity check the data pipeline

```bash
.venv\Scripts\python.exe -m pytest -q
```
Expects 145 passed (as of 2026-10-04): 3 `test_no_leakage.py`, 6
`test_retrieval_eligibility.py` (needs `retrieval/faiss_index.bin`; build it
first with `python -m retrieval.build_index`, see Step 4), 12 `test_manifest.py`
(frozen v1 files unchanged), 46 `test_stats.py`, 8 `test_predict_v1.py`, 16 `test_hourly_features.py` , 13 `test_download_era5_v2.py`, 13 `test_labels_v2.py`, 19 `test_folds.py` and 9 `test_train_unified.py`. To verify the freeze alone:
`.venv\Scripts\python.exe -m scripts.make_manifest --check`.

## Step 1-2 — data pipeline (P1, already run, re-run only if raw data changes)

```bash
python download_era5.py          # pulls raw ERA5 via Open-Meteo, resumable
python process_weather.py        # -> data/processed/weather_daily.parquet
python prepare_datasets.py       # -> events/, datasets/, tests/test_no_leakage.py (regenerated)
```
`prepare_datasets.py` overwrites `tests/test_no_leakage.py` — don't hand-edit that file.

## Step 3 — baseline LSTM (P2)

```bash
python -m training.data              # rebuilds/verifies window arrays + normalization stats
python -m training.baselines         # persistence + climatology sanity baselines
python -m training.train_lstm        # trains the LSTM, 5 seeds (~100s on CPU, no GPU needed)
python -m training.evaluate_lstm     # the 4 metrics that matter (see ml_notes.md)
python -m training.visualize_results # 4 PNG charts from the above
```
Only re-run `train_lstm` if you change `models/lstm.py`, the loss function,
or the training data/normalization — evaluate/visualize just re-read
whatever's currently in `models/baseline_lstm/`.

## Step 4 — retrieval system (P3)

```bash
python -m retrieval.build_index      # computes features for all 17,025 windows, builds FAISS index (~10s)
python -m retrieval.query            # manual sanity check: prints top-5 analogues for a real heatwave query date
python -m pytest tests/test_retrieval_eligibility.py -v   # automated validation, 6 tests
```
Re-run `build_index` only if `datasets/forecast_windows.parquet` or
`datasets/all_daily.parquet` change (e.g. a Step 8 conditional data fix) —
otherwise the index goes stale relative to the data it was built from.

To query a specific date programmatically:
```python
from retrieval.query import query_analogues
result = query_analogues("2022-04-11", k=5)
print(result)
```

## Step 5 onward — not yet built

Will be added here once Step 5 (joint integration) exists.

## Quick full-repo verification (run everything, confirm nothing is broken)

```bash
python -m pytest tests/ -v
python -m training.evaluate_lstm
python -m retrieval.query
```
This doesn't retrain or rebuild the index — it just confirms the existing
checkpoints, evaluation outputs, and retrieval index all still work
together. Use this after pulling changes from a teammate, or after editing
any file that reads (rather than produces) these artifacts.

## Week 1 (plan v5) — predictions and the G0 comparison

```bash
.venv\Scripts\python.exe -m retrieval.precompute_analogues   # gitignored analogue table (~30 s)
.venv\Scripts\python.exe -m evaluation.predict_v1            # per-seed val predictions -> predictions_v1/val/
.venv\Scripts\python.exe -m evaluation.compare_v1            # G0 table -> evaluation_v2/g0_val_comparison.md
.venv\Scripts\python.exe -m scripts.make_manifest --check    # frozen v1 files unchanged
```
`predict_v1` reads only the frozen checkpoints in `models/frozen/` and refuses the test split.

```bash
.venv\Scripts\python.exe -m evaluation.damped_persistence_v1   # strongest simple baseline -> predictions_v1/val/
.venv\Scripts\python.exe -m evaluation.anen_v1                 # analogue-ensemble baselines -> predictions_v1/val/
.venv\Scripts\python.exe -m evaluation.premise_checks          # trends + retention -> evaluation_v2/premise_report.md
.venv\Scripts\python.exe -m evaluation.mde_simulation          # power / MDE -> evaluation_v2/mde.md (~2 min)
.venv\Scripts\python.exe -m pipeline.download_era5_v2 --dry-run   # then without --dry-run to download (resumable)
```
Run the baselines before `compare_v1` so the G0 table includes them.

## Week 2 (plan v5) — v2 dataset, rolling folds, unified LSTM trainer

```bash
.venv\Scripts\python.exe -m pipeline.build_datasets_v2                        # -> datasets_v2/ (needs data/raw/era5_monthly)
.venv\Scripts\python.exe -m training.train_unified --config configs\A1prime.json   # 10 seeds x 4 folds
.venv\Scripts\python.exe -m training.train_unified --config configs\A2.json
.venv\Scripts\python.exe -m training.train_unified --config configs\A2r.json
```
Options: `--folds f4 --seeds 0 1` for a quick subset; `--threads N` (default: half the cores).
Outputs: `predictions_v2/<run_id>/<fold>.parquet`, a row per fold in `registry/runs.csv`,
gitignored checkpoints in `models/v2/`. `configs/A1_repro.json` reproduces frozen A1 exactly.
