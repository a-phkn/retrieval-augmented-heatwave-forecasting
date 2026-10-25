# Known deviations and issues (disclosed, not changed)

The code under `training/ models/ pipeline/ retrieval/ evaluation/ configs/` is frozen at the git tag `test-lock`
(2026-10-19). Changing it would break the test lock (`training/test_lock.py`) and the claim that the test used exactly
the pre-registered code. So the issues below, found in the critical reviews of 2026-10-22, are **disclosed here instead
of fixed**. None of them changes a reported number. Fixes for a future version are noted where useful.

Issues in files outside the frozen folders (docs, tests, CI, `post_test/`) were fixed directly on 2026-10-25.

## 1. Test-period handling

| Issue | Effect | Why it is harmless here |
|---|---|---|
| `retrieval/fold_retrieval.py` `humid_daily` reads the whole humid-air file (to 2026-09-06), then filters it to the fold's dates. The rule is "2019+ is never read" and "every reader calls the lock first". | 2019+ rows sit in memory briefly in development folds. | They are dropped before any statistic is computed (`h.index < _read_end(fold)`), and the tests that move test-period values show nothing fitted depends on them. Future fix: call `_read_end(fold)` first and read with a date filter. |
| The lock covers **code** folders, not **data** folders (`datasets_v2/`, `splits/`, `data/external/`). | A data file could in principle change without the lock noticing. | Checked: no pre-2019 input changed since the tag. The only change is the 2019–2021 500 hPa extract, downloaded by the guarded downloader during the test run and committed with the results (commit 7d91daa). Since 2026-10-25 all these inputs and the test outputs are fingerprinted in `data/TEST_MANIFEST.json`, checked by a test and in CI. |
| `training/test_run.py` only checks that a model's test forecasts exist before skipping it. | Deleting a file in `predictions_test/` would allow a second "one-time" run of that model. | Since 2026-10-25, `data/TEST_MANIFEST.json` and `tests/test_test_manifest.py` fail on any missing or changed test output, and a test re-derives H1–H4 from the committed forecasts. |
| Two early v1 sanity scripts looked at the test period before the freeze: `check_heatwaves.py` (the hottest 2019 and 2022 days, and a plot) and `retrieval/query.py` (analogues for one 2022 heatwave query date). | Some test-period temperatures were seen in September 2026. | No model forecast or score for the test period was seen before 2026-10-19, and no design choice used these values. "Used once" in the docs means scored once. |
| `process_weather.py` and `check_heatwaves.py` (v1 root scripts) can rewrite two frozen v1 files and lack the safety check `prepare_datasets.py` has. | A rerun could overwrite `data/processed/weather_daily.parquet` or `data/processing_log.json`. | `scripts/make_manifest.py --check` (CI) fails on any change to those files. |

## 2. Frozen code quality

| Issue | Where | Notes |
|---|---|---|
| Two of the four training loops do not guard against "no epoch had a finite validation loss" and would fail with an unclear `TypeError`. | `training/train_unified.py`, `train_one_seed` and `train_one_seed_ra` (the graph versions do guard) | Never happened: every registry run finished. Future fix: one shared loop with the guard. |
| `np.datetime64("NaT")` without a unit raises a DeprecationWarning (the suite's only warning); a future NumPy will raise an error. | `training/train_unified.py` line 634 | Works with the pinned NumPy 2.5.3. Future fix: `np.datetime64("NaT", "ns")`. |
| Duplicated code: the ~25-line `fold_family` setup copied in four screen scripts; four near-identical early-stopping loops; 11 scripts import `OUT_DIR` from `retrieval_information_check`; 8 import private helpers `_ci` / `_fmt_p` from `compare_v2`; 6 calls use `FoldRetriever`'s private state. | `evaluation/`, `training/train_unified.py`, `retrieval/fold_retrieval.py` | Harder to maintain, not wrong. Future fix: `evaluation/common.py` and one training-loop helper. |
| Docstring out of date: lists 5 retrieval modes (there are 8), names only folds f1–f4, and says "the test split is never built". | `training/train_unified.py` (module docstring) | The test fold exists and is guarded by the lock; the modes are listed in `retrieval/fold_retrieval.py`. |
| Unused imports. | `evaluation/test_evaluation.py` (`fold_daily`), `evaluation/calibration_check.py`, `evaluation/warning_retrieval.py`, `evaluation/select_wbgt_label.py`, `retrieval/fold_retrieval.py`, `retrieval/build_index.py`, `training/baselines.py`, `nn` in three v1 trainers | No effect. Unused imports in tests were removed. |
| Network retries without a wait, and the coordinate read uses only chunk 0. | `evaluation/graphcast_yardstick.py` | The extract is committed (`data/external/`), so this code does not need to run again. |
| `torch.load` without `weights_only`. | `training/evaluate_lstm.py`, `training/evaluate_retrieval_lstm.py` (v1) | Safe: the pinned torch 2.14.1 defaults to `weights_only=True`. |

## 3. Protocol and analysis

| Issue | What was done |
|---|---|
| The protocol calls the 10-seed ensemble the deliverable forecast and says it will be reported, but the test report has it only for the case studies. | Reported after the test, labelled as such: `evaluation_post_test/test_period_extras.md`. |
| Damped persistence, a development baseline, is not in the test report. | Not added; persistence and climatology are. |
| About 42 exploratory test comparisons are shown without a multiple-testing correction. | They are descriptive only. Only H1–H4 (Holm-corrected) are claims. |
| H1 compares C3-hop (settings tuned in development) with a Delhi-only LSTM whose settings were not retuned. | Disclosed in the README and the guide. The untuned graph relative C2 shows the same gain. |
| H3's ridge sees fewer inputs (Tmax anomalies only), and about 96% of H3's gap is a smaller cold bias. | Disclosed; the bias split is in `evaluation_post_test/test_period_extras.md`. |
| 47 development registry rows have `git_dirty=True`, including `A2Lr_hw5_Rg` (the development evidence for H4) and other retrieval rungs. | The code and data hashes are recorded in the same rows, so what ran is still auditable. Disclosed for G3 in `context/decisions.md`; noted here for the rest. |
| C3-hop was chosen after the development results, over the U1 tie-break rule. | Disclosed in `context/decisions.md` (2026-10-12) and the guide. |
| Hot weight 5 was a disclosed deviation from the sweep rule, which picked 1. | Disclosed in `context/decisions.md`; wording in `docs/MODELS_EXPLAINED.md` corrected. |
| GraphCast yardstick: our 2018 scores come from models whose design was chosen on 2016–2018, which flatters us slightly. | Does not change the conclusion (GraphCast ~40% better). |
| All truth is ERA5 (a 9-cell average), with no station check; the alert mapping applies IMD-style rules to it. | Disclosed. A station comparison (Safdarjung) needs a new data download and is left as an option. |
