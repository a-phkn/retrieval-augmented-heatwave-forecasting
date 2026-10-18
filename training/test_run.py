"""
The one-time test run (docs/CONFIRMATORY_PROTOCOL.md). Refuses to start unless the test lock is open:
the protocol is frozen as the git tag `test-lock` and the code is the frozen code (training.test_lock).

Every model is trained ONCE on the "test" fold (training.folds): fit 1980-2016, early stopping on
2017-2018, climatology / labels / normalisation from 1980-2018, forecasts for 2019-01-01 .. 2026-09-06,
with exactly the recipe and seeds of its development config. A model whose test forecasts already exist is
never retrained (one evaluation run; re-scoring the saved forecasts is allowed).

  main    C3-hop and the Delhi-only control for both families (with their early-stopping-block forecasts,
          used for recalibration, warning probabilities and range forecasts), plus both ridges
  extra   C2, U1 (tuned), C3-pool (tuned), LSTM + Rg, C3-hop + Rg, both families (exploratory)
  rz      the Rz side-check: 2019-2021 z500 download, then A2Lr_hw5_C3hop_Rz on the "test2021" fold

Outputs: predictions_test/<run>/<fold>.parquet (+ _stop / _analogues / _training_log), checkpoints in
models/test/ (gitignored), registry/test_runs.csv (code and data hashes, git commit; as registry/runs.csv).

Run from repo root (after the freeze):  python -m training.test_run --stage main|extra|rz|all
"""
from __future__ import annotations

import argparse
import subprocess
import sys

import pandas as pd

import training.train_unified as tu
from training import test_lock
from training.folds import TEST_FOLD

PRED_TEST = tu.REPO_ROOT / "predictions_test"
MODEL_TEST = tu.REPO_ROOT / "models" / "test"
REGISTRY_TEST = tu.REPO_ROOT / "registry" / "test_runs.csv"
FAMILIES = {"WBGT (physical)": ("A2Lr_hw5", "wbgt_lj_max", "wbgt"), "Tmax": ("A1prime_hw5", "t_max", "v2")}
MAIN = ["A2Lr_hw5_C3hop", "A2Lr_hw5", "A1prime_hw5_C3hop", "A1prime_hw5"]
EXTRA = [f"{c}_{m}" for m in ("C2", "U1_tuned", "C3pool_tuned", "Rg", "C3hop_Rg") for c in ("A2Lr_hw5", "A1prime_hw5")]
RZ_RUN, RZ_FOLD = "A2Lr_hw5_C3hop_Rz", "test2021"


def done(run_id: str, fold: str = TEST_FOLD) -> bool:
    return (PRED_TEST / run_id / f"{fold}.parquet").exists()


def train(run_id: str, fold: str = TEST_FOLD, stop_forecasts: bool = False) -> None:
    if done(run_id, fold):
        print(f"[{run_id} {fold}] test forecasts exist: not retrained (one evaluation run)", flush=True)
        return
    cfg = tu.load_config(tu.REPO_ROOT / "configs" / f"{run_id}.json")
    print(f"[{run_id} {fold}] training {len(cfg['seeds'])} seeds", flush=True)
    tu.run(cfg, folds=[fold], out_dir=PRED_TEST, model_dir=MODEL_TEST, registry=REGISTRY_TEST,
           threads=cfg.get("threads"), score_stop=stop_forecasts)


def ridges() -> None:
    """Both ridges (evaluation.ridge_baselines, unchanged) fitted on 1980-2018, forecasting the test period."""
    from evaluation.gd0_upstream_signal import NODE_IDS as GD0_NODES
    from evaluation.ridge_baselines import fold_frames
    from training.graph_data import load_upstream

    up = load_upstream(include_test=True)  # test_lock.require() inside
    tmax = up[[f"temperature_2m_max__{n}" for n in GD0_NODES]]
    tmax.columns = list(GD0_NODES)
    for control, target, labels in FAMILIES.values():
        out = {name: PRED_TEST / f"{control}_{name}" / f"{TEST_FOLD}.parquet" for name in ("ridge", "ridge_hw5")}
        if all(p.exists() for p in out.values()):
            print(f"[{control} ridges] test forecasts exist: not refitted", flush=True)
            continue
        for name, df in fold_frames(TEST_FOLD, target, labels, tmax).items():
            out[name].parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out[name], index=False)
        print(f"[{control} ridges] done", flush=True)


def rz_side_check() -> None:
    from retrieval.fold_retrieval import Z500_TEST_PATH

    if not Z500_TEST_PATH.exists():
        subprocess.run([sys.executable, "-m", "pipeline.download_era5_z500", "--test-years"], cwd=tu.REPO_ROOT, check=True)
    train(RZ_RUN, RZ_FOLD)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="The one-time test run (after the protocol freeze).")
    ap.add_argument("--stage", choices=("main", "extra", "rz", "all"), default="main")
    stage = ap.parse_args(argv).stage
    test_lock.require()  # refuses before anything is read
    print(f"test lock open at {pd.Timestamp.now():%Y-%m-%d %H:%M}; stage {stage}", flush=True)
    if stage in ("main", "all"):
        for run_id in MAIN:
            train(run_id, stop_forecasts=True)
        ridges()
    if stage in ("extra", "all"):
        for run_id in EXTRA:
            train(run_id)
    if stage in ("rz", "all"):
        rz_side_check()
    print(f"stage {stage} complete", flush=True)


if __name__ == "__main__":
    main()
