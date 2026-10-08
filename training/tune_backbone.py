"""
The one graph tuning round (context/decisions.md, 2026-10-07 and the C3-pool arm added
2026-10-08 after the G-D3 result). Per family (Tmax: A1prime_hw5; physical WBGT: A2Lr_hw5) and
arm, a fixed grid of 8 combinations is trained with 3 seeds on the 4 folds, and the combination
with the lowest mean all-days RMSE (°C) on the INNER early-stopping block (each fold's last 2
training years) is selected. Validation-block (2007-2018) forecasts are written by the trainer
but never read here.

Arms: "C3" (static edges, Delhi readout; the G-D3 candidate), "C3pool" (static edges, pooled
readout; added after the result, disclosed), "U1" (LSTM + flattened upstream).
Grids: graph hidden {32, 64}, U1 hidden {64, 128}; both lr {1e-3, 3e-4} x dropout {0, 0.2}.

Usage (repo root):
  python -m training.tune_backbone configs            # write configs/tuning/*.json + the queue file
  python -m training.tune_backbone run --config configs/tuning/<id>.json --folds f1
  python -m training.tune_backbone select             # selection report + 10-seed *_tuned configs
"""
from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

import training.train_unified as tu
from training.folds import FOLDS

CONFIG_DIR = tu.REPO_ROOT / "configs"
TUNE_CONFIG_DIR = CONFIG_DIR / "tuning"
TUNE_PRED_DIR = tu.PRED_DIR / "tuning"
TUNE_MODEL_DIR = tu.MODEL_DIR / "tuning"
TUNE_REGISTRY = tu.REGISTRY.parent / "tuning_runs.csv"
REPORT_DIR = tu.REPO_ROOT / "evaluation_v2"
TUNE_QUEUE = CONFIG_DIR / "week5_tuning_queue.txt"
FINAL_QUEUE = CONFIG_DIR / "week5_tuned_queue.txt"

FAMILIES = ("A1prime_hw5", "A2Lr_hw5")
ARMS = ("C3", "C3pool", "U1")
TUNE_SEEDS, FINAL_SEEDS = [0, 1, 2], list(range(10))
HIDDEN = {"C3": (32, 64), "C3pool": (32, 64), "U1": (64, 128)}
LRS, DROPOUTS = (1e-3, 3e-4), (0.0, 0.2)


def _lr_tag(lr: float) -> str:
    return f"{lr:.0e}".replace("e-0", "e-")  # 1e-3, 3e-4


def grid(arm: str) -> list[dict]:
    """The 8 hparams combinations of an arm, in a fixed order (ties go to the earlier one)."""
    return [{"hidden": h, "lr": lr, "dropout": d} for h, lr, d in product(HIDDEN[arm], LRS, DROPOUTS)]


def base_config(family: str, arm: str) -> dict:
    """The arm's untuned config: {family}_C3 or {family}_U1 as run, plus the pooled readout for C3pool."""
    cfg = json.loads((CONFIG_DIR / f"{family}_{'U1' if arm == 'U1' else 'C3'}.json").read_text(encoding="utf-8"))
    if arm == "C3pool":
        cfg["graph"] = {**cfg["graph"], "readout": "pool"}
    return cfg


def tuning_configs(family: str, arm: str) -> list[dict]:
    out = []
    for hp in grid(arm):
        cfg = base_config(family, arm)
        cfg.update(run_id=f"{family}_{arm}_h{hp['hidden']}_lr{_lr_tag(hp['lr'])}_d{hp['dropout']:g}",
                   seeds=TUNE_SEEDS, hparams=hp,
                   description=f"Graph tuning round (decisions.md 2026-10-07/08): {arm} of {family}, "
                               f"hparams {hp}; selection on the inner 2-year block only.")
        out.append(cfg)
    return out


def write_tuning_configs() -> list[str]:
    TUNE_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    ids = []
    for family, arm in product(FAMILIES, ARMS):
        for cfg in tuning_configs(family, arm):
            (TUNE_CONFIG_DIR / f"{cfg['run_id']}.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
            ids.append(cfg["run_id"])
    TUNE_QUEUE.write_text("\n".join(ids) + "\n", encoding="utf-8")
    return ids


def inner_score(run_dir: Path, n_seeds: int = len(TUNE_SEEDS)) -> float:
    """Mean over the 4 folds of the fold's mean per-seed all-days RMSE on the inner block.
    Reads only {fold}_stop.parquet; raises if a fold or seed is missing."""
    per_fold = []
    for fold in FOLDS:
        df = pd.read_parquet(run_dir / f"{fold}_stop.parquet")
        if df["seed"].nunique() != n_seeds:
            raise ValueError(f"{run_dir.name} {fold}: expected {n_seeds} seeds, got {df['seed'].nunique()}")
        per_fold.append(df.groupby("seed")["error"].apply(lambda e: np.sqrt(np.mean(e.to_numpy() ** 2))).mean())
    return float(np.mean(per_fold))


def select(pred_dir: Path = TUNE_PRED_DIR) -> dict:
    """{family: {arm: {"scores": {run_id: score}, "selected": run_id, "hparams": {...}}}}."""
    out: dict = {}
    for family, arm in product(FAMILIES, ARMS):
        cfgs = tuning_configs(family, arm)
        scores = {c["run_id"]: inner_score(pred_dir / c["run_id"]) for c in cfgs}
        best = min(cfgs, key=lambda c: scores[c["run_id"]])  # min keeps the first of equal scores
        out.setdefault(family, {})[arm] = {"scores": scores, "selected": best["run_id"], "hparams": best["hparams"]}
    return out


def tuned_config(family: str, arm: str, hp: dict) -> dict:
    cfg = base_config(family, arm)
    cfg.update(run_id=f"{family}_{arm}_tuned", seeds=FINAL_SEEDS, hparams=hp,
               description=f"Tuned {arm} of {family} (graph tuning round, decisions.md 2026-10-07/08): "
                           f"hparams {hp} selected on the inner 2-year block, 3 seeds; this run: 10 seeds.")
    return cfg


def write_selection(sel: dict) -> list[str]:
    ids, lines = [], ["# Graph tuning round: selection on the inner 2-year block", "",
                      "Mean all-days RMSE (°C) on each fold's last 2 training years, 3 seeds, averaged over the 4 "
                      "folds. No 2007-2018 validation number was read. Selected = lowest (ties: earlier in the grid).",
                      ""]
    for family in FAMILIES:
        for arm in ARMS:
            r = sel[family][arm]
            cfg = tuned_config(family, arm, r["hparams"])
            (CONFIG_DIR / f"{cfg['run_id']}.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
            ids.append(cfg["run_id"])
            lines += [f"## {family} {arm}: selected `{r['selected']}`", "", "| Combination | Inner RMSE |", "|---|---|"]
            lines += [f"| {k}{' **(selected)**' if k == r['selected'] else ''} | {v:.4f} |" for k, v in r["scores"].items()]
            lines.append("")
    FINAL_QUEUE.write_text("\n".join(ids) + "\n", encoding="utf-8")
    (REPORT_DIR / "graph_tuning.md").write_text("\n".join(lines), encoding="utf-8")
    (REPORT_DIR / "graph_tuning.json").write_text(json.dumps(sel, indent=2) + "\n", encoding="utf-8")
    return ids


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="The one graph tuning round (decisions.md 2026-10-07/08).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("configs")
    r = sub.add_parser("run")
    r.add_argument("--config", required=True, type=Path)
    r.add_argument("--folds", nargs="+", choices=list(FOLDS))
    sub.add_parser("select")
    args = ap.parse_args(argv)
    if args.cmd == "configs":
        print(f"{len(write_tuning_configs())} tuning configs")
    elif args.cmd == "run":
        cfg = tu.load_config(args.config)
        tu.run(cfg, args.folds, out_dir=TUNE_PRED_DIR, model_dir=TUNE_MODEL_DIR, threads=cfg.get("threads"),
               score_stop=True, registry=TUNE_REGISTRY)
    else:
        print(write_selection(select()))


if __name__ == "__main__":
    main()
