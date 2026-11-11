"""
RS-4 (pre-registered in context/decisions_retrieval.md): C3-hop with the 500 hPa pattern as a direct input. This is
not retrieval; it follows the information principle found in RS-1 and RS-2.

C3-hop is trained exactly as frozen, by training.train_unified.train_one_seed_graph with the run's own config. The
only change is 10 extra Delhi input channels on each of the 14 input days: that day's 500 hPa anomaly field (South
Asia, from retrieval.fold_retrieval.synoptic_daily) scored on its first 10 principal components. The PCA and the
scaling use the fold's training days only. Development folds only; nothing from 2019 on is read.

Run from repo root:
    python -m retrieval_study.z500_direct --family WBGT
    python -m retrieval_study.z500_direct --family Tmax
    python -m retrieval_study.z500_direct --report        -> retrieval_study/results/step4.{md,json}
A fold whose predictions already exist is skipped, so an interrupted run can be resumed.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from training.folds import FOLDS, INPUT_DAYS, SplitArrays

REPO_ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = REPO_ROOT / "retrieval_study" / "predictions"
CKPT_DIR = REPO_ROOT / "retrieval_study" / "checkpoints"
OUT_DIR = REPO_ROOT / "retrieval_study" / "results"
N_PCS = 10
BASE_RUNS = {"Tmax": "A1prime_hw5_C3hop", "WBGT": "A2Lr_hw5_C3hop"}
SUFFIX = "_Z500d"
CONTROL_SUFFIX = "_Z500shuf"  # the pre-registered control: the same dates' scores from another training year
CONTROL_SEED = 20261111


def daily_pcs(fold: str, d: pd.DataFrame, n_pcs: int = N_PCS) -> pd.DataFrame:
    """(dates, n_pcs) each day's weighted 500 hPa anomaly field scored on the first n_pcs principal components of the
    fold's training days, divided by the scores' training-day SD. Pre-2019 dates only (those of d)."""
    from retrieval.fold_retrieval import synoptic_daily
    from training.folds import fold_bounds

    za, w = synoptic_daily(fold, d)
    v = za.to_numpy(dtype=np.float64) * w[None, :]
    train = np.asarray(za.index <= fold_bounds(fold)[0])
    mean = v[train].mean(axis=0)
    _, _, vt = np.linalg.svd(v[train] - mean, full_matrices=False)
    scores = (v - mean) @ vt[:n_pcs].T
    return pd.DataFrame(scores / scores[train].std(axis=0), index=za.index)


def shuffled_pcs(pcs: pd.DataFrame, fold: str, seed: int = CONTROL_SEED) -> pd.DataFrame:
    """The control's scores: every year's dates take the scores of the same calendar dates in another of the fold's
    training years (one random donor year per year, never the year itself; 29 February takes 28 February). The
    season and each 14-day sequence's coherence are kept; what the right day's upper air says is not."""
    from training.folds import fold_bounds

    train_end = fold_bounds(fold)[0]
    years = sorted(set(pcs.index.year))
    train_years = [y for y in years if y <= train_end.year]
    rng = np.random.default_rng([seed, int(fold[1])])
    donor = {y: int(rng.choice([t for t in train_years if t != y])) for y in years}
    src = [pd.Timestamp(year=donor[d.year], month=d.month,
                        day=28 if (d.month == 2 and d.day == 29 and not pd.Timestamp(year=donor[d.year], month=3,
                                                                                          day=1).is_leap_year)
                        else d.day) for d in pcs.index]
    out = pcs.reindex(pd.DatetimeIndex(src))
    if out.isna().any().any():
        raise ValueError("a donor date has no 500 hPa scores")
    return pd.DataFrame(out.to_numpy(), index=pcs.index, columns=pcs.columns)


def with_z500(split: SplitArrays, pcs: pd.DataFrame) -> SplitArrays:
    """The split with the 500 hPa scores of each window's 14 input days appended as extra Delhi channels."""
    pos = pcs.index.get_indexer(pd.DatetimeIndex(split.query_dates))
    if (pos < INPUT_DAYS).any():
        raise ValueError("500 hPa scores missing for some input days")
    z = pcs.to_numpy(dtype=np.float32)[pos[:, None] + np.arange(-INPUT_DAYS, 0)[None, :]]
    return replace(split, X=np.concatenate([split.X, z], axis=2))


def run_family(short: str, folds: list[str], seeds: list[int] | None = None, control: bool = False) -> None:
    """Train and save C3-hop + the 500 hPa input for one family (control=True: the shuffled-year control)."""
    import torch

    from evaluation.predict_v1 import long_frame
    from training import train_unified as tu
    from training.folds import build_fold, fold_daily
    from training.graph_data import build_upstream, load_upstream

    cfg = tu.load_config(REPO_ROOT / "configs" / f"{BASE_RUNS[short]}.json")
    rid = cfg["run_id"] + (CONTROL_SUFFIX if control else SUFFIX)
    seeds = cfg["seeds"] if seeds is None else seeds
    torch.set_num_threads(int(cfg.get("threads", 8)))
    up_table = load_upstream()  # pre-2019 rows only
    for fold in folds:
        out = PRED_DIR / rid / f"{fold}.parquet"
        if out.exists():
            print(f"[{rid} {fold}] already done, skipped", flush=True)
            continue
        t0 = time.time()
        data = build_fold(fold, cfg["target"], cfg["labels"], target_form=cfg["target_form"])
        fit, stop = tu.fit_and_stop_sets(data, cfg["early_stop"])
        d, _ = fold_daily(fold, cfg["target"], cfg["labels"])
        pcs = daily_pcs(fold, d)
        if control:
            pcs = shuffled_pcs(pcs, fold)
        fit, stop, val = (with_z500(s, pcs) for s in (fit, stop, data.val))
        upd = build_upstream(fold, up_table)
        pos_fit, pos_stop, pos_val = (upd.positions(a.query_dates) for a in (fit, stop, val))
        upd_x, upd_adj = torch.from_numpy(upd.x), torch.from_numpy(upd.adj)
        frames, logs = [], []
        for seed in seeds:
            log: dict = {}
            model = tu.train_one_seed_graph(seed, fit, stop, float(cfg["hot_weight"]), cfg["graph"], upd,
                                            pos_fit, pos_stop, log=log, hp=cfg.get("hparams"))
            ckpt = CKPT_DIR / rid / fold / f"seed_{seed}" / "checkpoint.pt"
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt)
            pred = data.to_raw(tu.predict_graph(model, val.X, pos_val, upd_x, upd_adj), "val")
            frames.append(long_frame(rid, seed, data.val.query_dates, pred, data.val.y_raw, data.val.stratum))
            logs.append({"seed": seed, "epochs": len(log.get("val_curve", [])), "finite": bool(log.get("finite")),
                         "best_stop_loss": float(min(log.get("val_curve", [np.nan])))})
            rmse = float(np.sqrt(np.mean(frames[-1]["error"] ** 2)))
            print(f"  [{rid} {fold}] seed {seed}: val RMSE {rmse:.4f} ({time.time() - t0:.0f} s)", flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        (out.parent / f"{fold}_training_log.json").write_text(json.dumps(
            {"n_pcs": N_PCS, "n_features_delhi": int(fit.X.shape[2]), "seeds": logs,
             "wall_seconds": round(time.time() - t0, 1)}, indent=1) + "\n", encoding="utf-8")
        pd.concat(frames, ignore_index=True).to_parquet(out, index=False)
        print(f"[{rid} {fold}] done in {time.time() - t0:.0f} s", flush=True)


# ------------------------------------------------------------------ report


def load(run_dir: Path) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(run_dir / f"{f}.parquet") for f in FOLDS], ignore_index=True)


def report() -> dict:
    from evaluation.compare_v2 import compare, rmse_by

    res = {}
    for short, base in BASE_RUNS.items():
        mine = PRED_DIR / (base + SUFFIX)
        if not all((mine / f"{f}.parquet").exists() for f in FOLDS):
            res[short] = {"status": "not complete"}
            continue
        child = load(mine)
        refs = {"C3-hop": REPO_ROOT / "predictions_v2" / base}
        if short == "WBGT":
            refs["C3-hop + Rz (retrieval)"] = REPO_ROOT / "predictions_v2" / (base + "_Rz")
        control = PRED_DIR / (base + CONTROL_SUFFIX)
        if all((control / f"{f}.parquet").exists() for f in FOLDS):
            refs["the shuffled-year control"] = control
        r = {"rmse": {s: rmse_by(child if s == "all" else child[child["stratum"] == s]) for s in ("all", "extreme")},
             "by_lead": [rmse_by(child[child["lead"] == L]) for L in range(1, 6)], "tests": {}}
        for name, path in refs.items():
            parent = load(path)
            r[f"rmse {name}"] = {s: rmse_by(parent if s == "all" else parent[parent["stratum"] == s])
                                 for s in ("all", "extreme")}
            r[f"by_lead {name}"] = [rmse_by(parent[parent["lead"] == L]) for L in range(1, 6)]
            for s in ("all", "extreme"):
                t = compare(parent, child, s)
                r["tests"][f"vs {name}|{s}"] = {k: t[k] for k in ("parent", "child", "delta", "ci_low", "ci_high",
                                                                 "p_value", "n_clusters")}
        res[short] = r
    return res


def write_report(res: dict) -> None:
    def d(t: dict) -> str:
        mark = " **(CI < 0)**" if t["ci_high"] < 0 else (" (CI > 0)" if t["ci_low"] > 0 else "")
        return f"{t['delta']:+.3f} [{t['ci_low']:+.3f}, {t['ci_high']:+.3f}]{mark}"

    L = ["# Retrieval study, step 4: C3-hop with the 500 hPa pattern as a direct input (RS-4)", "",
         "Pre-registered in `context/decisions_retrieval.md` (RS-4) before any run. Not retrieval: the upper-air "
         "information given to the model directly. Development folds (validation 2007-2018 pooled), 10 seeds x 4 folds; "
         "RMSE in °C (mean of the 10 seeds' RMSE); Δ = new minus old (negative = the direct input helps); paired "
         "cluster jackknife with 10 seeds each.", ""]
    for short, r in res.items():
        L += [f"## {short}", ""]
        if r.get("status"):
            L += [f"Not complete yet ({r['status']}).", ""]
            continue
        L += ["| Compared with | Days | RMSE before | RMSE with the 500 hPa input | Δ [95% CI] |", "|---|---|---|---|---|"]
        for key, t in r["tests"].items():
            name, s = key[3:].split("|")
            L.append(f"| {name} | {'all days' if s == 'all' else 'heatwave days'} | {t['parent']:.3f} | {t['child']:.3f} | "
                     f"{d(t)} |")
        L += ["", "| Lead | " + " | ".join(str(i) for i in range(1, 6)) + " |", "|---|---|---|---|---|---|",
              "| C3-hop | " + " | ".join(f"{x:.3f}" for x in r["by_lead C3-hop"]) + " |",
              "| C3-hop + 500 hPa input | " + " | ".join(f"{x:.3f}" for x in r["by_lead"]) + " |", ""]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "step4.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="RS-4: C3-hop with the 500 hPa pattern as a direct input.")
    ap.add_argument("--family", choices=sorted(BASE_RUNS))
    ap.add_argument("--folds", nargs="*", default=list(FOLDS))
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--control", action="store_true", help="the shuffled-year control run")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args(argv)
    if args.family:
        run_family(args.family, args.folds, args.seeds, control=args.control)
    if args.report:
        res = report()
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "step4.json").write_text(json.dumps(res, indent=2, default=float) + "\n", encoding="utf-8")
        write_report(res)
        print(f"wrote {OUT_DIR / 'step4.md'}")


if __name__ == "__main__":
    main()
