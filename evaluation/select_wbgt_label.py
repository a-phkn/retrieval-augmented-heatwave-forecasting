"""
Choose the WBGT heatwave-label percentile by the pre-registered minimum-sample rule
(decision 2026-10-05; pipeline/labels_v2.py):

    P = the HIGHEST of WBGT_PERCENTILES (99, 98, 97.5, 95, 92.5, 90) whose labels give
        >= 25 episodes pooled over the four rolling-fold validation blocks, each block
        labelled with the threshold from ITS fold's training years.

The rule only counts labelled episodes; no forecast or model result is involved. Primary
variable (decision 2026-10-05): the physical Liljegren WBGT (wbgt_lj_max), the target of the
WBGT models. The same table for the BoM index (wbgt_bom_max) is kept as a sensitivity check
and does not choose P.
Only pre-2019 rows are read.

Writes configs/wbgt_label.json (chosen P, per-fold thresholds, the full count table).
Run from repo root:  python -m evaluation.select_wbgt_label
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.labels_v2 import MIN_POOLED_EPISODES, WBGT_PERCENTILES, label_frame_wbgt, wbgt_threshold
from training.folds import FOLDS, TEST_START, fold_bounds

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "configs" / "wbgt_label.json"
PRIMARY = "wbgt_lj_max"  # decision 2026-10-05: label on the physical WBGT


def load_pre_test() -> pd.DataFrame:
    filt = [("date", "<", TEST_START)]
    v2 = pd.read_parquet(REPO_ROOT / "datasets_v2/all_daily_v2.parquet", filters=filt).set_index("date")
    lj = pd.read_parquet(REPO_ROOT / "datasets_v2/wbgt_liljegren_daily.parquet", filters=filt).set_index("date")
    d = v2[["wbgt_bom_max"]].join(lj[["wbgt_lj_max"]], how="inner").sort_index()
    if d.index.max() >= TEST_START or d.isna().any().any():
        raise ValueError("unexpected test-period rows or gaps")
    return d


def count_table(d: pd.DataFrame, column: str) -> pd.DataFrame:
    """Episodes (and hot days) per fold validation block for every candidate percentile."""
    rows = []
    for p in WBGT_PERCENTILES:
        for fold in FOLDS:
            train_end, val_start, val_end = fold_bounds(fold)
            thr = wbgt_threshold(d.index, d[column], d.index <= train_end, p)
            lab = label_frame_wbgt(d.index, d[column], thr)
            blk = lab.loc[val_start:val_end]
            ids = blk["episode_id_wbgt"]
            rows.append({"percentile": p, "fold": fold, "threshold": round(thr, 3),
                         "episodes": int(ids[ids > 0].nunique()), "hot_days": int(blk["hot_wbgt"].sum())})
    return pd.DataFrame(rows)


def choose(table: pd.DataFrame) -> float | None:
    """Highest percentile with >= MIN_POOLED_EPISODES pooled episodes (None if none)."""
    pooled = table.groupby("percentile")["episodes"].sum()
    for p in WBGT_PERCENTILES:
        if pooled.get(p, 0) >= MIN_POOLED_EPISODES:
            return p
    return None


def main() -> None:
    d = load_pre_test()
    out = {"rule": f"highest percentile in {list(WBGT_PERCENTILES)} with >= {MIN_POOLED_EPISODES} pooled validation episodes",
           "season": "Mar 15 - Sep 30", "primary_variable": PRIMARY}
    for column in (PRIMARY, "wbgt_bom_max"):
        t = count_table(d, column)
        p = choose(t)
        pooled = t.groupby("percentile")[["episodes", "hot_days"]].sum().loc[list(WBGT_PERCENTILES)]
        out[column] = {"chosen_percentile": p,
                       "thresholds_by_fold": t[t["percentile"] == p].set_index("fold")["threshold"].to_dict() if p else None,
                       "pooled_by_percentile": {str(k): v for k, v in pooled.to_dict("index").items()},
                       "by_fold": t.to_dict("records")}
        print(f"{column}: chosen P = {p}")
        print(pooled.to_string())
    out["chosen_percentile"] = out[PRIMARY]["chosen_percentile"]
    out["thresholds_by_fold"] = out[PRIMARY]["thresholds_by_fold"]
    OUT_PATH.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"-> {OUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
