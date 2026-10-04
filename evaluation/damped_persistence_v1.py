"""
Damped anomaly persistence baseline (plan v5: strengthens A0, the reference floor).

    forecast(lead L) = clim_mean(target) + clim_std(target) * phi_L * z_last
    z_last = standardised Tmax anomaly of the last input day (query_date - 1)
    phi_L  = least-squares slope (through the origin) of z(target, lead L) on z_last,
             fitted on TRAIN windows only -- 5 numbers in total.

Why: plain persistence repeats yesterday's temperature, and climatology ignores it. The
damped version keeps the part of yesterday's anomaly that historically persists at each
lead. It is the honest "how much is just persistence?" baseline: an independent review
(2026-10-04) found it beats climatology, the analogue ensemble and both v1 neural models
on val, and that the analogue ensemble's skill sits in leads 1-2.

Writes predictions_v1/val/damped_persistence.parquet (predict_v1 format) and the fitted
phi values to evaluation_v2/damped_persistence_phi.json. Test split never read.

Run from repo root:  python -m evaluation.damped_persistence_v1
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from evaluation.predict_v1 import LEADS, OUT_DIR, REPO_ROOT, long_frame
from training.data import _load_daily, _load_windows, build_split_arrays, build_split_target_stratum

PHI_PATH = REPO_ROOT / "evaluation_v2" / "damped_persistence_phi.json"


def _z_series(daily: pd.DataFrame) -> np.ndarray:
    return (daily["t_max_anomaly"] / daily["clim_std_t_max"]).to_numpy()


def _design(daily: pd.DataFrame, query_dates: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """z_last (n,), z_target (n, LEADS), and target positions (n, LEADS)."""
    z = _z_series(daily)
    q = daily.index.get_indexer(pd.DatetimeIndex(query_dates))
    if (q < 1).any():
        raise ValueError("query dates must have a previous day in all_daily")
    pos = q[:, None] + np.arange(LEADS)[None, :]
    return z[q - 1], z[pos], pos


def fit_phi(daily: pd.DataFrame, windows: pd.DataFrame) -> np.ndarray:
    """phi_L for L = 1..5 from train windows only."""
    train_q = windows.loc[windows["split"] == "train", "query_date"]
    z_last, z_tgt, _ = _design(daily, train_q)
    return (z_last[:, None] * z_tgt).sum(axis=0) / np.sum(z_last**2)


def run(split: str = "val") -> pd.DataFrame:
    if split not in ("train", "val"):
        raise ValueError("test split is locked")
    daily, windows = _load_daily(), _load_windows()
    phi = fit_phi(daily, windows)
    _, y_raw, query_dates = build_split_arrays(split, daily, windows)
    z_last, _, pos = _design(daily, query_dates)
    pred = daily["clim_mean_t_max"].to_numpy()[pos] + daily["clim_std_t_max"].to_numpy()[pos] * (phi[None, :] * z_last[:, None])
    df = long_frame("damped_persistence", 0, query_dates, pred, y_raw, build_split_target_stratum(split, daily, windows))

    (OUT_DIR / split).mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / split / "damped_persistence.parquet"
    df.to_parquet(path, index=False)
    PHI_PATH.parent.mkdir(parents=True, exist_ok=True)
    PHI_PATH.write_text(json.dumps({"fitted_on": "train windows", "phi_by_lead": phi.tolist()}, indent=2) + "\n", encoding="utf-8")
    rmse = float(np.sqrt(np.mean(df["error"] ** 2)))
    print(f"  phi by lead: {np.round(phi, 3).tolist()}")
    print(f"  [{split}] damped_persistence rows={len(df)} RMSE={rmse:.4f} -> {path.relative_to(REPO_ROOT)}")
    return df


if __name__ == "__main__":
    run()
