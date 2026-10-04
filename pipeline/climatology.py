"""
Train-only day-of-year climatology, reusable per fold (plan v5, Week 2).

Reimplements exactly the v1 method in prepare_datasets.py so v1 numbers are reproduced
(tested): for each day-of-year 1..366, the mean and population SD (ddof=0) of the
variable over TRAINING dates whose day-of-year lies within +/- 7 days (circular over a
366-day year). Dates map to their climatology by day-of-year (v1 convention, including
its handling of leap years).

Generalised from v1 in two ways: any variable (Tmax, BoM WBGT, wet-bulb, ...) and any
training mask -- so each rolling-origin fold computes its climatology from its own
training years only (no leakage of a validation block into anomalies or labels).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CLIM_WINDOW_DAYS = 7
DAYS_IN_YEAR = 366


def doy_climatology(values: pd.Series, train_mask, window: int = CLIM_WINDOW_DAYS) -> pd.DataFrame:
    """values: date-indexed series; train_mask: boolean array/Series aligned with values.
    Returns a frame indexed by doy 1..366 with columns clim_mean, clim_std."""
    train_mask = np.asarray(train_mask, dtype=bool)
    if train_mask.shape[0] != len(values):
        raise ValueError("train_mask must align with values")
    if not train_mask.any():
        raise ValueError("training mask selects no dates")
    train = values[train_mask]
    doy = pd.DatetimeIndex(train.index).dayofyear.to_numpy()
    vals = train.to_numpy(dtype=np.float64)
    if np.isnan(vals).any():
        raise ValueError("training values contain NaN")
    rows = []
    for d in range(1, DAYS_IN_YEAR + 1):
        dist = np.abs(doy - d)
        sel = np.minimum(dist, DAYS_IN_YEAR - dist) <= window
        rows.append((d, vals[sel].mean(), vals[sel].std(ddof=0)))
    return pd.DataFrame(rows, columns=["doy", "clim_mean", "clim_std"]).set_index("doy")


def apply_climatology(dates, clim: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(clim_mean, clim_std) for each date, looked up by day-of-year."""
    doy = pd.DatetimeIndex(pd.to_datetime(dates)).dayofyear
    looked = clim.reindex(doy)
    if looked.isna().any().any():
        raise ValueError("climatology missing for some day-of-year")
    return looked["clim_mean"].to_numpy(), looked["clim_std"].to_numpy()
