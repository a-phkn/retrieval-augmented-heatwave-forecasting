"""
Daily exposure measures from hourly WBGT (PROTOCOL_V2 Section 1.2), shared by H-A, H-B and H-E.

Hour convention: hour-ending stamps (radiation is the mean of the preceding hour), so stamp h stands for
(h-1, h]. Working hours: stamps 06..20 IST. An hour is unsafe if WBGT >= the limit; an unsafe day has at
least 2 CONSECUTIVE unsafe working hours (NIOSH 2016, p.3: "two sequential measurements exceed").
ISO 7243:2017 Table A.1 limits by work class (deg C WBGT), acclimatised / unacclimatised.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WORK_HOURS = tuple(range(6, 21))
ISO_LIMITS = {  # (acclimatised, unacclimatised), ISO 7243:2017 Table A.1
    "low": (30.0, 29.0), "moderate": (28.0, 26.0), "high": (26.0, 23.0), "very_high": (25.0, 20.0),
}
PRIMARY_LIMIT = ISO_LIMITS["moderate"][0]


def longest_run(flags: np.ndarray) -> int:
    best = run = 0
    for f in flags:
        run = run + 1 if f else 0
        best = max(best, run)
    return best


def daily_exposure(hourly: pd.DataFrame, limit: float = PRIMARY_LIMIT, by=("cell",)) -> pd.DataFrame:
    """hourly: columns time (IST), t, wbgt and the `by` keys. Returns one row per (by..., date) with
    tmax (24-hour max temperature), wbgt_max (working hours), unsafe_hours, longest_unsafe_run, unsafe_day.
    Days without all 15 working hours are dropped."""
    h = hourly.assign(date=hourly["time"].dt.normalize(), hour=hourly["time"].dt.hour)
    keys = [*by, "date"]
    tmax = h.groupby(keys)["t"].max().rename("tmax")
    w = h[h["hour"].isin(WORK_HOURS)].sort_values([*keys, "hour"]).reset_index(drop=True)
    w["unsafe"] = (w["wbgt"] >= limit).astype(int)
    # vectorised longest run: a new run starts at every change of flag or of day
    new_day = (w[keys] != w[keys].shift()).any(axis=1)
    run_id = ((w["unsafe"] != w["unsafe"].shift()) | new_day).cumsum()
    run_len = w.groupby(run_id)["unsafe"].transform("sum")
    w["run"] = run_len * w["unsafe"]
    g = w.groupby(keys)
    out = pd.DataFrame({
        "n_hours": g["hour"].size(),
        "wbgt_max": g["wbgt"].max(),
        "unsafe_hours": g["unsafe"].sum(),
        "longest_unsafe_run": g["run"].max(),
    }).join(tmax)
    out = out[out["n_hours"] == len(WORK_HOURS)].drop(columns="n_hours")
    out["unsafe_day"] = out["longest_unsafe_run"] >= 2
    return out.reset_index()
