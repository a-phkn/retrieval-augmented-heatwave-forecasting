"""
Graded exposure: share of each working hour a worker must rest, from the ACGIH work-rest screening table
(PROTOCOL_V2 amendment A1, 24 Nov 2026, written after the H-A result).

Source (two copies, values identical, checked 24 Nov 2026):
  ACGIH TLV "Heat Stress and Heat Strain", Table 2, Screening Criteria for TLV and Action Limit (Cal/DOSH copy,
  https://www.dir.ca.gov/dosh/doshreg/Heat-illness-prevention-indoors/heat-tlv.pdf, p.215 by running header);
  CCOHS OSH Answers "Hot Environments - Assessment and Control Measures", Table 1, citing 2026 TLVs and BEIs p.242.
Rows: share of the hour spent working (75-100, 50-75, 25-50, 0-25 %). A WBGT at or below a row's value allows
that row's work share; "-" (None) means no criterion is given, so that row is never allowed.
Required rest share per hour = 0, 0.25, 0.50, 0.75 for the first row allowed, or 1.0 (stop) if none is.
The table is a screening tool, "not intended to prescribe work and recovery periods": here it is an exposure
scale, not advice. Assumes work clothes (clothing adjustment 0).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from v2.exposure import WORK_HOURS

REST_BY_ROW = (0.0, 0.25, 0.50, 0.75)
ACGIH = {  # (75-100 %, 50-75 %, 25-50 %, 0-25 % work), deg C WBGT
    "TLV": {"light": (31.0, 31.0, 32.0, 32.5), "moderate": (28.0, 29.0, 30.0, 31.5),
            "heavy": (None, 27.5, 29.0, 30.5), "very_heavy": (None, None, 28.0, 30.0)},
    "AL": {"light": (28.0, 28.5, 29.5, 30.0), "moderate": (25.0, 26.0, 27.0, 29.0),
           "heavy": (None, 24.0, 25.5, 28.0), "very_heavy": (None, None, 24.5, 27.0)},
}
PRIMARY = ("moderate", "TLV")   # acclimatised moderate work, as H-A's primary


def rest_share(wbgt, work: str = PRIMARY[0], level: str = PRIMARY[1]) -> np.ndarray:
    w = np.asarray(wbgt, float)
    rest = np.ones_like(w)
    for limit, r in reversed(list(zip(ACGIH[level][work], REST_BY_ROW))):   # lower rest rows override
        if limit is not None:
            rest = np.where(w <= limit, r, rest)
    return np.where(np.isnan(w), np.nan, rest)


def daily_rest(hourly: pd.DataFrame, work: str = PRIMARY[0], level: str = PRIMARY[1], col: str = "wbgt",
               by=("cell",)) -> pd.DataFrame:
    """One row per (by..., date): tmax (24 h), rest_hours (sum of rest shares over stamps 06..20),
    stop_hours (hours with rest share 1). Days without all 15 working hours are dropped."""
    h = hourly.assign(date=hourly["time"].dt.normalize(), hour=hourly["time"].dt.hour)
    keys = [*by, "date"]
    tmax = h.groupby(keys)["t"].max().rename("tmax")
    w = h[h["hour"].isin(WORK_HOURS)].assign(rest=lambda x: rest_share(x[col], work, level))
    w = w.assign(stop=(w["rest"] >= 1.0).astype(int))
    g = w.groupby(keys)
    out = pd.DataFrame({"n_hours": g["hour"].size(), "rest_hours": g["rest"].sum(),
                        "stop_hours": g["stop"].sum()}).join(tmax)
    return out[out["n_hours"] == len(WORK_HOURS)].drop(columns="n_hours").reset_index()
