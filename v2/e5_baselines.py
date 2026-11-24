"""
Post-hoc comparison for E5 (added 24 Nov AFTER the E5 result; descriptive, not a test): would a trigger without
the forecast model also meet the miss bound, and at what false-alarm rate? Same CRC procedure, same folds (each
calibrated on its own inner years, evaluated on its validation years), same event (rest-hours >= R), lead 1..5.
  persistence: score = observed rest-hours on the issue day (target - lead)
  calendar:    score = 1980-2006 day-of-year mean rest-hours (31-day smoothed), i.e. no weather information
Writes results_v2/e5_baselines.json.

Run from repo root:  python -m v2.e5_baselines
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from v2.calibration_v2 import ALPHA, MAX_FALSE_ALARM, in_season
from v2.risk_control import crc_threshold
from v2.run_calibration_v2 import R, RESULTS, rest_hours
from v2.window_test import doy_normal

FOLD_YEARS = {"f1": ((2005, 2006), (2007, 2009)), "f2": ((2008, 2009), (2010, 2012)),
              "f3": ((2011, 2012), (2013, 2015)), "f4": ((2014, 2015), (2016, 2018))}   # inner, validation


def scores(rest: pd.Series, kind: str, lead: int) -> pd.Series:
    if kind == "persistence":
        return rest.shift(lead, freq="D").reindex(rest.index)
    if kind == "calendar":
        clim = doy_normal(rest, (1980, 2006))
        return pd.Series(clim.reindex(rest.index.dayofyear).to_numpy(), index=rest.index)
    raise ValueError(kind)


def run(rest: pd.Series, kind: str, lead: int, alpha: float = ALPHA) -> dict:
    ev = rest >= R
    sc = scores(rest, kind, lead)
    fire, event = [], []
    for (i0, i1), (v0, v1) in FOLD_YEARS.values():
        def pick(a, b):
            m = (rest.index.year >= a) & (rest.index.year <= b) & in_season(rest.index) & sc.notna().to_numpy()
            return sc[m].to_numpy(), ev[m].to_numpy()
        s_cal, e_cal = pick(i0, i1)
        lam = crc_threshold(s_cal, e_cal, alpha)
        s_val, e_val = pick(v0, v1)
        fire.append(s_val >= lam)
        event.append(e_val)
    f, e = np.concatenate(fire), np.concatenate(event)
    miss = float(1 - f[e].mean())
    far = float(f[~e].mean())
    return {"kind": kind, "lead": lead, "alpha": alpha, "miss_rate": miss, "false_alarm_rate": far,
            "peirce_skill": 1 - miss - far, "fire_rate": float(f.mean()), "passes_false_alarm_rule": far <= MAX_FALSE_ALARM,
            "event_days": int(e.sum())}


def main() -> int:
    rest = rest_hours()
    rest = rest[rest.index.year <= 2018]
    out = [run(rest, k, lead) for k in ("persistence", "calendar") for lead in range(1, 6)]
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "e5_baselines.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(pd.DataFrame(out).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
