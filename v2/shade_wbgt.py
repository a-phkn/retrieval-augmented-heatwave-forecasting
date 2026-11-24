"""
Shade WBGT (PROTOCOL_V2 amendment A1): the worker is shaded from the direct beam, the sky is not.
Same Liljegren physics as pipeline/wbgt_liljegren.py (frozen, imported unchanged):
  - 2 m wind from the ACTUAL global radiation (the atmosphere's stability does not change because one worker is
    in shade);
  - globe and natural wet bulb with solar = diffuse only and fdir = 0;
  - WBGT = 0.7 Tnwb + 0.3 Tg, the shaded/indoor formula given by ACGIH (Cal/DOSH copy, Section 2) and CCOHS.
This is an approximation (ground-reflected beam is dropped with the beam); it is the lower bracket, sun WBGT
the upper.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.wbgt_liljegren import globe_temperature, natural_wet_bulb, wind_2m


def shade_wbgt(df: pd.DataFrame, chunk: int = 200_000) -> np.ndarray:
    """df columns as written by pipeline.build_wbgt_liljegren.cell_hourly_wbgt: t, rh, pressure, wind, ghi,
    diffuse, cosz."""
    out = np.empty(len(df))
    for s in range(0, len(df), chunk):
        d = df.iloc[s:s + chunk]
        tk = d["t"].to_numpy(float) + 273.15
        rh = np.clip(d["rh"].to_numpy(float), 0.5, 100.0) / 100.0
        p, cosz = d["pressure"].to_numpy(float), d["cosz"].to_numpy(float)
        ghi = np.maximum(d["ghi"].to_numpy(float), 0.0)
        dif = np.clip(d["diffuse"].to_numpy(float), 0.0, ghi)
        u2 = wind_2m(d["wind"].to_numpy(float), 10.0, cosz, ghi, urban=True)
        tg = globe_temperature(tk, rh, p, u2, dif, 0.0, cosz) - 273.15
        tnwb = natural_wet_bulb(tk, rh, p, u2, dif, 0.0, cosz) - 273.15
        out[s:s + chunk] = 0.7 * tnwb + 0.3 * tg
    if np.isnan(out).any():
        raise ValueError(f"shade solver failed for {int(np.isnan(out).sum())} hours")
    return out
