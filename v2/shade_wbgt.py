"""
Shade WBGT (PROTOCOL_V2 amendment A1): the worker is shaded from the direct beam, the sky is not.
Same Liljegren physics as pipeline/wbgt_liljegren.py (frozen, imported unchanged):
  - 2 m wind from the ACTUAL global radiation (the atmosphere's stability does not change because one worker is
    in shade);
  - globe and natural wet bulb with solar = diffuse only and fdir = 0;
  - WBGT = 0.7 Tnwb + 0.2 Tg + 0.1 Ta, the same outdoor weights as sun WBGT.
Correction (24 Nov, after the first run): the first version used the no-solar-load formula 0.7 Tnwb + 0.3 Tg.
A shaded worker still has a diffuse solar load, and with that formula shade WBGT exceeded sun WBGT by up to
1.62 C on bright overcast hours (globe well above air temperature, almost no direct beam). With the outdoor
weights, shade <= sun whenever the direct beam is >= 0 (each component is lower or equal).
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
        out[s:s + chunk] = 0.7 * tnwb + 0.2 * tg + 0.1 * (tk - 273.15)
    if np.isnan(out).any():
        raise ValueError(f"shade solver failed for {int(np.isnan(out).sum())} hours")
    return out
