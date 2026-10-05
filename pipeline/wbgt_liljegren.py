"""
Physical (Liljegren) outdoor WBGT from hourly reanalysis data (plan v5, Week 3).

    WBGT = 0.7 * Tnwb + 0.2 * Tg + 0.1 * Ta
    Tnwb natural wet-bulb temperature, Tg black-globe temperature, Ta air temperature.
Tg and Tnwb come from heat-balance equations for a wetted wick and a black globe in sun and
wind (Liljegren et al. 2008). This is the validated physical model; the BoM formula in
pipeline/hourly_features.py is only a temperature-humidity index (no sun or wind).

Sources and credit
------------------
* Algorithm and constants: ported from James C. Liljegren's original C code, WBGT version
  1.1 (wbgt.c.original, https://github.com/mdljts/wbgt, commit cd672a8). Solar position:
  Nels Larson's solarposition() in the same file (Astronomical Almanac 1990 low-precision
  formulas). Reference: Liljegren, Carhart, Lawday, Tschopp & Sharp (2008), J. Occup.
  Environ. Hyg. 5(10), 645-655.
* Adaptation to gridded, hourly-mean reanalysis input follows Kong & Huber (2022) and their
  reference implementation PyWBGT (https://github.com/QINQINKONG/PyWBGT, commit b40942b):
    - solve the Tg and Tnwb equations with a bracketing root finder, not Liljegren's damped
      fixed-point iteration (here: vectorised bisection; same root to within 1e-4 K);
    - use the mean cosine solar zenith angle over the SUNLIT part of each averaging interval,
      because reanalysis radiation is an hourly mean, not an instant;
    - take the direct-beam fraction from the reanalysis (direct / total), not Liljegren's
      empirical estimate from a single pyranometer;
    - convert 10 m wind to 2 m with the stability-class power law, using a fixed
      night-time class (no vertical temperature difference is available).
  Kong, Q. & Huber, M. (2022). Explicit calculations of wet-bulb globe temperature compared
  with approximations and why it matters for labor productivity. Earth's Future 10,
  e2021EF002334. https://doi.org/10.1029/2021EF002334
  No PyWBGT code is copied (it is licensed CC BY-NC-SA 4.0). This module ports the
  Argonne code and follows Kong & Huber's documented methodology. Their implementation was
  used as an independent reference to check our results (see tests/test_wbgt_liljegren.py).

Changes to the original (license condition 1): Python/numpy port; bisection solver;
sunlit-mean cosine zenith; reanalysis direct fraction; fixed night stability class;
vectorised over arrays.

------------------------------------------------------------------------------------------
Copyright (c) 2008, UChicago Argonne, LLC. All Rights Reserved. WBGT, Version 1.1,
James C. Liljegren, Decision & Information Sciences Division.
Redistribution and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this list of
   conditions and the following disclaimer. Software changes, modifications, or derivative
   works, should be noted with comments and the author and organization's name.
2. Redistributions in binary form must reproduce the above copyright notice, this list of
   conditions and the following disclaimer in the documentation and/or other materials
   provided with the distribution.
3. Neither the names of UChicago Argonne, LLC or the Department of Energy nor the names of
   its contributors may be used to endorse or promote products derived from this software
   without specific prior written permission.
4. The software and the end-user documentation included with the redistribution, if any,
   must include the following acknowledgment: "This product includes software produced by
   UChicago Argonne, LLC under Contract No. DE-AC02-06CH11357 with the Department of Energy."
DISCLAIMER: THE SOFTWARE IS SUPPLIED "AS IS" WITHOUT WARRANTY OF ANY KIND. NEITHER THE
UNITED STATES GOVERNMENT, NOR THE UNITED STATES DEPARTMENT OF ENERGY, NOR UCHICAGO ARGONNE,
LLC, NOR ANY OF THEIR EMPLOYEES, MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY
LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS, OR USEFULNESS OF ANY
INFORMATION, DATA, APPARATUS, PRODUCT, OR PROCESS DISCLOSED, OR REPRESENTS THAT ITS USE
WOULD NOT INFRINGE PRIVATELY OWNED RIGHTS.
------------------------------------------------------------------------------------------
"""
from __future__ import annotations

import numpy as np

# ---- physical constants (wbgt.c.original) ----
SOLAR_CONST = 1367.0
STEFANB = 5.6696e-8
CP = 1003.5
M_AIR = 28.97
M_H2O = 18.015
RATIO = CP * M_AIR / M_H2O
R_GAS = 8314.34
R_AIR = R_GAS / M_AIR
PR = CP / (CP + 1.25 * R_AIR)  # Prandtl number
# wick
EMIS_WICK, ALB_WICK, D_WICK, L_WICK = 0.95, 0.4, 0.007, 0.0254
# globe
EMIS_GLOBE, ALB_GLOBE, D_GLOBE = 0.95, 0.05, 0.0508
# surface
EMIS_SFC, ALB_SFC = 0.999, 0.45
# limits
CZA_MIN = 0.00873
REF_HEIGHT = 2.0
MIN_SPEED = 0.13
DEG_RAD = np.pi / 180.0

# Stability class table and wind-profile exponents (EPA-454/5-99-005, via wbgt.c.original)
_LSRDT = np.array([
    [1, 1, 2, 4, 0, 5, 6, 0],
    [1, 2, 3, 4, 0, 5, 6, 0],
    [2, 2, 3, 4, 0, 4, 4, 0],
    [3, 3, 4, 4, 0, 0, 0, 0],
    [3, 4, 4, 4, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
])
URBAN_EXP = np.array([0.15, 0.15, 0.20, 0.25, 0.30, 0.30])
RURAL_EXP = np.array([0.07, 0.07, 0.10, 0.15, 0.35, 0.55])


# ---- thermodynamic helpers (all temperatures in K, pressure in hPa = mb) ----

def esat(tk):
    """Saturation vapour pressure over liquid water (hPa); Buck (1981), x1.004 moist air."""
    return 1.004 * 6.1121 * np.exp(17.502 * (tk - 273.15) / (tk - 32.18))


def dew_point(e):
    """Dew point (K) from vapour pressure e (hPa); inverse of esat."""
    z = np.log(e / (6.1121 * 1.004))
    return 273.15 + 240.97 * z / (17.502 - z)


def viscosity(tk):
    """Viscosity of air, kg/(m s) (Bird, Stewart & Lightfoot)."""
    omega = (tk / 97.0 - 2.9) / 0.4 * (-0.034) + 1.048
    return 2.6693e-6 * np.sqrt(M_AIR * tk) / (3.617 * 3.617 * omega)


def thermal_cond(tk):
    """Thermal conductivity of air, W/(m K)."""
    return (CP + 1.25 * R_AIR) * viscosity(tk)


def diffusivity(tk, p_hpa):
    """Diffusivity of water vapour in air, m2/s."""
    pcrit13 = (36.4 * 218.0) ** (1.0 / 3.0)
    tcrit512 = (132.0 * 647.3) ** (5.0 / 12.0)
    tcrit12 = np.sqrt(132.0 * 647.3)
    mmix = np.sqrt(1.0 / M_AIR + 1.0 / M_H2O)
    return 3.640e-4 * (tk / tcrit12) ** 2.334 * pcrit13 * tcrit512 * mmix / (p_hpa / 1013.25) * 1e-4


def evap(tk):
    """Heat of evaporation, J/kg."""
    return (313.15 - tk) / 30.0 * (-71100.0) + 2.4073e6


def emis_atm(tk, rh):
    """Atmospheric emissivity (Oke); rh as a fraction."""
    return 0.575 * (rh * esat(tk)) ** 0.143


def h_sphere(tk, p_hpa, speed):
    """Convective heat-transfer coefficient of the globe, W/(m2 K)."""
    density = p_hpa * 100.0 / (R_AIR * tk)
    re = np.maximum(speed, MIN_SPEED) * density * D_GLOBE / viscosity(tk)
    nu = 2.0 + 0.6 * np.sqrt(re) * PR ** 0.3333
    return nu * thermal_cond(tk) / D_GLOBE


def h_cylinder(tk, p_hpa, speed):
    """Convective heat-transfer coefficient of the wick (long cylinder), W/(m2 K)."""
    density = p_hpa * 100.0 / (R_AIR * tk)
    re = np.maximum(speed, MIN_SPEED) * density * D_WICK / viscosity(tk)
    nu = 0.281 * re ** (1.0 - 0.4) * PR ** (1.0 - 0.56)
    return nu * thermal_cond(tk) / D_WICK


# ---- solar geometry ----

def solar_altitude_deg(times_utc, lat, lon):
    """Apparent solar altitude (deg, refraction added) at UTC instants: vectorised port of
    Larson's solarposition() (Astronomical Almanac 1990 low-precision formulas)."""
    t = np.asarray(times_utc, dtype="datetime64[ns]")
    days_j2000 = (t - np.datetime64("2000-01-01T12:00:00")).astype("timedelta64[ns]").astype(np.float64) / 86_400e9
    days_0h = np.floor(days_j2000 + 0.5) - 0.5  # 0 h UT of the date
    ut = (days_j2000 - days_0h) * 24.0
    cent = days_0h / 36525.0
    mean_anomaly = np.mod((357.528 + 0.9856003 * days_j2000) / 360.0, 1.0) * 2 * np.pi
    mean_long = np.mod((280.460 + 0.9856474 * days_j2000) / 360.0, 1.0) * 2 * np.pi
    obliq = (23.439 - 4.0e-7 * days_j2000) * DEG_RAD
    ecl = (1.915 * np.sin(mean_anomaly) + 0.020 * np.sin(2 * mean_anomaly)) * DEG_RAD + mean_long
    ra = np.arctan2(np.cos(obliq) * np.sin(ecl), np.cos(ecl))
    ra = np.mod(ra, 2 * np.pi) / (2 * np.pi) * 24.0
    dec = np.arcsin(np.sin(obliq) * np.sin(ecl))
    gmst0h = 24110.54841 + cent * (8640184.812866 + cent * (0.093104 - cent * 6.2e-6))
    gmst0h = np.mod(gmst0h / 3600.0 / 24.0, 1.0) * 24.0
    lmst = np.mod((gmst0h + ut * 1.00273790934 + lon / 15.0) / 24.0, 1.0) * 24.0
    ha = lmst - ra
    ha = np.where(ha < -12, ha + 24, np.where(ha > 12, ha - 24, ha)) / 24.0 * 2 * np.pi
    latr = lat * DEG_RAD
    alt = np.arcsin(np.sin(dec) * np.sin(latr) + np.cos(dec) * np.cos(ha) * np.cos(latr)) / DEG_RAD
    # refraction (standard atmosphere), as in the original; added to the altitude
    p, temp = 1013.25, 15.0
    tan_alt = np.tan(np.clip(alt, -89.99999, 89.99999) * DEG_RAD)
    low = (0.1594 + alt * (0.0196 + 0.00002 * alt)) * p / ((1.0 + alt * (0.505 + 0.0845 * alt)) * (273.0 + temp))
    high = 0.00452 * (p / (273.0 + temp)) / tan_alt
    refr = np.where(alt < -1.0, 0.0, np.where(alt < 19.225, low, high))
    return alt + refr


def sun_earth_distance(times_utc):
    """Sun-Earth distance in astronomical units."""
    t = np.asarray(times_utc, dtype="datetime64[ns]")
    d = (t - np.datetime64("2000-01-01T12:00:00")).astype("timedelta64[ns]").astype(np.float64) / 86_400e9
    g = np.mod((357.528 + 0.9856003 * d) / 360.0, 1.0) * 2 * np.pi
    return 1.00014 - 0.01671 * np.cos(g) - 0.00014 * np.cos(2 * g)


def mean_sunlit_cosz(interval_end_utc, lat, lon, minutes: int = 60, substeps: int = 30):
    """Mean cosine solar zenith over the SUNLIT part of the interval that ends at each
    timestamp (Kong & Huber's choice for interval-mean radiation). 0 if the sun is below
    CZA_MIN for the whole interval. Numerical: `substeps` midpoints per interval."""
    end = np.asarray(interval_end_utc, dtype="datetime64[ns]")
    step = np.timedelta64(int(minutes * 60e9 / substeps), "ns")
    offsets = (np.arange(substeps) + 0.5) * step - np.timedelta64(int(minutes * 60e9), "ns")
    t = end[:, None] + offsets[None, :]
    cz = np.cos((90.0 - solar_altitude_deg(t, lat, lon)) * DEG_RAD)
    lit = cz >= CZA_MIN
    n = lit.sum(axis=1)
    return np.where(n > 0, np.where(lit, cz, 0.0).sum(axis=1) / np.maximum(n, 1), 0.0)


# ---- wind ----

def wind_2m(speed_z, zspeed, cosz, solar, urban: bool = True):
    """Estimate 2 m wind from wind at height zspeed via the stability-class power law
    (EPA-454/5-99-005). Night uses a fixed class (no vertical temperature gradient)."""
    speed_z, cosz, solar = np.broadcast_arrays(np.asarray(speed_z, float), np.asarray(cosz, float), np.asarray(solar, float))
    day = cosz > 0
    j_day = np.select([solar >= 925.0, solar >= 675.0, solar >= 175.0], [0, 1, 2], 3)
    i_day = np.select([speed_z >= 6.0, speed_z >= 5.0, speed_z >= 3.0, speed_z >= 2.0], [4, 3, 2, 1], 0)
    i_night = np.select([speed_z >= 2.5, speed_z >= 2.0], [2, 1], 0)
    j = np.where(day, j_day, 5)
    i = np.where(day, i_day, i_night)
    cls = _LSRDT[i, j]
    exp = (URBAN_EXP if urban else RURAL_EXP)[cls - 1]
    return np.maximum(speed_z * (REF_HEIGHT / zspeed) ** exp, MIN_SPEED)


# ---- globe and natural wet-bulb temperatures ----

def _bisect(f, lo, hi, tol=1e-4, max_iter=60):
    """Vectorised bisection for a decreasing f with f(lo) > 0 > f(hi). Returns NaN where
    the bracket has no sign change."""
    flo, fhi = f(lo), f(hi)
    ok = (flo > 0) & (fhi < 0)
    lo, hi = lo.copy(), hi.copy()
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        pos = f(mid) > 0
        lo = np.where(pos, mid, lo)
        hi = np.where(pos, hi, mid)
        if np.all(hi - lo < tol):
            break
    return np.where(ok, 0.5 * (lo + hi), np.nan)


def globe_temperature(tk, rh, p_hpa, speed2m, solar, fdir, cosz):
    """Black-globe temperature (K). rh as a fraction, solar in W/m2, speed at 2 m."""
    tk, rh, p_hpa, speed2m, solar, fdir, cosz = np.broadcast_arrays(*(np.asarray(a, float) for a in (tk, rh, p_hpa, speed2m, solar, fdir, cosz)))
    cz = np.where(cosz >= CZA_MIN, cosz, 1.0)
    sun = np.where(cosz >= CZA_MIN, solar, 0.0)
    c0 = (0.5 * (emis_atm(tk, rh) * tk**4 + EMIS_SFC * tk**4)
          + sun / (2.0 * STEFANB * EMIS_GLOBE) * (1.0 - ALB_GLOBE) * (fdir * (1.0 / (2.0 * cz) - 1.0) + 1.0 + ALB_SFC))

    def f(x):
        h = h_sphere(0.5 * (x + tk), p_hpa, speed2m)
        return c0 - h / (STEFANB * EMIS_GLOBE) * (x - tk) - x**4

    return _bisect(f, tk - 50.0, tk + 90.0)


def natural_wet_bulb(tk, rh, p_hpa, speed2m, solar, fdir, cosz, radiative: bool = True):
    """Natural wet-bulb temperature (K); radiative=False gives the psychrometric wet bulb."""
    tk, rh, p_hpa, speed2m, solar, fdir, cosz = np.broadcast_arrays(*(np.asarray(a, float) for a in (tk, rh, p_hpa, speed2m, solar, fdir, cosz)))
    eair = rh * esat(tk)
    cz = np.where(cosz >= CZA_MIN, cosz, 1.0)
    sun = np.where(cosz >= CZA_MIN, solar, 0.0)
    sza = np.arccos(np.clip(cz, -1.0, 1.0))
    solar_term = (1.0 - ALB_WICK) * sun * ((1.0 - fdir) * (1.0 + 0.25 * D_WICK / L_WICK)
                                         + fdir * (np.tan(sza) / np.pi + 0.25 * D_WICK / L_WICK) + ALB_SFC)
    lw_down_up = 0.5 * (emis_atm(tk, rh) * tk**4 + EMIS_SFC * tk**4)

    def g(x):
        tref = 0.5 * (x + tk)
        h = h_cylinder(tref, p_hpa, speed2m)
        fatm = STEFANB * EMIS_WICK * (lw_down_up - x**4) + solar_term
        ewick = esat(x)
        density = p_hpa * 100.0 / (R_AIR * tref)
        sc = viscosity(tref) / (density * diffusivity(tref, p_hpa))
        new = tk - evap(tref) / RATIO * (ewick - eair) / (p_hpa - ewick) * (PR / sc) ** 0.56 + (fatm / h) * float(radiative)
        return new - x

    lo = np.minimum(dew_point(np.maximum(eair, 1e-3)), tk) - 5.0
    return _bisect(g, lo, tk + 40.0)


def wbgt(t_c, rh_pct, p_hpa, wind10m, ghi, direct_horizontal, cosz, urban: bool = True) -> dict[str, np.ndarray]:
    """Outdoor WBGT and its components (deg C) from hourly reanalysis inputs.

    t_c air temperature (C); rh_pct relative humidity (%); p_hpa surface pressure (hPa);
    wind10m 10 m wind speed (m/s); ghi global horizontal irradiance and direct_horizontal
    its direct component (W/m2, interval means); cosz mean sunlit cosine zenith of the
    same interval (mean_sunlit_cosz)."""
    tk = np.asarray(t_c, float) + 273.15
    rh = np.clip(np.asarray(rh_pct, float), 0.5, 100.0) / 100.0
    ghi = np.maximum(np.asarray(ghi, float), 0.0)
    fdir = np.clip(np.divide(direct_horizontal, ghi, out=np.zeros_like(ghi), where=ghi > 0), 0.0, 1.0)
    u2 = wind_2m(wind10m, 10.0, cosz, ghi, urban=urban)
    tg = globe_temperature(tk, rh, p_hpa, u2, ghi, fdir, cosz) - 273.15
    tnwb = natural_wet_bulb(tk, rh, p_hpa, u2, ghi, fdir, cosz) - 273.15
    t = tk - 273.15
    return {"wbgt": 0.7 * tnwb + 0.2 * tg + 0.1 * t, "tg": tg, "tnwb": tnwb, "wind2m": u2, "fdir": fdir}
