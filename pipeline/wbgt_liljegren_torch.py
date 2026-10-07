"""
Differentiable (PyTorch) port of the physical Liljegren WBGT in pipeline/wbgt_liljegren.py,
for the physics head (decision 2026-10-07: exact formula, option A). Same equations,
constants and solver as the numpy module, which stays the reference; this file adds
gradients. Credit, licence and the list of changes to Liljegren's original code: see
pipeline/wbgt_liljegren.py (Copyright (c) 2008, UChicago Argonne, LLC; WBGT version 1.1,
James C. Liljegren). Further change here: PyTorch port with implicit-function gradients.

Gradients through the two implicit temperatures. The globe temperature Tg and natural
wet-bulb Tnwb solve f(x, inputs) = 0. Both are found by bisection with no autograd (as the
numpy code), then returned as one Newton step from the root,

    x = x* - f(x*, inputs) / f'(x*)        (x* and f'(x*) detached),

whose value is x* (f(x*) = 0 to solver precision) and whose gradient with respect to the
inputs is exactly the implicit-function-theorem gradient -f_inputs / f_x. No gradient flows
through the bisection iterations themselves.

Not differentiable by design: the 10 m -> 2 m wind exponent switches at the stability-class
thresholds (piecewise constant in the class, as in the original), and the sun/no-sun switch
at CZA_MIN. Gradients are exact inside each piece.

Inputs (tensors, broadcastable): air temperature t_c (C), relative humidity rh_pct (%),
surface pressure p_hpa (hPa), 10 m wind (m/s), global horizontal irradiance ghi (W/m2),
direct fraction fdir (0-1, direct / global), mean sunlit cosine zenith cosz.
"""
from __future__ import annotations

import math

import torch

from pipeline.wbgt_liljegren import (
    ALB_GLOBE, ALB_SFC, ALB_WICK, CP, CZA_MIN, D_GLOBE, D_WICK, EMIS_GLOBE, EMIS_SFC, EMIS_WICK, L_WICK, M_AIR,
    M_H2O, MIN_SPEED, PR, R_AIR, RATIO, REF_HEIGHT, RURAL_EXP, STEFANB, URBAN_EXP, _LSRDT,
)


def esat(tk):
    return 1.004 * 6.1121 * torch.exp(17.502 * (tk - 273.15) / (tk - 32.18))


def dew_point(e):
    z = torch.log(e / (6.1121 * 1.004))
    return 273.15 + 240.97 * z / (17.502 - z)


def viscosity(tk):
    omega = (tk / 97.0 - 2.9) / 0.4 * (-0.034) + 1.048
    return 2.6693e-6 * torch.sqrt(M_AIR * tk) / (3.617 * 3.617 * omega)


def thermal_cond(tk):
    return (CP + 1.25 * R_AIR) * viscosity(tk)


def diffusivity(tk, p_hpa):
    pcrit13 = (36.4 * 218.0) ** (1.0 / 3.0)
    tcrit512 = (132.0 * 647.3) ** (5.0 / 12.0)
    tcrit12 = math.sqrt(132.0 * 647.3)
    mmix = math.sqrt(1.0 / M_AIR + 1.0 / M_H2O)
    return 3.640e-4 * (tk / tcrit12) ** 2.334 * pcrit13 * tcrit512 * mmix / (p_hpa / 1013.25) * 1e-4


def evap(tk):
    return (313.15 - tk) / 30.0 * (-71100.0) + 2.4073e6


def emis_atm(tk, rh):
    return 0.575 * (rh * esat(tk)) ** 0.143


def h_sphere(tk, p_hpa, speed):
    density = p_hpa * 100.0 / (R_AIR * tk)
    re = torch.clamp(speed, min=MIN_SPEED) * density * D_GLOBE / viscosity(tk)
    nu = 2.0 + 0.6 * torch.sqrt(re) * PR ** 0.3333
    return nu * thermal_cond(tk) / D_GLOBE


def h_cylinder(tk, p_hpa, speed):
    density = p_hpa * 100.0 / (R_AIR * tk)
    re = torch.clamp(speed, min=MIN_SPEED) * density * D_WICK / viscosity(tk)
    nu = 0.281 * re ** (1.0 - 0.4) * PR ** (1.0 - 0.56)
    return nu * thermal_cond(tk) / D_WICK


def wind_2m(speed_z, zspeed: float, cosz, solar, urban: bool = True):
    """Stability-class power law, as the numpy wind_2m (the exponent is piecewise constant)."""
    with torch.no_grad():
        day = cosz > 0
        j_day = torch.where(solar >= 925.0, 0, torch.where(solar >= 675.0, 1, torch.where(solar >= 175.0, 2, 3)))
        i_day = torch.where(speed_z >= 6.0, 4, torch.where(speed_z >= 5.0, 3, torch.where(
            speed_z >= 3.0, 2, torch.where(speed_z >= 2.0, 1, 0))))
        i_night = torch.where(speed_z >= 2.5, 2, torch.where(speed_z >= 2.0, 1, 0))
        j = torch.where(day, j_day, torch.full_like(j_day, 5))
        i = torch.where(day, i_day, i_night)
        cls = torch.as_tensor(_LSRDT, device=speed_z.device)[i, j]
        exp = torch.as_tensor(URBAN_EXP if urban else RURAL_EXP, dtype=speed_z.dtype, device=speed_z.device)[cls - 1]
    return torch.clamp(speed_z * (REF_HEIGHT / zspeed) ** exp, min=MIN_SPEED)


def _bisect(f, lo, hi, tol: float = 1e-4, max_iter: int = 200):
    """Bisection for a decreasing f with f(lo) > 0 > f(hi), no autograd (as the numpy _bisect)."""
    with torch.no_grad():
        ok = (f(lo) > 0) & (f(hi) < 0)
        lo, hi = lo.clone(), hi.clone()
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            pos = f(mid) > 0
            lo = torch.where(pos, mid, lo)
            hi = torch.where(pos, hi, mid)
            if bool(torch.all(hi - lo < tol)):
                break
        root = 0.5 * (lo + hi)
    return torch.where(ok, root, torch.full_like(root, float("nan")))


def _implicit(f, root):
    """One Newton step from the detached root: value = root, gradient = -f_inputs / f_x."""
    x = root.detach().requires_grad_(True)
    with torch.enable_grad():
        fx = f(x)
        (dfdx,) = torch.autograd.grad(fx.sum(), x, retain_graph=False, create_graph=False)
    return root.detach() - f(root.detach()) / dfdx.detach()


def globe_temperature(tk, rh, p_hpa, speed2m, solar, fdir, cosz, tol: float = 1e-4):
    lit = cosz >= CZA_MIN
    cz = torch.where(lit, cosz, torch.ones_like(cosz))
    sun = torch.where(lit, solar, torch.zeros_like(solar))
    c0 = (0.5 * (emis_atm(tk, rh) * tk**4 + EMIS_SFC * tk**4)
          + sun / (2.0 * STEFANB * EMIS_GLOBE) * (1.0 - ALB_GLOBE) * (fdir * (1.0 / (2.0 * cz) - 1.0) + 1.0 + ALB_SFC))

    def f(x):
        h = h_sphere(0.5 * (x + tk), p_hpa, speed2m)
        return c0 - h / (STEFANB * EMIS_GLOBE) * (x - tk) - x**4

    return _implicit(f, _bisect(f, tk.detach() - 50.0, tk.detach() + 90.0, tol))


def natural_wet_bulb(tk, rh, p_hpa, speed2m, solar, fdir, cosz, tol: float = 1e-4):
    eair = rh * esat(tk)
    lit = cosz >= CZA_MIN
    cz = torch.where(lit, cosz, torch.ones_like(cosz))
    sun = torch.where(lit, solar, torch.zeros_like(solar))
    sza = torch.arccos(torch.clamp(cz, -1.0, 1.0))
    solar_term = (1.0 - ALB_WICK) * sun * ((1.0 - fdir) * (1.0 + 0.25 * D_WICK / L_WICK)
                                         + fdir * (torch.tan(sza) / math.pi + 0.25 * D_WICK / L_WICK) + ALB_SFC)
    lw_down_up = 0.5 * (emis_atm(tk, rh) * tk**4 + EMIS_SFC * tk**4)

    def g(x):
        tref = 0.5 * (x + tk)
        h = h_cylinder(tref, p_hpa, speed2m)
        fatm = STEFANB * EMIS_WICK * (lw_down_up - x**4) + solar_term
        ewick = esat(x)
        density = p_hpa * 100.0 / (R_AIR * tref)
        sc = viscosity(tref) / (density * diffusivity(tref, p_hpa))
        new = tk - evap(tref) / RATIO * (ewick - eair) / (p_hpa - ewick) * (PR / sc) ** 0.56 + fatm / h
        return new - x

    with torch.no_grad():
        lo = torch.minimum(dew_point(torch.clamp(eair, min=1e-3)), tk) - 5.0
    return _implicit(g, _bisect(g, lo, tk.detach() + 40.0, tol))


def wbgt(t_c, rh_pct, p_hpa, wind10m, ghi, fdir, cosz, urban: bool = True, tol: float = 1e-4) -> dict[str, torch.Tensor]:
    """Outdoor WBGT and components (deg C); same inputs as the numpy wbgt except the direct
    FRACTION fdir is given (the head predicts it) instead of the direct irradiance. `tol`: bisection
    tolerance in K (1e-4 as the numpy module; tighter only for gradient tests)."""
    tk = t_c + 273.15
    rh = torch.clamp(rh_pct, 0.5, 100.0) / 100.0
    ghi = torch.clamp(ghi, min=0.0)
    fdir = torch.clamp(fdir, 0.0, 1.0)
    u2 = wind_2m(wind10m, 10.0, cosz, ghi, urban=urban)
    tg = globe_temperature(tk, rh, p_hpa, u2, ghi, fdir, cosz, tol) - 273.15
    tnwb = natural_wet_bulb(tk, rh, p_hpa, u2, ghi, fdir, cosz, tol) - 273.15
    return {"wbgt": 0.7 * tnwb + 0.2 * tg + 0.1 * t_c, "tg": tg, "tnwb": tnwb, "wind2m": u2}
