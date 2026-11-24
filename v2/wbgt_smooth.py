"""
E2 (PROTOCOL_V2, H-N): the differentiable Liljegren WBGT of pipeline/wbgt_liljegren_torch.py with
the one remaining hard switch on a learned input made smooth.

The frozen torch port has two switches that its gradients cannot see:
  1. the 10 m -> 2 m wind exponent, piecewise constant in the EPA stability class, which is set by
     thresholds on 10 m wind speed (2, 3, 5, 6 m/s by day; 2, 2.5 m/s by night) and on global
     irradiance (175, 675, 925 W/m2);
  2. the sun / no-sun switch at cos(zenith) = CZA_MIN.
In v2 the sun angle is not predicted: cos(zenith) is computed from date, hour and location
(pipeline.wbgt_liljegren.mean_sunlit_cosz) and passed in as data, so switch 2 never sits on a
learned quantity and is left as it is. Day / night in switch 1 is decided by that same given
cos(zenith) and is also left hard.

Switch 1 is replaced by a soft class membership: each threshold becomes a logistic step of width
`tau_wind` (m/s) or `tau_ghi` (W/m2), the class weights are differences of those steps (non-negative,
summing to 1), and the exponent is the weighted mean of the class exponents. As tau -> 0 this is
exactly the original. The bias it adds at the default widths is measured in tests_v2 and reported
(PROTOCOL_V2: "report the bias this adds").

Nothing in pipeline/ is edited (frozen folders, training/test_lock.py); this module imports from it.
"""
from __future__ import annotations

import torch

from pipeline.wbgt_liljegren import _LSRDT, MIN_SPEED, REF_HEIGHT, RURAL_EXP, URBAN_EXP
from pipeline.wbgt_liljegren_torch import globe_temperature, natural_wet_bulb

DAY_WIND_THR = (2.0, 3.0, 5.0, 6.0)      # i = 0..4 (numpy wind_2m: i_day)
NIGHT_WIND_THR = (2.0, 2.5)              # i = 0..2 (i_night)
GHI_THR = (925.0, 675.0, 175.0)          # j = 0..3, from strongest sun down (j_day)
TAU_WIND, TAU_GHI = 0.10, 10.0           # default step widths


def _band_weights(x: torch.Tensor, thresholds, tau: float) -> torch.Tensor:
    """Soft one-hot over the len(thresholds)+1 bands of increasing x, shape (..., n_bands).
    tau == 0 gives the hard bands (x >= threshold is in the upper band, as the numpy code)."""
    if tau == 0:
        steps = [(x >= t).to(x.dtype) for t in thresholds]
    else:
        steps = [torch.sigmoid((x - t) / tau) for t in thresholds]
    upper = [torch.ones_like(x), *steps]
    lower = [*steps, torch.zeros_like(x)]
    return torch.stack([u - l for u, l in zip(upper, lower)], dim=-1)


def wind_exponent(speed_z: torch.Tensor, cosz: torch.Tensor, solar: torch.Tensor, urban: bool = True,
                  tau_wind: float = TAU_WIND, tau_ghi: float = TAU_GHI) -> torch.Tensor:
    exps = torch.as_tensor(URBAN_EXP if urban else RURAL_EXP, dtype=speed_z.dtype, device=speed_z.device)
    table = torch.as_tensor(_LSRDT, device=speed_z.device)
    # day: rows i = 0..4 (wind bands), columns j = 0..3 (sun bands, strongest first)
    day_tab = exps[table[:5, :4] - 1]                                    # (5, 4)
    w_i = _band_weights(speed_z, DAY_WIND_THR, tau_wind)                 # (..., 5)
    # j = 0 is the strongest sun: build bands on increasing solar, then reverse
    v_j = _band_weights(solar, tuple(sorted(GHI_THR)), tau_ghi).flip(-1)  # (..., 4)
    day_exp = torch.einsum("...i,ij,...j->...", w_i, day_tab, v_j)
    night_tab = exps[table[:3, 5] - 1]                                   # (3,)
    u_i = _band_weights(speed_z, NIGHT_WIND_THR, tau_wind)               # (..., 3)
    night_exp = (u_i * night_tab).sum(-1)
    return torch.where(cosz > 0, day_exp, night_exp)


def wind_2m(speed_z, zspeed: float, cosz, solar, urban: bool = True, tau_wind: float = TAU_WIND,
            tau_ghi: float = TAU_GHI):
    exp = wind_exponent(speed_z, cosz, solar, urban, tau_wind, tau_ghi)
    u2 = speed_z * (REF_HEIGHT / zspeed) ** exp
    # smooth floor at MIN_SPEED (softplus, width 0.01 m/s) instead of a hard clamp
    return MIN_SPEED + 0.01 * torch.nn.functional.softplus((u2 - MIN_SPEED) / 0.01) if tau_wind > 0 \
        else torch.clamp(u2, min=MIN_SPEED)


def wbgt(t_c, rh_pct, p_hpa, wind10m, ghi, fdir, cosz, urban: bool = True, tol: float = 1e-4,
         tau_wind: float = TAU_WIND, tau_ghi: float = TAU_GHI) -> dict[str, torch.Tensor]:
    """Same inputs and outputs as pipeline.wbgt_liljegren_torch.wbgt; tau_wind = tau_ghi = 0 reproduces it."""
    tk = t_c + 273.15
    rh = torch.clamp(rh_pct, 0.5, 100.0) / 100.0
    ghi = torch.clamp(ghi, min=0.0)
    fdir = torch.clamp(fdir, 0.0, 1.0)
    u2 = wind_2m(wind10m, 10.0, cosz, ghi, urban=urban, tau_wind=tau_wind, tau_ghi=tau_ghi)
    tg = globe_temperature(tk, rh, p_hpa, u2, ghi, fdir, cosz, tol) - 273.15
    tnwb = natural_wet_bulb(tk, rh, p_hpa, u2, ghi, fdir, cosz, tol) - 273.15
    return {"wbgt": 0.7 * tnwb + 0.2 * tg + 0.1 * t_c, "tg": tg, "tnwb": tnwb, "wind2m": u2}
