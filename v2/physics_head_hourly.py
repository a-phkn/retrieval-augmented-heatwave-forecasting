"""
E3 (PROTOCOL_V2, H-N): the physics head moved from one peak hour per cell to every working hour.

Based on models/physics_head.py (frozen; copied here, not edited). Changes, each for a stated reason:
  * Hours: one output per hour stamp in HOURS (06..20 IST, hour-ending; PROTOCOL_V2 Section 1.2),
    per forecast day and per cell, instead of one peak hour.
  * Sun angle: no longer predicted. cos(zenith) is a deterministic function of date, hour and
    location, so it is passed in (computed with pipeline.wbgt_liljegren.mean_sunlit_cosz). This
    removes a learned quantity that has a true value known in advance, and it moves the
    sun / no-sun switch off learned inputs (see v2/wbgt_smooth.py).
  * WBGT through v2.wbgt_smooth (E2: smooth stability-class exponent).
  * Tmax: per cell, max over the forecast hours of the hourly temperature + softplus(gap), so it is
    never below any forecast hour (same idea as the October head, which used the peak hour only).
  * Unsafe hours (E4): soft count sum_h sigmoid((WBGT_h - limit) / sharpness) on the 9-cell mean
    hourly WBGT, for each ISO 7243 limit given; and a soft "unsafe day" indicator for
    >= 2 consecutive unsafe hours (PROTOCOL_V2 Section 1.2). Both tend to the hard counts as the
    sharpness -> 0 (annealed in training).
Hidden width 64, as the October head (models/physics_head.py); any change is fixed in configs_v2/ before training.
Output ranges are physical by construction, as in the October head:
  t, pressure  fold mean + SD x raw, smoothly clamped      rh    100 x sigmoid
  wind         softplus (> 0)                              ghi   1100 x cosz x sigmoid (0 at night)
  fdir         sigmoid (0-1)
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from models.physics_head import GHI_MAX, P_RANGE, T_RANGE, soft_clamp
from v2.wbgt_smooth import wbgt

HOURS = tuple(range(6, 21))          # hour-ending stamps 06..20 IST: 15 hours
LEARNED = ("t", "rh", "pressure", "wind", "ghi", "fdir")
N_CELLS, FORECAST_DAYS = 9, 5


def soft_unsafe_hours(w: torch.Tensor, limit: float, sharpness: float) -> torch.Tensor:
    """w: (..., hours) WBGT. Smooth count of hours with WBGT >= limit; sharpness 0 = hard count."""
    if sharpness == 0:
        return (w >= limit).to(w.dtype).sum(-1)
    return torch.sigmoid((w - limit) / sharpness).sum(-1)


def soft_unsafe_day(w: torch.Tensor, limit: float, sharpness: float) -> torch.Tensor:
    """Smooth indicator of >= 2 consecutive hours at or above the limit:
    1 - prod_h (1 - s_h s_{h+1}), with s_h the per-hour soft exceedance; exactly the hard rule at sharpness 0."""
    s = (w >= limit).to(w.dtype) if sharpness == 0 else torch.sigmoid((w - limit) / sharpness)
    pairs = s[..., :-1] * s[..., 1:]
    return 1.0 - torch.prod(1.0 - pairs, dim=-1)


class HourlyPhysicsHead(nn.Module):
    def __init__(self, in_dim: int, t_mean: torch.Tensor, t_std: torch.Tensor, p_mean: torch.Tensor,
                 p_std: torch.Tensor, hidden: int = 64, n_cells: int = N_CELLS, forecast_days: int = FORECAST_DAYS,
                 n_hours: int = len(HOURS)):
        """t_mean, t_std, p_mean, p_std: (n_hours, n_cells) hourly temperature (C) and pressure (hPa)
        statistics from the fold's training years (they only set the output scale)."""
        super().__init__()
        self.n_cells, self.forecast_days, self.n_hours = n_cells, forecast_days, n_hours
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, forecast_days * n_hours * n_cells * len(LEARNED)
                                           + forecast_days * n_cells))
        for name, v in (("t_mean", t_mean), ("t_std", t_std), ("p_mean", p_mean), ("p_std", p_std)):
            self.register_buffer(name, torch.as_tensor(v, dtype=torch.float32).reshape(n_hours, n_cells))

    def ingredients(self, h: torch.Tensor, cosz: torch.Tensor) -> dict[str, torch.Tensor]:
        """h (B, D_in); cosz (B, days, hours, cells), given. Returns each ingredient (B, days, hours, cells)
        in physical units, plus tmax (B, days, cells)."""
        B, D, H, C = h.shape[0], self.forecast_days, self.n_hours, self.n_cells
        out = self.net(h)
        n_hourly = D * H * C * len(LEARNED)
        raw = out[:, :n_hourly].reshape(B, D, H, C, len(LEARNED))
        gap = out[:, n_hourly:].reshape(B, D, C)
        r = dict(zip(LEARNED, raw.unbind(-1)))
        t = soft_clamp(self.t_mean + self.t_std * r["t"], *T_RANGE)
        return {"t": t,
                "rh": 100.0 * torch.sigmoid(r["rh"]),
                "pressure": soft_clamp(self.p_mean + self.p_std * r["pressure"], *P_RANGE),
                "wind": F.softplus(r["wind"]),
                "ghi": GHI_MAX * cosz * torch.sigmoid(r["ghi"]),
                "fdir": torch.sigmoid(r["fdir"]),
                "cosz": cosz,
                "tmax": t.amax(dim=2) + F.softplus(gap)}

    def forward(self, h: torch.Tensor, cosz: torch.Tensor, limits=(28.0,), sharpness: float = 0.5,
                tol: float = 1e-4) -> dict[str, torch.Tensor]:
        """Returns wbgt_cells (B, days, hours, cells), wbgt_hourly (9-cell mean, B, days, hours),
        wbgt_max (B, days), tmax (B, days, 9-cell mean of per-cell Tmax), and per limit L:
        unsafe_hours[L] (B, days) and unsafe_day[L] (B, days)."""
        g = self.ingredients(h, cosz)
        w_cells = wbgt(g["t"], g["rh"], g["pressure"], g["wind"], g["ghi"], g["fdir"], g["cosz"], tol=tol)["wbgt"]
        w = w_cells.mean(dim=-1)
        return {"wbgt_cells": w_cells, "wbgt_hourly": w, "wbgt_max": w.amax(dim=-1),
                "tmax": g["tmax"].mean(dim=-1), "ingredients": g,
                "unsafe_hours": {L: soft_unsafe_hours(w, L, sharpness) for L in limits},
                "unsafe_day": {L: soft_unsafe_day(w, L, sharpness) for L in limits}}
