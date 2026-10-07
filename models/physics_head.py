"""
Physics head (plan v5 option 2a; design decided 2026-10-07: exact Liljegren formula (A),
per cell at each cell's own peak hour (D3)).

From a backbone's final representation, the head predicts, for each forecast day and each of
Delhi's 9 cells, the 7 ingredients of that cell's WBGT at its peak hour (air temperature,
relative humidity, surface pressure, 10 m wind, global irradiance, direct fraction, sun
angle), applies the exact formula (pipeline.wbgt_liljegren_torch) per cell, and averages the
9 cells: the same definition as the target wbgt_lj_max. With perfect ingredients the head
reproduces the target exactly (tests/test_build_peak_ingredients.py).

Output ranges are physical by construction:
  t, pressure     fold-training mean + SD x raw, then clamped to a physical range
  rh              100 x sigmoid           (0-100 %)
  wind            softplus                (> 0 m/s)
  cosz            sigmoid                 (0-1)
  ghi             1100 x cosz x sigmoid   (0 at night, at most a clear-sky bound)
  fdir            sigmoid                 (0-1)
The ingredients are also returned, so training can supervise them against the per-cell peak
ingredients (pipeline/build_peak_ingredients.py) if the pre-registered loss asks for it.

LSTMPhysics = the control LSTM's encoder (same sizes as models/lstm.py, which is not changed)
with this head in place of the direct 5-value output.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from pipeline.wbgt_liljegren_torch import wbgt

INGREDIENTS = ("t", "rh", "pressure", "wind", "ghi", "fdir", "cosz")
N_CELLS, FORECAST_DAYS = 9, 5
GHI_MAX = 1100.0  # W/m2: upper bound per unit cos(zenith), above clear-sky values at Delhi
T_RANGE, P_RANGE = (-10.0, 55.0), (850.0, 1050.0)


class PhysicsHead(nn.Module):
    def __init__(self, in_dim: int, t_mean: torch.Tensor, t_std: torch.Tensor, p_mean: torch.Tensor,
                 p_std: torch.Tensor, hidden: int = 64, n_cells: int = N_CELLS, forecast_days: int = FORECAST_DAYS):
        """t_mean, t_std, p_mean, p_std: (n_cells,) peak-hour temperature (C) and pressure
        (hPa) statistics from the fold's training years (they only set the output scale)."""
        super().__init__()
        self.n_cells, self.forecast_days = n_cells, forecast_days
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, forecast_days * n_cells * len(INGREDIENTS)))
        for name, v in (("t_mean", t_mean), ("t_std", t_std), ("p_mean", p_mean), ("p_std", p_std)):
            self.register_buffer(name, torch.as_tensor(v, dtype=torch.float32).reshape(n_cells))

    def ingredients(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        """(B, D) -> each ingredient (B, forecast_days, n_cells), in physical units."""
        raw = self.net(h).reshape(h.shape[0], self.forecast_days, self.n_cells, len(INGREDIENTS))
        r = dict(zip(INGREDIENTS, raw.unbind(-1)))
        cosz = torch.sigmoid(r["cosz"])
        return {"t": torch.clamp(self.t_mean + self.t_std * r["t"], *T_RANGE),
                "rh": 100.0 * torch.sigmoid(r["rh"]),
                "pressure": torch.clamp(self.p_mean + self.p_std * r["pressure"], *P_RANGE),
                "wind": F.softplus(r["wind"]),
                "ghi": GHI_MAX * cosz * torch.sigmoid(r["ghi"]),
                "fdir": torch.sigmoid(r["fdir"]),
                "cosz": cosz}

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """(WBGT (B, forecast_days) in deg C = 9-cell mean of per-cell WBGT, ingredients)."""
        g = self.ingredients(h)
        per_cell = wbgt(g["t"], g["rh"], g["pressure"], g["wind"], g["ghi"], g["fdir"], g["cosz"])["wbgt"]
        return per_cell.mean(dim=-1), g


class LSTMPhysics(nn.Module):
    """Control LSTM encoder (2 layers, hidden 64, dropout 0.2, as models/lstm.py) + PhysicsHead."""

    def __init__(self, n_features: int, head_stats: dict[str, torch.Tensor], hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.head = PhysicsHead(hidden_size, **head_stats)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        out, _ = self.lstm(x)
        return self.head(out[:, -1])
