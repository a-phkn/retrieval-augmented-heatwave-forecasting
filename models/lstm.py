"""
Baseline LSTM forecaster, per the roadmap spec:
- 2-layer LSTM, hidden size 64, dropout 0.2 (dropout applies between LSTM
  layers; with num_layers=2 this is PyTorch's standard inter-layer dropout).
- MLP head: 64 -> 32 -> 5, taking the final hidden state.
- Input: (batch, 14, 13) z-normalized. Output: (batch, 5) z-normalized
  t_max forecast (de-normalize downstream via training.data.denormalize_y).
"""
from __future__ import annotations

import torch
from torch import nn


class LSTMForecaster(nn.Module):
    def __init__(
        self,
        n_features: int = 13,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        forecast_days: int = 5,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, forecast_days),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 14, n_features)
        _, (h_n, _) = self.lstm(x)
        last_layer_hidden = h_n[-1]  # (batch, hidden_size) -- top layer's final hidden state
        return self.head(last_layer_hidden)  # (batch, forecast_days)
