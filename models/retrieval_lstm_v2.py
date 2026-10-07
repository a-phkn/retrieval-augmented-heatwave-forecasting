"""
Retrieval-augmented LSTM for v2 (plan v5, Week 3). Same architecture as RA-v1
(models/retrieval_lstm.py, unchanged and frozen): one shared LSTM encoder for the query
window and each analogue window, single-head scaled dot-product attention of the query
encoding over the analogue encodings, attention values = projection of [analogue encoding ;
the analogue's own 5-day outcome in model-target units], MLP head on [query ; context].

One deliberate change (pre-registered 2026-10-06): a query with NO eligible analogue gets
a zero context vector. RA-v1 instead attends uniformly over the zero-filled padding slots,
which adds value_proj's bias (a learned constant) to those queries' context. Only the
earliest training windows (start of 1980) have no analogue, but the fix keeps "no
retrieval evidence" meaning exactly zero evidence.
"""
from __future__ import annotations

import torch
from torch import nn


class RetrievalAugmentedLSTMv2(nn.Module):
    def __init__(self, n_features: int = 13, hidden_size: int = 64, num_layers: int = 2,
                 dropout: float = 0.2, forecast_days: int = 5):
        super().__init__()
        self.hidden_size = hidden_size
        self.forecast_days = forecast_days
        self.encoder = nn.LSTM(input_size=n_features, hidden_size=hidden_size, num_layers=num_layers,
                               batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.value_proj = nn.Linear(hidden_size + forecast_days, hidden_size)
        self.head = nn.Sequential(nn.Linear(hidden_size * 2, 32), nn.ReLU(), nn.Linear(32, forecast_days))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 14, n_features) -> (batch, hidden_size)"""
        _, (h_n, _) = self.encoder(x)
        return h_n[-1]

    def forward(self, x_query: torch.Tensor, x_analogues: torch.Tensor, y_analogues: torch.Tensor,
                analogue_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x_query (B, 14, F); x_analogues (B, K, 14, F); y_analogues (B, K, 5) in the same
        normalised units as the model's target; analogue_mask (B, K) bool, True = real analogue.
        Returns (prediction (B, 5), attention weights (B, K); rows without analogues are all 0).
        """
        b, k, seq_len, n_features = x_analogues.shape
        h_query = self.encode(x_query)
        h_an = self.encode(x_analogues.reshape(b * k, seq_len, n_features)).reshape(b, k, self.hidden_size)

        scores = torch.einsum("bh,bkh->bk", h_query, h_an) / (self.hidden_size ** 0.5)
        has_any = analogue_mask.any(dim=1, keepdim=True)
        # rows with no analogue: neutral scores (avoids an all -inf softmax), then zeroed below
        scores = torch.where(analogue_mask | ~has_any, scores, torch.full_like(scores, float("-inf")))
        weights = torch.softmax(scores, dim=1) * analogue_mask.float()  # all-zero rows stay zero

        values = self.value_proj(torch.cat([h_an, y_analogues], dim=-1))
        context = torch.einsum("bk,bkh->bh", weights, values)  # zero vector when no analogue
        return self.head(torch.cat([h_query, context], dim=-1)), weights
