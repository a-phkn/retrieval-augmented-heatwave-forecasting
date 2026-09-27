"""
Retrieval-augmented forecaster: shared LSTM encoder (identical architecture
to models.lstm.LSTMForecaster's encoder) + scaled dot-product attention
over K retrieved analogues + MLP head. Per Execution_Pipeline.md Step 5:
"LSTM encoder + attention over P3's top-K analogues + MLP head."

DESIGN CHOICE (not explicitly pinned down by the roadmap's one-line
description, documented here since it materially affects what "retrieval"
contributes):
  - The SAME encoder (shared weights) encodes the query window AND every
    analogue window -- an analogue is "just another 14-day window," so
    reusing the encoder is both simpler and forces the attention mechanism
    to do the retrieval-specific work, rather than learning two encoders.
  - Attention keys = analogue encodings (so similarity is judged in the
    model's own learned representation space, not just the retriever's
    fixed 17-dim statistical space).
  - Attention VALUES = [analogue encoding ; analogue's OWN actual 5-day
    outcome (z-normalized)], concatenated. This is the specific mechanism
    meant to address the tail-shrinkage problem: the context vector the
    model receives carries not just "what did a similar window look like"
    but "what actually happened after it" -- letting the model shift its
    prediction toward real historical outcomes for similar situations,
    which a plain point-estimate regressor has no way to access.
  - Query attends over analogues using its own encoding as the attention
    query vector (single-head scaled dot-product, not multi-head -- K is
    small (<=20) and the analogue set is already curated by the retriever,
    so multi-head attention's usual benefit -- attending to different
    subspaces of a large context -- has little to bite on here).
  - Padded analogue slots (when fewer than K are eligible -- see
    training/retrieval_data.py) are masked out of the softmax with -inf,
    never contributing to the context vector.
  - Final prediction: MLP head over concat(query encoding, context vector)
    -- same head shape as the baseline's MLP head, doubled input width.

Output includes per-analogue attention weights (Execution_Pipeline.md Step
5's output spec: "includes attention weights per prediction").
"""
from __future__ import annotations

import torch
from torch import nn


class RetrievalAugmentedLSTM(nn.Module):
    def __init__(
        self,
        n_features: int = 13,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        forecast_days: int = 5,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.forecast_days = forecast_days

        # Shared encoder for both the query window and every analogue window.
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Projects an analogue's [encoding ; its own actual outcome] into
        # the attention value space (same width as the encoding, so the
        # final concat below has a fixed, predictable size).
        self.value_proj = nn.Linear(hidden_size + forecast_days, hidden_size)

        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Linear(32, forecast_days),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 14, n_features) -> (batch, hidden_size)"""
        _, (h_n, _) = self.encoder(x)
        return h_n[-1]

    def forward(
        self,
        x_query: torch.Tensor,
        x_analogues: torch.Tensor,
        y_analogues: torch.Tensor,
        analogue_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x_query:       (batch, 14, n_features)
        x_analogues:   (batch, K, 14, n_features)
        y_analogues:   (batch, K, forecast_days) -- analogues' own actual
                       outcomes, z-normalized the SAME way as the main
                       target (caller's responsibility, see training script)
        analogue_mask: (batch, K) bool, True where a real (non-padded)
                       analogue exists

        Returns: (prediction (batch, forecast_days),
                  attention_weights (batch, K))
        """
        batch, k, seq_len, n_features = x_analogues.shape

        h_query = self.encode(x_query)  # (batch, hidden)

        x_analogues_flat = x_analogues.reshape(batch * k, seq_len, n_features)
        h_analogues_flat = self.encode(x_analogues_flat)
        h_analogues = h_analogues_flat.reshape(batch, k, self.hidden_size)  # (batch, K, hidden)

        # Attention scores: scaled dot product between query encoding and
        # each analogue encoding.
        scores = torch.einsum("bh,bkh->bk", h_query, h_analogues) / (self.hidden_size ** 0.5)
        scores = scores.masked_fill(~analogue_mask, float("-inf"))

        # A query with ZERO eligible analogues (all masked) would produce an
        # all -inf row -> softmax NaN. Doesn't happen in practice (see
        # training/retrieval_data.py -- worst case is a few analogues short
        # of K, never zero, for any split), but guarded explicitly rather
        # than silently producing NaN losses if that assumption ever breaks.
        no_analogues = ~analogue_mask.any(dim=1)
        if no_analogues.any():
            scores = scores.clone()
            scores[no_analogues] = 0.0  # uniform attention as a safe fallback
            safe_mask = analogue_mask.clone()
            safe_mask[no_analogues] = True
        else:
            safe_mask = analogue_mask

        attention_weights = torch.softmax(scores, dim=1)  # (batch, K)
        attention_weights = attention_weights * safe_mask.float()  # zero out any fallback-only mass on real pads
        attention_weights = attention_weights / attention_weights.sum(dim=1, keepdim=True).clamp(min=1e-8)

        values = self.value_proj(torch.cat([h_analogues, y_analogues], dim=-1))  # (batch, K, hidden)
        context = torch.einsum("bk,bkh->bh", attention_weights, values)  # (batch, hidden)

        fused = torch.cat([h_query, context], dim=-1)  # (batch, hidden*2)
        prediction = self.head(fused)
        return prediction, attention_weights
