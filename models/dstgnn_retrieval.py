"""
G-R*: the C3-hop backbone with analogue retrieval at the Delhi readout (pre-registered
2026-10-13, context/decisions.md).

C3-hop (models.dstgnn_multihop) is unchanged up to its readout r = [s_T(Delhi) ; ring transport
of the last day's states into Delhi]. Retrieval then works like models.retrieval_lstm_v2:
  query  q   = W_q r                                   (B, H)
  keys   k_j = last state of a 1-layer LSTM over analogue j's 14 Delhi input days   (B, K, H)
  values v_j = W_v [k_j ; y_j], y_j = analogue j's own 5-day outcome in model-target units
  context    = sum_j softmax_j(q . k_j / sqrt(H)) v_j  (zero when a window has no analogue)
  forecast   = head([r ; context])
Analogues are not passed through the graph: that would cost ~6x the training time, and the
retrieval information check showed their value lies in their outcomes. So retrieval acts only at
the Delhi readout; the graph's message passing never sees it.
"""
from __future__ import annotations

import torch
from torch import nn

from models.dstgnn_multihop import MultiHopDSTGNN


class RetrievalMultiHopDSTGNN(MultiHopDSTGNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        h, out = self.hidden, self.head[-1].out_features
        self.analogue_encoder = nn.LSTM(input_size=self.embed_delhi.in_features, hidden_size=h, batch_first=True)
        self.query_proj = nn.Linear(2 * h, h)
        self.value_proj = nn.Linear(h + out, h)
        self.head = nn.Sequential(nn.Linear(3 * h, 32), nn.ReLU(), nn.Linear(32, out))

    def encode_analogues(self, x: torch.Tensor) -> torch.Tensor:
        """x (B, K, 14, F_delhi) -> (B, K, H)."""
        b, k, t, f = x.shape
        _, (h_n, _) = self.analogue_encoder(x.reshape(b * k, t, f))
        return h_n[-1].reshape(b, k, self.hidden)

    def forward(self, x_delhi: torch.Tensor, x_up: torch.Tensor, x_analogues: torch.Tensor,
                y_analogues: torch.Tensor, analogue_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x_delhi (B, 14, F); x_up (B, 14, N-1, F_up); x_analogues (B, K, 14, F); y_analogues (B, K, 5);
        analogue_mask (B, K) bool, True = real analogue. Returns (prediction (B, 5), attention (B, K);
        rows without analogues are all 0)."""
        last = self.node_states(x_delhi, x_up)[:, -1]  # (B, N, H)
        r = torch.cat([last[:, 0], self._transport(last, self.w_read)[:, 0]], dim=-1)
        if self.drop is not None:
            r = self.drop(r)
        keys = self.encode_analogues(x_analogues)
        scores = torch.einsum("bh,bkh->bk", self.query_proj(r), keys) / (self.hidden ** 0.5)
        has_any = analogue_mask.any(dim=1, keepdim=True)
        scores = torch.where(analogue_mask | ~has_any, scores, torch.full_like(scores, float("-inf")))
        weights = torch.softmax(scores, dim=1) * analogue_mask.float()  # all-zero rows stay zero
        context = torch.einsum("bk,bkh->bh", weights, self.value_proj(torch.cat([keys, y_analogues], dim=-1)))
        return self.head(torch.cat([r, context], dim=-1)), weights
