"""
Multi-hop DSTGNN, "C3-hop" (decisions.md 2026-10-08: exploratory arm added after the G-D3
result and the independent ML review; disclosed as post-result).

Why: in models.dstgnn with the Delhi readout, a message on day t carries the neighbours'
states of day t-1 and moves ONE hop, so an upstream point h hops away reaches Delhi's final
state with data at least h + 1 days old, and no upstream point's LAST input day ever reaches
the forecast. G-D0 found the useful upstream signal is exactly the last input day, at points
500-1000 km (2-4 hops) away. This variant changes only how far and how late information
travels; the readout stays Delhi-only, so upstream data still reaches the forecast only along
the edges (the graph's contribution stays testable against C2).

  1. K hops per day (DCRNN-style K-step diffusion; Li et al. 2018): on day t the message to
     node i is sum_k W_k (A_hat^k s_{t-1})_i, with A_hat the previous day's edges normalised as
     in models.dstgnn (a_ji / (1 + sum_j a_ji)), applied k times.
  2. A final K-hop transport of the LAST day's states (with the last day's edges) into Delhi
     before the head: readout = [s_T(Delhi) ; sum_k V_k (A_hat^k s_T)(Delhi)]. Causal: the
     forecast is for the days after the input window.
K = 4, the largest hop distance from any upstream point to Delhi on the real graph, so every
point's last input day can reach the forecast. Everything else (embeddings, GRU cell, dropout,
edge locality checks) is models.dstgnn's.
"""
from __future__ import annotations

import torch
from torch import nn

from models.dstgnn import DSTGNN

HOPS = 4


class MultiHopDSTGNN(DSTGNN):
    def __init__(self, *args, hops: int = HOPS, **kwargs):
        super().__init__(*args, **kwargs)
        if self.graph == "none":
            raise ValueError("the multi-hop model needs edges (graph 'static' or 'dynamic')")
        if self.readout != "delhi":
            raise ValueError("the multi-hop model keeps the Delhi-only readout (edges stay the only upstream path)")
        if hops < 1:
            raise ValueError("hops must be >= 1")
        self.hops, h = hops, self.hidden
        del self.w_msg  # replaced by one weight per hop
        self.w_hop = nn.ModuleList(nn.Linear(h, h, bias=False) for _ in range(hops))
        self.w_read = nn.ModuleList(nn.Linear(h, h, bias=False) for _ in range(hops))
        self.head = nn.Sequential(nn.Linear(2 * h, 32), nn.ReLU(), nn.Linear(32, self.head[-1].out_features))

    @staticmethod
    def _transport(at: torch.Tensor, s: torch.Tensor, weights: nn.ModuleList) -> torch.Tensor:
        """sum_k W_k (A_hat^k s): at (B, N, N) source -> destination, s (B, N, H)."""
        norm = (1.0 + at.sum(dim=1)).unsqueeze(-1)  # (B, N, 1): per destination, as models.dstgnn
        h, out = s, torch.zeros_like(s)
        for w in weights:
            h = torch.einsum("bji,bjh->bih", at, h) / norm
            out = out + w(h)
        return out

    def node_states(self, x_delhi: torch.Tensor, x_up: torch.Tensor,
                    adj_dynamic: torch.Tensor | None = None) -> torch.Tensor:
        b, t, _ = x_delhi.shape
        if x_up.shape[2] != self.n_nodes - 1:
            raise ValueError(f"expected {self.n_nodes - 1} upstream nodes, got {x_up.shape[2]}")
        e = torch.cat([self.embed_delhi(x_delhi).unsqueeze(2), self.embed_up(x_up)], dim=2)
        if self.drop is not None:
            e = self.drop(e)
        a = self.adjacency(adj_dynamic, b, t)
        s = e.new_zeros(b, self.n_nodes, self.hidden)
        states = []
        for day in range(t):
            msg = torch.zeros_like(s) if day == 0 else self._transport(a[:, day - 1], s, self.w_hop)
            inp = torch.cat([e[:, day], msg], dim=-1).reshape(b * self.n_nodes, 2 * self.hidden)
            s = self.cell(inp, s.reshape(b * self.n_nodes, self.hidden)).reshape(b, self.n_nodes, self.hidden)
            states.append(s)
        return torch.stack(states, dim=1)

    def forward(self, x_delhi: torch.Tensor, x_up: torch.Tensor, adj_dynamic: torch.Tensor | None = None) -> torch.Tensor:
        b, t, _ = x_delhi.shape
        last = self.node_states(x_delhi, x_up, adj_dynamic)[:, -1]  # (B, N, H)
        a_last = self.adjacency(adj_dynamic, b, t)[:, -1]
        r = torch.cat([last[:, 0], self._transport(a_last, last, self.w_read)[:, 0]], dim=-1)
        if self.drop is not None:
            r = self.drop(r)
        return self.head(r)
