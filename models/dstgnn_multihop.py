"""
Multi-hop DSTGNN, "C3-hop" (decisions.md 2026-10-08: exploratory arm added after the G-D3
result and the independent ML review; ring averaging decided 2026-10-10 after a focused code
review, before any C3-hop result; disclosed as post-result).

Why: in models.dstgnn with the Delhi readout, a message on day t carries the neighbours'
states of day t-1 and moves ONE hop, so an upstream point h hops away reaches Delhi's final
state with data at least h + 1 days old, and no upstream point's LAST input day ever reaches
the forecast. G-D0 found the useful upstream signal is exactly the last input day, at points
500-1000 km (2-4 hops) away.

A first version used DCRNN-style diffusion (powers of the normalised adjacency). The focused
review showed it shrinks a 4-hop point's weight ~40x relative to a 1-hop neighbour (and W_k is
shared by every walk of length k, so training cannot undo it), so it could not test the
hypothesis. This version uses exact-distance "ring" averaging instead (shortest-path message
passing; Abboud, Dimitrov & Ceylan 2022, "Shortest Path Networks for Graph Property Prediction"):

  ring_k(i) = the nodes whose shortest-path distance to node i on the graph is exactly k;
  R_k[j, i] = 1 / |ring_k(i)| for j in ring_k(i), else 0 (each destination's ring averages to 1).

  1. Inside the recurrence, on day t, node i receives sum_k W_k mean_{j in ring_k(i)} s_j(t-1),
     k = 1..K: every node within K hops reaches it in one day, each distance with O(1) weight.
  2. A final ring transport of the LAST day's states into Delhi before the head:
     readout = [s_T(Delhi) ; sum_k V_k mean_{j in ring_k(Delhi)} s_j(T)]. Causal: the forecast
     is for the days after the input window.
K = 4, the largest hop distance from any upstream point to Delhi on the real graph, so every
point's last input day reaches the forecast. The rings come from the graph's neighbour mask (C3's
topology: who neighbours whom, within 370 km); C3's geographic edge weights are not used (they vary
only ~0.5-0.8 across neighbours). Upstream data reaches the forecast only through the rings, which
group points by graph distance; the comparison with C2 (one plain mean over all points) asks whether
that distance structure helps. Everything else (embeddings, GRU cell, dropout) is models.dstgnn's.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from models.dstgnn import DSTGNN

HOPS = 4


def hop_distances(mask: np.ndarray) -> np.ndarray:
    """(N, N) shortest-path hop counts d[j, i] from source j to destination i along mask[j, i]
    edges (unreachable: a large number)."""
    n = len(mask)
    d = np.full((n, n), 10**6, dtype=np.int64)
    for i in range(n):
        d[i, i], frontier = 0, [i]
        while frontier:
            nxt = []
            for v in frontier:
                for u in np.flatnonzero(mask[:, v]):
                    if d[u, i] > d[v, i] + 1:
                        d[u, i] = d[v, i] + 1
                        nxt.append(u)
            frontier = nxt
    return d


def ring_matrices(mask: np.ndarray, hops: int) -> np.ndarray:
    """(hops, N, N): R[k-1, j, i] = 1 / |ring_k(i)| for d(j, i) == k, else 0."""
    d = hop_distances(mask)
    rings = np.zeros((hops, *d.shape), dtype=np.float32)
    for k in range(1, hops + 1):
        r = (d == k).astype(np.float32)
        size = r.sum(axis=0, keepdims=True)
        rings[k - 1] = np.divide(r, size, out=np.zeros_like(r), where=size > 0)
    return rings


class MultiHopDSTGNN(DSTGNN):
    def __init__(self, *args, hops: int = HOPS, **kwargs):
        super().__init__(*args, **kwargs)
        if self.graph != "static":
            raise ValueError("the multi-hop model uses the static graph's topology (graph 'static')")
        if self.readout != "delhi":
            raise ValueError("the multi-hop model keeps the Delhi-only readout (rings stay the only upstream path)")
        if self.neighbour_mask is None:
            raise ValueError("the multi-hop model needs neighbour_mask to build its distance rings")
        if hops < 1:
            raise ValueError("hops must be >= 1")
        self.hops, h = hops, self.hidden
        self.register_buffer("rings", torch.from_numpy(ring_matrices(self.neighbour_mask.cpu().numpy(), hops)))
        del self.w_msg  # replaced by one weight per ring
        self.w_hop = nn.ModuleList(nn.Linear(h, h, bias=False) for _ in range(hops))
        self.w_read = nn.ModuleList(nn.Linear(h, h, bias=False) for _ in range(hops))
        self.head = nn.Sequential(nn.Linear(2 * h, 32), nn.ReLU(), nn.Linear(32, self.head[-1].out_features))

    def _transport(self, s: torch.Tensor, weights: nn.ModuleList) -> torch.Tensor:
        """sum_k W_k (ring-k mean of s) per destination: s (B, N, H) -> (B, N, H)."""
        out = torch.zeros_like(s)
        for k, w in enumerate(weights):
            out = out + w(torch.einsum("ji,bjh->bih", self.rings[k], s))
        return out

    def node_states(self, x_delhi: torch.Tensor, x_up: torch.Tensor,
                    adj_dynamic: torch.Tensor | None = None) -> torch.Tensor:
        b, t, _ = x_delhi.shape
        if x_up.shape[2] != self.n_nodes - 1:
            raise ValueError(f"expected {self.n_nodes - 1} upstream nodes, got {x_up.shape[2]}")
        e = torch.cat([self.embed_delhi(x_delhi).unsqueeze(2), self.embed_up(x_up)], dim=2)
        if self.drop is not None:
            e = self.drop(e)
        s = e.new_zeros(b, self.n_nodes, self.hidden)
        states = []
        for day in range(t):
            msg = torch.zeros_like(s) if day == 0 else self._transport(s, self.w_hop)  # s = previous day's states
            inp = torch.cat([e[:, day], msg], dim=-1).reshape(b * self.n_nodes, 2 * self.hidden)
            s = self.cell(inp, s.reshape(b * self.n_nodes, self.hidden)).reshape(b, self.n_nodes, self.hidden)
            states.append(s)
        return torch.stack(states, dim=1)

    def forward(self, x_delhi: torch.Tensor, x_up: torch.Tensor, adj_dynamic: torch.Tensor | None = None) -> torch.Tensor:
        last = self.node_states(x_delhi, x_up)[:, -1]  # (B, N, H)
        r = torch.cat([last[:, 0], self._transport(last, self.w_read)[:, 0]], dim=-1)
        if self.drop is not None:
            r = self.drop(r)
        return self.head(r)
