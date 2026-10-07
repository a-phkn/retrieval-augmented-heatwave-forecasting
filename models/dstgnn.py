"""
Dynamic spatio-temporal graph neural network (DSTGNN) skeleton for the regional upstream
graph (plan v5, Option 1; Week 3: skeleton + G-D1 correctness tests, no training yet).

Nodes: 0 = Delhi (full Delhi feature set), 1..N-1 = upstream points (their own daily
features). A graph GRU (message passing inside the recurrence, as in DCRNN, Li et al.
2018) runs over the 14 input days. On day t:
  1. each node's features are embedded to `hidden` (separate embeddings for Delhi and the
     upstream nodes, shared across upstream nodes so they are interchangeable);
  2. node i receives a message from its in-neighbours' states of the PREVIOUS day, weighted
     by the PREVIOUS day's edges (the wind that carried air from t-1 to t):
     m_i = sum_j a_ji(t-1) s_j(t-1) / (1 + sum_j a_ji(t-1));
  3. a GRU cell shared by all nodes updates s_i(t) from [embedding ; message].
So information moves exactly one graph hop per day (2 degrees ~ 220 km per day ~ 2.5 m/s of
advection) and day t never sees day t+1. The forecast (5 leads) is read out from Delhi's
final state; with edges, upstream data can only reach Delhi through them, which is what
makes the graph's contribution testable.

Graph modes (the plan's controls, docs/GLOSSARY.md C2-C4):
  "none"     C2: no edges; readout = [Delhi ; mean of upstream states] (upstream data used,
             but without any spatial structure);
  "static"   C3: fixed geographic edges (pipeline.graph.geographic_weights);
  "dynamic"  C4: wind-gated advective edges per day (pipeline.graph.advective_weights),
             optionally plus a learned adaptive adjacency (Graph WaveNet style,
             Wu et al. 2019): softmax(relu(E1 E2^T)) over each node's in-neighbours,
             restricted to the same neighbour mask as the physical edges (so it cannot
             create long-range one-day links) and summing to 1 per destination, the same
             order of magnitude as the dimensionless physical weights. Results are to be
             reported with and without it.
All edges must be between neighbours (pipeline.graph.neighbour_mask); pass the mask as
`neighbour_mask` so the adaptive part respects it.
C1 (plain Delhi LSTM) is the existing control model.
"""
from __future__ import annotations

import torch
from torch import nn

GRAPH_MODES = ("none", "static", "dynamic")


class DSTGNN(nn.Module):
    def __init__(self, n_nodes: int, f_delhi: int, f_upstream: int, hidden: int = 32, graph: str = "dynamic",
                 adaptive: bool = False, embed_dim: int = 8, forecast_days: int = 5,
                 static_adj: torch.Tensor | None = None, neighbour_mask: torch.Tensor | None = None):
        super().__init__()
        if graph not in GRAPH_MODES:
            raise ValueError(f"graph must be one of {GRAPH_MODES}")
        if graph == "static" and static_adj is None:
            raise ValueError("graph 'static' needs static_adj (N, N)")
        if adaptive and graph != "dynamic":
            raise ValueError("the adaptive adjacency is part of the dynamic (C4) model only")
        if adaptive and neighbour_mask is None:
            raise ValueError("the adaptive adjacency needs neighbour_mask (N, N) to stay local")
        self.n_nodes, self.hidden, self.graph, self.adaptive = n_nodes, hidden, graph, adaptive
        self.embed_delhi = nn.Linear(f_delhi, hidden)
        self.embed_up = nn.Linear(f_upstream, hidden)
        self.w_msg = nn.Linear(hidden, hidden, bias=False)
        self.cell = nn.GRUCell(hidden * 2, hidden)
        readout = hidden * 2 if graph == "none" else hidden
        self.head = nn.Sequential(nn.Linear(readout, 32), nn.ReLU(), nn.Linear(32, forecast_days))
        if static_adj is not None:
            self.register_buffer("static_adj", torch.as_tensor(static_adj, dtype=torch.float32))
        self.register_buffer("neighbour_mask", None if neighbour_mask is None
                             else torch.as_tensor(neighbour_mask, dtype=torch.bool))
        if static_adj is not None:
            self._check_local(self.static_adj)
        if adaptive:
            self.e_src = nn.Parameter(torch.randn(n_nodes, embed_dim) * 0.1)
            self.e_dst = nn.Parameter(torch.randn(n_nodes, embed_dim) * 0.1)

    def _check_local(self, adj: torch.Tensor) -> None:
        """Reject any edge weight outside the neighbour mask (when a mask was given)."""
        if self.neighbour_mask is not None and (adj[..., ~self.neighbour_mask] != 0).any():
            raise ValueError("adjacency has edges between non-neighbours (outside neighbour_mask)")

    def adjacency(self, adj_dynamic: torch.Tensor | None, batch: int, steps: int) -> torch.Tensor | None:
        """(B, T, N, N) source -> destination weights used for message passing (None: no edges)."""
        if self.graph == "none":
            return None
        if self.graph == "static":
            return self.static_adj.expand(batch, steps, -1, -1)
        if adj_dynamic is None:
            raise ValueError("graph 'dynamic' needs adj_dynamic (B, T, N, N)")
        self._check_local(adj_dynamic)
        a = adj_dynamic
        if self.adaptive:
            a = a + self.adaptive_adjacency()
        return a

    def adaptive_adjacency(self) -> torch.Tensor:
        """(N, N) learned weights on neighbour edges only; each destination's in-weights sum
        to 1 (softmax over its in-neighbours)."""
        logits = torch.relu(self.e_src @ self.e_dst.T)
        logits = logits.masked_fill(~self.neighbour_mask, float("-inf"))
        w = torch.softmax(logits, dim=0)
        return torch.nan_to_num(w, nan=0.0)  # a node with no in-neighbour gets no learned edge

    def node_states(self, x_delhi: torch.Tensor, x_up: torch.Tensor,
                    adj_dynamic: torch.Tensor | None = None) -> torch.Tensor:
        """x_delhi (B, T, Fd), x_up (B, T, N-1, Fu), adj_dynamic (B, T, N, N) -> node states (B, T, N, H)."""
        b, t, _ = x_delhi.shape
        if x_up.shape[2] != self.n_nodes - 1:
            raise ValueError(f"expected {self.n_nodes - 1} upstream nodes, got {x_up.shape[2]}")
        e = torch.cat([self.embed_delhi(x_delhi).unsqueeze(2), self.embed_up(x_up)], dim=2)  # (B, T, N, H)
        a = self.adjacency(adj_dynamic, b, t)
        s = e.new_zeros(b, self.n_nodes, self.hidden)
        states = []
        for day in range(t):
            if a is None or day == 0:  # day 0: no previous-day states yet
                msg = torch.zeros_like(s)
            else:
                at = a[:, day - 1]  # (B, N, N), source -> destination: previous day's edges
                msg = self.w_msg(torch.einsum("bji,bjh->bih", at, s) / (1.0 + at.sum(dim=1)).unsqueeze(-1))
            inp = torch.cat([e[:, day], msg], dim=-1).reshape(b * self.n_nodes, 2 * self.hidden)
            s = self.cell(inp, s.reshape(b * self.n_nodes, self.hidden)).reshape(b, self.n_nodes, self.hidden)
            states.append(s)
        return torch.stack(states, dim=1)

    def forward(self, x_delhi: torch.Tensor, x_up: torch.Tensor, adj_dynamic: torch.Tensor | None = None) -> torch.Tensor:
        last = self.node_states(x_delhi, x_up, adj_dynamic)[:, -1]  # (B, N, H)
        if self.graph == "none":
            return self.head(torch.cat([last[:, 0], last[:, 1:].mean(dim=1)], dim=-1))
        return self.head(last[:, 0])
