"""models/dstgnn_multihop.py ("C3-hop"): correctness on synthetic inputs and the real graph."""
import numpy as np
import pytest
import torch

from models.dstgnn import DSTGNN
from models.dstgnn_multihop import HOPS, MultiHopDSTGNN
from pipeline.graph import geographic_weights, neighbour_mask, node_coords

FD, FU, T = 7, 4, 14


def _real(hops=HOPS, seed=0, **kw):
    c = node_coords()
    torch.manual_seed(seed)
    return MultiHopDSTGNN(len(c), FD, FU, graph="static", hops=hops, neighbour_mask=torch.from_numpy(neighbour_mask(c)),
                          static_adj=torch.as_tensor(geographic_weights(c), dtype=torch.float32), **kw).eval(), c


def _batch(n, b=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, T, FD, generator=g), torch.randn(b, T, n - 1, FU, generator=g)


def _hops_to_delhi(mask: np.ndarray) -> np.ndarray:
    d = np.full(len(mask), 99)
    d[0], frontier = 0, [0]
    while frontier:
        nxt = []
        for v in frontier:
            for u in np.flatnonzero(mask[:, v]):
                if d[u] == 99:
                    d[u] = d[v] + 1
                    nxt.append(u)
        frontier = nxt
    return d


def test_k_is_the_largest_hop_distance_to_delhi():
    assert _hops_to_delhi(neighbour_mask(node_coords())).max() == HOPS == 4


def test_output_shape_and_gradients_reach_every_parameter():
    m, c = _real()
    m.train()
    xd, xu = _batch(len(c))
    out = m(xd, xu)
    assert out.shape == (2, 5)
    out.sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())
    assert not hasattr(m, "w_msg")


def test_every_upstream_points_last_input_day_reaches_the_forecast():
    """The defect it fixes: with models.dstgnn's Delhi readout, no upstream point's last day
    reaches the forecast; here every point's does (K = its hop distance or more)."""
    m, c = _real()
    xd, xu = _batch(len(c))
    torch.manual_seed(0)
    base_model = DSTGNN(len(c), FD, FU, graph="static",
                        static_adj=torch.as_tensor(geographic_weights(c), dtype=torch.float32)).eval()
    with torch.no_grad():
        base, base_old = m(xd, xu), base_model(xd, xu)
        for j in range(len(c) - 1):
            xu2 = xu.clone()
            xu2[:, -1, j] += 5.0
            assert not torch.equal(m(xd, xu2), base), j
            assert torch.equal(base_model(xd, xu2), base_old), j  # the original C3 cannot see it


def test_too_few_hops_cannot_reach_the_farthest_point_on_the_last_day():
    c = node_coords()
    far = list(map(tuple, c)).index((24.0, 68.0))
    assert _hops_to_delhi(neighbour_mask(c))[far] == 4
    xd, xu = _batch(len(c))
    xu2 = xu.clone()
    xu2[:, -1, far - 1] += 5.0
    for hops, reaches in ((3, False), (4, True)):
        m, _ = _real(hops=hops)
        with torch.no_grad():
            assert torch.equal(m(xd, xu2), m(xd, xu)) != reaches, hops


def test_states_are_causal_in_time():
    m, c = _real()
    xd, xu = _batch(len(c))
    with torch.no_grad():
        s = m.node_states(xd, xu)
        xu2 = xu.clone()
        xu2[:, 9:] += 3.0
        s2 = m.node_states(xd, xu2)
    assert torch.allclose(s[:, :9], s2[:, :9]) and not torch.allclose(s[:, 9:], s2[:, 9:])


def test_invalid_setups_are_rejected():
    c = node_coords()
    adj = torch.as_tensor(geographic_weights(c), dtype=torch.float32)
    with pytest.raises(ValueError, match="edges"):
        MultiHopDSTGNN(len(c), FD, FU, graph="none")
    with pytest.raises(ValueError, match="readout"):
        MultiHopDSTGNN(len(c), FD, FU, graph="static", static_adj=adj, readout="pool")
    with pytest.raises(ValueError, match="hops"):
        MultiHopDSTGNN(len(c), FD, FU, graph="static", static_adj=adj, hops=0)
    bad = adj.clone()
    bad[1, 27] = 1.0  # a 1000+ km one-day link
    with pytest.raises(ValueError, match="neighbours"):
        MultiHopDSTGNN(len(c), FD, FU, graph="static", static_adj=bad, neighbour_mask=torch.from_numpy(neighbour_mask(c)))


def test_dropout_and_dynamic_edges_work():
    c = node_coords()
    torch.manual_seed(0)
    m = MultiHopDSTGNN(len(c), FD, FU, graph="dynamic", dropout=0.2, neighbour_mask=torch.from_numpy(neighbour_mask(c)))
    xd, xu = _batch(len(c))
    a = torch.rand(2, T, len(c), len(c)) * torch.from_numpy(neighbour_mask(c)).float()
    assert m.train()(xd, xu, a).shape == (2, 5)
