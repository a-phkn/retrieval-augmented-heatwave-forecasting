"""models/dstgnn_multihop.py ("C3-hop", ring averaging): correctness on the real 28-node graph."""
import numpy as np
import pytest
import torch

from models.dstgnn import DSTGNN
from models.dstgnn_multihop import HOPS, MultiHopDSTGNN, hop_distances, ring_matrices
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


def test_rings_are_exact_distance_averages_and_k_is_the_largest_distance_to_delhi():
    mask = neighbour_mask(node_coords())
    d = hop_distances(mask)
    assert d[:, 0].max() == HOPS == 4  # every upstream point is within 4 hops of Delhi
    assert np.bincount(d[:, 0]).tolist() == [1, 8, 10, 5, 4]
    r = ring_matrices(mask, HOPS)
    for k in range(1, HOPS + 1):
        sizes = (d == k).sum(axis=0)
        np.testing.assert_allclose(r[k - 1].sum(axis=0), (sizes > 0).astype(float), atol=1e-6)  # averages
        assert ((r[k - 1] > 0) == (d == k)).all()


def test_output_shape_and_gradients_reach_every_parameter():
    m, c = _real()
    m.train()
    xd, xu = _batch(len(c))
    out = m(xd, xu)
    assert out.shape == (2, 5)
    out.sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())
    assert not hasattr(m, "w_msg")


def test_every_upstream_points_last_input_day_reaches_the_forecast_and_c3s_does_not():
    m, c = _real()
    torch.manual_seed(0)
    c3 = DSTGNN(len(c), FD, FU, graph="static", static_adj=torch.as_tensor(geographic_weights(c), dtype=torch.float32)).eval()
    xd, xu = _batch(len(c))
    with torch.no_grad():
        base, base_c3 = m(xd, xu), c3(xd, xu)
        for j in range(len(c) - 1):
            xu2 = xu.clone()
            xu2[:, -1, j] += 5.0
            assert not torch.equal(m(xd, xu2), base), j
            assert torch.equal(c3(xd, xu2), base_c3), j  # the original C3 cannot see it


def _last_day_gradient_by_hop(model, c, seeds=range(3)) -> np.ndarray:
    d = hop_distances(neighbour_mask(c))[:, 0]
    out = np.zeros(HOPS)
    for s in seeds:
        xd, xu = _batch(len(c), b=8, seed=s)
        xu.requires_grad_(True)
        model(xd, xu).abs().sum().backward()
        g = xu.grad[:, -1].abs().mean(dim=(0, 2)).numpy()  # per upstream point
        out += np.array([g[d[1:] == k].mean() for k in range(1, HOPS + 1)])
    return out / len(seeds)


def test_distant_points_are_not_drowned_out():
    """The design point of ring averaging (focused review M1): at initialisation a 4-hop point's
    last day moves the forecast about as much as a 1-hop neighbour's (DCRNN-style diffusion gave
    ~100x less)."""
    m, c = _real()
    g = _last_day_gradient_by_hop(m, c)
    assert (g > 0).all() and g.max() / g.min() < 5, g


def test_too_few_hops_cannot_reach_the_farthest_point_on_the_last_day():
    c = node_coords()
    far = list(map(tuple, c)).index((24.0, 68.0))
    assert hop_distances(neighbour_mask(c))[far, 0] == 4
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
    mask = torch.from_numpy(neighbour_mask(c))
    with pytest.raises(ValueError, match="static"):
        MultiHopDSTGNN(len(c), FD, FU, graph="none")
    with pytest.raises(ValueError, match="static"):
        MultiHopDSTGNN(len(c), FD, FU, graph="dynamic", neighbour_mask=mask)
    with pytest.raises(ValueError, match="readout"):
        MultiHopDSTGNN(len(c), FD, FU, graph="static", static_adj=adj, neighbour_mask=mask, readout="pool")
    with pytest.raises(ValueError, match="neighbour_mask"):
        MultiHopDSTGNN(len(c), FD, FU, graph="static", static_adj=adj)
    with pytest.raises(ValueError, match="hops"):
        MultiHopDSTGNN(len(c), FD, FU, graph="static", static_adj=adj, neighbour_mask=mask, hops=0)


def test_dropout_acts_in_training_only():
    m, c = _real(dropout=0.2)
    xd, xu = _batch(len(c))
    m.eval()
    assert torch.equal(m(xd, xu), m(xd, xu))
    m.train()
    torch.manual_seed(1)
    o1 = m(xd, xu)
    torch.manual_seed(2)
    assert not torch.allclose(o1, m(xd, xu))
