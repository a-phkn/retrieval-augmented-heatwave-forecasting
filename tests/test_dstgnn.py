"""Gate G-D1 (plan v5): the DSTGNN skeleton and its graph geometry are correct, checked on
synthetic inputs (no upstream data needed). models/dstgnn.py, pipeline/graph.py."""
import numpy as np
import pytest
import torch

from models.dstgnn import DSTGNN
from pipeline.graph import (DELHI, advective_weights, bearing_deg, distance_km, geographic_weights,
                            neighbour_mask, node_coords)

T, FD, FU = 14, 13, 10


def _hops(mask: np.ndarray, src: int, dst: int) -> int:
    """Shortest number of directed edges from src to dst (99 if unreachable)."""
    seen, frontier, h = {src}, {src}, 0
    while frontier and dst not in frontier:
        frontier = {j for i in frontier for j in np.flatnonzero(mask[i])} - seen
        seen |= frontier
        h += 1
    return h if dst in frontier else 99


# ------------------------------------------------------------------ geometry


def test_node_set_is_delhi_plus_the_27_upstream_points():
    c = node_coords()
    assert c.shape == (28, 2) and tuple(c[0]) == DELHI
    assert len({tuple(x) for x in c}) == 28


def test_distance_and_bearing_known_values():
    c = np.array([[28.0, 76.0], [28.0, 78.0], [30.0, 76.0]])
    d = distance_km(c)
    assert np.allclose(d, d.T) and np.all(np.diag(d) == 0)
    assert 195 < d[0, 1] < 197  # 2 deg of longitude at 28 N ~ 196 km
    assert 221 < d[0, 2] < 224  # 2 deg of latitude ~ 222 km
    b = bearing_deg(c)
    assert abs(b[0, 1] - 89.5) < 1.0  # due east (great circle starts slightly north of east)
    assert abs(b[0, 2] - 0.0) < 1e-6 and abs(b[2, 0] - 180.0) < 1e-6


def test_advective_edges_only_point_downwind():
    c = np.array([[28.0, 76.0], [28.0, 74.0]])  # node 1 is west of node 0
    speed = np.array([5.0, 5.0])
    from_west = advective_weights(speed, np.array([270.0, 270.0]), c)  # westerly: blows towards the east
    assert from_west[1, 0] > 0 and from_west[0, 1] == 0  # west node feeds the east node, not back
    from_east = advective_weights(speed, np.array([90.0, 90.0]), c)
    assert from_east[1, 0] == 0 and from_east[0, 1] > 0
    calm = advective_weights(np.zeros(2), np.array([270.0, 270.0]), c)
    assert np.all(calm == 0)
    assert np.all(np.diagonal(from_west) == 0)


def test_advective_weights_broadcast_over_batch_and_days():
    c = node_coords()
    rng = np.random.default_rng(0)
    w = advective_weights(rng.uniform(0, 8, (3, T, 28)), rng.uniform(0, 360, (3, T, 28)), c)
    assert w.shape == (3, T, 28, 28) and (w >= 0).all()
    one = advective_weights(rng.uniform(0, 8, 28), rng.uniform(0, 360, 28), c)
    assert one.shape == (28, 28)
    g = geographic_weights(c)
    assert np.allclose(g, g.T) and np.all(np.diag(g) == 0)
    assert np.array_equal(g > 0, neighbour_mask(c))  # edges only between neighbours
    assert np.all(w[..., ~neighbour_mask(c)] == 0)


def test_graph_is_local_and_connected():
    """Edges only between points <= 370 km apart: each lattice point's 8 surrounding points,
    Delhi's 8 nearest; a point 1,000 km away is several hops from Delhi. The cutoff sits in a
    distance gap, so moving Delhi by 0.15 degrees does not change the graph."""
    c = node_coords()
    m = neighbour_mask(c)
    assert m.sum(axis=0)[0] == 8 and m.sum(axis=0)[1:].max() <= 9
    for shift in ((0.15, 0), (-0.15, 0), (0, 0.15), (0, -0.15)):
        c2 = c.copy()
        c2[0] += shift
        assert np.array_equal(neighbour_mask(c2), m), shift
    far = list(map(tuple, c)).index((24.0, 68.0))
    assert not m[far, 0]
    assert _hops(m, far, 0) >= 3
    assert all(_hops(m, j, 0) < 99 for j in range(1, 28))  # every node can reach Delhi


# ------------------------------------------------------------------ model


def _batch(b=2, n=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, T, FD, generator=g), torch.randn(b, T, n - 1, FU, generator=g), \
        torch.rand(b, T, n, n, generator=g)


def _model(graph="dynamic", n=6, seed=0, **kw):
    torch.manual_seed(seed)
    if graph == "static" and "static_adj" not in kw:
        kw["static_adj"] = torch.rand(n, n)
    return DSTGNN(n, FD, FU, graph=graph, **kw).eval()


@pytest.mark.parametrize("graph", ["none", "static", "dynamic"])
def test_output_shape_and_gradients_reach_every_parameter(graph):
    m = _model(graph).train()
    xd, xu, a = _batch()
    out = m(xd, xu, a)
    assert out.shape == (2, 5)
    out.sum().backward()
    used = {n for n, p in m.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
    expected = {n for n, _ in m.named_parameters()}
    if graph == "none":
        expected.discard("w_msg.weight")  # no edges -> the message weight is unused by design
    assert used == expected


def test_states_are_causal_in_time():
    m = _model()
    xd, xu, a = _batch()
    s = m.node_states(xd, xu, a)
    xd2, xu2 = xd.clone(), xu.clone()
    xd2[:, 9:] += 3.0
    xu2[:, 9:] -= 3.0
    s2 = m.node_states(xd2, xu2, a)
    assert torch.allclose(s[:, :9], s2[:, :9])  # days before the change are untouched
    assert not torch.allclose(s[:, 9:], s2[:, 9:])


def test_upstream_reaches_delhi_only_through_edges():
    m = _model()
    xd, xu, _ = _batch()
    zero = torch.zeros(2, T, 6, 6)
    xu2 = xu + 5.0
    with torch.no_grad():
        assert torch.allclose(m(xd, xu, zero), m(xd, xu2, zero))  # no edges: upstream cannot matter
        a = zero.clone()
        a[:, :, 3, 0] = 1.0  # one edge, node 3 -> Delhi
        assert not torch.allclose(m(xd, xu, a), m(xd, xu2, a))


def test_information_travels_one_hop_per_day():
    """Chain node 2 -> node 1 -> Delhi: a change at node 2 on the last day cannot reach
    Delhi; on the second-to-last day it reaches node 1 only; earlier it reaches Delhi."""
    m = _model(n=3)
    xd, xu, _ = _batch(n=3)
    a = torch.zeros(2, T, 3, 3)
    a[:, :, 2, 1] = 1.0
    a[:, :, 1, 0] = 1.0
    with torch.no_grad():
        base = m.node_states(xd, xu, a)
        for day, reaches in ((T - 1, False), (T - 2, False), (T - 3, True)):
            xu2 = xu.clone()
            xu2[:, day, 1] += 5.0  # upstream index 1 = node 2
            s = m.node_states(xd, xu2, a)
            assert torch.allclose(s[:, -1, 0], base[:, -1, 0]) != reaches, day


def test_far_upstream_heat_reaches_delhi_after_as_many_days_as_graph_hops():
    """On the real 28-node graph: a change at the farthest point reaches Delhi's final state
    only if it happened at least `hops` days before the last input day."""
    c = node_coords()
    m = neighbour_mask(c)
    far = list(map(tuple, c)).index((24.0, 68.0))
    h = _hops(m, far, 0)
    model = _model("static", n=28, static_adj=torch.as_tensor(geographic_weights(c), dtype=torch.float32))
    xd, xu, _ = _batch(n=28)
    with torch.no_grad():
        base = model.node_states(xd, xu)[:, -1, 0]
        for day, reaches in ((T - h, False), (T - 1 - h, True)):
            xu2 = xu.clone()
            xu2[:, day, far - 1] += 5.0
            # exact comparison: after 5 untrained hops the effect is real but tiny
            assert torch.equal(model.node_states(xd, xu2)[:, -1, 0], base) != reaches, (day, h)


def test_transport_uses_the_previous_days_edges():
    """The message into day t uses day t-1's edges (the wind that carried air from t-1 to t)."""
    m = _model()
    xd, xu, _ = _batch()
    for edge_day, reaches in ((T - 2, True), (T - 1, False)):
        a = torch.zeros(2, T, 6, 6)
        a[:, edge_day, 3, 0] = 1.0  # node 3 -> Delhi on one day only
        xu2 = xu.clone()
        xu2[:, T - 2, 2] += 5.0  # upstream index 2 = node 3, changed on day T-2
        with torch.no_grad():
            same = torch.allclose(m(xd, xu, a), m(xd, xu2, a))
        assert same != reaches, edge_day


def test_adaptive_edges_stay_local_and_sum_to_one():
    c = node_coords()
    mask = torch.as_tensor(neighbour_mask(c))
    torch.manual_seed(0)
    model = DSTGNN(28, FD, FU, graph="dynamic", adaptive=True, neighbour_mask=mask)
    w = model.adaptive_adjacency().detach()
    assert torch.all(w[~mask] == 0)
    assert torch.allclose(w.sum(dim=0), torch.ones(28))
    with pytest.raises(ValueError, match="neighbour_mask"):
        DSTGNN(28, FD, FU, graph="dynamic", adaptive=True)


def test_non_local_edges_are_rejected_when_a_mask_is_given():
    c = node_coords()
    mask = torch.as_tensor(neighbour_mask(c))
    far = list(map(tuple, c)).index((24.0, 68.0))
    dense = torch.ones(28, 28) - torch.eye(28)
    with pytest.raises(ValueError, match="non-neighbours"):
        DSTGNN(28, FD, FU, graph="static", static_adj=dense, neighbour_mask=mask)
    model = DSTGNN(28, FD, FU, graph="dynamic", neighbour_mask=mask)
    xd, xu, _ = _batch(n=28)
    a = torch.as_tensor(geographic_weights(c), dtype=torch.float32).expand(2, T, 28, 28).clone()
    model(xd, xu, a)  # local edges are fine
    a[:, 3, far, 0] = 1.0  # a one-day jump from 24N 68E to Delhi
    with pytest.raises(ValueError, match="non-neighbours"):
        model(xd, xu, a)


def test_upstream_nodes_are_interchangeable():
    """Reordering upstream nodes (and their edges) leaves Delhi's forecast unchanged."""
    m = _model()
    xd, xu, a = _batch()
    perm = torch.tensor([3, 0, 4, 1, 2])  # permutation of the 5 upstream nodes
    full = torch.cat([torch.tensor([0]), perm + 1])
    a_p = a[:, :, full][:, :, :, full]
    with torch.no_grad():
        assert torch.allclose(m(xd, xu, a), m(xd, xu[:, :, perm], a_p), atol=1e-6)


def test_adaptive_adjacency_is_learned_and_dynamic_only():
    m = _model(adaptive=True, neighbour_mask=~torch.eye(6, dtype=torch.bool))
    xd, xu, a = _batch()
    a = a * (1 - torch.eye(6))  # no self-loops: they lie outside the mask
    with torch.no_grad():
        base = m(xd, xu, a)
        m.e_src.add_(1.0)
        assert not torch.allclose(base, m(xd, xu, a))
    with pytest.raises(ValueError, match="adaptive"):
        DSTGNN(6, FD, FU, graph="static", static_adj=torch.rand(6, 6), adaptive=True)


def test_invalid_configurations_are_rejected():
    with pytest.raises(ValueError):
        DSTGNN(6, FD, FU, graph="grid")
    with pytest.raises(ValueError, match="static_adj"):
        DSTGNN(6, FD, FU, graph="static")
    m = _model()
    xd, xu, _ = _batch()
    with pytest.raises(ValueError, match="adj_dynamic"):
        m(xd, xu)
    with pytest.raises(ValueError, match="upstream nodes"):
        m(xd, xu[:, :, :3], torch.rand(2, T, 6, 6))
