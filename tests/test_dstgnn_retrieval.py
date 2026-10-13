"""models/dstgnn_retrieval.py (G-R*): C3-hop + analogue attention at the Delhi readout."""
import torch

from models.dstgnn_multihop import MultiHopDSTGNN
from models.dstgnn_retrieval import RetrievalMultiHopDSTGNN
from pipeline.graph import geographic_weights, neighbour_mask, node_coords

FD, FU, T, K = 7, 4, 14, 5


def _model(seed=0):
    c = node_coords()
    torch.manual_seed(seed)
    m = RetrievalMultiHopDSTGNN(len(c), FD, FU, graph="static", hops=4,
                                neighbour_mask=torch.from_numpy(neighbour_mask(c)),
                                static_adj=torch.as_tensor(geographic_weights(c), dtype=torch.float32))
    return m.eval(), len(c)


def _batch(n, b=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    mask = torch.ones(b, K, dtype=torch.bool)
    mask[1, 3:] = False  # window 1: only 3 analogues
    mask[2] = False  # window 2: no analogue at all
    return (torch.randn(b, T, FD, generator=g), torch.randn(b, T, n - 1, FU, generator=g),
            torch.randn(b, K, T, FD, generator=g), torch.randn(b, K, 5, generator=g), mask)


def test_shapes_and_attention_respects_the_mask():
    m, n = _model()
    with torch.no_grad():
        pred, att = m(*_batch(n))
    assert pred.shape == (3, 5) and att.shape == (3, K)
    torch.testing.assert_close(att[:2].sum(dim=1), torch.ones(2))
    assert (att[1, 3:] == 0).all() and (att[2] == 0).all()


def test_no_analogue_means_zero_context_and_masked_slots_are_ignored():
    m, n = _model()
    xd, xu, xa, ya, mask = _batch(n)
    with torch.no_grad():
        pred, _ = m(xd, xu, xa, ya, mask)
        last = m.node_states(xd, xu)[:, -1]
        r = torch.cat([last[:, 0], m._transport(last, m.w_read)[:, 0]], dim=-1)
        torch.testing.assert_close(pred[2], m.head(torch.cat([r[2], torch.zeros(m.hidden)])))
        xa2, ya2 = xa.clone(), ya.clone()
        xa2[1, 3:] += 10.0
        ya2[1, 3:] += 10.0
        xa2[2] += 10.0
        torch.testing.assert_close(m(xd, xu, xa2, ya2, mask)[0], pred)


def test_analogue_outcomes_reach_the_forecast_and_so_does_every_upstream_last_day():
    m, n = _model()
    xd, xu, xa, ya, mask = _batch(n)
    with torch.no_grad():
        base, _ = m(xd, xu, xa, ya, mask)
        ya2 = ya.clone()
        ya2[0, 0] += 3.0
        assert not torch.equal(m(xd, xu, xa, ya2, mask)[0][0], base[0])
        for j in range(n - 1):  # C3-hop's property is kept: every point's last input day reaches Delhi
            xu2 = xu.clone()
            xu2[:, -1, j] += 5.0
            assert not torch.equal(m(xd, xu2, xa, ya, mask)[0][:2], base[:2]), j


def test_gradients_reach_every_parameter_and_backbone_is_c3_hop():
    m, n = _model()
    m.train()
    pred, _ = m(*_batch(n))
    pred.sum().backward()
    assert isinstance(m, MultiHopDSTGNN) and m.hops == 4
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())
