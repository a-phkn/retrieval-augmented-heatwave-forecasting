"""models/physics_head.py: shapes, physical ranges, the exact-formula path and gradients."""
import numpy as np
import torch

from models.physics_head import GHI_MAX, INGREDIENTS, LSTMPhysics, PhysicsHead
from pipeline.wbgt_liljegren_torch import wbgt

STATS = {"t_mean": torch.full((9,), 33.0), "t_std": torch.full((9,), 7.0),
         "p_mean": torch.full((9,), 985.0), "p_std": torch.full((9,), 8.0)}


def test_shapes_and_physical_ranges():
    torch.manual_seed(0)
    head = PhysicsHead(16, **STATS)
    out, g = head(torch.randn(32, 16) * 3)
    assert out.shape == (32, 5) and torch.isfinite(out).all()
    assert set(g) == set(INGREDIENTS) and all(v.shape == (32, 5, 9) for v in g.values())
    assert ((g["rh"] > 0) & (g["rh"] < 100)).all() and (g["wind"] > 0).all()
    assert ((g["fdir"] >= 0) & (g["fdir"] <= 1)).all() and ((g["cosz"] >= 0) & (g["cosz"] <= 1)).all()
    assert (g["ghi"] <= GHI_MAX * g["cosz"] + 1e-4).all() and (g["ghi"] >= 0).all()


def test_output_is_the_cell_mean_of_the_exact_formula():
    torch.manual_seed(1)
    head = PhysicsHead(8, **STATS)
    out, g = head(torch.randn(4, 8))
    per_cell = wbgt(g["t"], g["rh"], g["pressure"], g["wind"], g["ghi"], g["fdir"], g["cosz"])["wbgt"]
    assert torch.allclose(out, per_cell.mean(dim=-1))


def test_gradients_reach_the_encoder_and_are_finite():
    torch.manual_seed(2)
    model = LSTMPhysics(n_features=13, head_stats=STATS)
    x = torch.randn(16, 14, 13)
    out, g = model(x)
    loss = (out - 30.0).pow(2).mean() + sum(v.pow(2).mean() for v in g.values()) * 1e-6
    loss.backward()
    grads = [p.grad for p in model.parameters()]
    assert all(gr is not None and torch.isfinite(gr).all() for gr in grads)
    assert model.lstm.weight_ih_l0.grad.abs().sum() > 0


def test_head_can_learn_a_simple_wbgt_target_quickly():
    """A sanity check that the exact-formula head is trainable: fit 64 fixed targets."""
    torch.manual_seed(3)
    head = PhysicsHead(8, **STATS)
    h = torch.randn(64, 8)
    target = torch.from_numpy(np.random.default_rng(0).uniform(25, 33, (64, 5)).astype(np.float32))
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    first = None
    for _ in range(150):
        opt.zero_grad()
        loss = (head(h)[0] - target).pow(2).mean()
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < 0.25 * first
