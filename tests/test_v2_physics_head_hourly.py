"""v2/physics_head_hourly.py (E3, E4): shapes, physical ranges, soft counts, gradients."""
import torch

from v2.physics_head_hourly import HOURS, HourlyPhysicsHead, soft_unsafe_day, soft_unsafe_hours

D, H, C = 5, len(HOURS), 9


def _head(in_dim=32):
    torch.manual_seed(0)
    stats = dict(t_mean=torch.full((H, C), 35.0), t_std=torch.full((H, C), 4.0),
                 p_mean=torch.full((H, C), 980.0), p_std=torch.full((H, C), 5.0))
    return HourlyPhysicsHead(in_dim, **stats)


def _cosz(B):
    hours = torch.tensor(HOURS, dtype=torch.float32)
    cz = torch.clamp(torch.sin((hours - 6.0) / 13.0 * torch.pi), min=0.0)  # 0 at 06 and ~19, peak near noon
    return cz.reshape(1, 1, H, 1).expand(B, D, H, C).contiguous()


def test_shapes_and_physical_ranges():
    head, B = _head(), 4
    out = head(torch.randn(B, 32), _cosz(B), limits=(26.0, 28.0, 30.0))
    g = out["ingredients"]
    assert out["wbgt_cells"].shape == (B, D, H, C) and out["wbgt_hourly"].shape == (B, D, H)
    assert out["wbgt_max"].shape == (B, D) and out["tmax"].shape == (B, D)
    assert ((g["rh"] >= 0) & (g["rh"] <= 100)).all() and (g["wind"] > 0).all()
    assert (g["ghi"] <= 1100.0 * g["cosz"] + 1e-4).all() and (g["ghi"][g["cosz"] == 0] == 0).all()
    assert (g["tmax"] >= g["t"].amax(dim=2) - 1e-5).all()
    assert torch.isfinite(out["wbgt_cells"]).all()


def test_unsafe_counts_are_ordered_across_limits():
    """One WBGT series, several limits: a stricter limit can never give fewer unsafe hours."""
    head, B = _head(), 3
    out = head(torch.randn(B, 32), _cosz(B), limits=(25.0, 26.0, 28.0, 30.0))
    u = out["unsafe_hours"]
    assert (u[25.0] >= u[26.0]).all() and (u[26.0] >= u[28.0]).all() and (u[28.0] >= u[30.0]).all()


def test_soft_counts_approach_hard_counts():
    w = torch.tensor([[25.0, 27.0, 29.0, 29.5, 27.9, 28.1, 30.0, 26.0]])
    assert soft_unsafe_hours(w, 28.0, 0.0).item() == 4.0
    assert abs(soft_unsafe_hours(w, 28.0, 1e-3).item() - 4.0) < 1e-3
    assert soft_unsafe_day(w, 28.0, 0.0).item() == 1.0              # 29.0, 29.5 consecutive
    lone = torch.tensor([[29.0, 27.0, 29.0, 27.0]])
    assert soft_unsafe_day(lone, 28.0, 0.0).item() == 0.0           # no two in a row
    assert soft_unsafe_day(lone, 28.0, 1e-3).item() < 1e-3


def test_gradients_reach_every_parameter():
    head, B = _head(), 2
    out = head(torch.randn(B, 32), _cosz(B), limits=(28.0,), sharpness=0.5)
    loss = out["unsafe_hours"][28.0].sum() + out["tmax"].sum() + out["unsafe_day"][28.0].sum()
    loss.backward()
    for name, p in head.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name
