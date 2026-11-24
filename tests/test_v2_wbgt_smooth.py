"""v2/wbgt_smooth.py (E2): exact at tau = 0, small bias at the default widths, gradients see the class switch."""
import numpy as np
import torch

from pipeline import wbgt_liljegren_torch as lt
from tests.test_wbgt_liljegren_torch import _inputs, _torch
from v2 import wbgt_smooth as s

ARGS = ("t", "rh", "p", "wind", "ghi", "fdir", "cosz")


def test_tau_zero_reproduces_the_frozen_port_exactly():
    x = _torch(_inputs(5000, seed=11))
    hard = lt.wbgt(*(x[k] for k in ARGS))
    soft0 = s.wbgt(*(x[k] for k in ARGS), tau_wind=0.0, tau_ghi=0.0)
    for k in ("wbgt", "tg", "tnwb", "wind2m"):
        assert torch.equal(hard[k], soft0[k]), k


def test_bias_at_default_widths_is_small():
    """Measured 24 Nov on 20,000 random Delhi-like hours: mean 0.0025 C, p99 0.048 C, max 0.125 C.
    The bound here guards against regressions."""
    x = _torch(_inputs(20000, seed=7))
    d = (s.wbgt(*(x[k] for k in ARGS))["wbgt"] - lt.wbgt(*(x[k] for k in ARGS))["wbgt"]).abs()
    assert d.mean() < 0.005 and d.quantile(0.99) < 0.06 and d.max() < 0.15


def test_away_from_thresholds_values_are_unchanged():
    x = _inputs(2000, seed=2)
    """At least 9 widths from every threshold (sigmoid tail < 1.3e-4), tight solver tolerance."""
    x["wind"] = np.clip(x["wind"], 3.9, 4.1)                        # inside one day wind band
    x["ghi"] = np.where(x["cosz"] > 0, np.clip(x["ghi"], 300, 550), 0.0)  # inside one sun band
    t = _torch(x)
    d = (s.wbgt(*(t[k] for k in ARGS), tol=1e-9)["wbgt"] - lt.wbgt(*(t[k] for k in ARGS), tol=1e-9)["wbgt"]).abs()
    assert d.max() < 1e-3


def test_gradient_crosses_the_class_switch_and_matches_finite_differences():
    """At a wind threshold the frozen port's exponent jumps with zero gradient; the smooth one has a
    gradient that agrees with central differences of its own value."""
    one = lambda v: torch.tensor([v], dtype=torch.float64)  # noqa: E731
    base = dict(t=one(40.0), rh=one(30.0), p=one(975.0), ghi=one(800.0), fdir=one(0.7), cosz=one(0.8))
    for w0 in (1.98, 2.0, 2.02, 3.0, 5.0, 6.0):
        wind = one(w0).requires_grad_(True)
        f = lambda w: s.wbgt(base["t"], base["rh"], base["p"], w, base["ghi"], base["fdir"], base["cosz"],  # noqa: E731
                             tol=1e-11)["wbgt"]
        (g,) = torch.autograd.grad(f(wind).sum(), wind)
        eps = 1e-5
        fd = (f(one(w0 + eps)) - f(one(w0 - eps))) / (2 * eps)
        assert torch.isfinite(g).all()
        assert np.isclose(g.item(), fd.item(), rtol=1e-3, atol=1e-5), w0


def test_ghi_gradient_includes_the_stability_class():
    one = lambda v: torch.tensor([v], dtype=torch.float64)  # noqa: E731
    ghi = one(675.0).requires_grad_(True)
    f = lambda q: s.wbgt(one(40.0), one(30.0), one(975.0), one(4.0), q, one(0.7), one(0.8), tol=1e-11)["wbgt"]  # noqa: E731
    (g,) = torch.autograd.grad(f(ghi).sum(), ghi)
    eps = 1e-3
    fd = (f(one(675.0 + eps)) - f(one(675.0 - eps))) / (2 * eps)
    assert np.isclose(g.item(), fd.item(), rtol=1e-3, atol=1e-6)


def test_float32_training_batch():
    x = _inputs(512, seed=5)
    t = {k: torch.tensor(v, dtype=torch.float32, requires_grad=True) for k, v in x.items()}
    out = s.wbgt(*(t[k] for k in ARGS))["wbgt"]
    out.mean().backward()
    assert torch.isfinite(out).all()
    assert all(torch.isfinite(t[k].grad).all() for k in ("t", "rh", "p", "wind", "ghi", "fdir"))
