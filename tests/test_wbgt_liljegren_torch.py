"""pipeline/wbgt_liljegren_torch.py: same values as the numpy reference, and correct gradients
(the physics head's exact formula, decision 2026-10-07)."""
import numpy as np
import pytest
import torch

from pipeline import wbgt_liljegren as ref
from pipeline import wbgt_liljegren_torch as lt


def _inputs(n: int, seed: int = 0) -> dict[str, np.ndarray]:
    """Plausible hourly Delhi-like inputs, day and night, dry and humid."""
    rng = np.random.default_rng(seed)
    cosz = np.where(rng.random(n) < 0.8, rng.uniform(0.02, 1.0, n), 0.0)
    return {"t": rng.uniform(10, 47, n), "rh": rng.uniform(5, 98, n), "p": rng.uniform(960, 1005, n),
            "wind": rng.uniform(0.2, 9, n), "ghi": np.where(cosz > 0, rng.uniform(0, 1050, n) * cosz, 0.0),
            "fdir": rng.uniform(0, 1, n), "cosz": cosz}


def _torch(x: dict[str, np.ndarray], requires_grad: bool = False) -> dict[str, torch.Tensor]:
    return {k: torch.tensor(v, dtype=torch.float64, requires_grad=requires_grad) for k, v in x.items()}


def test_values_match_the_numpy_reference():
    x = _inputs(3000)
    expected = ref.wbgt(x["t"], x["rh"], x["p"], x["wind"], x["ghi"], x["ghi"] * x["fdir"], x["cosz"])
    t = _torch(x)
    got = lt.wbgt(t["t"], t["rh"], t["p"], t["wind"], t["ghi"], t["fdir"], t["cosz"])
    for k in ("wbgt", "tg", "tnwb", "wind2m"):
        assert np.allclose(got[k].detach().numpy(), expected[k], atol=2e-4, equal_nan=False), k


def test_gradients_match_finite_differences():
    """Analytic (implicit-function) gradients vs central differences of the forward value,
    with a tight solver tolerance so the differences are not solver noise. Points are kept
    away from the wind-class thresholds, where the exponent switches by design."""
    x = _inputs(40, seed=3)
    x["wind"] = np.clip(x["wind"], 3.3, 4.7)  # one stability band for speed
    x["ghi"] = np.where(x["cosz"] > 0, np.clip(x["ghi"], 200, 650), 0.0)  # one band for sun
    tol = 1e-11
    t = _torch(x, requires_grad=True)
    out = lt.wbgt(t["t"], t["rh"], t["p"], t["wind"], t["ghi"], t["fdir"], t["cosz"], tol=tol)["wbgt"]
    grads = torch.autograd.grad(out.sum(), [t[k] for k in ("t", "rh", "p", "wind", "ghi", "fdir")])
    for name, g in zip(("t", "rh", "p", "wind", "ghi", "fdir"), grads):
        eps = {"t": 1e-4, "rh": 1e-3, "p": 1e-2, "wind": 1e-4, "ghi": 1e-2, "fdir": 1e-5}[name]
        up, dn = dict(x), dict(x)
        up[name], dn[name] = x[name] + eps, x[name] - eps
        f = lambda z: lt.wbgt(*(torch.tensor(z[k], dtype=torch.float64)  # noqa: E731
                                for k in ("t", "rh", "p", "wind", "ghi", "fdir", "cosz")), tol=tol)["wbgt"].numpy()
        fd = (f(up) - f(dn)) / (2 * eps)
        mask = np.isfinite(fd)
        if name in ("ghi", "fdir"):
            mask &= x["cosz"] > 0  # no sun term at night
        assert np.allclose(g.numpy()[mask], fd[mask], rtol=1e-4, atol=1e-6), name


def test_known_directions_of_the_physics():
    """More humidity, more sun or hotter air -> higher WBGT; more wind cools in the sun."""
    x = _torch({k: v[:1] for k, v in _inputs(1).items()})
    base = dict(t=torch.tensor([38.0]), rh=torch.tensor([40.0]), p=torch.tensor([980.0]), wind=torch.tensor([2.5]),
                ghi=torch.tensor([800.0]), fdir=torch.tensor([0.7]), cosz=torch.tensor([0.9]))
    for k in base:
        base[k] = base[k].double().requires_grad_(True)
    w = lt.wbgt(*(base[k] for k in ("t", "rh", "p", "wind", "ghi", "fdir", "cosz")))["wbgt"]
    g = dict(zip(("t", "rh", "wind", "ghi"), torch.autograd.grad(w.sum(), [base[k] for k in ("t", "rh", "wind", "ghi")])))
    assert g["t"] > 0 and g["rh"] > 0 and g["ghi"] > 0 and g["wind"] < 0
    assert x["t"].dtype == torch.float64


def test_float32_batches_work_for_training():
    x = _inputs(256, seed=5)
    t = {k: torch.tensor(v, dtype=torch.float32, requires_grad=True) for k, v in x.items()}
    out = lt.wbgt(t["t"], t["rh"], t["p"], t["wind"], t["ghi"], t["fdir"], t["cosz"])["wbgt"]
    out.mean().backward()
    assert torch.isfinite(out).all() and all(torch.isfinite(t[k].grad).all() for k in ("t", "rh", "p", "wind", "ghi", "fdir"))
    expected = ref.wbgt(x["t"], x["rh"], x["p"], x["wind"], x["ghi"], x["ghi"] * x["fdir"], x["cosz"])["wbgt"]
    assert np.abs(out.detach().numpy() - expected).max() < 0.01
