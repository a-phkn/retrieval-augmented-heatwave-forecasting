"""evaluation/ridge_baselines.py: the weighted ridge reduces to the G-D0 ridge, and weights act."""
import numpy as np
import pandas as pd

import evaluation.ridge_baselines as rb
from evaluation.gd0_upstream_signal import fit_with_inner_alpha, ridge


def _data(n=400, p=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    Y = X @ rng.normal(size=(p, 5)) + rng.normal(scale=0.5, size=(n, 5))
    q = pd.date_range("1990-01-01", periods=n, freq="7D")
    return X, Y, q


def test_unit_weights_reproduce_the_gd0_ridge():
    X, Y, q = _data()
    for a in (0.1, 100.0):
        c0, b0 = ridge(X, Y[:, [0]], a)
        c1, b1 = rb.weighted_ridge(X, Y[:, [0]], a, np.ones(len(X)))
        np.testing.assert_allclose(c1, c0, atol=1e-10)
        np.testing.assert_allclose(b1, b0, atol=1e-10)
    end = q[-1]
    for got, want in zip(rb.fit_weighted_inner_alpha(X, Y, np.ones_like(Y), q, end), fit_with_inner_alpha(X, Y, q, end)):
        np.testing.assert_allclose(got, want, atol=1e-10)


def test_weights_pull_the_fit_toward_the_weighted_rows():
    X, Y, _ = _data()
    hot = X[:, 0] > 1.0
    y = Y[:, [0]].copy()
    y[hot] += 3.0  # the weighted rows follow a shifted relation
    w = np.where(hot, 5.0, 1.0)
    for coef_b in (rb.weighted_ridge(X, y, 0.1, w),):
        err_w = np.abs((X[hot] @ coef_b[0] + coef_b[1] - y[hot])).mean()
    c, b = ridge(X, y, 0.1)
    assert err_w < np.abs((X[hot] @ c + b - y[hot])).mean()
    # scaling all weights does not change the fit
    c2, b2 = rb.weighted_ridge(X, y, 0.1, 3.0 * w)
    np.testing.assert_allclose(c2, coef_b[0], atol=1e-10)
