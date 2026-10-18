"""evaluation/range_forecasts.py: CRPS formulas against brute-force references."""
import numpy as np
import pytest
from scipy.stats import norm

import evaluation.range_forecasts as rf


def test_gaussian_crps_matches_numerical_integration():
    y, mu, sd = np.array([1.3]), np.array([0.2]), np.array([0.8])
    grid = np.linspace(-10, 10, 200001)
    num = np.trapezoid((norm.cdf(grid, mu[0], sd[0]) - (grid >= y[0])) ** 2, grid)
    assert rf.crps_gaussian(y, mu, sd)[0] == pytest.approx(num, rel=1e-4)


def test_empirical_crps_matches_the_pairwise_definition():
    rng = np.random.default_rng(0)
    resid = rng.normal(0.3, 1.2, 300)
    y, c = np.array([0.5, -1.0]), np.array([0.0, 0.4])
    x = c[:, None] - resid[None, :]
    brute = np.abs(x - y[:, None]).mean(axis=1) - 0.5 * np.abs(x[:, :, None] - x[:, None, :]).mean(axis=(1, 2))
    np.testing.assert_allclose(rf.crps_empirical(y, c, resid), brute, rtol=1e-10)


def test_crps_of_a_perfect_point_forecast_is_zero_and_grows_with_error():
    assert rf.crps_empirical(np.array([2.0]), np.array([2.0]), np.zeros(10))[0] == pytest.approx(0.0)
    assert rf.crps_empirical(np.array([3.0]), np.array([2.0]), np.zeros(10))[0] == pytest.approx(1.0)
