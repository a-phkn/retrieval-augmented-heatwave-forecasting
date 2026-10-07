import numpy as np

import evaluation.retrieval_information_check as ric


def test_ridge_recovers_a_linear_signal_and_ignores_noise():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2000, 3))
    y = (2.0 * x[:, :1] + 0.5) + 0.01 * rng.normal(size=(2000, 1))
    pred = ric.ridge_fit_predict(x[:1500], y[:1500], x[1500:])
    assert np.sqrt(np.mean((pred - y[1500:]) ** 2)) < 0.05


def test_analogue_signal_averages_the_analogues_next_days():
    z = np.arange(30, dtype=float)
    cand_pos = np.array([0, 10, 20])
    sig = ric.analogue_signal(z, np.array([[0, 2], [1, 1]]), cand_pos)
    assert sig.shape == (2, ric.FORECAST_DAYS)
    assert np.allclose(sig[0], (z[0:5] + z[20:25]) / 2) and np.allclose(sig[1], z[10:15])


def test_cluster_bootstrap_centres_on_the_point_estimate_and_brackets_it():
    rng = np.random.default_rng(1)
    years = np.repeat(np.arange(8), 50)
    vals = rng.normal(-1.0, 0.1, size=years.size)
    point, lo, hi = ric.cluster_boot(lambda w: float(np.sum(vals * w) / np.sum(w)), years, rng)
    assert lo < point < hi and hi < 0


def test_rmse_seedmean_weights_rows():
    se = np.array([[1.0, 9.0], [4.0, 4.0]])
    assert ric.rmse_seedmean(se) == (np.sqrt(5.0) + 2.0) / 2
    assert ric.rmse_seedmean(se, np.array([1.0, 0.0])) == (1.0 + 2.0) / 2
