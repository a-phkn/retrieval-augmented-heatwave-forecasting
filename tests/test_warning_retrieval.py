"""evaluation/warning_retrieval.py (W): logistic blend helpers and the analogue hot-day share."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import evaluation.warning_retrieval as wr


def test_logistic_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    x1, x2 = rng.normal(size=20000), rng.uniform(size=20000)
    p = 1 / (1 + np.exp(-(-1.0 + 0.8 * x1 + 2.0 * x2)))
    o = (rng.uniform(size=20000) < p).astype(float)
    par = wr.fit_logistic(o, x1, x2)
    np.testing.assert_allclose(par, [-1.0, 0.8, 2.0], atol=0.1)
    assert wr.predict_logistic(par, np.array([0.0]), np.array([0.5]))[0] == pytest.approx(0.5, abs=0.05)


def test_logit_is_clipped():
    assert np.isfinite(wr.logit(np.array([0.0, 1.0]))).all()


def test_analog_prob_is_the_hot_share_of_each_rows_analogues_at_its_lead():
    days = pd.date_range("2000-01-01", periods=40, freq="D")
    hot = np.zeros(40)
    hot[[11, 12, 21]] = 1
    cand = days[[10, 20, 30]]

    class FakeRetriever(SimpleNamespace):
        def retrieve(self, q, mode, k, seed=None):
            return SimpleNamespace(idx=np.array([[0, 1, -1]] * len(q)))  # analogues: windows starting day 10 and 20

    fr = FakeRetriever(d=pd.DataFrame({"hot": hot}, index=days), cand_dates=cand)
    q = pd.Series([days[35], days[35]])
    p = wr.analog_prob(fr, q, np.array([1, 2]), "region", None)
    # lead 1: days 10 (no) and 20 (no) -> 0; lead 2: days 11 (hot) and 21 (hot) -> 1
    np.testing.assert_allclose(p, [0.0, 1.0])
