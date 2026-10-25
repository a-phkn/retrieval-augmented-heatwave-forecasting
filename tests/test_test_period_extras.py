"""post_test/test_period_extras.py: warning scores and the bias share of an MSE gap, on hand-made numbers."""
import numpy as np
import pandas as pd
import pytest

import post_test.test_period_extras as tx


def test_event_scores_on_a_hand_counted_table():
    obs = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], bool)
    fc = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0], bool)  # 2 hits, 2 misses, 1 false alarm, 5 correct "no"
    s = tx.event_scores(obs, fc)
    assert (s["hits"], s["misses"], s["false_alarms"], s["n"]) == (2, 2, 1, 10)
    assert s["hit_rate"] == pytest.approx(0.5) and s["false_alarm_ratio"] == pytest.approx(1 / 3)
    chance = 4 * 3 / 10
    assert s["ets"] == pytest.approx((2 - chance) / (2 + 2 + 1 - chance))


def test_event_scores_perfect_and_chance():
    obs = np.array([1, 0, 1, 0], bool)
    assert tx.event_scores(obs, obs)["ets"] == pytest.approx(1.0)
    assert tx.event_scores(obs, np.zeros(4, bool))["hit_rate"] == 0.0
    assert np.isnan(tx.event_scores(obs, np.zeros(4, bool))["false_alarm_ratio"])  # no warnings issued
    with pytest.raises(ValueError):
        tx.event_scores(obs, np.zeros(3, bool))


def test_bias_share_is_all_bias_when_models_differ_only_by_a_constant():
    rng = np.random.default_rng(0)
    e = rng.normal(0, 1, 500)
    e -= e.mean()
    a = pd.DataFrame({"seed": 0, "error": e - 1.0})
    b = pd.DataFrame({"seed": 0, "error": e - 2.0})  # same spread, colder by 1
    r = tx.bias_share(a, b)
    assert r["a_bias2"] == pytest.approx(1.0) and r["b_bias2"] == pytest.approx(4.0)
    assert r["bias_share_of_gap"] == pytest.approx(1.0)


def test_bias_share_is_zero_when_only_the_spread_differs():
    e = np.array([-1.0, 1.0] * 50)
    r = tx.bias_share(pd.DataFrame({"seed": 0, "error": e}), pd.DataFrame({"seed": 0, "error": 2 * e}))
    assert r["bias_share_of_gap"] == pytest.approx(0.0)
