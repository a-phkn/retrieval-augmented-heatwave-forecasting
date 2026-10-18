"""evaluation/heatwave_info_screen.py (S2): onset windows, onset matching rules, weighted ridge, pass rule."""
import numpy as np
import pandas as pd
import pytest

import evaluation.heatwave_info_screen as hs
from evaluation.retrieval_information_check import ridge_fit_predict


def test_onset_windows_flag_windows_whose_targets_contain_an_episode_start():
    idx = pd.date_range("2000-05-01", periods=20, freq="D")
    ep = np.zeros(20, int)
    ep[8:12] = 3  # episode 3 starts on day 8
    d = pd.DataFrame({"episode_id": ep}, index=idx)
    flags = hs.onset_windows(d, idx[:15])
    assert flags[np.arange(4, 9)].all() and not flags[:4].any() and not flags[9:].any()  # targets q..q+4 cover day 8


def test_match_score_respects_the_chronological_buffer_and_own_episode():
    cand = pd.date_range("2000-01-01", periods=60, freq="D")
    v = np.eye(3)[np.r_[np.zeros(59, int), [1]]].astype(float)  # all candidates point along x, the last along y
    members = np.zeros(60, bool)
    members[[5, 50]] = True
    q = pd.DatetimeIndex([cand[40]])
    ep = np.zeros(60, int)
    s = hs.match_score(v[[0]], v, members, q, cand, np.array([0]), ep)
    assert s[0] == pytest.approx(1.0)  # member 5 is eligible (>= 19 days earlier); member 50 is in the future
    ep[5] = 7
    assert hs.match_score(v[[0]], v, members, q, cand, np.array([7]), ep)[0] == 0.0  # own episode excluded


def test_weighted_ridge_with_unit_weights_matches_the_plain_ridge():
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(200, 4)), rng.normal(size=200)
    np.testing.assert_allclose(hs.wridge_fit_predict(x, y, np.ones(200), x[:5]),
                               ridge_fit_predict(x, y[:, None], x[:5])[:, 0], atol=1e-8)


def test_pass_rule_needs_extreme_gain_no_all_days_loss_and_beating_the_random_control():
    res = {"extreme": {"B vs base": (-0.2, -0.3, -0.1), "B vs B-rand": (-0.1, -0.2, -0.01), "D vs base": (-0.2, -0.3, -0.1)},
           "all": {"B vs base": (0.0, -0.01, 0.01), "D vs base": (0.02, 0.01, 0.03)}}
    assert hs.passes(res, "B") == (True, True, True)
    assert hs.passes(res, "D") == (True, False)  # significantly worse on all days
