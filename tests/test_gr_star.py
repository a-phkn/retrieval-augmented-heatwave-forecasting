"""evaluation/gr_star.py: the G-R* decision wiring, the 2x2 overlap jackknife and the attention stats."""
import numpy as np
import pandas as pd
import pytest

import evaluation.gr_star as gs


def _cmp(ci_high, ci_low=-1.0):
    return {"delta": 0.0, "ci_low": ci_low, "ci_high": ci_high, "p_value": 0.5, "n_clusters": 36}


def test_decision_uses_the_three_g3_conditions_and_names_the_final_model(monkeypatch):
    df = pd.DataFrame({"seed": [0], "error": [1.0], "pred": [1.0], "actual": [0.0], "stratum": ["extreme"]})
    monkeypatch.setattr(gs, "load_run", lambda rid: df)
    monkeypatch.setattr(gs, "mde80", lambda r: 0.2)
    for (ctrl_all_low, ext_hi, rand_ext_hi), passes in (((-0.01, -0.01, -0.01), True), ((-0.01, -0.01, 0.02), False),
                                                         ((0.001, -0.01, -0.01), False), ((-0.01, 0.0, -0.01), False)):
        # compare call order per family: ctrl all, ctrl extreme, rand extreme, rand all, rand-vs-ctrl all
        seq = iter([_cmp(1.0, ctrl_all_low), _cmp(ext_hi), _cmp(rand_ext_hi), _cmp(1.0), _cmp(1.0)] * 2)
        monkeypatch.setattr(gs, "compare", lambda a, b, s: next(seq))
        out = gs.gate_rows()
        for fam, base in gs.FAMILIES.items():
            assert out[fam]["passes"] is passes
            assert out[fam]["final_model"] == (f"{base}_C3hop_Rg" if passes else f"{base}_C3hop")


def test_did_jackknife_is_zero_when_retrieval_gains_are_equal_and_signs_a_smaller_graph_gain():
    rng = np.random.default_rng(0)
    clusters = np.repeat(np.arange(12), 50)
    base = rng.normal(0, 2, clusters.size)
    errs = {"lstm": [base], "lstm_rg": [base * 0.9], "c3": [base * 0.8], "c3_rg": [base * 0.72]}
    # gains: lstm 0.1 RMSE(base) vs c3 0.08 RMSE(base) -> difference +0.02 RMSE(base) > 0 (smaller gain on graph)
    r = gs.did_jackknife(errs, clusters)
    assert r["delta"] == pytest.approx(0.02 * np.sqrt(np.mean(base ** 2)), rel=1e-9)
    same = gs.did_jackknife({**errs, "c3_rg": [base * 0.7]}, clusters)  # equal gains: 0.8 -> 0.7 and 1.0 -> 0.9
    assert same["delta"] == pytest.approx(0.0, abs=1e-12)
    assert r["n_clusters"] == 12 and r["ci_low"] <= r["delta"] <= r["ci_high"]


def test_attention_stats_uniform_and_concentrated():
    q = pd.Timestamp("2010-05-01")
    uni = pd.DataFrame({"seed": 0, "query_date": q, "attention": [0.2] * 5})
    s = gs.attention_stats(uni)
    assert s["mean_max_weight"] == pytest.approx(0.2) and s["mean_norm_entropy"] == pytest.approx(1.0)
    conc = pd.DataFrame({"seed": 0, "query_date": q, "attention": [0.96, 0.01, 0.01, 0.01, 0.01, 0.0]})
    c = gs.attention_stats(conc)  # the zero-weight slot (no analogue) is ignored
    assert c["mean_max_weight"] == pytest.approx(0.96) and c["mean_norm_entropy"] < 0.2
    assert c["uniform_max_weight"] == pytest.approx(0.2)


def test_top_quarter_uses_the_training_windows_quantile():
    train = np.arange(100, dtype=float)
    assert gs.top_quarter(train, np.array([74.0, 74.25, 99.0, 150.0])).tolist() == [False, True, True, True]


def test_upstream_heat_dates_are_validation_dates_and_a_plausible_share():
    from training.folds import fold_windows

    hot = gs.upstream_heat_dates()
    val = pd.DatetimeIndex(np.concatenate([fold_windows(f)[1]["query_date"].values for f in gs.FOLDS]))
    assert hot.isin(val).all() and hot.is_unique
    assert 0.10 < len(hot) / len(val) < 0.50
    assert (hot < pd.Timestamp("2019-01-01")).all()
