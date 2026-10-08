"""evaluation/graph_gates.py: the pre-registered decision logic, on made-up inputs only
(no graph result is read)."""
import numpy as np

import evaluation.graph_gates as gg

W, T = "WBGT (physical)", "Tmax"


def test_g_d2_conditions():
    good = [{"finite": True, "val_curve": [1.0, 0.8, 0.9]}] * 10
    seeds, ctrl = np.array([2.0, 2.01, 2.02]), np.array([2.0, 2.01, 2.02])
    assert gg.g_d2(good, seeds, ctrl)["pass"]
    assert not gg.g_d2(good[:9] + [{"finite": False, "val_curve": [1.0, 0.5]}], seeds, ctrl)["pass"]
    flat = [{"finite": True, "val_curve": [1.0, 1.1]}] * 2 + good[:8]  # 80% improved < 90%
    assert not gg.g_d2(flat, seeds, ctrl)["pass"]
    assert not gg.g_d2(good, np.array([2.0, 2.1, 2.2]), ctrl)["pass"]  # seed SD 10x the control's


def test_candidate_choice_prefers_simpler_within_the_tie_margin():
    rmse = {"C3": 2.10, "C4": 2.09, "C4a": 2.00}
    assert gg.choose_candidate(rmse, {"C3": True, "C4": True, "C4a": True}) == "C4a"
    assert gg.choose_candidate({**rmse, "C4a": 2.085}, {"C3": True, "C4": True, "C4a": True}) == "C3"  # C3 within 0.02
    assert gg.choose_candidate({"C3": 2.12, "C4": 2.09, "C4a": 2.085}, {"C3": True, "C4": True, "C4a": True}) == "C4"
    assert gg.choose_candidate({"C3": 2.10, "C4": 2.085, "C4a": 2.0}, {"C3": True, "C4": True, "C4a": False}) == "C3"
    assert gg.choose_candidate(rmse, {"C3": False, "C4": False, "C4a": False}) is None


def test_non_inferiority_uses_the_upper_ci_end():
    assert gg.non_inferior({"ci_high": 0.049}) and not gg.non_inferior({"ci_high": 0.05})
    assert not gg.non_inferior({"ci_high": float("nan")})


def test_backbone_decision_across_families():
    ok = {"candidate": "C4", "a": True, "b": True}
    assert gg.backbone_decision({T: ok, W: ok})["backbone"] == "graph C4"
    assert gg.backbone_decision({T: {**ok, "candidate": "C3"}, W: ok})["backbone"] == "graph C4"  # WBGT's pick
    assert gg.backbone_decision({T: {**ok, "a": False}, W: ok})["backbone"] == "control LSTM"
    assert gg.backbone_decision({T: {"candidate": None, "a": None, "b": None}, W: ok})["backbone"] == "control LSTM"
    assert gg.backbone_decision({T: ok, W: {**ok, "b": False}})["backbone"] == "U1"



def _fam(c2=None, c3=None, pool=None, hop=None, u1_a=True, ctrl_b=False):
    ok = {"g_d2": True, "a": True, "b": True}
    return {"C2": c2 or ok, "C3": c3 or ok, "C3pool": pool or ok, "C3hop": hop or ok, "U1": {"a": u1_a},
            "control": {"b": ctrl_b}}


def test_eligibility_needs_both_checks_in_both_families():
    bad_b = {"g_d2": True, "a": True, "b": False}
    ok = gg.eligible({T: _fam(), W: _fam(c3=bad_b)})
    assert ok == {"control": False, "U1": True, "C2": True, "C3": False, "C3pool": True, "C3hop": True}
    no_d2 = {"g_d2": False, "a": True, "b": True}
    assert not gg.eligible({T: _fam(pool=no_d2), W: _fam()})["C3pool"]  # graph code must pass G-D2
    assert gg.eligible({T: _fam(ctrl_b=True), W: _fam(ctrl_b=True)})["control"]


def test_amended_decision_picks_lowest_wbgt_rmse_with_simplicity_ties_and_stays_provisional():
    ok = {"control": False, "U1": True, "C2": True, "C3": True, "C3pool": True, "C3hop": False}
    rmse = {"control": 2.37, "U1": 2.27, "C2": 2.254, "C3": 2.30, "C3pool": 2.20, "C3hop": 2.10}
    d = gg.amended_decision(ok, rmse)
    assert d["model"] == "C3pool" and d["provisional"]
    assert gg.amended_decision(ok, {**rmse, "C3pool": 2.24})["model"] == "C2"  # within 0.02 -> simpler C2
    assert gg.amended_decision({**ok, "C2": False}, {**rmse, "C3pool": 2.24})["model"] == "C3pool"
    assert gg.amended_decision({k: False for k in ok}, rmse)["backbone"] == "control"


def test_selection_free_years_exclude_every_inner_block():
    import pandas as pd

    from training.folds import FOLDS, fold_bounds

    inner = {y for f in FOLDS for y in (fold_bounds(f)[0].year - 1, fold_bounds(f)[0].year)}
    assert not inner & set(gg.SELECTION_FREE_YEARS)
    assert set(gg.SELECTION_FREE_YEARS) | (inner & set(range(2007, 2019))) == set(range(2007, 2019))
    df = pd.DataFrame({"target_date": pd.to_datetime(["2008-05-01", "2010-05-01", "2016-01-01"])})
    assert gg.selection_free(df)["target_date"].dt.year.tolist() == [2010, 2016]


def test_multi_hop_arm_is_a_candidate_and_needs_g_d2():
    ok = gg.eligible({T: _fam(hop={"g_d2": False, "a": True, "b": True}), W: _fam()})
    assert not ok["C3hop"]
    ok = {m: True for m in gg.SIMPLICITY}
    rmse = {"control": 2.37, "U1": 2.27, "C2": 2.254, "C3": 2.30, "C3pool": 2.26, "C3hop": 2.20}
    assert gg.amended_decision(ok, rmse)["model"] == "C3hop"
    assert gg.amended_decision(ok, {**rmse, "C3hop": 2.24})["model"] == "C2"  # within 0.02 of C2 -> simpler
