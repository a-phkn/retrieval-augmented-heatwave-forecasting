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


def test_tuned_decision_follows_the_amended_rule():
    ok = {"g_d2": True, "a": True, "b": True}
    both = lambda c3, pool: {T: {"C3": c3, "C3pool": pool}, W: {"C3": c3, "C3pool": pool}}  # noqa: E731
    assert gg.tuned_decision(both(ok, ok))["backbone"] == "graph C3 (tuned)"  # the agreed arm first
    assert gg.tuned_decision(both({**ok, "b": False}, ok))["backbone"] == "graph C3-pool (tuned)"
    assert gg.tuned_decision(both({**ok, "b": False}, {**ok, "g_d2": False}))["backbone"] == "U1 (tuned)"
    assert gg.tuned_decision(both({**ok, "a": False}, {**ok, "b": False}))["backbone"] == "control LSTM"
    mixed = {T: {"C3": ok, "C3pool": ok}, W: {"C3": {**ok, "b": False}, "C3pool": {**ok, "a": False}}}
    assert gg.tuned_decision(mixed)["backbone"] == "U1 (tuned)"  # an arm must pass in BOTH families
