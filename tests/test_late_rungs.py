"""evaluation/late_rungs.py: the physics-head keep rule and R2's G3 check, on made-up comparisons."""
import evaluation.late_rungs as lr


def _cmp(ci_high, ci_low=-1.0):
    return {"delta": 0.0, "ci_low": ci_low, "ci_high": ci_high, "p_value": 0.5, "n_clusters": 36}


def _fake_physics(monkeypatch, wbgt_hi, tmax_hi):
    import pandas as pd

    df = pd.DataFrame({"seed": [0], "error": [1.0], "pred": [1.0], "actual": [0.0], "stratum": ["normal"]})
    monkeypatch.setattr(lr, "load_run", lambda rid: df)
    monkeypatch.setattr(lr, "load_physics_tmax", lambda run_id="PH_lstm": df)
    his = iter([wbgt_hi, 0.0, tmax_hi, 0.0])  # order of compare calls: WBGT all, WBGT extreme, Tmax all, Tmax extreme
    monkeypatch.setattr(lr, "compare", lambda a, b, s: _cmp(next(his)))


def test_keep_rule_drops_the_head_if_either_target_reaches_the_margin(monkeypatch, tmp_path):
    import json

    (tmp_path / "PH_lstm").mkdir()
    for f in lr.FOLDS:
        (tmp_path / "PH_lstm" / f"{f}_training_log.json").write_text(json.dumps([{"finite": True, "skipped_batches": 0}]))
    monkeypatch.setattr(lr, "PRED_DIR", tmp_path)
    for wbgt_hi, tmax_hi, keep in ((0.049, 0.049, True), (0.05, 0.0, False), (0.0, 0.08, False)):
        _fake_physics(monkeypatch, wbgt_hi, tmax_hi)
        out = lr.physics_head()
        assert out["keep"] is keep, (wbgt_hi, tmax_hi)
        assert out["skipped_batches"] == 0 and out["finite"]


def test_r2_uses_the_three_g3_conditions():
    row = {"ctrl_all_ci_low": -0.01, "ctrl_ext_ci_high": -0.001, "rand_ext_ci_high": -0.002}
    assert all(lr.g3_conditions(row))
    assert lr.g3_conditions({**row, "rand_ext_ci_high": 0.01}) == (True, True, False)
    assert lr.g3_conditions({**row, "ctrl_all_ci_low": 0.001})[0] is False  # significantly worse on all days
