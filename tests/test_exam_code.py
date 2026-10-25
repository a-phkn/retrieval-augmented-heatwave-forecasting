"""training/test_run.py and evaluation/test_evaluation.py: refusal while locked, Holm, plan contents.
No test-period data is read."""
import pytest

import evaluation.test_evaluation as te
import training.test_run as tr
from training import test_lock


def test_test_run_and_scoring_refuse_while_locked(monkeypatch):
    # Simulate "not frozen yet" by pointing the lock at a tag that does not exist (works with or without the real tag)
    monkeypatch.setattr(test_lock, "TAG", "no-such-tag-for-this-test")
    monkeypatch.setattr(test_lock, "_unlocked", False)
    with pytest.raises(test_lock.TestLocked):  # guard first: if this did not raise, the mains below would train/score
        test_lock.require()
    with pytest.raises(test_lock.TestLocked):
        tr.main(["--stage", "main"])
    with pytest.raises(test_lock.TestLocked):
        te.main()


def test_holm_adjusts_step_down_and_stays_monotone():
    adj = te.holm({"H1": 0.01, "H2": 0.04, "H3": 0.03})
    assert adj == pytest.approx({"H1": 0.03, "H2": 0.06, "H3": 0.06})
    assert te.holm({"a": 0.5, "b": 0.6}) == pytest.approx({"a": 1.0, "b": 1.0})  # 2 * 0.5 = 1.0; b kept monotone
    assert te.holm({"a": 0.6, "b": 0.7}) == pytest.approx({"a": 1.0, "b": 1.0})  # capped at 1
    assert te.holm({"a": 0.02, "b": 0.02}) == pytest.approx({"a": 0.04, "b": 0.04})  # ties
    assert te.holm({"only": 0.03}) == pytest.approx({"only": 0.03})  # one test: unchanged
    assert te.holm({}) == {}


def test_plan_trains_the_protocol_models_with_existing_configs():
    assert tr.MAIN == ["A2Lr_hw5_C3hop", "A2Lr_hw5", "A1prime_hw5_C3hop", "A1prime_hw5"]
    assert tr.MAIN_RETRIEVAL == ["A2Lr_hw5_Rg"] and "A2Lr_hw5_Rg" not in tr.EXTRA
    for run_id in tr.MAIN + tr.MAIN_RETRIEVAL + tr.EXTRA + [tr.RZ_RUN]:
        assert (tr.tu.REPO_ROOT / "configs" / f"{run_id}.json").exists(), run_id
    assert "test" in tr.tu.TEST_FOLDS and tr.RZ_FOLD in tr.tu.TEST_FOLDS
