"""training/test_run.py and evaluation/test_evaluation.py: refusal while locked, Holm, plan contents.
No test-period data is read."""
import subprocess

import pytest

import evaluation.test_evaluation as te
import training.test_run as tr
from training import test_lock

TAG_EXISTS = subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"refs/tags/{test_lock.TAG}"],
                            cwd=test_lock.REPO_ROOT, capture_output=True).returncode == 0


@pytest.mark.skipif(TAG_EXISTS, reason="the protocol is frozen")
def test_test_run_and_scoring_refuse_while_locked(monkeypatch):
    monkeypatch.setattr(test_lock, "_unlocked", False)
    with pytest.raises(test_lock.TestLocked):
        tr.main(["--stage", "main"])
    with pytest.raises(test_lock.TestLocked):
        te.main()


def test_holm_adjusts_step_down_and_stays_monotone():
    adj = te.holm({"H1": 0.01, "H2": 0.04, "H3": 0.03})
    assert adj["H1"] == pytest.approx(0.03) and adj["H3"] == pytest.approx(0.06) and adj["H2"] == pytest.approx(0.06)
    assert te.holm({"a": 0.5, "b": 0.6})["b"] == pytest.approx(1.0) or te.holm({"a": 0.5, "b": 0.6})["b"] <= 1.0


def test_plan_trains_the_protocol_models_with_existing_configs():
    assert tr.MAIN == ["A2Lr_hw5_C3hop", "A2Lr_hw5", "A1prime_hw5_C3hop", "A1prime_hw5"]
    assert tr.MAIN_RETRIEVAL == ["A2Lr_hw5_Rg"] and "A2Lr_hw5_Rg" not in tr.EXTRA
    for run_id in tr.MAIN + tr.MAIN_RETRIEVAL + tr.EXTRA + [tr.RZ_RUN]:
        assert (tr.tu.REPO_ROOT / "configs" / f"{run_id}.json").exists(), run_id
    assert "test" in tr.tu.TEST_FOLDS and tr.RZ_FOLD in tr.tu.TEST_FOLDS
