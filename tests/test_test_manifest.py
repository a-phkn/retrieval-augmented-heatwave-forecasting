"""The committed one-time test results stay exactly as produced (scripts/make_test_manifest.py), and the
confirmatory H1-H4 numbers still follow from the committed test forecasts. Reads only committed files; it does not
call the test lock, so it also runs in CI."""
import json

import pytest

import evaluation.test_evaluation as te
from scripts.make_manifest import load_manifest, verify_manifest
from scripts.make_test_manifest import COVERED, TEST_MANIFEST_PATH, covered_files


def test_test_outputs_and_inputs_match_the_manifest():
    problems, _ = verify_manifest(load_manifest(TEST_MANIFEST_PATH))
    assert not problems, "\n".join(problems)


def test_manifest_covers_every_tracked_test_file():
    listed = {e["path"] for e in load_manifest(TEST_MANIFEST_PATH)["files"]}
    assert set(covered_files()) <= listed, "a tracked file under the covered folders is not fingerprinted"
    assert set(COVERED) >= {"predictions_test", "evaluation_test", "registry/test_runs.csv"}


def test_confirmatory_result_reproduces_from_the_committed_forecasts():
    committed = json.loads((te.OUT_DIR / "test_results.json").read_text(encoding="utf-8"))["primary"]
    recomputed = te.primary()
    assert set(recomputed) == {"H1", "H2", "H3", "H4"} == set(committed)
    for h, r in recomputed.items():
        c = committed[h]
        assert r["delta"] == pytest.approx(c["delta"], abs=1e-9), h
        assert r["ci_low"] == pytest.approx(c["ci_low"], abs=1e-9) and r["ci_high"] == pytest.approx(c["ci_high"], abs=1e-9), h
        assert r["p_holm"] == pytest.approx(c["p_holm"], abs=1e-9), h
        assert r["supported"] is c["supported"], h
    assert [h for h, r in recomputed.items() if r["supported"]] == ["H1", "H3"]
