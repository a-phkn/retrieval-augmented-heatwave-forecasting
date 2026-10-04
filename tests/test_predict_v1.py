"""
evaluation/predict_v1.py: predictions from the frozen checkpoints must reproduce
the stored v1 evaluation exactly, be aligned to the frozen window index, and
never touch the test split.
"""
import json

import numpy as np
import pandas as pd
import pytest

import evaluation.predict_v1 as pv
from scripts.make_manifest import REPO_ROOT

pytestmark = pytest.mark.skipif(not pv.seed_dirs(pv.LSTM_DIR), reason="frozen A1 checkpoints not present")


@pytest.fixture(scope="module")
def val_outputs(tmp_path_factory):
    out = tmp_path_factory.mktemp("pred")
    original = pv.OUT_DIR
    pv.OUT_DIR = out
    try:
        written = pv.run_split("val")
    finally:
        pv.OUT_DIR = original
    return {name: pd.read_parquet(path) for name, path in written.items()}


def test_refuses_test_split():
    with pytest.raises(ValueError, match="refused"):
        pv.run_split("test")


def test_schema_and_shape(val_outputs):
    lstm = val_outputs["lstm_a1"]
    assert sorted(lstm["seed"].unique()) == [0, 1, 2, 3, 4]
    assert len(lstm) == 5 * 1092 * 5
    for col in ("model", "seed", "query_date", "lead", "target_date", "pred", "actual", "error", "stratum", "cluster_id"):
        assert col in lstm.columns
    assert np.allclose(lstm["error"], lstm["pred"] - lstm["actual"])


def test_lead_days_align_with_window_index(val_outputs):
    windows = pd.read_parquet(REPO_ROOT / "splits/window_index_v1.parquet")
    val = windows[windows["split"] == "val"].set_index("query_date")
    p = val_outputs["persistence"]
    for lead, col in ((1, "forecast_start"), (5, "forecast_end")):
        got = p[p["lead"] == lead].set_index("query_date")["target_date"]
        assert (got.reindex(val.index) == val[col]).all()


def test_reproduces_stored_v1_evaluation(val_outputs):
    stored = json.loads((REPO_ROOT / "archive_v1/evaluation/baseline_lstm/val_extended_metrics.json").read_text())
    rmse = lambda e: float(np.sqrt(np.mean(e**2)))  # noqa: E731
    lstm = val_outputs["lstm_a1"]
    per_seed = lstm.groupby("seed")["error"].apply(lambda e: rmse(e.to_numpy()))
    assert per_seed.mean() == pytest.approx(stored["1_global_accuracy"]["lstm"]["rmse"]["mean"], abs=1e-5)
    for s in ("normal", "unusual", "extreme"):
        got = lstm[lstm["stratum"] == s].groupby("seed")["error"].apply(lambda e: rmse(e.to_numpy())).mean()
        assert got == pytest.approx(stored["2_stratified_rmse"]["lstm"][s]["rmse_mean"], abs=1e-5)
    for base in ("persistence", "climatology"):
        assert rmse(val_outputs[base]["error"].to_numpy()) == pytest.approx(
            stored["1_global_accuracy"][base]["rmse"], abs=1e-5
        )


def test_ra_provenance_if_present(val_outputs):
    if "ra_v1" not in val_outputs:
        pytest.skip("frozen RA checkpoints not present")
    ra = val_outputs["ra_v1"]
    attn = ra[[f"analogue{k}_attention" for k in range(1, pv.K + 1)]].to_numpy()
    assert np.allclose(attn.sum(axis=1), 1.0, atol=1e-5)  # attention over real analogues sums to 1
    for k in range(1, pv.K + 1):
        dates = ra[f"analogue{k}_date"]
        assert dates.notna().all()  # every val query has K eligible analogues
        # Split rule: val queries may only retrieve train windows (query_date <= 2015-12-31),
        # which also implies the 19-day no-future rule.
        assert (dates <= pd.Timestamp("2015-12-31")).all()
        assert ((ra["query_date"] - dates).dt.days >= 19).all()


def test_ra_reproduces_stored_evaluation(val_outputs):
    if "ra_v1" not in val_outputs:
        pytest.skip("frozen RA checkpoints not present")
    stored = json.loads((REPO_ROOT / "evaluation/retrieval_augmented/val_extended_metrics.json").read_text())
    per_seed = val_outputs["ra_v1"].groupby("seed")["error"].apply(lambda e: float(np.sqrt(np.mean(e**2))))
    assert per_seed.mean() == pytest.approx(stored["1_global_accuracy"]["rmse"]["mean"], abs=1e-5)


def test_compare_v1_pairs_identical_rows_and_reproduces_independent_pairing(val_outputs):
    from evaluation.compare_v1 import compare
    from evaluation.stats import paired_cluster_test

    parent, child = val_outputs["lstm_a1"], val_outputs["climatology"]
    row = compare(parent, child, "extreme")

    # Independent pairing: sort both by (seed, query_date, lead) and assert identical keys.
    def by_seed(df):
        df = df[df["stratum"] == "extreme"].sort_values(["seed", "query_date", "lead"])
        return [g["error"].to_numpy() for _, g in df.groupby("seed")], df[df["seed"] == df["seed"].min()]

    p_err, p_ref = by_seed(parent)
    c_err, c_ref = by_seed(child)
    assert (p_ref[["query_date", "lead"]].to_numpy() == c_ref[["query_date", "lead"]].to_numpy()).all()
    ref = paired_cluster_test(parent=p_err, child=c_err, cluster_ids=p_ref["cluster_id"].to_numpy())
    assert row["delta"] == pytest.approx(ref.delta) and row["se"] == pytest.approx(ref.se)


def test_compare_v1_rejects_mismatched_windows(val_outputs):
    from evaluation.compare_v1 import compare

    child = val_outputs["climatology"].iloc[5:]  # drop a window's rows
    with pytest.raises(ValueError, match="disagree"):
        compare(val_outputs["lstm_a1"], child, "all")
