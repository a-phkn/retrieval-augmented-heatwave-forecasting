"""training/train_unified.py: config validation, v1 reproduction, end-to-end smoke run."""
import json
import os

import numpy as np
import pandas as pd
import pytest
import torch

import training.train_unified as tu
from scripts.make_manifest import REPO_ROOT


def test_config_validation(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"run_id": "x"}))
    with pytest.raises(ValueError, match="missing keys"):
        tu.load_config(bad)
    cfg = json.loads((REPO_ROOT / "configs/A2r.json").read_text())
    cfg["folds"] = ["f9"]
    bad.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="folds"):
        tu.load_config(bad)
    cfg["folds"], cfg["early_stop"] = ["f1"], "test_block"
    bad.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="early_stop"):
        tu.load_config(bad)


@pytest.mark.parametrize("name", ["A1_repro", "A1prime", "A2", "A2r"])
def test_shipped_configs_are_valid(name):
    cfg = tu.load_config(REPO_ROOT / f"configs/{name}.json")
    assert cfg["threads"] == 8  # bit-reproducibility holds only at a fixed thread count
    if name == "A1_repro":
        assert cfg["early_stop"] == "val_block"  # v1 behaviour, kept only to reproduce v1
    else:
        assert cfg["early_stop"] == "inner_2y"  # never choose the checkpoint on the reported block
        assert cfg["seeds"] == list(range(10))  # decision 2026-10-04: 10 seeds for decision runs
        assert cfg["folds"] == ["f1", "f2", "f3", "f4"]


@pytest.mark.skipif(os.environ.get("GITHUB_ACTIONS") == "true",
                    reason="bit-exact float training holds only on the CPU type it was frozen on; "
                           "GitHub runners vary (CI runs 7523bd6, ba6659b), so this runs locally")
def test_reproduces_frozen_a1_seed0_bit_for_bit(tmp_path):
    """Configured as v1 A1 (8 threads, as frozen), seed 0 must give the frozen A1 weights
    exactly and the archived seed-0 predictions."""
    before = torch.get_num_threads()
    torch.set_num_threads(3)  # not 8, so the restore check below is meaningful
    try:
        cfg = tu.load_config(REPO_ROOT / "configs/A1_repro.json")
        res = tu.run(cfg, folds=["f4"], seeds=[0], out_dir=tmp_path / "p", model_dir=tmp_path / "m", write_registry=False)
        assert torch.get_num_threads() == 3  # run() used 8 and restored the caller's count
    finally:
        torch.set_num_threads(before)
    new_w = torch.load(tmp_path / "m/A1_repro/f4/seed_0/checkpoint.pt", weights_only=True)
    old_w = torch.load(REPO_ROOT / "models/frozen/lstm_tmax_v1/seed_0/checkpoint.pt", weights_only=True)
    assert new_w.keys() == old_w.keys()
    assert all(torch.equal(new_w[k], old_w[k]) for k in old_w)
    new = res["f4"].sort_values(["query_date", "lead"]).reset_index(drop=True)
    old = pd.read_parquet(REPO_ROOT / "predictions_v1/val/lstm_a1.parquet")
    old = old[old["seed"] == 0].sort_values(["query_date", "lead"]).reset_index(drop=True)
    assert (new[["query_date", "lead"]].to_numpy() == old[["query_date", "lead"]].to_numpy()).all()
    assert np.array_equal(new["pred"].to_numpy(), old["pred"].to_numpy())


def test_inner_split_never_touches_the_validation_block():
    from training.folds import build_fold, inner_split

    data = build_fold("f1", "t_max", "v2")
    fit, stop = tu.fit_and_stop_sets(data, "inner_2y")
    q_fit, q_stop = pd.DatetimeIndex(fit.query_dates), pd.DatetimeIndex(stop.query_dates)
    assert q_stop.min() == pd.Timestamp("2005-01-01") and q_stop.max() <= pd.Timestamp("2006-12-27")
    assert (q_fit + pd.Timedelta(days=4)).max() < pd.Timestamp("2005-01-01")  # fit targets end before stop block
    assert len(fit.query_dates) + len(stop.query_dates) == len(data.train.query_dates) - 4  # 4 crossing windows dropped
    f, s = inner_split(data.train, data.train_end)
    assert np.array_equal(f.X, fit.X) and np.array_equal(s.y, stop.y)
    assert tu.fit_and_stop_sets(data, "val_block")[1] is data.val


def test_smoke_wbgt_anomaly_run_writes_outputs_and_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(tu, "MAX_EPOCHS", 2)
    monkeypatch.setattr(tu, "REGISTRY", tmp_path / "runs.csv")
    cfg = tu.load_config(REPO_ROOT / "configs/A2r.json")
    res = tu.run(cfg, folds=["f1"], seeds=[0], out_dir=tmp_path / "p", model_dir=tmp_path / "m")
    df = res["f1"]
    assert (tmp_path / "p" / "A2r" / "f1.parquet").exists()
    assert (tmp_path / "m" / "A2r" / "f1" / "seed_0" / "checkpoint.pt").exists()
    assert df["query_date"].min() >= pd.Timestamp("2007-01-01") and df["target_date"].max() <= pd.Timestamp("2009-12-31")
    assert np.isfinite(df["pred"]).all() and df["pred"].between(5, 45).all()  # back-transformed to deg C
    reg = pd.read_csv(tmp_path / "runs.csv")
    row = reg.loc[0]
    assert row["run_id"] == "A2r" and row["fold"] == "f1" and row["early_stop"] == "inner_2y"
    assert len(row["data_sha256"]) == 64 and len(row["code_sha256"]) == 64
    assert row["torch_threads"] == 8 and row["n_stop_windows"] > 700
    # skill columns are consistent with the RMSE columns they are computed from
    assert row["skill_clim_all"] == pytest.approx(1 - row["val_rmse_all"] / row["clim_rmse_all"], abs=1e-5)
    assert row["skill_persist_extreme"] == pytest.approx(1 - row["val_rmse_extreme"] / row["persist_rmse_extreme"], abs=1e-5)


def test_reference_rmse_on_a_toy_split():
    from training.folds import SplitArrays

    y = np.array([[30.0, 31, 32, 33, 34]], dtype=np.float32)
    split = SplitArrays(
        X=np.zeros((1, 14, 1), np.float32), y=y, y_raw=y, clim_target=y.astype(np.float64) - 1.0,
        hot=np.zeros((1, 5), bool), stratum=np.array([["extreme"] + ["normal"] * 4]),
        query_dates=pd.Series([pd.Timestamp("2010-05-01")]), persist=np.array([30.0]),
    )
    ref = tu._reference_rmse(split)
    assert ref["clim_rmse_all"] == pytest.approx(1.0) and ref["clim_rmse_extreme"] == pytest.approx(1.0)
    assert ref["persist_rmse_all"] == pytest.approx(np.sqrt(np.mean(np.arange(5.0) ** 2)))
    assert ref["persist_rmse_extreme"] == pytest.approx(0.0)


def test_registry_rewrites_an_old_header_instead_of_misaligning(tmp_path, monkeypatch):
    reg = tmp_path / "runs.csv"
    monkeypatch.setattr(tu, "REGISTRY", reg)
    old_fields = [f for f in tu.REGISTRY_FIELDS if f != "target_form"]
    pd.DataFrame([{f: f"old_{f}" for f in old_fields}]).to_csv(reg, index=False)
    row = {f: f"new_{f}" for f in tu.REGISTRY_FIELDS}
    tu._append_registry(row)
    df = pd.read_csv(reg, dtype=str, keep_default_na=False)
    assert list(df.columns) == tu.REGISTRY_FIELDS
    assert df.loc[0, "run_id"] == "old_run_id" and df.loc[0, "target_form"] == ""
    assert df.loc[1, "target_form"] == "new_target_form" and df.loc[1, "val_rmse_all"] == "new_val_rmse_all"


def test_config_target_form_defaults_and_validation(tmp_path):
    cfg = json.loads((REPO_ROOT / "configs/A2r.json").read_text())
    assert tu.load_config(REPO_ROOT / "configs/A2r.json")["target_form"] == "anomaly"
    cfg["target_form"] = "dp_residual"  # contradicts anomaly_target = true
    bad = tmp_path / "c.json"
    bad.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="target_form"):
        tu.load_config(bad)


def test_every_shipped_run_config_follows_the_decision_rules():
    """All v2 run configs: valid, 10 seeds (decision 2026-10-04), 4 folds, inner early stop."""
    paths = [p for p in sorted((REPO_ROOT / "configs").glob("*.json")) if p.stem not in ("A1_repro", "wbgt_label")]
    assert len(paths) >= 18
    for p in paths:
        cfg = tu.load_config(p)
        assert cfg["run_id"] == p.stem, p
        assert cfg["seeds"] == list(range(10)) and cfg["folds"] == ["f1", "f2", "f3", "f4"], p
        assert cfg["early_stop"] == "inner_2y" and cfg["threads"] == 8, p


def test_data_files_include_upstream_only_for_regional_retrieval():
    from retrieval.fold_retrieval import UPSTREAM_PATH
    base = {"labels": "v2"}
    assert UPSTREAM_PATH not in tu._data_files(base)
    assert UPSTREAM_PATH not in tu._data_files({**base, "retrieval": {"mode": "sim", "k": 5}})
    assert UPSTREAM_PATH in tu._data_files({**base, "retrieval": {"mode": "region", "k": 5}})
    assert "pipeline/download_era5_upstream.py" in tu.CODE_FILES
