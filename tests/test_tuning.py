"""The graph tuning round (decisions.md 2026-10-07/08): DSTGNN readout + dropout, the trainer's
hparams / inner-block scoring, and training/tune_backbone.py's grid and selection."""
import json

import numpy as np
import pandas as pd
import pytest
import torch

import training.train_unified as tu
import training.tune_backbone as tb
from models.dstgnn import DSTGNN
from training.folds import FOLDS, build_fold

FD, FU, T, N = 7, 4, 14, 6


def _batch(b=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, T, FD, generator=g), torch.randn(b, T, N - 1, FU, generator=g)


def _static(**kw):
    torch.manual_seed(0)
    return DSTGNN(N, FD, FU, graph="static", static_adj=torch.rand(N, N), **kw)


# ------------------------------------------------------------------ model


def test_default_model_is_unchanged_by_the_new_options():
    """dropout 0 adds no module call and no parameter: the runs before 2026-10-08 stay reproducible."""
    a, b = _static(), _static(readout="delhi", dropout=0.0)
    assert a.drop is None and a.state_dict().keys() == b.state_dict().keys()
    xd, xu = _batch()
    assert torch.equal(a.train()(xd, xu), b.train()(xd, xu))


def test_pooled_readout_sees_upstream_states_directly():
    m = _static(readout="pool").eval()
    assert m.head[0].in_features == 2 * m.hidden
    xd, xu = _batch()
    # Upstream input on the LAST day only: with the Delhi readout it cannot reach the forecast
    # (one hop per day); with the pooled readout it does.
    xu2 = xu.clone()
    xu2[:, -1] += 5.0
    assert not torch.allclose(m(xd, xu), m(xd, xu2))
    d = _static().eval()
    assert torch.allclose(d(xd, xu), d(xd, xu2))


def test_mode_none_always_pools_and_bad_options_are_rejected():
    torch.manual_seed(0)
    assert DSTGNN(N, FD, FU, graph="none").readout == "pool"
    with pytest.raises(ValueError, match="readout"):
        _static(readout="max")
    with pytest.raises(ValueError, match="dropout"):
        _static(dropout=1.0)


def test_dropout_acts_in_training_only():
    m = _static(dropout=0.2)
    xd, xu = _batch()
    m.eval()
    assert torch.equal(m(xd, xu), m(xd, xu))
    m.train()
    torch.manual_seed(1)
    o1 = m(xd, xu)
    torch.manual_seed(2)
    assert not torch.allclose(o1, m(xd, xu))


# ------------------------------------------------------------------ trainer


def _cfg(tmp_path, name, **extra):
    cfg = {**json.loads((tu.REPO_ROOT / "configs" / f"{name}.json").read_text(encoding="utf-8")), **extra}
    p = tmp_path / f"{cfg['run_id']}.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return p


def test_hparams_and_readout_validation(tmp_path):
    hp = {"hidden": 64, "lr": 3e-4, "dropout": 0.2}
    assert tu.load_config(_cfg(tmp_path, "A2Lr_hw5_C3", hparams=hp, graph={"mode": "static", "adaptive": False,
                                                                           "readout": "pool"}))["hparams"] == hp
    assert tu.load_config(_cfg(tmp_path, "A2Lr_hw5_U1", hparams=hp))
    for bad in ({"hparams": hp, "run_id": "ctl"},):  # the control LSTM is never retuned
        with pytest.raises(ValueError, match="hparams"):
            tu.load_config(_cfg(tmp_path, "A2Lr_hw5", **bad))
    for bad_hp in ({"hidden": 64, "lr": 3e-4}, {"hidden": 0, "lr": 3e-4, "dropout": 0.0},
                   {"hidden": 64, "lr": 3e-4, "dropout": 1.0}, {"hidden": 64, "lr": 1, "dropout": 0.0}):
        with pytest.raises(ValueError, match="hparams"):
            tu.load_config(_cfg(tmp_path, "A2Lr_hw5_C3", hparams=bad_hp))
    with pytest.raises(ValueError, match="readout"):
        tu.load_config(_cfg(tmp_path, "A2Lr_hw5_C3", graph={"mode": "static", "adaptive": False, "readout": "max"}))


def test_graph_run_configs_keep_the_hash_they_ran_with():
    """The new optional keys must not change the hashes of the configs already run."""
    import hashlib

    reg = pd.read_csv(tu.REGISTRY, dtype=str, keep_default_na=False)
    for name in ("A2Lr_hw5_C3", "A1prime_hw5_U1", "A2Lr_hw5_C4a"):
        cfg = tu.load_config(tu.REPO_ROOT / "configs" / f"{name}.json")
        h = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
        assert set(reg.loc[reg["run_id"] == name, "config_sha256"]) == {h}


@pytest.mark.parametrize("name", ["A2Lr_hw5_U1", "A2Lr_hw5_C3"])
def test_score_stop_writes_inner_block_forecasts(tmp_path, monkeypatch, name):
    monkeypatch.setattr(tu, "MAX_EPOCHS", 1)
    extra = {"hparams": {"hidden": 16, "lr": 3e-4, "dropout": 0.2}, "run_id": f"t_{name}"}
    if name.endswith("C3"):
        extra["graph"] = {"mode": "static", "adaptive": False, "readout": "pool"}
    cfg = tu.load_config(_cfg(tmp_path, name, **extra))
    reg = tmp_path / "tuning_runs.csv"
    tu.run(cfg, folds=["f1"], seeds=[0], out_dir=tmp_path / "p", model_dir=tmp_path / "m", threads=2,
           score_stop=True, registry=reg)
    stop = pd.read_parquet(tmp_path / "p" / cfg["run_id"] / "f1_stop.parquet")
    data = build_fold("f1", cfg["target"], cfg["labels"], target_form=cfg["target_form"])
    _, inner = tu.fit_and_stop_sets(data, "inner_2y")
    np.testing.assert_allclose(stop["actual"].to_numpy(), inner.y_raw.reshape(-1), atol=1e-4)
    assert stop["query_date"].max() < pd.Timestamp("2007-01-01")  # never the validation block
    assert np.isfinite(stop["pred"]).all() and stop["pred"].between(0, 50).all()  # back in deg C
    row = pd.read_csv(reg).loc[0]
    assert row["hparams"] == "hidden=16 lr=0.0003 dropout=0.2"
    assert row["backbone"] == ("lstm_upstream" if name.endswith("U1") else "dstgnn static pool")


def test_score_stop_refuses_retrieval_and_heads(tmp_path):
    cfg = tu.load_config(tu.REPO_ROOT / "configs" / "A2Lr_hw5_R1.json")
    with pytest.raises(ValueError, match="score_stop"):
        tu.run(cfg, folds=["f1"], seeds=[0], out_dir=tmp_path, model_dir=tmp_path, score_stop=True)


# ------------------------------------------------------------------ tune_backbone


def test_grids_are_the_agreed_eight_and_ids_are_unique():
    for arm in tb.ARMS:
        g = tb.grid(arm)
        assert len(g) == 8 and len({json.dumps(x, sort_keys=True) for x in g}) == 8
        assert {x["hidden"] for x in g} == ({64, 128} if arm == "U1" else {32, 64})
        assert {x["lr"] for x in g} == {1e-3, 3e-4} and {x["dropout"] for x in g} == {0.0, 0.2}
    ids = [c["run_id"] for f in tb.FAMILIES for a in tb.ARMS for c in tb.tuning_configs(f, a)]
    assert len(ids) == len(set(ids)) == 48


def test_tuning_configs_are_valid_and_differ_from_their_base_only_as_intended(tmp_path):
    for family in tb.FAMILIES:
        for arm in tb.ARMS:
            base = json.loads((tb.CONFIG_DIR / f"{family}_{'U1' if arm == 'U1' else 'C3'}.json").read_text())
            for cfg in tb.tuning_configs(family, arm):
                p = tmp_path / "c.json"
                p.write_text(json.dumps(cfg))
                tu.load_config(p)
                changed = {k for k in cfg if cfg[k] != base.get(k)}
                assert changed <= {"run_id", "seeds", "hparams", "description", "graph"}
                assert cfg["seeds"] == [0, 1, 2]
                assert cfg.get("graph", {}).get("readout", "delhi") == ("pool" if arm == "C3pool" else "delhi")


def _fake_stop(run_dir, rmse, seeds=(0, 1, 2), with_val=False):
    run_dir.mkdir(parents=True, exist_ok=True)
    for fold in FOLDS:
        pd.DataFrame({"seed": np.repeat(seeds, 4), "error": rmse}).to_parquet(run_dir / f"{fold}_stop.parquet")
        if with_val:  # a validation file that would flip the choice if it were read
            pd.DataFrame({"seed": np.repeat(seeds, 4), "error": 100.0 - rmse}).to_parquet(run_dir / f"{fold}.parquet")


def test_selection_reads_only_the_inner_block_and_breaks_ties_by_grid_order(tmp_path):
    for family in tb.FAMILIES:
        for arm in tb.ARMS:
            for i, cfg in enumerate(tb.tuning_configs(family, arm)):
                rmse = {2: 1.5, 5: 1.5}.get(i, 2.0 + i)  # combos 2 and 5 tie for best
                _fake_stop(tmp_path / cfg["run_id"], rmse, with_val=True)
    sel = tb.select(tmp_path)
    for family in tb.FAMILIES:
        for arm in tb.ARMS:
            r = sel[family][arm]
            assert r["selected"] == tb.tuning_configs(family, arm)[2]["run_id"]
            assert r["scores"][r["selected"]] == pytest.approx(1.5)


def test_inner_score_needs_every_fold_and_seed(tmp_path):
    _fake_stop(tmp_path / "r", 1.0, seeds=(0, 1))
    with pytest.raises(ValueError, match="seeds"):
        tb.inner_score(tmp_path / "r")
    (tmp_path / "r" / "f4_stop.parquet").unlink()
    with pytest.raises(FileNotFoundError):
        tb.inner_score(tmp_path / "r", n_seeds=2)


def test_write_selection_writes_tuned_configs_queue_and_report(tmp_path, monkeypatch):
    for family in tb.FAMILIES:  # base configs the tuned ones are built from
        for base in ("C3", "U1"):
            (tmp_path / f"{family}_{base}.json").write_text((tb.CONFIG_DIR / f"{family}_{base}.json").read_text())
    monkeypatch.setattr(tb, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(tb, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(tb, "FINAL_QUEUE", tmp_path / "q.txt")
    sel = {f: {a: {"scores": {"x": 1.0, "y": 2.0}, "selected": "x", "hparams": tb.grid(a)[3]} for a in tb.ARMS}
           for f in tb.FAMILIES}
    ids = tb.write_selection(sel)
    assert ids == [f"{f}_{a}_tuned" for f in tb.FAMILIES for a in tb.ARMS]
    assert (tmp_path / "q.txt").read_text().split() == ids
    for i in ids:
        cfg = tu.load_config(tmp_path / f"{i}.json")
        assert cfg["hparams"] == tb.grid(i.split("_")[-2])[3] and cfg["seeds"] == list(range(10))
    assert "x **(selected)**" in (tmp_path / "graph_tuning.md").read_text(encoding="utf-8")


def test_tuned_config_is_a_valid_ten_seed_run(tmp_path):
    for family in tb.FAMILIES:
        for arm in tb.ARMS:
            cfg = tb.tuned_config(family, arm, tb.grid(arm)[5])
            p = tmp_path / f"{cfg['run_id']}.json"
            p.write_text(json.dumps(cfg))
            out = tu.load_config(p)
            assert out["seeds"] == list(range(10)) and out["folds"] == list(FOLDS) and out["threads"] == 8
