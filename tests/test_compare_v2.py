"""evaluation/compare_v2.py: per-fold baselines, pooling and paired comparison."""
import json

import numpy as np
import pandas as pd
import pytest

import evaluation.compare_v2 as cv
from scripts.make_manifest import REPO_ROOT
from training.folds import build_fold

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def test_damped_persistence_on_primary_fold_reproduces_v1_phi():
    """f4 + Tmax uses v1's training windows and climatology, so phi must equal the v1 fit."""
    v1 = json.loads((REPO_ROOT / "evaluation_v2/damped_persistence_phi.json").read_text())["phi_by_lead"]
    assert np.allclose(cv.fit_phi("f4", "t_max"), v1, atol=1e-10)


@pytest.mark.parametrize("target", ["t_max", "wbgt_bom_max"])
def test_baselines_sit_on_the_scored_windows(target):
    data = build_fold("f1", target, "v2")
    b = cv.fold_baselines("f1", target)
    n = len(data.val.query_dates) * 5
    for name, df in b.items():
        assert len(df) == n, name
        # long format is window-major, lead-minor: same order as the model's y_raw.reshape(-1)
        assert np.array_equal(df["actual"].to_numpy(), data.val.y_raw.reshape(-1).astype(np.float64)), name
        assert np.array_equal(df["stratum"].to_numpy(), data.val.stratum.reshape(-1)), name
        assert df["target_date"].between("2007-01-01", "2009-12-31").all()
    clim = b["climatology"]
    assert np.allclose(clim["pred"].to_numpy(), data.val.clim_target.reshape(-1))
    pers = b["persistence"]
    assert np.allclose(pers["pred"].to_numpy(), np.repeat(data.val.persist, 5))
    # all three baselines share windows, actuals and strata exactly
    ref = cv._labels(b["climatology"])
    for name in cv.BASELINES:
        assert cv._labels(b[name]).equals(ref)
        assert np.array_equal(b[name]["actual"].to_numpy(), b["climatology"]["actual"].to_numpy())


def test_damped_persistence_lies_between_persistence_and_climatology_anomalies():
    """0 < phi < 1 at every lead, so the forecast anomaly shrinks towards climatology."""
    for target in ("t_max", "wbgt_bom_max"):
        phi = cv.fit_phi("f2", target)
        assert ((phi > 0) & (phi < 1)).all() and (np.diff(phi) < 0).all(), (target, phi)


def test_phi_uses_only_training_years(monkeypatch):
    import training.folds as tf

    before = cv.fit_phi("f1", "wbgt_bom_max")
    original = tf._load_daily_pre_test

    def altered():
        d = original().copy()
        late = d.index > "2006-12-31"
        d.loc[late, ["t_max", "wbgt_bom_max"]] += 7.0
        return d

    monkeypatch.setattr(tf, "_load_daily_pre_test", altered)
    assert np.array_equal(cv.fit_phi("f1", "wbgt_bom_max"), before)


def _toy(model, seeds, errors_by_seed, strata):
    rows = []
    dates = pd.date_range("2010-05-01", periods=len(strata) // 5)
    for s, err in zip(seeds, errors_by_seed):
        for i, (q, lead) in enumerate([(q, l) for q in dates for l in range(1, 6)]):
            rows.append({"model": model, "seed": s, "query_date": q, "lead": lead,
                         "target_date": q + pd.Timedelta(days=lead - 1), "pred": 0.0, "actual": 0.0,
                         "error": err[i], "stratum": strata[i],
                         "cluster_id": 20100 + lead % 2})  # toy: 2 clusters (the test needs >= 2)
    return pd.DataFrame(rows)


def test_compare_rejects_misaligned_inputs_and_counts_instances():
    strata = ["normal"] * 8 + ["extreme"] * 2
    rng = np.random.default_rng(0)
    p = _toy("p", [0], [rng.normal(size=10)], strata)
    c = _toy("c", [0, 1], [rng.normal(size=10), rng.normal(size=10)], strata)
    row = cv.compare(p, c, "extreme")
    assert row["n_instances"] == 2
    bad = c.copy()
    bad.loc[bad["lead"] == 1, "stratum"] = "unusual"
    with pytest.raises(ValueError, match="disagree"):
        cv.compare(p, bad, "all")


def test_rmse_by_averages_per_seed_rmse():
    strata = ["normal"] * 10
    df = _toy("m", [0, 1], [np.full(10, 1.0), np.full(10, 3.0)], strata)
    assert cv.rmse_by(df) == pytest.approx(2.0)  # mean of per-seed RMSEs 1 and 3, not RMSE of pooled errors
    assert cv.rmse_by(df, "lead").to_numpy() == pytest.approx([2.0] * 5)


def test_forecast_conditioned_uses_the_v2_hot_rule_on_the_forecast():
    dates = pd.to_datetime(["2010-05-01", "2010-05-02", "2010-01-15"])
    df = pd.DataFrame({"seed": 0, "target_date": dates, "pred": [41.0, 39.0, 46.0], "error": [2.0, 1.0, 5.0]})
    clim = pd.Series([37.0, 37.0, 20.0], index=dates)
    out = cv.forecast_conditioned(df, clim)
    # only 1 May qualifies: 2 May is below 40 C, 15 Jan is out of season
    assert out == {"n_per_seed": 1.0, "rmse": 2.0, "bias": 2.0}


def test_load_run_pools_folds_and_requires_matching_seeds(tmp_path):
    for fold, seeds in (("f1", [0, 1]), ("f2", [0, 1])):
        (tmp_path / "R").mkdir(exist_ok=True)
        pd.DataFrame({"seed": seeds, "x": [1, 2]}).to_parquet(tmp_path / "R" / f"{fold}.parquet")
    df = cv.load_run("R", folds=("f1", "f2"), pred_dir=tmp_path)
    assert len(df) == 4 and set(df["fold"]) == {"f1", "f2"}
    assert cv.load_run("R", folds=("f1", "f3"), pred_dir=tmp_path) is None  # missing fold
    pd.DataFrame({"seed": [0], "x": [1]}).to_parquet(tmp_path / "R" / "f2.parquet")
    with pytest.raises(ValueError, match="seed sets"):
        cv.load_run("R", folds=("f1", "f2"), pred_dir=tmp_path)


def test_load_runs_reads_every_v2_config_but_not_the_repro_or_label_config():
    runs = cv.load_runs()
    assert {"A1prime", "A2", "A2r"} <= set(runs)
    assert "A1_repro" not in runs and "wbgt_label" not in runs
    assert all(c["early_stop"] == "inner_2y" for c in runs.values())


def _row(run, rmse, delta, ci_low, complexity=0):
    return {"run_id": run, "rmse_all": rmse, "dp_all_delta": delta, "complexity": complexity,
            "floor_pass": not (delta > 0 and ci_low > 0)}


def test_control_choice_follows_the_preregistered_rule():
    rows = [_row("base", 1.55, -0.04, -0.09), _row("fancy", 1.54, -0.05, -0.10, complexity=2),
            _row("worse", 1.50, +0.10, +0.04)]
    c = cv.choose_control(rows)
    # 'worse' is lowest RMSE but fails the floor; base and fancy tie within 0.02 C -> simpler
    assert c["choice"] == "base" and c["passes_floor"]
    rows = [_row("a", 2.3, +0.11, +0.05), _row("b", 2.2, +0.07, +0.01)]
    c = cv.choose_control(rows)
    assert c["choice"] == "b" and not c["passes_floor"]


def test_adopted_control_fixes_hot_weight_then_applies_the_rule():
    """Decision 2026-10-06: hot_weight = 5, then the pre-registered rule picks the form."""
    rows = [{**_row("hw1", 2.29, -0.07, -0.11, complexity=1), "hot_weight": 1},
            {**_row("raw5", 2.40, +0.03, -0.04, complexity=1), "hot_weight": 5},
            {**_row("anom5", 2.37, +0.00, -0.06, complexity=2), "hot_weight": 5}]
    assert cv.choose_control(rows)["choice"] == "hw1"  # the rule as written
    c = cv.adopted_control(rows)
    assert c["choice"] == "anom5" and c["passes_floor"]  # 0.03 C apart: beyond the tie margin
    assert "decision 2026-10-06" in c["reason"]
    assert cv.adopted_control([rows[0]]) is None  # no hot_weight-5 run in the family


def _rung(run, mode, ext, all_d=-0.01, all_lo=-0.05, ext_hi=-0.1, rand_hi=-0.05):
    return {"run_id": run, "mode": mode, "rmse_extreme": ext, "ctrl_all_delta": all_d, "ctrl_all_ci_low": all_lo,
            "ctrl_ext_ci_high": ext_hi, "rand_ext_ci_high": rand_hi}


def test_retrieval_rung_choice_follows_the_preregistered_g3_rule():
    # R1 is lowest but within 0.02 C of R0 -> the simpler R0
    c = cv.choose_retrieval_rung([_rung("r0", "sim", 3.10), _rung("r1", "time", 3.09)])
    assert c["choice"] == "r0" and c["helps"] == ["r0", "r1"]
    # each condition alone disqualifies a rung
    assert cv.choose_retrieval_rung([_rung("a", "sim", 3.0, all_d=0.05, all_lo=0.01)])["choice"] is None  # worse all days
    assert cv.choose_retrieval_rung([_rung("b", "sim", 3.0, ext_hi=0.02)])["choice"] is None  # extreme gain not significant
    # amendment: beating the random control must be significant, not a point difference
    assert cv.choose_retrieval_rung([_rung("c", "sim", 3.0, rand_hi=0.01)])["choice"] is None
    no_rand = _rung("d", "sim", 3.0)
    del no_rand["rand_ext_ci_high"]
    assert cv.choose_retrieval_rung([no_rand])["choice"] is None  # no random control run -> cannot pass
    c = cv.choose_retrieval_rung([_rung("r0", "sim", 3.20), _rung("r1", "time", 3.10)])
    assert c["choice"] == "r1"  # beyond the tie margin
    # a barely significant all-days loss (CI low just above 0) fails condition 1
    assert cv.g3_conditions(_rung("e", "sim", 3.0, all_d=0.015, all_lo=0.0004)) == (False, True, True)
    # a missing CI bound fails its condition, consistently for all three
    assert cv.g3_conditions(_rung("f", "sim", 3.0, all_lo=np.nan, ext_hi=np.nan, rand_hi=np.nan)) == (False, False, False)


def test_mde80_scales_with_the_ci_width():
    res = {"n_clusters": 11, "ci_low": -0.1, "ci_high": 0.1}
    t975 = 2.228138851986274  # t(0.975, 10 df)
    t80 = 0.8790578285505887  # t(0.80, 10 df)
    assert cv.mde80(res) == pytest.approx((t975 + t80) * 0.1 / t975)
    assert cv.mde80({**res, "ci_low": -0.2, "ci_high": 0.2}) == pytest.approx(2 * cv.mde80(res))
    assert np.isnan(cv.mde80({**res, "n_clusters": 1})) and np.isnan(cv.mde80({**res, "ci_low": np.nan}))


def test_retrieval_rows_pair_each_rung_with_its_own_random_control():
    strata = ["normal"] * 6 + ["extreme"] * 4
    rng = np.random.default_rng(1)

    def run(name):
        return _toy(name, [0, 1], [rng.normal(size=10), rng.normal(size=10)], strata)

    preds = {k: run(k) for k in ("C", "C_R0", "C_R0rand", "C_R1", "C_R1rand")}
    base = {"target": "t", "labels": "l"}
    runs = {"C": {**base, "parent": "x"},
            "C_R0": {**base, "parent": "C", "retrieval": {"mode": "sim", "k": 5}},
            "C_R0rand": {**base, "parent": "C", "retrieval": {"mode": "rand", "k": 5}},
            "C_R1": {**base, "parent": "C_R0", "retrieval": {"mode": "time", "k": 5}},
            "C_R1rand": {**base, "parent": "C", "retrieval": {"mode": "time_rand", "k": 5}}}
    cv.RETRIEVAL_CONTROLS["toy"] = "C"
    try:
        rows = {r["mode"]: r for r in cv.retrieval_rows("toy", runs, preds, [], None)}
        assert rows["sim"]["rand_run"] == "C_R0rand" and rows["time"]["rand_run"] == "C_R1rand"
        assert "rand_run" not in rows["rand"] and "rand_run" not in rows["time_rand"]
        assert all(f"rand_{s}_delta" in rows["sim"] for s in cv.STRATA)  # every stratum reported
        exp = cv.compare(preds["C_R1rand"], preds["C_R1"], "extreme")["delta"]
        assert rows["time"]["rand_extreme_delta"] == pytest.approx(exp)
        bad_parent = {**runs, "C_R1rand": {**runs["C_R1rand"], "parent": "C_R1"}}
        with pytest.raises(ValueError, match="parent"):  # a mis-parented control must not vanish silently
            cv.retrieval_rows("toy", bad_parent, preds, [], None)
        runs["C_R0k10"] = {**runs["C_R0"]}
        preds["C_R0k10"] = run("C_R0k10")
        with pytest.raises(ValueError, match="both claim"):
            cv.retrieval_rows("toy", runs, preds, [], None)
    finally:
        del cv.RETRIEVAL_CONTROLS["toy"]


def test_anen_run_sits_on_the_scored_windows():
    an = cv.anen_run("t_max", "v2", folds=("f1",))
    base = cv.fold_baselines("f1", "t_max")["climatology"]
    assert cv._labels(an).equals(cv._labels(base))
    assert np.array_equal(an["actual"].to_numpy(), base["actual"].to_numpy())
    assert np.isfinite(an["pred"]).all() and (an["pred"] != base["pred"]).mean() > 0.9


def test_relabel_and_ensemble():
    strata = ["normal"] * 10
    df = _toy("m", [0, 1], [np.full(10, 1.0), np.full(10, 3.0)], strata)
    df["pred"] = df["actual"] + df["error"]  # make the toy forecasts consistent with their errors
    by_date = pd.Series("extreme", index=pd.date_range("2010-04-01", "2010-06-30"))
    assert (cv.relabel(df, by_date)["stratum"] == "extreme").all()
    ens = cv.ensemble(df)
    assert len(ens) == 10 and (ens["error"] == 2.0).all() and set(ens["seed"]) == {0}


def test_variant_runs_are_not_control_candidates():
    assert not cv.is_variant({"hot_weight": 5})
    assert not cv.is_variant({"backbone": "lstm"})
    for cfg in ({"retrieval": {"mode": "sim", "k": 5}}, {"backbone": "dstgnn", "graph": {"mode": "none", "adaptive": False}},
                {"backbone": "lstm_upstream"}, {"head": "physics"}):
        assert cv.is_variant(cfg)
