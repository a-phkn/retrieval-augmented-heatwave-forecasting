"""followup/global_models.py: daily values, pairing and tests on small hand-made data (offline, no test-period reads)."""
import json

import numpy as np
import pandas as pd
import pytest

from followup import global_models as gm
from followup import open_meteo as om

SPOT = {"latitude": 28.25, "longitude": 77.0, "elevation": 274.0}


def weather(times: pd.DatetimeIndex) -> pd.DataFrame:
    """Plausible Delhi pre-monsoon hours (IST): hot afternoons, dry air, sunshine by day (past-hour means)."""
    h = times.hour.to_numpy() + times.minute.to_numpy() / 60
    sun = np.clip(np.sin(np.pi * (h - 6.0) / 13.0), 0.0, None) * ((h >= 6) & (h <= 19))
    return pd.DataFrame({"time": times, "t": 32 + 8 * np.cos(2 * np.pi * (h - 15) / 24),
                         "rh": 30 - 10 * np.cos(2 * np.pi * (h - 15) / 24), "pressure": 975.0,
                         "wind": 2.5 + np.sin(2 * np.pi * h / 24), "ghi": 950 * sun, "diffuse": 190 * sun})


def nine_cells(df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([df.assign(cell=c) for c in range(1, 10)], ignore_index=True)


def test_daily_tmax_needs_24_hours_in_every_cell():
    times = pd.date_range("2024-05-01", periods=48, freq="h")
    rows = [pd.DataFrame({"cell": c, "time": times, "t": c + times.hour / 10 + 5 * (times.day - 1)}) for c in range(1, 10)]
    hourly = pd.concat(rows, ignore_index=True)
    out = gm.daily_tmax(hourly)
    assert list(out.index) == list(pd.to_datetime(["2024-05-01", "2024-05-02"])) and out.iloc[0] == pytest.approx(7.3)
    gap = hourly.drop(hourly[(hourly["cell"] == 3) & (hourly["time"] == pd.Timestamp("2024-05-02 13:00"))].index)
    assert list(gm.daily_tmax(gap).index) == [pd.Timestamp("2024-05-01")]  # day 2 incomplete in one cell -> dropped


def test_daily_wbgt_matches_the_frozen_pipeline(monkeypatch):
    import pipeline.build_wbgt_liljegren as bw

    w = weather(pd.date_range("2024-05-01", periods=48, freq="h"))
    monkeypatch.setattr(bw, "load_cell_hourly", lambda cell, raw_dir=None: w[["time", "t", "rh", "wind", "pressure"]])
    monkeypatch.setattr(bw, "load_cell_radiation", lambda cell, raw_dir=None: (w[["time", "ghi", "diffuse"]], (28.25, 77.0)))
    frozen = bw.daily_from_hourly(bw.cell_hourly_wbgt(1))["wbgt_lj_max"]
    ours, failed = gm.daily_wbgt(nine_cells(w), {c: SPOT for c in range(1, 10)})
    assert failed == 0 and len(ours) == 2
    np.testing.assert_allclose(ours.to_numpy(), frozen.to_numpy(), rtol=0, atol=1e-12)


def test_align_uses_the_lead_L_value_of_the_target_day_and_drops_gaps():
    q = pd.to_datetime(["2024-05-01", "2024-05-01", "2024-05-02", "2024-05-02"])
    lead = np.array([1, 2, 1, 2])
    ours = pd.DataFrame({"query_date": q, "lead": lead, "target_date": q + pd.to_timedelta(lead - 1, unit="D"),
                         "pred": 0.0, "actual": 40.0, "stratum": "normal", "cluster_id": 20241, "seed": 0, "error": 0.0})
    by_lead = {1: pd.Series({pd.Timestamp("2024-05-01"): 41.0, pd.Timestamp("2024-05-02"): 42.0}),
               2: pd.Series({pd.Timestamp("2024-05-02"): 52.0})}  # lead 2 of 05-02 (target 05-03) is missing
    g = gm.align(ours, by_lead, "X")
    assert list(zip(g["query_date"].dt.day, g["lead"], g["pred"])) == [(1, 1, 41.0), (1, 2, 52.0), (2, 1, 42.0)]
    assert (g["error"] == g["pred"] - 40.0).all() and (g["seed"] == 0).all()
    assert gm.restrict(ours, g)[gm.KEY].equals(g[gm.KEY])


def test_paired_tests_compare_the_same_rows():
    from evaluation.stats import make_cluster_ids
    from pipeline.labels_v2 import in_season

    rng = np.random.default_rng(0)
    d = pd.date_range("2024-01-01", "2026-06-30", freq="D")
    base = pd.DataFrame({"query_date": d, "lead": 1, "target_date": d, "actual": 30.0, "stratum": "normal",
                         "cluster_id": make_cluster_ids(d), "seed": 0})
    base.loc[(d.month == 5) & (d.day <= 5), "stratum"] = "extreme"
    c3 = base.assign(pred=30 + rng.normal(0, 2.0, len(d)))
    g = base.assign(pred=30 + rng.normal(0, 1.0, len(d)))
    c3, g = (x.assign(error=x["pred"] - x["actual"]) for x in (c3, g))
    t = gm.paired_tests(c3, g, in_season)
    rmse = lambda x: float(np.sqrt(np.mean(x["error"] ** 2)))  # noqa: E731
    assert t["all"]["delta"] == pytest.approx(rmse(g) - rmse(c3)) and t["all"]["ci_high"] < 0
    assert t["all"]["n_clusters"] == 8 and t["all"]["few_clusters"]
    assert not np.isfinite(t["season"]["ci_low"])  # 3 hot-season clusters: no CI, by design
    assert gm._delta(t["season"]).endswith("no CI (< 5 clusters), fragile (< 10 clusters)")


def test_cached_previous_runs_become_daily_values_per_lead(tmp_path):
    times = pd.date_range("2024-05-01", periods=48, freq="h")
    w = weather(times)
    cols = {"temperature_2m": "t", "relative_humidity_2m": "rh", "surface_pressure": "pressure",
            "wind_speed_10m": "wind", "shortwave_radiation": "ghi", "diffuse_radiation": "diffuse"}
    for cell in range(1, 10):
        for group in om.column_groups(om.WBGT_VARS):
            hourly = {"time": times.strftime("%Y-%m-%dT%H:%M").tolist()}
            for c in group:
                v, L = c.split("_previous_day")
                hourly[c] = (w[cols[v]] + (int(L) if v == "temperature_2m" else 0)).round(6).tolist()
            path = om.previous_path("m", cell, times[0].date(), group, tmp_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({**SPOT, "hourly": hourly}), encoding="utf-8")
    daily, where, failed = gm.model_daily("m", om.WBGT_VARS, ("tmax", "wbgt"), tmp_path)
    assert failed == 0 and sorted(daily["tmax"]) == [1, 2, 3, 4, 5] and where[5] == SPOT
    assert daily["tmax"][3].iloc[0] == pytest.approx(w["t"].iloc[:24].max() + 3)  # lead L's own column
    assert len(daily["wbgt"][1]) == 2 and daily["wbgt"][1].between(20, 45).all()


def test_single_runs_become_leads_counted_from_the_run_date(tmp_path):
    run = pd.Timestamp("2024-05-01")
    times = pd.date_range("2024-05-01 17:30", periods=168, freq="h")  # 12 UTC run, hours at XX:30 IST
    w = weather(times)
    w["t"] = w["t"] + (times.normalize() - run).days  # day r + k is k degrees warmer
    loc = {**SPOT, "hourly": {"time": times.strftime("%Y-%m-%dT%H:%M").tolist(),
                              **{v: w[gm.SHORT[v]].round(6).tolist() for v in om.WBGT_VARS}}}
    (tmp_path / om.SINGLE_DIR).mkdir()
    (tmp_path / om.SINGLE_DIR / "2024-05-01.json").write_text(json.dumps([loc] * 9), encoding="utf-8")
    daily, _, failed, unavailable = gm.single_daily(tmp_path)
    assert failed == 0 and unavailable == []
    for L in range(1, 6):
        s = daily["tmax"][L]
        assert list(s.index) == [run + pd.Timedelta(days=L)]
        day = w[times.normalize() == run + pd.Timedelta(days=L)]
        assert s.iloc[0] == pytest.approx(day["t"].max())
        assert len(daily["wbgt"][L]) == 1


def test_a_model_added_after_the_first_results_is_marked_and_downloaded():
    assert gm.ADDED_LATER <= set(gm.MODELS)
    for name in gm.ADDED_LATER:
        assert gm._label(name) == f"{name} (added after the first results)"
        assert gm.MODELS[name][0] == om.PULLS[name][0]  # the analysis reads the model id the download fetches
    assert gm._label("IFS") == "IFS"


def test_extract_has_one_row_per_model_target_lead_and_day():
    s = pd.Series([1.0, 2.0], index=pd.to_datetime(["2024-05-01", "2024-05-02"]))
    out = gm.extract_frame({"IFS": {"tmax": {1: s, 2: s}}, "AIFS": {"tmax": {1: s}}})
    assert list(out.columns) == ["model", "target", "lead", "date", "value"] and len(out) == 6
