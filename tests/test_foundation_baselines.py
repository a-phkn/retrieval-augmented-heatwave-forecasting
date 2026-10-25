"""post_test/foundation_baselines.py: contexts end the day before the query, median extraction, cache location."""
import numpy as np
import pandas as pd
import pytest

import post_test.foundation_baselines as fb


def test_context_ends_the_day_before_the_query_date():
    s = pd.Series(np.arange(100.0), index=pd.date_range("2000-01-01", periods=100, freq="D"))
    (c,) = fb.contexts(s, pd.DatetimeIndex(["2000-03-01"]), n=10)
    q_pos = s.index.get_loc(pd.Timestamp("2000-03-01"))
    np.testing.assert_array_equal(c, np.arange(q_pos - 10, q_pos))  # last value = q - 1
    with pytest.raises(ValueError):
        fb.contexts(s, pd.DatetimeIndex(["2000-01-05"]), n=10)


def test_median_is_taken_from_any_quantile_layout():
    q = np.array([[[1.0, 2.0, 3.0]] * 5])  # (1, horizon 5, 3 levels)
    np.testing.assert_array_equal(fb.median_from_quantiles(q, [0.1, 0.5, 0.9]), np.full(5, 2.0))
    assert fb.median_from_quantiles(np.arange(5.0)[:, None], [0.5]).tolist() == [0, 1, 2, 3, 4]


def test_model_weights_are_cached_inside_the_project_even_with_a_global_hf_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "global_cache"))
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "0")  # both are restored after the test
    fb.use_project_cache()
    import os
    assert os.environ["HF_HOME"] == str(fb.HF_HOME) and fb.HF_HOME.parent == fb.REPO_ROOT
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"


def test_a_forecast_must_have_exactly_five_days():
    assert fb.five_days([[1, 2, 3, 4, 5]], "m").tolist() == [1, 2, 3, 4, 5]
    for bad in ([1, 2, 3, 4], list(range(64))):
        with pytest.raises(ValueError):
            fb.five_days(bad, "m")


def test_cached_forecasts_are_reused_only_with_the_same_settings(tmp_path):
    import json
    path = tmp_path / "chronos2_X.parquet"
    assert fb.cached_forecasts("chronos2", path) is None  # nothing cached
    pd.DataFrame({"pred": [1.0]}).to_parquet(path)
    with pytest.raises(ValueError):  # cache without a settings record
        fb.cached_forecasts("chronos2", path)
    meta = path.with_suffix(".meta.json")
    meta.write_text(json.dumps({**fb.run_settings("chronos2"), "context_days": 512}), encoding="utf-8")
    with pytest.raises(ValueError):  # other settings
        fb.cached_forecasts("chronos2", path)
    meta.write_text(json.dumps(fb.run_settings("chronos2")), encoding="utf-8")
    assert fb.cached_forecasts("chronos2", path)["pred"].tolist() == [1.0]


def test_committed_forecasts_record_the_pinned_settings():
    for m in fb.MODELS:
        for ctrl, _, _ in fb.FAMILIES.values():
            assert fb.cached_forecasts(m, fb.OUT_PRED / f"{m}_{ctrl}.parquet") is not None
