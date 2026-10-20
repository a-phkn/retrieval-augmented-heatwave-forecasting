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


def test_model_weights_are_cached_inside_the_project():
    import os
    assert os.environ["HF_HOME"].startswith(str(fb.REPO_ROOT))
