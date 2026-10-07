"""evaluation/retrieval_mechanisms.py: the metric helpers."""
import numpy as np
import pandas as pd

import evaluation.retrieval_mechanisms as rm


def test_doy_distance_is_circular():
    a = pd.DatetimeIndex(["2010-01-03", "2010-07-01", "2010-12-30"])
    b = pd.DatetimeIndex(["2001-12-30", "2001-07-11", "2001-01-03"])
    assert list(rm.doy_distance(a, b)) == [4, 10, 4]


def test_mean_pairwise_similarity():
    same = np.tile(np.array([1.0, 0.0]), (1, 3, 1))
    ortho = np.array([[[1.0, 0.0], [0.0, 1.0]]])
    assert rm.mean_pairwise(same)[0] == 1.0 and rm.mean_pairwise(ortho)[0] == 0.0


def test_attention_unevenness_bounds():
    assert np.isclose(rm.attention_unevenness(np.full((1, 5), 0.2))[0], 0.0)
    assert rm.attention_unevenness(np.array([[1.0, 0, 0, 0, 0]]))[0] > 0.99


def test_rowwise_spearman():
    a = np.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]], dtype=float)
    b = np.array([[10, 20, 30, 40, 50], [10, 20, 30, 40, 50]], dtype=float)
    assert np.allclose(rm.rowwise_spearman(a, b), [1.0, -1.0])
