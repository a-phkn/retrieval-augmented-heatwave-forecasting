"""retrieval_study/raft.py: RAFT's pooling, Pearson retrieval, eligibility, random control and weighting (offline)."""
import numpy as np
import pytest

from retrieval_study import raft


def test_keys_pool_from_the_end_and_values_from_the_start():
    x = np.arange(1, 15, dtype=float)[None, :]  # 14 input days
    np.testing.assert_allclose(raft.pool(x, 4, "end"), [[4.5, 8.5, 12.5]])  # the first 2 days are dropped
    np.testing.assert_allclose(raft.pool(x, 2, "end"), [[1.5, 3.5, 5.5, 7.5, 9.5, 11.5, 13.5]])
    v = np.arange(1, 6, dtype=float)[None, :]  # 5 forecast days
    np.testing.assert_allclose(raft.pool(v, 2, "start"), [[1.5, 3.5]])  # day 5 is dropped
    np.testing.assert_allclose(raft.pool(v, 4, "start"), [[2.5]])
    assert raft.n_values() == 8
    with pytest.raises(ValueError):
        raft.pool(v, 6, "start")


def test_standardised_rows_give_pearson_correlations_and_constant_rows_give_zero():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(3, 14)), rng.normal(size=(4, 14))
    r = raft.standardise_rows(a) @ raft.standardise_rows(b).T / 14
    expected = np.corrcoef(np.vstack([a, b]))[:3, 3:]
    np.testing.assert_allclose(r, expected, atol=1e-12)
    flat = raft.standardise_rows(np.full((1, 14), 30.0))
    assert np.all(flat == 0)


def test_the_best_match_is_found_but_an_overlapping_window_is_excluded():
    rng = np.random.default_rng(1)
    t = np.arange(14)
    query = np.sin(t / 2.0)[None, :]
    cands = rng.normal(size=(50, 14))
    cands[7] = 3.0 * np.sin(t / 2.0) + 40.0  # same shape, other level and scale: Pearson 1
    cands[8] = np.sin(t / 2.0) + 0.01 * rng.normal(size=14)  # nearly the same, but it overlaps the query in time
    q_day = np.array([1000])
    c_day = np.arange(50) * 30  # far apart ...
    c_day[8] = 1000 - (raft.OVERLAP_DAYS - 1)  # ... except candidate 8
    nb = raft.neighbours(query, q_day, cands, c_day, m_max=5)
    assert nb.idx[1][0, 0] == 7 and nb.sim[1][0, 0] == pytest.approx(1.0)
    assert 8 not in nb.idx[1][0] and all(8 not in nb.idx[p][0] for p in raft.PERIODS)
    assert np.all(np.diff(nb.sim[1][0]) <= 0)  # best first


def test_the_random_control_draws_eligible_candidates_and_keeps_their_correlations():
    rng = np.random.default_rng(2)
    keys = rng.normal(size=(40, 14))
    day = np.arange(40) * 7  # neighbours within 19 days overlap
    a = raft.neighbours(keys, day, keys, day, m_max=5, seed=11)
    b = raft.neighbours(keys, day, keys, day, m_max=5, seed=11)
    c = raft.neighbours(keys, day, keys, day, m_max=5, seed=12)
    np.testing.assert_array_equal(a.idx[1], b.idx[1])
    assert not np.array_equal(a.idx[1], c.idx[1])
    for i in range(40):
        chosen = a.idx[1][i][a.idx[1][i] >= 0]
        assert len(set(chosen)) == len(chosen) and np.all(np.abs(day[chosen] - day[i]) >= raft.OVERLAP_DAYS)
        np.testing.assert_array_equal(a.idx[2][i], a.idx[1][i])  # the same draw for every period
    std = raft.standardise_rows(keys)
    i, j = 0, a.idx[1][0, 0]
    assert a.sim[1][0, 0] == pytest.approx(std[i] @ std[j] / 14)


def test_softmax_weights_sharpen_with_temperature_and_skip_missing_neighbours():
    sim = np.array([[0.9, 0.5, np.nan], [np.nan, np.nan, np.nan]])
    w = raft.weights(sim, 3, tau=0.1)
    assert w[0].sum() == pytest.approx(1.0) and w[0, 2] == 0 and w[0, 0] > w[0, 1]
    assert w[0, 0] == pytest.approx(np.exp(9) / (np.exp(9) + np.exp(5)))
    assert np.all(w[1] == 0)
    flat = raft.weights(sim, 2, tau=1e6)
    np.testing.assert_allclose(flat[0], [0.5, 0.5], atol=1e-6)


def test_retrieved_values_are_the_weighted_pooled_changes():
    keys = np.zeros((4, 14))
    values = np.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [0, 0, 0, 0, 0], [9, 9, 9, 9, 9]], dtype=float)
    nb = raft.Neighbours(idx={p: np.array([[0, 1, -1]]) for p in raft.PERIODS},
                         sim={p: np.array([[0.8, 0.6, np.nan]]) for p in raft.PERIODS})
    out = raft.retrieved_values(nb, values, m=3)
    w0 = np.exp(8) / (np.exp(8) + np.exp(6))
    mix = w0 * values[0] + (1 - w0) * values[1]
    expected = np.concatenate([mix, [mix[:2].mean(), mix[2:4].mean()], [mix[:4].mean()]])
    np.testing.assert_allclose(out[0], expected)
    assert keys.shape[0] == 4
    with pytest.raises(ValueError):
        raft.retrieved_values(nb, values, m=4)


def test_offset_values_are_changes_from_the_last_input_day():
    fut = np.array([[41.0, 42.0, 43.0, 44.0, 45.0]])
    np.testing.assert_allclose(raft.offset_values(fut, np.array([40.0])), [[1, 2, 3, 4, 5]])
