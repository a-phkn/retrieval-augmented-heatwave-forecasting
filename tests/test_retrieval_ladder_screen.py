"""evaluation/retrieval_ladder_screen.py: the R2-R4 selection policies match their
specification (decisions.md 2026-10-07) and Rg's eligibility rules."""
import numpy as np
import pandas as pd
import pytest

import evaluation.retrieval_ladder_screen as rls
from retrieval.fold_retrieval import MIN_DAYS_APART, FoldRetriever

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(scope="module")
def setup():
    fr = FoldRetriever("f1", "v2")
    q = pd.DatetimeIndex(fr.train_w["query_date"].iloc[-400::20])
    return fr, q, rls.Selector(fr, q)


def test_eligibility_and_top_selection_match_rg(setup):
    fr, q, sel = setup
    rg = fr.retrieve(q, "region")
    for i in range(len(q)):
        assert sel.eligible(i).sum() == rg.n_eligible[i]
        assert sel.top(i, recent=False) == [int(c) for c in rg.idx[i] if c >= 0]


def test_mmr_starts_at_the_most_similar_and_keeps_the_dedup_rules(setup):
    fr, q, sel = setup
    rg = fr.retrieve(q, "region")
    differs = 0
    for i in range(len(q)):
        chosen = sel.mmr(i)
        assert len(chosen) == 5 and chosen[0] == rg.idx[i, 0]
        days = np.sort(sel.cand_day[chosen])
        assert (np.diff(days) >= MIN_DAYS_APART).all()
        eps = fr.cand_episode[chosen]
        assert all((eps == e).sum() <= 2 for e in set(eps[eps > 0]))
        assert set(chosen) <= set(np.flatnonzero(sel.eligible(i)))
        differs += set(chosen) != set(rg.idx[i])
    assert differs > 0  # diversity changes some selections


def test_recent_gate_and_random_draws_stay_in_the_gated_pool(setup):
    fr, q, sel = setup
    rng = np.random.default_rng(0)
    for i in range(len(q)):
        oldest = q[i] - pd.DateOffset(years=rls.RECENT_YEARS)
        for chosen in (sel.top(i, recent=True), sel.random(i, True, rng)):
            assert (fr.cand_dates[chosen] >= oldest).all()
            assert set(chosen) <= set(np.flatnonzero(sel.eligible(i)))


def test_drift_gate_fires_on_random_walks_and_rarely_on_white_noise():
    """KPSS alone flags ~5% of white-noise windows by design (p < 0.05), so the gate's
    false-firing rate is checked over many windows, not on one."""
    rng = np.random.default_rng(1)
    pos = np.arange(400, 20_000, 400)  # 49 non-overlapping 365-day windows
    walk_rate = rls.drift_flags(np.cumsum(rng.normal(size=20_000)), pos).mean()
    noise_rate = rls.drift_flags(rng.normal(size=20_000), pos).mean()
    assert walk_rate > 0.9 and noise_rate < 0.15
    assert not rls.drift_flags(np.cumsum(rng.normal(size=500)), np.array([100]))[0]  # too little history


def test_trend_beta_recovers_a_planted_heat_season_trend():
    days = pd.date_range("1980-01-01", "2015-12-31")
    z = 0.03 * (days.year - 1980).to_numpy(float) * days.month.isin(rls.HEAT_SEASON)
    d = pd.DataFrame(index=days)
    assert rls.trend_beta(d, z, pd.Timestamp("2015-12-31")) == pytest.approx(0.03, abs=1e-9)
    assert rls.trend_beta(d, z, pd.Timestamp("1999-12-31")) == pytest.approx(0.03, abs=1e-9)
