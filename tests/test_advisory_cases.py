"""advisory/cases.py: the stratified, gap-respecting sample and the committed evaluation cases."""
import json

import pandas as pd
import pytest

import advisory.cases as ac
from advisory.library import COLOURS


def toy_headline(per_colour_runs: int = 6) -> pd.Series:
    """Season dates 10 days apart with colours cycling, so every colour has many well-separated candidates."""
    dates = pd.date_range("2020-03-20", periods=4 * per_colour_runs, freq="10D")
    return pd.Series([COLOURS[i % 4] for i in range(len(dates))], index=dates)


def test_sample_is_stratified_deterministic_and_respects_the_gap():
    h = toy_headline()
    a, b = ac.sample(h, per_colour=3, min_gap=3, seed=1), ac.sample(h, per_colour=3, min_gap=3, seed=1)
    assert a == b and len(a) == 12
    assert pd.Series(h.loc[a]).value_counts().to_dict() == {c: 3 for c in COLOURS}
    gaps = [(y - x).days for x, y in zip(a, a[1:])]
    assert min(gaps) >= 3


def test_sample_avoids_given_dates_and_refuses_when_a_colour_cannot_be_filled():
    h = toy_headline()
    taken = ac.sample(h, per_colour=3, min_gap=3, seed=1)
    extra = ac.sample(h.drop(taken), per_colour=1, min_gap=10, seed=2, avoid=taken)
    assert all(abs((e - t).days) >= 10 for e in extra for t in taken) and not set(extra) & set(taken)
    with pytest.raises(ValueError):
        ac.sample(h.drop(taken), per_colour=1, min_gap=11, seed=2, avoid=taken)  # every free date is 10 days from one
    with pytest.raises(ValueError):
        ac.sample(h, per_colour=7, min_gap=3, seed=1)  # only 6 per colour exist


def test_draw_uses_the_first_seed_that_fills_every_colour(monkeypatch):
    h = toy_headline()
    monkeypatch.setattr(ac, "CASES_PER_COLOUR", 5)
    calls = []
    real = ac.sample

    def flaky(headline, per_colour=5, min_gap=3, seed=0, avoid=None):
        calls.append(seed)
        if seed < 102:  # pretend the first seeds cannot be filled
            raise ValueError("infeasible")
        return real(headline, per_colour=per_colour, min_gap=min_gap, seed=seed, avoid=avoid)

    monkeypatch.setattr(ac, "sample", flaky)
    chosen, practice, seed = ac.draw(h, first_seed=100, tries=10)
    assert seed == 102 and calls[:3] == [100, 101, 102] and len(chosen) == 20 and len(practice) == 4


@pytest.fixture(scope="module")
def doc():
    return ac.load_cases()


def test_committed_cases_follow_the_design(doc):
    cases, practice = doc["cases"], doc["practice"]
    assert (doc["cases_per_colour"], doc["min_gap_days"], doc["forecast_rule"]) == (13, 3, "probability")
    assert pd.Series([c["headline_colour"] for c in cases]).value_counts().to_dict() == {c: 13 for c in COLOURS}
    assert sorted(c["headline_colour"] for c in practice) == sorted(COLOURS)
    dates = sorted(pd.Timestamp(c["query_date"]) for c in cases + practice)
    assert min((b - a).days for a, b in zip(dates, dates[1:])) >= 3
    assert len({c["case_id"] for c in cases + practice}) == len(cases) + len(practice)


def test_each_case_has_the_7_days_before_and_the_5_days_from_its_query_date(doc):
    for c in doc["cases"] + doc["practice"]:
        q = pd.Timestamp(c["query_date"])
        obs = pd.to_datetime([r["date"] for r in c["observed"]])
        fc = pd.to_datetime([r["date"] for r in c["forecast"]])
        assert list(obs) == list(pd.date_range(q - pd.Timedelta(days=7), periods=7))
        assert list(fc) == list(pd.date_range(q, periods=5))
        assert all(0.0 <= r["p_hot"] <= 1.0 for r in c["forecast"])


def test_stored_colours_follow_from_the_stored_numbers(doc):
    for c in doc["cases"] + doc["practice"]:
        assert ac.assessment_of(c, doc["wbgt_threshold"]).headline_colour == c["headline_colour"], c["case_id"]


@pytest.mark.slow
def test_committed_cases_match_a_rebuild(doc):
    from training import test_lock

    try:
        test_lock.require()
    except test_lock.TestLocked:
        pytest.skip("the test lock is not open here")
    assert json.loads(json.dumps(ac.build_cases())) == doc
