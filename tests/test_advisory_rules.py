"""advisory/rules.py: day classes, IMD-style run colours, and which actions each colour unlocks."""
import numpy as np
import pandas as pd
import pytest

from advisory import rules
from advisory.library import IMD_STYLE_LABEL, colour_rank, load_library

NORMAL = 38.0  # a late-May normal; anomaly = tmax - 38


@pytest.fixture(scope="module")
def library():
    return load_library()


def frames(tmax12, start="2024-05-20", p_hot=None, wbgt=None):
    """7 observed + 5 forecast days with a flat normal. Without p_hot, each forecast day gets probability 1 if its
    forecast meets the label rule and 0 otherwise, so the run tests below describe the same situations under the
    probability rule."""
    dates = pd.date_range(start, periods=12, freq="D")
    df = pd.DataFrame({"date": dates, "tmax": np.asarray(tmax12, float), "normal": NORMAL})
    obs, fc = df.iloc[:7].reset_index(drop=True), df.iloc[7:].reset_index(drop=True)
    if p_hot is None:
        p_hot = rules.classify(fc["date"], fc["tmax"], fc["normal"])[0].astype(float)
    fc["p_hot"] = p_hot
    if wbgt is not None:
        fc["wbgt"] = wbgt
    return obs, fc


COOL, HW, SEV = 36.0, 41.5, 43.0  # anomaly -2 (no), +3.5 (heat wave), +5 (severe)


@pytest.mark.parametrize("severe,expected", [
    ([False], "GREEN"),                      # one heat-wave day: GREEN, as in IMD's table
    ([False, False], "YELLOW"),
    ([False] * 3, "YELLOW"),
    ([False] * 4, "ORANGE"),
    ([True, False, True], "YELLOW"),         # severe days not consecutive
    ([True, True], "ORANGE"),
    ([True, True, False], "ORANGE"),
    ([True, True, True], "RED"),
    ([False] * 6, "ORANGE"),
    ([False] * 7, "RED"),
])
def test_run_colour(severe, expected):
    assert rules.run_colour(np.array(severe)) == expected


def test_every_day_of_a_run_gets_the_run_colour_and_gaps_split_runs():
    hw = np.array([0, 1, 1, 0, 1, 1, 1, 1, 0, 1], bool)
    sev = np.zeros(10, bool)
    assert rules.day_colours(hw, sev) == ["GREEN", "YELLOW", "YELLOW", "GREEN"] + ["ORANGE"] * 4 + ["GREEN", "GREEN"]
    with pytest.raises(ValueError):
        rules.day_colours(np.zeros(3, bool), np.array([0, 1, 0], bool))


def test_classify_uses_the_labels_v2_rule():
    dates = pd.to_datetime(["2024-05-20"] * 4 + ["2024-01-15"])
    tmax = [41.5, 41.0, 45.0, 43.0, 46.0]
    normal = [38.0, 38.5, 43.0, 38.0, 20.0]
    hw, sev = rules.classify(dates, tmax, normal)
    # +3.5: heat wave; +2.5: no; 45.0 qualifies by the absolute rule; +5.0: severe; January is outside the season
    assert hw.tolist() == [True, False, True, True, False]
    assert sev.tolist() == [False, False, False, True, False]  # 45.0 is only +2.0 above normal: not severe


def test_observed_days_count_towards_the_run():
    obs, fc = frames([COOL] * 4 + [HW] * 3 + [HW] + [COOL] * 4)  # 3 observed + 1 forecast heat-wave day = 4
    a = rules.assess(obs, fc)
    assert [d.colour for d in a.days] == ["ORANGE", "GREEN", "GREEN", "GREEN", "GREEN"]
    assert a.headline_colour == "ORANGE" and a.label == IMD_STYLE_LABEL


def test_isolated_forecast_heatwave_day_stays_green_and_severe_run_is_red():
    a = rules.assess(*frames([COOL] * 7 + [COOL, HW, COOL, COOL, COOL]))
    assert a.headline_colour == "GREEN" and a.days[1].heatwave_day
    b = rules.assess(*frames([COOL] * 7 + [SEV, SEV, SEV, COOL, COOL]))
    assert [d.colour for d in b.days] == ["RED"] * 3 + ["GREEN"] * 2 and b.days[0].severe_day


def test_forecast_days_are_classified_by_the_probability():
    level = rules.P_HOT_LEVEL
    # forecasts below the threshold, but likely enough: a 2-day run -> YELLOW under the probability rule only
    obs, fc = frames([COOL] * 7 + [39.0, 39.0, COOL, COOL, COOL], p_hot=[level, level + 0.2, 0.1, 0.0, 0.0])
    a = rules.assess(obs, fc)
    assert [d.colour for d in a.days] == ["YELLOW", "YELLOW", "GREEN", "GREEN", "GREEN"]
    assert a.forecast_rule == "probability" and a.p_hot_level == level
    assert rules.assess(obs, fc, forecast_rule="value").headline_colour == "GREEN"  # the old rule: never crosses
    # a hot forecast with a low probability is not a heatwave day under the probability rule
    obs, fc = frames([COOL] * 7 + [SEV, SEV, SEV, COOL, COOL], p_hot=[level - 0.01] * 3 + [0.0, 0.0])
    assert rules.assess(obs, fc).headline_colour == "GREEN"
    assert rules.assess(obs, fc, forecast_rule="value").headline_colour == "RED"
    # severe still needs a forecast departure >= 4.5 C: likely days at +3.5 C make a run, not severe days
    obs, fc = frames([COOL] * 7 + [HW, HW, HW, COOL, COOL], p_hot=[0.9, 0.9, 0.9, 0.0, 0.0])
    a = rules.assess(obs, fc)
    assert a.headline_colour == "YELLOW" and not any(d.severe_day for d in a.days)


def test_probability_outside_the_season_never_makes_a_heatwave_day():
    obs, fc = frames([COOL] * 12, start="2024-11-01", p_hot=[0.9] * 5)
    assert rules.assess(obs, fc).headline_colour == "GREEN"


def test_the_probability_rule_refuses_a_forecast_without_probabilities():
    obs, fc = frames([COOL] * 12)
    with pytest.raises(ValueError):
        rules.assess(obs, fc.drop(columns="p_hot"))
    with pytest.raises(ValueError):
        rules.assess(obs, fc.assign(p_hot=[0.1, np.nan, 0.1, 0.1, 0.1]))
    with pytest.raises(ValueError):
        rules.assess(obs, fc, forecast_rule="guess")
    assert rules.assess(obs, fc.drop(columns="p_hot"), forecast_rule="value").headline_colour == "GREEN"


def test_probability_and_humid_heat_note_are_carried_separately():
    a = rules.assess(*frames([COOL] * 12, p_hot=[0.1, 0.2, 0.3, 0.3, 0.3], wbgt=[30, 36.3, 30, 30, 30]),
                     wbgt_threshold=36.25)
    assert [d.p_hot for d in a.days] == [0.1, 0.2, 0.3, 0.3, 0.3]
    assert [d.humid_heat for d in a.days] == [False, True, False, False, False]
    assert a.headline_colour == "GREEN"  # humid heat never changes the IMD-style colour
    assert not rules.assess(*frames([COOL] * 12, wbgt=[40] * 5)).any_humid_heat  # no threshold given: no note


def test_input_checks():
    obs, fc = frames([COOL] * 12)
    with pytest.raises(ValueError):
        rules.assess(obs.iloc[:6], fc)
    with pytest.raises(ValueError):
        rules.assess(obs, fc.assign(date=fc["date"] + pd.Timedelta(days=1)))  # a gap
    with pytest.raises(ValueError):
        rules.assess(obs, fc.assign(p_hot=1.5))
    with pytest.raises(ValueError):
        rules.assess(obs.assign(tmax=np.nan), fc)


def test_higher_colours_unlock_more_actions_and_green_unlocks_none(library):
    sets = {c: {a.action_id for a in rules.forecast_actions(c, library)} for c in ("GREEN", "YELLOW", "ORANGE", "RED")}
    assert sets["GREEN"] == set()
    assert sets["YELLOW"] < sets["ORANGE"] < sets["RED"]
    assert all(library[i].section == "forecast" for i in sets["RED"])
    for c, ids in sets.items():
        assert all(colour_rank(library[i].min_colour) <= colour_rank(c) for i in ids)


def test_preparedness_only_in_heat_season_and_if_this_happens_split(library):
    may, dec = rules.assess(*frames([COOL] * 12)), rules.assess(*frames([COOL] * 12, start="2024-12-01"))
    prep = rules.preparedness_actions(may, library)
    assert prep and all(a.section == "preparedness" for a in prep) and rules.preparedness_actions(dec, library) == []
    public, pro = rules.if_this_happens_actions(library)
    assert public and pro and not any(a.for_professionals for a in public) and all(a.for_professionals for a in pro)
