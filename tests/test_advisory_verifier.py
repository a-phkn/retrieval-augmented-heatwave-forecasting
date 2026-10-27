"""advisory/verifier.py: the rule-only template passes, and each kind of unfaithful advisory is rejected."""
import copy

import numpy as np
import pandas as pd
import pytest

from advisory import rules
from advisory import verifier as vf
from advisory.library import CLINICAL_DISCLAIMER, PRECEDENT_LABEL, load_library

COOL, HW, SEV = 36.0, 41.5, 43.0  # normal 38: anomaly -2 / +3.5 / +5


@pytest.fixture(scope="module")
def library():
    return load_library()


def assessment(tmax12, wbgt=None, start="2024-05-20", thr=36.25):
    dates = pd.date_range(start, periods=12, freq="D")
    df = pd.DataFrame({"date": dates, "tmax": np.asarray(tmax12, float), "normal": 38.0})
    fc = df.iloc[7:].reset_index(drop=True).assign(p_hot=[0.05, 0.42, 0.61, 0.33, 0.1])
    if wbgt is not None:
        fc["wbgt"] = wbgt
    return rules.assess(df.iloc[:7].reset_index(drop=True), fc, wbgt_threshold=thr)


@pytest.fixture(scope="module")
def orange(library):
    a = assessment([COOL] * 7 + [HW, SEV, SEV, COOL, COOL], wbgt=[30, 36.8, 30, 30, 30])
    assert a.headline_colour == "ORANGE" and a.any_humid_heat
    return a, vf.template_advisory(a, library)


def test_template_passes_at_every_colour(library):
    for tmax in ([COOL] * 12, [COOL] * 7 + [HW, HW, COOL, COOL, COOL], [COOL] * 7 + [HW] * 4 + [COOL],
                 [COOL] * 7 + [SEV] * 3 + [COOL] * 2):
        a = assessment(tmax)
        assert vf.verify(vf.template_advisory(a, library), a, library) == [], a.headline_colour


def test_template_passes_with_humid_heat(orange, library):
    a, adv = orange
    assert vf.verify(adv, a, library) == []
    assert adv["forecast_actions"] and adv["humid_heat_note"]


def _broken(adv, change):
    bad = copy.deepcopy(adv)
    change(bad)
    return bad


def _first(library, **kw):
    return next(x for x in library.values() if all(getattr(x, k) == v for k, v in kw.items()))


@pytest.mark.parametrize("name,change,expect", [
    ("changed colour", lambda d: d["days"][1].update(colour="RED"), "colour RED differs"),
    ("changed headline", lambda d: d.update(headline_colour="RED"), "headline colour"),
    ("label dropped", lambda d: d.update(label="IMD heat warning"), "IMD-style label"),
    ("probability changed", lambda d: d["days"][2].update(p_hot=0.9), "probability differs"),
    ("invented number", lambda d: d["days"][1].update(text="ORANGE. Up to 47 °C."), "number 47"),
    ("wrong colour word", lambda d: d["days"][0].update(text="Red alert tomorrow."), "mentions RED"),
    ("official claim", lambda d: d.update(summary="IMD has issued an ORANGE alert."), "claims an IMD warning"),
    ("unknown action", lambda d: d["forecast_actions"].append({"action_id": "INVENTED"}), "unknown action id"),
    ("missing field", lambda d: d.pop("preparedness"), "missing field preparedness"),
    ("extra field", lambda d: d.update(extra=1), "unexpected field"),
    ("humid note uses alert word", lambda d: d["humid_heat_note"].update(
        text=f"Humid heat alert ({vf.HUMID_SOURCE_PHRASE})."), "alert word"),
    ("humid note without source", lambda d: d["humid_heat_note"].update(text="High WBGT."), "physical WBGT"),
    ("humid note dropped", lambda d: d.update(humid_heat_note=None), "no humid-heat note"),
    ("disclaimer altered", lambda d: d["if_this_happens"]["professionals"].update(disclaimer="FYI"),
     "clinical disclaimer"),
])
def test_rejects(orange, library, name, change, expect):
    a, adv = orange
    v = vf.verify(_broken(adv, change), a, library)
    assert any(expect in m for m in v), (name, v)


def test_rejects_wrong_sections_and_colours(orange, library):
    a, adv = orange
    red_only = _first(library, section="forecast", min_colour="RED")
    observed = _first(library, section="if_this_happens", for_professionals=False)
    clinical = _first(library, section="if_this_happens", for_professionals=True)
    prep = _first(library, section="preparedness")
    cases = [
        (lambda d: d["forecast_actions"].append({"action_id": red_only.action_id, "label": PRECEDENT_LABEL}), "needs RED"),
        (lambda d: d["forecast_actions"].append({"action_id": observed.action_id}), "belongs to section"),
        (lambda d: d["preparedness"].append({"action_id": d["forecast_actions"][0]["action_id"]}), "listed twice"),
        (lambda d: d["if_this_happens"]["public"].append({"action_id": clinical.action_id}), "clinical guidance"),
        (lambda d: d["if_this_happens"]["professionals"]["actions"].remove({"action_id": clinical.action_id})
         or d["if_this_happens"]["professionals"]["actions"].append({"action_id": observed.action_id}), "public guidance"),
        (lambda d: d["preparedness"].remove({"action_id": prep.action_id})
         or d["forecast_actions"].append({"action_id": prep.action_id}), "belongs to section"),
    ]
    for change, expect in cases:
        v = vf.verify(_broken(adv, change), a, library)
        assert any(expect in m for m in v), (expect, v)


def test_precedent_label_and_quotes(orange, library):
    a, adv = orange
    i = next(k for k, e in enumerate(adv["forecast_actions"]) if "label" in e)
    v = vf.verify(_broken(adv, lambda d: d["forecast_actions"][i].pop("label")), a, library)
    assert any("precedent label" in m for m in v)
    e = adv["forecast_actions"][0]
    src = library[e["action_id"]].source.text
    ok = _broken(adv, lambda d: d["forecast_actions"][0].update(quote=" ".join(src.split()[:6])))
    assert vf.verify(ok, a, library) == []
    bad = _broken(adv, lambda d: d["forecast_actions"][0].update(quote="Stay indoors at all times"))
    assert any("not an exact part" in m for m in vf.verify(bad, a, library))


def test_preparedness_outside_the_season_is_rejected(library):
    winter = assessment([COOL] * 12, start="2024-12-01")
    adv = vf.template_advisory(winter, library)
    assert adv["preparedness"] == [] and vf.verify(adv, winter, library) == []
    adv["preparedness"] = [{"action_id": _first(library, section="preparedness").action_id}]
    assert any("heat season" in m for m in vf.verify(adv, winter, library))


def test_numbers_are_matched_at_the_written_precision(orange):
    a, _ = orange
    allowed = vf.allowed_numbers(a)
    assert vf.unsupported_numbers("43.0 °C, about 43 °C, 61% chance, call 108", allowed) == []
    assert vf.unsupported_numbers("43.4 °C", allowed) == ["43.4"]
