"""advisory/library.py: the committed library is exactly what the sources and the decided rules give."""
import copy
import json

import pytest

import advisory.library as lib


@pytest.fixture(scope="module")
def candidates():
    return json.loads(lib.CANDIDATES_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def library():
    return lib.load_library()


def test_committed_library_matches_a_fresh_build():
    committed = json.loads(lib.LIBRARY_PATH.read_text(encoding="utf-8"))
    assert committed == json.loads(json.dumps(lib.build_document(), ensure_ascii=False)), \
        "advisory/action_library.json is stale: run python -m advisory.library"


def test_every_candidate_is_in_the_library_or_explicitly_excluded(candidates, library):
    assert set(library) | set(lib.EXCLUDED) == {c["action_id"] for c in candidates}
    assert not set(library) & set(lib.EXCLUDED)


def test_quotes_are_copied_word_for_word(candidates, library):
    by_id = {c["action_id"]: c for c in candidates}
    for a in library.values():
        c = by_id[a.action_id]
        assert (a.source.doc_id, a.source.page, a.source.text) == (c["doc_id"], c["page"], c["verbatim_excerpt"])
        assert a.text == c["text"]


def test_colour_rules(candidates, library):
    by_id = {c["action_id"]: c for c in candidates}
    for a in library.values():
        c = by_id[a.action_id]
        if c["tier_min"] == "UNSPECIFIED":
            assert (a.min_colour, a.tier_basis, a.colour_source) == ("YELLOW", "delhi_any_alert", None)
        else:  # a sourced colour is used as sourced, with its own quote
            assert a.min_colour == c["tier_min"] and a.tier_basis == c["tier_source"]["tier_basis"]
            assert a.colour_source.text == c["tier_source"]["quote"]
    assert sum(a.tier_basis == "ahmedabad_precedent" for a in library.values()) == 30


def test_sections_follow_the_trigger_and_never_mix(candidates, library):
    by_id = {c["action_id"]: c for c in candidates}
    for a in library.values():
        trig, sourced = by_id[a.action_id]["trigger_type"], by_id[a.action_id]["tier_min"] != "UNSPECIFIED"
        expected = {"forecast": "forecast", "observation": "if_this_happens",
                    "standing": "forecast" if sourced else "preparedness"}[trig]
        assert a.section == expected, a.action_id
        assert a.for_professionals == (a.section == "if_this_happens" and a.audience == "health")
    # first aid for the public is shown directly, clinical protocols are not
    assert not library["PUBLIC_HEAT_STROKE_EMERGENCY"].for_professionals
    assert library["CLINICAL_40C_IV_TRANSPORT"].for_professionals


def test_build_refuses_bad_rows(candidates):
    row = copy.deepcopy(next(c for c in candidates if c["tier_min"] == "UNSPECIFIED"))
    for change in ({"verification": "NOT FOUND"}, {"trigger_type": "rumour"}, {"audience": "everyone"},
                   {"tier_min": "RED"}):  # a colour with no colour source
        with pytest.raises(ValueError):
            lib.build_action({**row, **change})
    with pytest.raises(ValueError):
        lib.build_library([row, row])  # duplicate ids


def test_colour_rank():
    assert [lib.colour_rank(c) for c in lib.COLOURS] == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        lib.colour_rank("PURPLE")
