"""advisory/rating.py: the blind pack, the key, the scoring sheets and the statistics, on small hand-made data."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from advisory import rating as rt
from advisory.cases import assessment_of, load_cases
from advisory.library import load_library
from advisory.verifier import template_advisory

MARKER = "Zebra crossing sentence."  # stands in for the model's text, so a test can see which version was shown


@pytest.fixture(scope="module")
def library():
    return load_library()


@pytest.fixture
def small(tmp_path, library):
    """Two rated cases (Red and Green) + the 4 practice items, with fake LLM runs."""
    doc = load_cases()
    red = next(c for c in doc["cases"] if c["headline_colour"] == "RED")
    green = next(c for c in doc["cases"] if c["headline_colour"] == "GREEN")
    small_doc = {**doc, "cases": [red, green]}
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps(small_doc), encoding="utf-8")
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    for c in small_doc["cases"] + small_doc["practice"]:
        adv = template_advisory(assessment_of(c, doc["wbgt_threshold"]), library)
        adv["summary"] = MARKER
        (runs_dir / f"{c['case_id']}.json").write_text(json.dumps({"outcome": "passed_first_try", "advisory": adv}),
                                                       encoding="utf-8")
    return cases_path, runs_dir, tmp_path / "out", [red["case_id"], green["case_id"]]


def test_key_has_both_versions_of_every_case_in_a_shuffled_order():
    ids = [f"C{i:02d}" for i in range(1, 53)]
    key = rt.make_key(ids, seed=7)
    assert key == rt.make_key(ids, seed=7) and list(key)[0] == "A001" and list(key)[-1] == "A104" and len(key) == 104
    assert key != rt.make_key(ids, seed=8)
    df = pd.DataFrame.from_dict(key, orient="index")
    assert (df.groupby("case_id")["version"].nunique() == 2).all() and (df["case_id"].value_counts() == 2).all()
    assert [k["case_id"] for k in key.values()][:4] != ["C01", "C01", "C02", "C02"]  # really shuffled


def test_the_real_build_uses_a_secret_seed_kept_only_in_the_key(small):
    cases_path, runs_dir, out, _ = small
    rt.build(cases_path, runs_dir, out)
    first = json.loads((out / "key.json").read_text(encoding="utf-8"))
    rt.build(cases_path, runs_dir, out)
    second = json.loads((out / "key.json").read_text(encoding="utf-8"))
    assert first["seed"] != second["seed"]  # drawn at random each build (a 1 in 2**32 clash is negligible)
    assert str(first["seed"]) not in (rt.Path(rt.__file__)).read_text(encoding="utf-8")


def test_pack_is_blind_and_complete(small):
    cases_path, runs_dir, out, _ = small
    sha = rt.build(cases_path, runs_dir, out, seed=1)
    key_text = (out / "key.json").read_text(encoding="utf-8")
    assert hashlib.sha256(key_text.encode("utf-8")).hexdigest() == sha
    key = json.loads(key_text)["items"]
    page = (out / "rating_pack.html").read_text(encoding="utf-8")
    assert all(f'id="{code}"' in page for code in key) and all(f'id="practice{i}"' in page for i in range(1, 5))
    low = page.lower()
    for word in ("llm", "rule-based", "rule_based", "template", "groq", "gpt-oss"):
        assert word not in low, word


def test_each_item_shows_the_version_in_the_key(small):
    cases_path, runs_dir, out, _ = small
    rt.build(cases_path, runs_dir, out, seed=2)
    key = json.loads((out / "key.json").read_text(encoding="utf-8"))["items"]
    page = (out / "rating_pack.html").read_text(encoding="utf-8")
    sections = page.split('<section class="item" id="')[1:]
    for code, k in key.items():
        section = next(s for s in sections if s.startswith(f'{code}"'))
        assert (MARKER in section) == (k["version"] == rt.LLM), code
    practice = {i: next(s for s in sections if s.startswith(f'practice{i}"')) for i in range(1, 5)}
    assert [MARKER in practice[i] for i in (1, 2, 3, 4)] == [True, False, True, False]  # the versions alternate


def test_sheets_list_every_item_once_and_the_build_refuses_missing_runs(small):
    cases_path, runs_dir, out, _ = small
    rt.build(cases_path, runs_dir, out, seed=3)
    key = json.loads((out / "key.json").read_text(encoding="utf-8"))["items"]
    for rater in rt.RATERS:
        sheet = pd.read_csv(out / f"ratings_{rater}.csv", dtype=str, encoding="utf-8-sig", keep_default_na=False)
        assert list(sheet.columns) == list(rt.COLUMNS) and list(sheet["item"]) == list(key)
    next(runs_dir.glob("*.json")).unlink()
    with pytest.raises(SystemExit):
        rt.build(cases_path, runs_dir, out)


def write_sheet(path, rows):
    pd.DataFrame(rows, columns=rt.COLUMNS).to_csv(path, index=False, encoding="utf-8-sig")


def test_read_sheet_reports_bad_values_and_missing_notes(tmp_path):
    good = ["A001", "yes", "yes", "no", "4", "5", ""]
    write_sheet(tmp_path / "s.csv", [good, ["A002", "No", "yes", "no", "na", "3", ""],
                                     ["A003", "maybe", "yes", "no", "6", "0", "x"]])
    df, errors, warnings = rt.read_sheet(tmp_path / "s.csv", ["A001", "A002", "A003"])
    assert len(errors) == 3 and all("A003" in e for e in errors)
    assert warnings == ["s.csv A002: a note is required"] and df.loc["A002", "q1_forecast"] == "no"
    _, errors, _ = rt.read_sheet(tmp_path / "s.csv", ["A001", "A002"])
    assert any("do not match" in e for e in errors)


def test_kappas_and_mcnemar_match_hand_calculations():
    assert rt.cohen_kappa(["y", "y", "n", "n"], ["y", "n", "n", "n"]) == pytest.approx(0.5)
    assert rt.cohen_kappa(["y", "n"], ["y", "n"]) == pytest.approx(1.0)
    assert rt.weighted_kappa([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert rt.weighted_kappa([1, 5], [5, 1]) == pytest.approx(-1.0)
    assert rt.mcnemar_p(0, 0) == 1.0 and rt.mcnemar_p(5, 0) == pytest.approx(0.0625)


def test_an_item_counts_as_unfaithful_or_harmful_if_either_rater_says_so():
    idx = pd.Index(["A001", "A002"], name="item")
    s1 = pd.DataFrame({"q1_forecast": ["yes", "yes"], "q2_sources": ["yes", "no"], "q3_harmful": ["no", "no"],
                       "q4_actions": ["4", "na"], "q5_clear": ["5", "3"], "notes": ""}, index=idx)
    s2 = pd.DataFrame({"q1_forecast": ["yes", "no"], "q2_sources": ["yes", "no"], "q3_harmful": ["no", "yes"],
                       "q4_actions": ["2", "na"], "q5_clear": ["4", "3"], "notes": ""}, index=idx)
    c = rt.combine(s1, s2)
    assert c["q1_forecast"].tolist() == [True, False] and c["q2_sources"].tolist() == [True, False]
    assert c["q3_harmful"].tolist() == [False, True]
    assert c.loc["A001", "q4_actions"] == 3.0 and np.isnan(c.loc["A002", "q4_actions"])
    assert c["q5_clear"].tolist() == [4.5, 3.0]


def test_analyse_on_a_small_paired_example():
    key = {}
    rows = []
    for i in range(1, 11):  # 10 cases: the LLM version is unfaithful on case 1 only and clearer everywhere
        for version, code in ((rt.LLM, f"A{2 * i - 1:03d}"), (rt.RULES, f"A{2 * i:03d}")):
            key[code] = {"case_id": f"C{i:02d}", "version": version}
            faithful = "no" if (version == rt.LLM and i == 1) else "yes"
            rows.append([code, faithful, "yes", "no", "4", "5" if version == rt.LLM else "3", ""])
    idx = "item"
    s = pd.DataFrame(rows, columns=rt.COLUMNS).set_index(idx)
    res = rt.analyse(s, s.copy(), key)
    q1, q5 = res["questions"]["q1_forecast"], res["questions"]["q5_clear"]
    assert (res["n_items"], res["n_cases"]) == (20, 10)
    assert q1["llm"] == pytest.approx(0.9) and q1["rule_based"] == pytest.approx(1.0)
    assert q1["difference"] == pytest.approx(-0.1) and (q1["llm_only"], q1["rule_based_only"]) == (0, 1)
    assert q5["difference"] == pytest.approx(2.0) and q5["p"] < 0.01
    assert res["agreement"]["q1_forecast"]["raw"] == 1.0
