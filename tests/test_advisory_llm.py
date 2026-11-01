"""advisory/llm.py, offline: the schema and prompt only offer unlocked actions, the assembly keeps colours, labels
and quotes from the rules, bad replies are repaired once and then replaced by the template, the cache prevents repeat
calls, rate limits are respected, and the API key never reaches a saved file. No network is used."""
import json

import pytest

from advisory import llm
from advisory.cases import assessment_of, load_cases
from advisory.library import PRECEDENT_LABEL, load_library
from advisory.rules import MAX_HIGHLIGHTS, forecast_actions, highlight_order

FAKE_KEY = "gsk_test_not_a_real_key_123"


@pytest.fixture(scope="module")
def library():
    return load_library()


@pytest.fixture(scope="module")
def doc():
    return load_cases()


def first(doc, colour, humid=None):
    for c in doc["cases"]:
        a = assessment_of(c, doc["wbgt_threshold"])
        if a.headline_colour == colour and (humid is None or a.any_humid_heat == humid):
            return c, a
    pytest.skip(f"no {colour} case with humid={humid}")


def good_reply(a, library):
    """A reply that passes the verifier: the template's own sentences and the first 8 actions in the fixed order."""
    r = {"summary": f"Highest level in the next 5 days: {a.headline_colour}."}
    for i, d in enumerate(a.days, start=1):
        r[f"day_{i}"] = f"{d.colour}. Forecast maximum {d.tmax:.1f} °C."
    ids = [x.action_id for x in highlight_order(forecast_actions(a.headline_colour, library))][:MAX_HIGHLIGHTS]
    if ids:
        r["highlight_action_ids"] = ids
    if a.any_humid_heat:
        r["humid_heat_note"] = f"High humid heat stress is expected ({llm.HUMID_SOURCE_PHRASE})."
    return r


class FakeResponse:
    def __init__(self, status=200, body=None, headers=None):
        self.status_code, self._body, self.headers = status, body or {}, headers or {}
        self.text = json.dumps(self._body)

    def json(self):
        return self._body


def completion(reply: dict | str, tokens=1000):
    content = reply if isinstance(reply, str) else json.dumps(reply)
    return FakeResponse(200, {"choices": [{"message": {"content": content}}], "usage": {"total_tokens": tokens},
                              "system_fingerprint": "fp_test"})


class FakePost:
    """Returns the queued responses in order and records every request."""

    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def __call__(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "json": json})
        return self.responses.pop(0)


def test_schema_offers_only_the_unlocked_actions(doc, library):
    _, red = first(doc, "RED")
    sch = llm.schema(red, library)
    ids = [x.action_id for x in forecast_actions("RED", library)]
    assert sch["properties"]["highlight_action_ids"]["items"]["enum"] == ids
    assert sch["additionalProperties"] is False and set(sch["required"]) == set(sch["properties"])
    assert {f"day_{i}" for i in range(1, 6)} <= set(sch["properties"])
    _, green = first(doc, "GREEN")
    assert "highlight_action_ids" not in llm.schema(green, library)["properties"]


def test_prompt_lists_no_locked_action(doc, library):
    _, yellow = first(doc, "YELLOW")
    msg = llm.user_message(yellow, library)
    unlocked = {x.action_id for x in forecast_actions("YELLOW", library)}
    locked = {a for a, x in library.items() if x.section == "forecast"} - unlocked
    assert all(i in msg for i in unlocked) and not any(i in msg for i in locked)
    assert "IMD" in llm.SYSTEM_PROMPT and llm.HUMID_SOURCE_PHRASE in llm.SYSTEM_PROMPT


def test_humid_note_field_only_when_the_rule_engine_flags_humid_heat(doc, library):
    for c in doc["cases"]:
        a = assessment_of(c, doc["wbgt_threshold"])
        assert ("humid_heat_note" in llm.schema(a, library)["properties"]) == a.any_humid_heat


def test_assembled_good_reply_passes_and_keeps_rule_based_parts(doc, library):
    from advisory.verifier import template_advisory, verify

    _, red = first(doc, "RED")
    adv = llm.assemble(good_reply(red, library), red, library)
    assert verify(adv, red, library) == []
    t = template_advisory(red, library)
    assert [d["colour"] for d in adv["days"]] == [d["colour"] for d in t["days"]]
    assert adv["if_this_happens"] == t["if_this_happens"] and adv["label"] == t["label"]
    ahm = [e for e in adv["forecast_actions"] if library[e["action_id"]].tier_basis == "ahmedabad_precedent"]
    assert all(e["label"] == PRECEDENT_LABEL for e in ahm)


def _run(case, doc, library, responses):
    post = FakePost(responses)
    rec = llm.run_case(case, doc["wbgt_threshold"], library, FAKE_KEY, llm.Throttle(sleep=lambda s: None),
                       post=post, sleep=lambda s: None)
    return rec, post


def test_passes_first_try(doc, library):
    case, a = first(doc, "ORANGE")
    rec, post = _run(case, doc, library, [completion(good_reply(a, library))])
    assert rec["outcome"] == "passed_first_try" and len(post.calls) == 1
    body = post.calls[0]["json"]
    assert body["model"] == llm.MODEL and body["response_format"]["json_schema"]["strict"] is True
    assert body["temperature"] == 0 and body["reasoning_effort"] == "low"


@pytest.mark.parametrize("break_it", [
    lambda r: r.update(day_1="Up to 47 °C."),                        # a number not in the forecast
    lambda r: r.update(summary="IMD has issued a heat warning."),     # an official-warning claim
    lambda r: r.update(day_2="Red skies ahead."),                    # a colour word not that day's colour
])
def test_bad_reply_is_repaired_once(doc, library, break_it):
    case, a = first(doc, "RED")
    bad = good_reply(a, library)
    break_it(bad)
    rec, post = _run(case, doc, library, [completion(bad), completion(good_reply(a, library))])
    assert rec["outcome"] == "passed_after_repair" and len(post.calls) == 2
    repair = post.calls[1]["json"]["messages"]
    assert repair[-2]["role"] == "assistant" and "rejected" in repair[-1]["content"]
    assert rec["attempts"][0]["violations"] and not rec["attempts"][1]["violations"]


def test_two_bad_replies_fall_back_to_the_template(doc, library):
    from advisory.verifier import template_advisory

    case, a = first(doc, "YELLOW")
    rec, post = _run(case, doc, library, [completion("not json"), completion({"summary": "x"})])
    assert rec["outcome"] == "template_fallback" and len(post.calls) == 2
    assert rec["advisory"] == template_advisory(a, library)


def test_run_caches_and_never_saves_the_key(doc, library, tmp_path, monkeypatch):
    case, a = first(doc, "GREEN")
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps({**doc, "cases": [case], "practice": []}), encoding="utf-8")
    monkeypatch.setenv(llm.KEY_ENV, FAKE_KEY)
    post = FakePost([completion(good_reply(a, library))])
    counts = llm.run(cases_path, post=post, sleep=lambda s: None, runs_dir=tmp_path / "runs")
    assert counts == {"passed_first_try": 1} and len(post.calls) == 1
    assert post.calls[0]["headers"]["Authorization"] == f"Bearer {FAKE_KEY}"
    saved = (tmp_path / "runs" / f"{case['case_id']}.json").read_text(encoding="utf-8")
    assert FAKE_KEY not in saved and "Authorization" not in saved
    again = FakePost([])  # any call would fail: the cache must be used
    assert llm.run(cases_path, post=again, sleep=lambda s: None, runs_dir=tmp_path / "runs") == counts
    assert again.calls == []


def test_missing_key_stops_before_any_call(tmp_path, monkeypatch):
    monkeypatch.delenv(llm.KEY_ENV, raising=False)
    monkeypatch.setattr(llm, "ENV_FILE", tmp_path / "no_such.env")
    with pytest.raises(SystemExit):
        llm.run(post=FakePost([]), runs_dir=tmp_path)


def test_the_environment_variable_wins_over_the_env_file(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# a comment\nOTHER=1\nGROQ_API_KEY = "gsk_from_file"\n', encoding="utf-8")
    monkeypatch.delenv(llm.KEY_ENV, raising=False)
    assert llm.read_key(env) == "gsk_from_file"
    monkeypatch.setenv(llm.KEY_ENV, FAKE_KEY)
    assert llm.read_key(env) == FAKE_KEY


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "utf-16"])
def test_env_file_is_read_in_the_encodings_windows_tools_write(tmp_path, monkeypatch, encoding):
    env = tmp_path / ".env"
    env.write_text("GROQ_API_KEY=gsk_abc\r\n", encoding=encoding)
    monkeypatch.delenv(llm.KEY_ENV, raising=False)
    assert llm.read_key(env) == "gsk_abc"


@pytest.mark.parametrize("content", ["", "# GROQ_API_KEY=old\n", "GROQ_API_KEY=\n", "OTHER_KEY=abc\n"])
def test_env_file_without_a_usable_key_gives_none(tmp_path, monkeypatch, content):
    env = tmp_path / ".env"
    env.write_text(content, encoding="utf-8")
    monkeypatch.delenv(llm.KEY_ENV, raising=False)
    assert llm.read_key(env) is None
    assert llm.read_key(tmp_path / "no_such.env") is None


def test_run_uses_the_key_from_the_env_file_and_never_saves_it(doc, library, tmp_path, monkeypatch):
    case, a = first(doc, "GREEN")
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps({**doc, "cases": [case], "practice": []}), encoding="utf-8")
    env = tmp_path / ".env"
    env.write_text(f"{llm.KEY_ENV}={FAKE_KEY}\n", encoding="utf-8")
    monkeypatch.delenv(llm.KEY_ENV, raising=False)
    monkeypatch.setattr(llm, "ENV_FILE", env)
    post = FakePost([completion(good_reply(a, library))])
    llm.run(cases_path, post=post, sleep=lambda s: None, runs_dir=tmp_path / "runs")
    assert post.calls[0]["headers"]["Authorization"] == f"Bearer {FAKE_KEY}"
    assert FAKE_KEY not in (tmp_path / "runs" / f"{case['case_id']}.json").read_text(encoding="utf-8")


def test_rate_limits(monkeypatch):
    slept = []
    ok = FakeResponse(200, {"choices": [{"message": {"content": "{}"}}]})
    post = FakePost([FakeResponse(429, headers={"retry-after": "3"}), FakeResponse(503), ok])
    assert llm.call_api({}, FAKE_KEY, post=post, sleep=slept.append) == ok.json()
    assert slept[0] == 3.0 and len(post.calls) == 3
    with pytest.raises(llm.DailyLimit):
        llm.call_api({}, FAKE_KEY, post=FakePost([FakeResponse(429, headers={"retry-after": "3600"})]), sleep=slept.append)
    with pytest.raises(RuntimeError):
        llm.call_api({}, FAKE_KEY, post=FakePost([FakeResponse(400, {"error": {"message": "bad"}})]), sleep=slept.append)


class TimesOut:
    """A post that times out `times` times, then returns ok."""

    def __init__(self, times, ok=None):
        self.times, self.ok, self.calls = times, ok, 0

    def __call__(self, url, headers=None, json=None, timeout=None):
        import requests

        self.calls += 1
        if self.calls <= self.times:
            raise requests.exceptions.ReadTimeout("read timed out")
        return self.ok


def test_timeouts_are_retried_and_then_stop_cleanly():
    ok = FakeResponse(200, {"choices": [{"message": {"content": "{}"}}]})
    post = TimesOut(2, ok)
    assert llm.call_api({}, FAKE_KEY, post=post, sleep=lambda s: None) == ok.json() and post.calls == 3
    with pytest.raises(llm.ProviderUnavailable):
        llm.call_api({}, FAKE_KEY, post=TimesOut(99), sleep=lambda s: None)
    with pytest.raises(llm.ProviderUnavailable):  # 429s that never clear also stop cleanly
        llm.call_api({}, FAKE_KEY, post=FakePost([FakeResponse(429, headers={"retry-after": "1"})] * 4),
                     sleep=lambda s: None)


def test_run_stops_cleanly_and_keeps_finished_cases_when_the_provider_is_down(doc, library, tmp_path, monkeypatch):
    (c1, a1), c2 = first(doc, "GREEN"), next(c for c in doc["cases"] if c["headline_colour"] == "ORANGE")
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps({**doc, "cases": [c1, c2], "practice": []}), encoding="utf-8")
    monkeypatch.setenv(llm.KEY_ENV, FAKE_KEY)
    responses = iter([completion(good_reply(a1, library))])

    def post(url, headers=None, json=None, timeout=None):
        import requests

        try:
            return next(responses)
        except StopIteration:
            raise requests.exceptions.ConnectTimeout("down") from None

    counts = llm.run(cases_path, post=post, sleep=lambda s: None, runs_dir=tmp_path / "runs")
    assert counts == {"passed_first_try": 1}
    assert (tmp_path / "runs" / f"{c1['case_id']}.json").exists()
    assert not (tmp_path / "runs" / f"{c2['case_id']}.json").exists()  # no half-finished record: it is retried later


def test_a_repair_too_large_to_send_falls_back_to_the_template(doc, library):
    from advisory.verifier import template_advisory

    case, a = first(doc, "RED")
    bad = good_reply(a, library)
    bad["day_1"] = "Up to 47 °C."
    rec, post = _run(case, doc, library, [completion(bad), FakeResponse(413)])
    assert rec["outcome"] == "template_fallback" and len(rec["attempts"]) == 2
    assert "not sent" in rec["attempts"][1]["violations"][0]
    assert rec["advisory"] == template_advisory(a, library)


def test_throttle_waits_until_the_minute_window_has_room():
    now = [0.0]
    waits = []

    def sleep(s):
        waits.append(s)
        now[0] += s

    t = llm.Throttle(budget=7000, clock=lambda: now[0], sleep=sleep)
    t.wait_for(5000)
    t.record(5000)
    now[0] = 10.0
    t.wait_for(4000)  # 5000 + 4000 > 7000: must wait until the first call is 60 s old
    assert waits and now[0] >= 60.0
