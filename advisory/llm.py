"""
The constrained LLM advisory (Phase 6 part 4; pre-registered in context/decisions.md before any call).

The model never sets a colour, never writes an action's text or quote, and never adds actions. For each case it
returns strict JSON: a short summary, one sentence per forecast day, the ids of up to MAX_HIGHLIGHTS actions to
highlight (an enum of the actions unlocked for that case only), and, only when the rule engine flags humid heat, a
humid-heat note. Code assembles the advisory around it (colours, labels, quotes, preparedness and "If this happens"
come from the rule-based template), the verifier checks it, a rejected advisory gets ONE repair call listing the
violations, and after that the rule-based template is used instead (recorded as a fallback).

Provider: Groq's OpenAI-compatible chat completions endpoint, model MODEL (open weights), called with `requests`
(already pinned; no new dependency). The API key comes from the environment variable GROQ_API_KEY or, failing that,
from a GROQ_API_KEY=... line in the project's gitignored .env file; this code never prints, logs or saves it.

Every response is cached in advisory/eval/llm_runs/<case_id>.json (model, settings, prompt and schema hashes, raw
reply, usage, verifier result). A rerun reuses the cache and never calls the API again for a finished case, so the
run can be stopped and resumed (for example after the free tier's daily limit).

Run from repo root (with the key in .env, or in $env:GROQ_API_KEY in the same PowerShell window):
    python -m advisory.llm                 # all cases and practice items
    python -m advisory.llm --dry-run C01   # print the prompt and schema for one case, no call
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from advisory.cases import CASES_PATH, assessment_of, load_cases
from advisory.library import PRECEDENT_LABEL, Action, load_library
from advisory.rules import FORECAST_DAYS, MAX_HIGHLIGHTS, Assessment, forecast_actions
from advisory.verifier import HUMID_SOURCE_PHRASE, template_advisory, verify

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "advisory" / "eval" / "llm_runs"
ENV_FILE = REPO_ROOT / ".env"  # gitignored
API_URL = "https://api.groq.com/openai/v1/chat/completions"
KEY_ENV = "GROQ_API_KEY"
MODEL = "openai/gpt-oss-120b"
SETTINGS = {"temperature": 0, "seed": 20261030, "max_completion_tokens": 4000, "reasoning_effort": "low"}
TOKENS_PER_MINUTE_BUDGET = 7000  # below the free tier's 8,000 per minute
LONG_WAIT_S = 120  # a 429 asking to wait longer than this is treated as the daily limit: stop, resume later
TIMEOUT_S = 120

SYSTEM_PROMPT = f"""You write short public heat advisories for Delhi from a 5-day forecast that has already been checked.
Follow every rule:
1. Use only the numbers given to you: the dates, the forecast maximum temperatures, the normals, the departures from
   normal and the chances of a heatwave day. You may round them to whole numbers or one decimal. Never write any
   other number.
2. Each day's alert colour is fixed. In a day's sentence, name no colour except that day's own colour. In the summary,
   name only colours that occur in the 5 days. Never change, soften or exaggerate a colour.
3. Never call this an official IMD warning and never say that IMD issued, warned or declared anything.
4. Pick up to {MAX_HIGHLIGHTS} actions to highlight, by id, from the list given, most important first. Choose ones that
   suit the highest alert level and cover different audiences (public, health, municipal, energy). Do not restate
   actions or give any advice, medical or otherwise, in the summary or the day sentences.
5. The summary is 2 or 3 short sentences for the public: the highest alert level, on which days, and what to watch.
   Each day sentence is one short sentence about that day's forecast.
6. If you are asked for a humid-heat note, write one or two sentences that contain exactly this phrase:
   "{HUMID_SOURCE_PHRASE}". In that note, do not use the words heat wave, heatwave, alert or warning, or any colour.
7. Plain English. No exclamation marks, no emojis, no headings."""

REPAIR_INSTRUCTION = ("Your advisory was rejected by the checker for these reasons:\n{violations}\n"
                      "Return a corrected advisory that follows every rule.")


# ------------------------------------------------------------------ prompt and schema

def unlocked(a: Assessment, library: dict[str, Action]) -> list[Action]:
    """The forecast actions the model may highlight for this case (none on a Green headline)."""
    return forecast_actions(a.headline_colour, library)


def user_message(a: Assessment, library: dict[str, Action]) -> str:
    lines = [f"Highest alert level in the next {FORECAST_DAYS} days: {a.headline_colour}.", "",
             "Forecast (date | forecast max C | normal C | departure C | chance of a heatwave day | alert colour):"]
    for d in a.days:
        p = "n/a" if d.p_hot is None else f"{100 * d.p_hot:.0f}%"
        lines.append(f"{d.date:%Y-%m-%d} ({d.date:%A}) | {d.tmax:.1f} | {d.normal:.1f} | {d.anomaly:+.1f} | {p} | {d.colour}")
    acts = unlocked(a, library)
    lines.append("")
    if acts:
        lines.append(f"Actions you may highlight (up to {MAX_HIGHLIGHTS}; id | audience | from colour | action):")
        lines += [f"{x.action_id} | {x.audience} | {x.min_colour} | {x.text}" for x in acts]
    else:
        lines.append("No forecast actions apply at this level; do not highlight any.")
    if a.any_humid_heat:
        humid = ", ".join(f"{d.date:%Y-%m-%d}" for d in a.days if d.humid_heat)
        lines += ["", f"Humid heat stress is forecast on: {humid}. Write the humid-heat note."]
    return "\n".join(lines)


def schema(a: Assessment, library: dict[str, Action]) -> dict:
    """The strict JSON schema for this case: day texts as separate required fields; the highlight field only when
    actions are unlocked (an enum of their ids); the humid note only when the rule engine flags humid heat."""
    props: dict = {"summary": {"type": "string"}}
    for i in range(1, FORECAST_DAYS + 1):
        props[f"day_{i}"] = {"type": "string"}
    ids = [x.action_id for x in unlocked(a, library)]
    if ids:
        props["highlight_action_ids"] = {"type": "array", "items": {"type": "string", "enum": ids}}
    if a.any_humid_heat:
        props["humid_heat_note"] = {"type": "string"}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def request_body(messages: list[dict], sch: dict) -> dict:
    return {"model": MODEL, "messages": messages,
            "response_format": {"type": "json_schema", "json_schema": {"name": "heat_advisory", "strict": True,
                                                                       "schema": sch}},
            **SETTINGS}


def _sha(obj) -> str:
    text = obj if isinstance(obj, str) else json.dumps(obj, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ assembly

def assemble(reply: dict, a: Assessment, library: dict[str, Action]) -> dict:
    """The full advisory from the model's JSON: everything except the free text and the highlight choice comes from
    the rule-based template, so colours, labels and quotes cannot be changed by the model."""
    adv = template_advisory(a, library)
    adv["summary"] = reply.get("summary")
    for i, day in enumerate(adv["days"], start=1):
        day["text"] = reply.get(f"day_{i}")
    adv["forecast_actions"] = []
    for aid in reply.get("highlight_action_ids", []):
        e = {"action_id": aid}
        act = library.get(aid)
        if act is not None and act.tier_basis == "ahmedabad_precedent":
            e["label"] = PRECEDENT_LABEL
        adv["forecast_actions"].append(e)
    adv["humid_heat_note"] = {"text": reply["humid_heat_note"]} if "humid_heat_note" in reply else None
    return adv


def parse_reply(content: str | None) -> tuple[dict | None, list[str]]:
    """The model's JSON, or (None, problems) when it is not a JSON object."""
    try:
        obj = json.loads(content or "")
    except json.JSONDecodeError as e:
        return None, [f"the reply is not valid JSON ({e.msg})"]
    if not isinstance(obj, dict):
        return None, ["the reply is not a JSON object"]
    return obj, []


# ------------------------------------------------------------------ calling the API

def read_key(env_file: Path | None = None) -> str | None:
    """The API key: the environment variable first, else a KEY_ENV=value line in the .env file. None if neither has
    it. Handles the UTF-8 BOM and UTF-16 that Windows editors and PowerShell can write."""
    key = os.environ.get(KEY_ENV, "").strip()
    if key:
        return key
    path = env_file or ENV_FILE
    if not path.exists():
        return None
    raw = path.read_bytes()
    text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8-sig")
    for line in text.splitlines():
        name, sep, value = line.strip().removeprefix("export ").partition("=")
        if sep and name.strip() == KEY_ENV:
            return value.strip().strip("'\"") or None
    return None


class DailyLimit(RuntimeError):
    """The provider asked for a long wait: stop now, resume later (finished cases are cached)."""


class ProviderUnavailable(RuntimeError):
    """No usable answer after several tries (timeouts, connection errors): stop now, resume later."""


class RequestTooLarge(RuntimeError):
    """HTTP 413: this request alone exceeds the provider's size limit (possible for a repair call on a Red case)."""


class Throttle:
    """Keeps the tokens used in the last 60 s under TOKENS_PER_MINUTE_BUDGET."""

    def __init__(self, budget: int = TOKENS_PER_MINUTE_BUDGET, clock=time.monotonic, sleep=time.sleep):
        self.budget, self.clock, self.sleep, self.used = budget, clock, sleep, deque()

    def wait_for(self, tokens: int) -> None:
        while True:
            now = self.clock()
            while self.used and now - self.used[0][0] >= 60:
                self.used.popleft()
            if not self.used or sum(t for _, t in self.used) + tokens <= self.budget:
                return
            self.sleep(max(0.5, 60 - (now - self.used[0][0])))

    def record(self, tokens: int) -> None:
        self.used.append((self.clock(), tokens))


def call_api(body: dict, key: str, post=None, sleep=time.sleep, max_tries: int = 4) -> dict:
    """One chat completion; retries timeouts, connection errors, 5xx and short 429 waits. Raises DailyLimit on a long
    wait, ProviderUnavailable when the tries run out, RequestTooLarge on a 413."""
    import requests

    post = post or requests.post
    for attempt in range(1, max_tries + 1):
        try:
            r = post(API_URL, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                     json=body, timeout=TIMEOUT_S)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            if attempt == max_tries:
                raise ProviderUnavailable(f"no answer after {max_tries} tries ({type(e).__name__})") from e
            sleep(10 * attempt)
            continue
        if r.status_code == 413:
            raise RequestTooLarge("the request exceeds the provider's size limit")
        if r.status_code == 429:
            wait = float(r.headers.get("retry-after", 60))
            if wait > LONG_WAIT_S:
                raise DailyLimit(f"the provider asked to wait {wait:.0f} s (likely the daily limit)")
            sleep(wait)
            continue
        if r.status_code >= 500 and attempt < max_tries:
            sleep(5 * attempt)
            continue
        if r.status_code != 200:
            raise RuntimeError(f"API error {r.status_code}: {r.text[:300]}")
        return r.json()
    raise ProviderUnavailable(f"the provider kept refusing the request after {max_tries} tries")


def run_case(case: dict, wbgt_threshold: float, library: dict[str, Action], key: str, throttle: Throttle,
             post=None, sleep=time.sleep, now=lambda: datetime.now(timezone.utc)) -> dict:
    """Generate, verify, repair once, fall back; returns the record that is cached."""
    a = assessment_of(case, wbgt_threshold)
    sch, msg = schema(a, library), user_message(a, library)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": msg}]
    rec = {"case_id": case["case_id"], "model": MODEL, "settings": SETTINGS, "api_url": API_URL,
           "system_prompt_sha256": _sha(SYSTEM_PROMPT), "user_message_sha256": _sha(msg), "schema_sha256": _sha(sch),
           "headline_colour": a.headline_colour, "attempts": []}
    advisory = None
    for attempt in (1, 2):
        estimate = (len(SYSTEM_PROMPT) + sum(len(m["content"]) for m in messages)) // 3 + 1500
        throttle.wait_for(estimate)
        try:
            resp = call_api(request_body(messages, sch), key, post=post, sleep=sleep)
        except RequestTooLarge as e:  # recorded; this case then falls back to the rule-based advisory
            rec["attempts"].append({"attempt": attempt, "utc": now().isoformat(timespec="seconds"), "raw_content": None,
                                    "reasoning": None, "usage": {}, "system_fingerprint": None,
                                    "violations": [f"not sent: {e}"]})
            break
        usage = resp.get("usage", {})
        throttle.record(int(usage.get("total_tokens", estimate)))
        message = (resp.get("choices") or [{}])[0].get("message", {})
        content = message.get("content")
        reply, problems = parse_reply(content)
        candidate = assemble(reply, a, library) if reply is not None else None
        violations = problems if candidate is None else verify(candidate, a, library)
        rec["attempts"].append({"attempt": attempt, "utc": now().isoformat(timespec="seconds"), "raw_content": content,
                                "reasoning": message.get("reasoning"), "usage": usage,
                                "system_fingerprint": resp.get("system_fingerprint"), "violations": violations})
        if not violations:
            advisory = candidate
            break
        messages = messages + [{"role": "assistant", "content": content or ""},
                               {"role": "user", "content": REPAIR_INSTRUCTION.format(
                                   violations="\n".join(f"- {v}" for v in violations))}]
    rec["outcome"] = ("passed_first_try" if advisory is not None and len(rec["attempts"]) == 1
                      else "passed_after_repair" if advisory is not None else "template_fallback")
    rec["advisory"] = advisory if advisory is not None else template_advisory(a, library)
    return rec


def run(cases_path: Path = CASES_PATH, only: list[str] | None = None, post=None, sleep=time.sleep,
        runs_dir: Path = RUNS_DIR) -> dict:
    """Every case and practice item without a cached record; returns counts by outcome."""
    key = read_key()
    if not key:
        raise SystemExit(f"No API key found. Add a line {KEY_ENV}=<your key> to the .env file in the project folder, "
                         f"or set $env:{KEY_ENV} in this terminal.")
    doc, library = load_cases(cases_path), load_library()
    runs_dir.mkdir(parents=True, exist_ok=True)
    throttle, counts = Throttle(sleep=sleep), {}
    for case in doc["cases"] + doc["practice"]:
        if only and case["case_id"] not in only:
            continue
        path = runs_dir / f"{case['case_id']}.json"
        if path.exists():
            outcome = json.loads(path.read_text(encoding="utf-8"))["outcome"]
        else:
            try:
                rec = run_case(case, doc["wbgt_threshold"], library, key, throttle, post=post, sleep=sleep)
            except (DailyLimit, ProviderUnavailable) as e:
                print(f"Stopped at {case['case_id']}: {e}. Finished cases are saved; run the same command later.")
                break
            path.write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
            outcome = rec["outcome"]
            print(f"{case['case_id']} ({rec['headline_colour']}): {outcome}", flush=True)
        counts[outcome] = counts.get(outcome, 0) + 1
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate the LLM advisories (Groq).")
    ap.add_argument("--dry-run", metavar="CASE_ID", help="print the prompt and schema for one case; no API call")
    ap.add_argument("--only", nargs="+", metavar="CASE_ID", help="run only these cases")
    args = ap.parse_args(argv)
    if args.dry_run:
        doc, library = load_cases(), load_library()
        case = next(c for c in doc["cases"] + doc["practice"] if c["case_id"] == args.dry_run)
        a = assessment_of(case, doc["wbgt_threshold"])
        print(SYSTEM_PROMPT, "\n\n---\n", user_message(a, library), "\n\n---\n", json.dumps(schema(a, library), indent=1))
        return 0
    counts = run(only=args.only)
    print("Outcomes so far:", counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
