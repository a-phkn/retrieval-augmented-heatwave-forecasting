"""
The verifier: checks a finished advisory (the structure the constrained LLM will fill in) against the rule engine and
the action library, and rejects it on any violation (decision 2026-10-26 in context/decisions.md).

The advisory's actions are chosen BY ID; the reader sees the library's text and quote, so the LLM cannot reword an
instruction. What the LLM writes freely (the summary, one sentence per day, the humid-heat note) is checked for
numbers that are not in the forecast, colour words that differ from the rule engine, and claims of official status.

An advisory is a dict:
  label             IMD_STYLE_LABEL, word for word
  headline_colour   the rule engine's headline colour
  summary           free text
  days              5 x {date "YYYY-MM-DD", colour, p_hot (or None), text (free)}
  forecast_actions  [{action_id, label (PRECEDENT_LABEL, required for Ahmedabad-coloured actions), quote (optional)}]
  preparedness      [{action_id, quote (optional)}]                       (only in the heat season)
  if_this_happens   {public: [{action_id, ...}], professionals: {disclaimer, actions: [{action_id, ...}]}}
  humid_heat_note   None, or {text (free; must include HUMID_SOURCE_PHRASE)}  (exactly when the engine flags humid heat)

template_advisory() builds a valid advisory from the rules alone (no LLM): the reference the LLM output is compared
with, and the fallback when the LLM output is rejected.
"""
from __future__ import annotations

import math
import re

from advisory.library import CLINICAL_DISCLAIMER, IMD_STYLE_LABEL, PRECEDENT_LABEL, Action, colour_rank
from advisory.rules import (
    FORECAST_DAYS, Assessment, forecast_actions, if_this_happens_actions, preparedness_actions,
)
from pipeline.labels_v2 import ANOMALY_HOT_C, ANOMALY_SEVERE_C, TMAX_ABSOLUTE_C, TMAX_FLOOR_C

HUMID_SOURCE_PHRASE = "from our physical WBGT forecast, not an IMD category"
TOP_KEYS = {"label", "headline_colour", "summary", "days", "forecast_actions", "preparedness", "if_this_happens",
            "humid_heat_note"}
_NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")
_COLOUR_WORD = re.compile(r"\b(green|yellow|orange|red)\b", re.IGNORECASE)
_HUMID_FORBIDDEN = re.compile(r"\b(heat[\s-]?waves?|alerts?|warnings?|green|yellow|orange|red)\b", re.IGNORECASE)
_OFFICIAL_CLAIM = re.compile(r"\bIMD\s+(?:has\s+)?(?:issued|issues|warns|declared|declares|announced)\b", re.IGNORECASE)
ALWAYS_ALLOWED = (108, 102, TMAX_FLOOR_C, TMAX_ABSOLUTE_C, ANOMALY_HOT_C, ANOMALY_SEVERE_C, *range(0, 13))


def _norm(s: str) -> str:
    return " ".join(str(s).split())


def allowed_numbers(a: Assessment) -> list[float]:
    """Every number the free text may mention: the forecast values, dates, probabilities (as %), thresholds,
    emergency numbers and small day counts."""
    vals = [float(v) for v in ALWAYS_ALLOWED]
    for d in a.days:
        vals += [d.tmax, d.normal, abs(d.anomaly), d.date.day, d.date.year, d.lead]
        if d.p_hot is not None:
            vals += [100 * d.p_hot]
        if d.wbgt is not None:
            vals += [d.wbgt]
    if a.wbgt_threshold is not None:
        vals.append(a.wbgt_threshold)
    return vals


def unsupported_numbers(text: str, allowed: list[float]) -> list[str]:
    """Numbers in the text that do not match an allowed value at the precision written ("38" matches 38.4)."""
    bad = []
    for tok in _NUMBER.findall(text or ""):
        x, nd = float(tok), len(tok.split(".")[1]) if "." in tok else 0
        if not any(math.isclose(round(v, nd), x, abs_tol=1e-9) for v in allowed):
            bad.append(tok)
    return bad


def _check_entry(e: dict, where: str, section: str, a: Assessment, library: dict[str, Action], out: list[str],
                 seen: set[str]) -> Action | None:
    if not isinstance(e, dict) or "action_id" not in e:
        out.append(f"{where}: an entry without action_id")
        return None
    aid = e["action_id"]
    if aid in seen:
        out.append(f"{where}: {aid} listed twice")
    seen.add(aid)
    act = library.get(aid)
    if act is None:
        out.append(f"{where}: unknown action id {aid}")
        return None
    if act.section != section:
        out.append(f"{where}: {aid} belongs to section '{act.section}', not '{section}'")
    if section == "forecast" and colour_rank(act.min_colour) > colour_rank(a.headline_colour):
        out.append(f"{where}: {aid} needs {act.min_colour}, but the rule engine's colour is {a.headline_colour}")
    if act.tier_basis == "ahmedabad_precedent" and e.get("label") != PRECEDENT_LABEL:
        out.append(f"{where}: {aid} has an Ahmedabad colour and must carry the precedent label")
    if "quote" in e and _norm(e["quote"]) not in _norm(act.source.text):
        out.append(f"{where}: {aid} quote is not an exact part of the source excerpt")
    return act


def _check_free_text(text: str, where: str, allowed: list[float], colours_ok: set[str], out: list[str]) -> None:
    if not isinstance(text, str):
        out.append(f"{where}: text missing")
        return
    for tok in unsupported_numbers(text, allowed):
        out.append(f"{where}: number {tok} is not in the forecast")
    for w in {m.lower() for m in _COLOUR_WORD.findall(text)}:
        if w.upper() not in colours_ok:
            out.append(f"{where}: mentions {w.upper()}, but the rule engine gives {sorted(colours_ok)}")
    if _OFFICIAL_CLAIM.search(text):
        out.append(f"{where}: claims an IMD warning was issued (this is not an official IMD warning)")


def verify(adv: dict, a: Assessment, library: dict[str, Action]) -> list[str]:
    """All violations (an empty list means the advisory passes)."""
    out: list[str] = []
    if not isinstance(adv, dict):
        return ["advisory is not an object"]
    keys = set(adv)
    if keys != TOP_KEYS:
        out += [f"missing field {k}" for k in sorted(TOP_KEYS - keys)] + [f"unexpected field {k}" for k in sorted(keys - TOP_KEYS)]
        return out
    allowed = allowed_numbers(a)
    if adv["label"] != IMD_STYLE_LABEL:
        out.append("the IMD-style label is missing or altered")
    if adv["headline_colour"] != a.headline_colour:
        out.append(f"headline colour {adv['headline_colour']} differs from the rule engine ({a.headline_colour})")
    _check_free_text(adv["summary"], "summary", allowed, {d.colour for d in a.days}, out)

    days = adv["days"]
    if not isinstance(days, list) or len(days) != FORECAST_DAYS:
        out.append(f"days must list the {FORECAST_DAYS} forecast days")
    else:
        for i, (d, ref) in enumerate(zip(days, a.days), start=1):
            if d.get("date") != ref.date.strftime("%Y-%m-%d"):
                out.append(f"day {i}: date {d.get('date')} should be {ref.date:%Y-%m-%d}")
            if d.get("colour") != ref.colour:
                out.append(f"day {i}: colour {d.get('colour')} differs from the rule engine ({ref.colour})")
            p = d.get("p_hot")
            if (p is None) != (ref.p_hot is None) or (p is not None and not math.isclose(p, ref.p_hot, abs_tol=1e-9)):
                out.append(f"day {i}: probability differs from the forecast")
            _check_free_text(d.get("text"), f"day {i}", allowed, {ref.colour}, out)

    seen: set[str] = set()
    for e in adv["forecast_actions"]:
        _check_entry(e, "forecast_actions", "forecast", a, library, out, seen)
    if adv["preparedness"] and not a.in_heat_season:
        out.append("preparedness actions are shown only in the heat season")
    for e in adv["preparedness"]:
        _check_entry(e, "preparedness", "preparedness", a, library, out, seen)

    ith = adv["if_this_happens"]
    if not isinstance(ith, dict) or set(ith) != {"public", "professionals"}:
        out.append("if_this_happens must have exactly 'public' and 'professionals'")
    else:
        for e in ith["public"]:
            act = _check_entry(e, "if_this_happens.public", "if_this_happens", a, library, out, seen)
            if act is not None and act.for_professionals:
                out.append(f"if_this_happens.public: {act.action_id} is clinical guidance for health professionals")
        pro = ith["professionals"]
        if not isinstance(pro, dict) or set(pro) != {"disclaimer", "actions"}:
            out.append("if_this_happens.professionals must have 'disclaimer' and 'actions'")
        else:
            if pro["actions"] and pro["disclaimer"] != CLINICAL_DISCLAIMER:
                out.append("the clinical disclaimer is missing or altered")
            for e in pro["actions"]:
                act = _check_entry(e, "if_this_happens.professionals", "if_this_happens", a, library, out, seen)
                if act is not None and not act.for_professionals:
                    out.append(f"if_this_happens.professionals: {act.action_id} is public guidance")

    note = adv["humid_heat_note"]
    if a.any_humid_heat and note is None:
        out.append("the rule engine flags humid heat but there is no humid-heat note")
    if not a.any_humid_heat and note is not None:
        out.append("a humid-heat note is given but the rule engine flags no humid heat")
    if note is not None:
        text = note.get("text") if isinstance(note, dict) else None
        if not isinstance(text, str) or HUMID_SOURCE_PHRASE not in text:
            out.append("the humid-heat note must say it comes from our physical WBGT, not an IMD category")
        else:
            for w in sorted({m.group(0).lower() for m in _HUMID_FORBIDDEN.finditer(text)}):
                out.append(f"humid-heat note uses the IMD alert word '{w}' for WBGT")
            for tok in unsupported_numbers(text, allowed):
                out.append(f"humid-heat note: number {tok} is not in the forecast")
    return out


def template_advisory(a: Assessment, library: dict[str, Action]) -> dict:
    """A valid advisory from the rules alone: every unlocked action, fixed sentences, no LLM."""
    def entry(act: Action) -> dict:
        e = {"action_id": act.action_id}
        if act.tier_basis == "ahmedabad_precedent":
            e["label"] = PRECEDENT_LABEL
        return e

    days = []
    for d in a.days:
        chance = "" if d.p_hot is None else f" Chance of heat-wave conditions: {100 * d.p_hot:.0f}%."
        days.append({"date": d.date.strftime("%Y-%m-%d"), "colour": d.colour, "p_hot": d.p_hot,
                     "text": f"{d.colour}. Forecast maximum {d.tmax:.1f} °C, {abs(d.anomaly):.1f} °C "
                             f"{'above' if d.anomaly >= 0 else 'below'} normal.{chance}"})
    public, pro = if_this_happens_actions(library)
    humid = [d for d in a.days if d.humid_heat]
    note = None if not humid else {
        "text": "High humid heat stress expected on " + ", ".join(f"{d.date.day} {d.date:%B}" for d in humid)
                + f" ({HUMID_SOURCE_PHRASE})."}
    return {
        "label": IMD_STYLE_LABEL,
        "headline_colour": a.headline_colour,
        "summary": f"Highest level in the next {FORECAST_DAYS} days: {a.headline_colour}.",
        "days": days,
        "forecast_actions": [entry(x) for x in forecast_actions(a.headline_colour, library)],
        "preparedness": [entry(x) for x in preparedness_actions(a, library)],
        "if_this_happens": {"public": [entry(x) for x in public],
                            "professionals": {"disclaimer": CLINICAL_DISCLAIMER, "actions": [entry(x) for x in pro]}},
        "humid_heat_note": note,
    }
