"""
The action library: every action the heat advisory may recommend, each tied to a word-for-word quote from an official
document (decisions 2026-10-06, 2026-10-13 and 2026-10-26 in context/decisions.md).

Built only from sources/phase6_candidate_actions.json (Phase 6 source research, every quote verified against its
page); nothing is invented here. What the build adds is the decided presentation rules:
  - section: forecast actions, heat-season preparedness, or "If this happens" (never mixed);
  - minimum colour: as sourced; an action no source gives a colour starts at YELLOW ("delhi_any_alert", because
    Delhi's plan lists it for "during a heat alert");
  - tier_basis: where the colour comes from (delhi_source, national_source, ahmedabad_precedent, delhi_any_alert);
  - for_professionals: clinical / health-system steps shown behind the clinical disclaimer.

Run from repo root:  python -m advisory.library   ->  writes advisory/action_library.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CANDIDATES_PATH = REPO_ROOT / "sources" / "phase6_candidate_actions.json"
LIBRARY_PATH = REPO_ROOT / "advisory" / "action_library.json"

COLOURS = ("GREEN", "YELLOW", "ORANGE", "RED")  # IMD's colour-coded warning levels, lowest first
SECTIONS = ("forecast", "preparedness", "if_this_happens")
TIER_BASES = ("delhi_source", "national_source", "ahmedabad_precedent", "delhi_any_alert")
AUDIENCES = ("health", "public", "municipal", "energy")

# Labels the advisory must show (checked by advisory.verifier)
IMD_STYLE_LABEL = "IMD-style, adjusted for area-average data; not an official IMD warning"
PRECEDENT_LABEL = "Precedent from Ahmedabad's Heat Action Plan (2019 update), not Delhi policy"
CLINICAL_DISCLAIMER = ("Clinical and health-system guidance quoted from official health documents, for trained health "
                       "workers only. Members of the public: call 108 or 102.")

# Decision 2026-10-26 (user option 2a): practice abroad, not Delhi or Indian instructions
EXCLUDED = {
    "WHO_ENERGY_UTILITIES_SUSPEND_DISCONNECTIONS": "low confidence: international practice (WHO/WMO), not a Delhi instruction",
    "WHO_ENERGY_PHILADELPHIA_HALT_SERVICE_SUSPENSIONS": "low confidence: a Philadelphia (USA) example, not a Delhi instruction",
}
_SECTION_BY_TRIGGER = {"forecast": "forecast", "standing": "preparedness", "observation": "if_this_happens"}


@dataclass(frozen=True)
class Quote:
    """A verified word-for-word quote and where it is."""
    doc_id: str
    page: int
    text: str
    printed_page: int | None = None


@dataclass(frozen=True)
class Action:
    action_id: str
    text: str  # our plain summary of the quoted instruction (shown to readers; the quote is the evidence)
    audience: str
    section: str
    min_colour: str  # GREEN never: forecast actions start at YELLOW
    tier_basis: str
    trigger: str
    for_professionals: bool
    confidence: str
    source: Quote  # the instruction itself
    colour_source: Quote | None  # the quote that gives the colour (None for delhi_any_alert)
    also_in: tuple[str, ...]
    notes: str


def colour_rank(colour: str) -> int:
    """GREEN 0 < YELLOW 1 < ORANGE 2 < RED 3."""
    if colour not in COLOURS:
        raise ValueError(f"unknown colour {colour!r}")
    return COLOURS.index(colour)


def _quote(doc_id: str, page, text: str, printed_page=None) -> Quote:
    return Quote(doc_id=str(doc_id), page=int(page), text=str(text),
                 printed_page=None if printed_page in (None, "") else int(printed_page))


def build_action(c: dict) -> Action:
    """One candidate row -> a library Action, applying the decided rules. Raises on anything unexpected."""
    trig = c["trigger_type"]
    if trig not in _SECTION_BY_TRIGGER:
        raise ValueError(f"{c['action_id']}: unknown trigger_type {trig!r}")
    if not str(c.get("verification", "")).startswith("VERIFIED"):
        raise ValueError(f"{c['action_id']}: quote not verified")
    sourced = c["tier_min"] != "UNSPECIFIED"
    ts = c.get("tier_source")
    if sourced:
        if not ts or ts.get("tier_basis") not in ("delhi_source", "national_source", "ahmedabad_precedent"):
            raise ValueError(f"{c['action_id']}: a colour without a recorded colour source")
        min_colour, basis = c["tier_min"], ts["tier_basis"]
        colour_source = _quote(ts["doc_id"], ts["page"], ts["quote"])
    else:
        if ts:
            raise ValueError(f"{c['action_id']}: colour source recorded but no colour")
        min_colour, basis, colour_source = "YELLOW", "delhi_any_alert", None
    if min_colour not in COLOURS[1:]:
        raise ValueError(f"{c['action_id']}: minimum colour {min_colour!r}")
    section = _SECTION_BY_TRIGGER[trig]
    if section == "preparedness" and sourced:  # decision 2026-10-26: a sourced colour is kept
        section = "forecast"
    audience = c["audience"]
    if audience not in AUDIENCES:
        raise ValueError(f"{c['action_id']}: unknown audience {audience!r}")
    return Action(
        action_id=c["action_id"], text=c["text"], audience=audience, section=section, min_colour=min_colour,
        tier_basis=basis, trigger=c["trigger"],
        for_professionals=section == "if_this_happens" and audience == "health",
        confidence=c["confidence"],
        source=_quote(c["doc_id"], c["page"], c["verbatim_excerpt"], c.get("printed_page")),
        colour_source=colour_source,
        also_in=tuple(sorted(f"{a['doc_id']} p.{a['page']}" for a in c.get("also_in", []))),
        notes=c.get("notes", ""))


def build_library(candidates: list[dict]) -> list[Action]:
    """All candidates except the excluded ones, sorted by id; ids must be unique."""
    ids = [c["action_id"] for c in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate action ids in the candidates")
    missing = set(EXCLUDED) - set(ids)
    if missing:
        raise ValueError(f"excluded ids not found in the candidates: {sorted(missing)}")
    return sorted((build_action(c) for c in candidates if c["action_id"] not in EXCLUDED), key=lambda a: a.action_id)


def library_document(actions: list[Action], candidates_bytes: bytes) -> dict:
    """The JSON written to advisory/action_library.json."""
    return {
        "description": "Heat-advisory action library (Phase 6). Built by advisory/library.py from "
                       "sources/phase6_candidate_actions.json; do not edit by hand.",
        "built_from_sha256": hashlib.sha256(candidates_bytes.replace(b"\r\n", b"\n")).hexdigest(),
        "decisions": "context/decisions.md: 2026-10-06 (sections, energy), 2026-10-13 (colours), 2026-10-26 (library)",
        "labels": {"imd_style": IMD_STYLE_LABEL, "precedent": PRECEDENT_LABEL, "clinical": CLINICAL_DISCLAIMER},
        "excluded": EXCLUDED,
        "actions": [asdict(a) for a in actions],
    }


def build_document() -> dict:
    raw = CANDIDATES_PATH.read_bytes()
    return library_document(build_library(json.loads(raw.decode("utf-8"))), raw)


def _action_from_dict(d: dict) -> Action:
    d = dict(d)
    d["source"] = Quote(**d["source"])
    d["colour_source"] = Quote(**d["colour_source"]) if d["colour_source"] else None
    d["also_in"] = tuple(d["also_in"])
    return Action(**d)


def load_library(path: Path = LIBRARY_PATH) -> dict[str, Action]:
    """The committed library, by action id."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    return {d["action_id"]: _action_from_dict(d) for d in doc["actions"]}


def main() -> int:
    doc = build_document()
    with open(LIBRARY_PATH, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
        f.write("\n")
    acts = doc["actions"]
    by = lambda k: {v: sum(a[k] == v for a in acts) for v in sorted({a[k] for a in acts})}  # noqa: E731
    print(f"Wrote {LIBRARY_PATH.relative_to(REPO_ROOT)}: {len(acts)} actions ({len(EXCLUDED)} excluded)")
    print("  by section:", by("section"))
    print("  by minimum colour:", by("min_colour"))
    print("  by tier_basis:", by("tier_basis"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
