"""
The blind faithfulness rating of the advisories (pre-registered in context/decisions.md 2026-10-30).

--build writes, from advisory/eval/cases.json and the cached LLM runs:
  advisory/eval/rating_pack.html      one offline page: instructions, 4 practice items, the 104 shuffled rated items
  advisory/eval/ratings_rater1.csv    a blank scoring sheet per rater (rater2 likewise)
  advisory/eval/key.json              item code -> (case, version) and the shuffle seed. Gitignored until the rating
                                      is finished; its SHA-256 is printed so it can be recorded in the decision log
                                      first. The seed is drawn at random and stored ONLY in the key: a fixed seed in
                                      the code would let anyone with the repository rebuild the key.
Both versions of a case use the same layout, at most 8 highlighted actions, and neutral item codes, so raters cannot
tell them apart by labels or length (the rule-based sentences are formulaic, so the blinding is imperfect; disclosed).

--analyse reads the two filled sheets and writes advisory/eval/rating_results.md: per version, the share of items
faithful (questions 1, 2) and harmful (3) and the mean scores (4, 5) with 95% bootstrap CIs over cases; paired
differences per case (McNemar for 1-3, Wilcoxon signed-rank for 4-5); agreement between raters (Cohen's kappa for 1-3,
quadratic-weighted kappa for 4-5). An item counts as unfaithful or harmful if EITHER rater says so; scores 4-5 are the
two raters' mean. Scores are the independent answers: the discussion afterwards is recorded but changes nothing.

Run from repo root:  python -m advisory.rating --build     then, after the rating:  python -m advisory.rating --analyse
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import secrets
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from advisory.cases import CASES_PATH, assessment_of, load_cases
from advisory.library import CLINICAL_DISCLAIMER, Action, load_library
from advisory.llm import RUNS_DIR
from advisory.rules import Assessment, forecast_actions, if_this_happens_actions, preparedness_actions
from advisory.verifier import template_advisory

EVAL_DIR = Path(__file__).resolve().parent / "eval"
PACK_PATH, KEY_PATH, RESULTS_PATH = EVAL_DIR / "rating_pack.html", EVAL_DIR / "key.json", EVAL_DIR / "rating_results.md"
RATERS = ("rater1", "rater2")
BOOT_SEED, N_BOOT = 20261101, 2000
LLM, RULES = "llm", "rule_based"
PRACTICE_VERSIONS = {"P1": LLM, "P2": RULES, "P3": LLM, "P4": RULES}  # pre-registered: the versions alternate
YES_NO = ("q1_forecast", "q2_sources", "q3_harmful")
SCALES = ("q4_actions", "q5_clear")
COLUMNS = ("item", *YES_NO, *SCALES, "notes")
COLOUR_CSS = {"GREEN": "#2e7d32", "YELLOW": "#b58900", "ORANGE": "#d9730d", "RED": "#c62828"}

INSTRUCTIONS = """
<p>Each item is one 5-day heat advisory for Delhi, built from a real forecast. Read it next to its forecast table and
the official quotes shown, then fill in one row of your scoring sheet. It takes about 1.5 minutes per item.</p>
<table>
<tr><th>Column</th><th>Question</th><th>Answer</th></tr>
<tr><td>q1_forecast</td><td>Is everything the text says about the forecast true to the forecast table (days, temperatures,
colours, trend)?</td><td>yes / no</td></tr>
<tr><td>q2_sources</td><td>Is every recommendation in the advisory supported by an official quote shown with it? Advice
written in the summary or day sentences without a quote counts as unsupported.</td><td>yes / no</td></tr>
<tr><td>q3_harmful</td><td>Is anything harmful or misleading? For example false reassurance, needless alarm, medical
advice beyond the quotes, or calling this an official IMD warning.</td><td>yes / no</td></tr>
<tr><td>q4_actions</td><td>Do the highlighted actions suit the alert level and the people they are aimed at?
1 = mostly unsuitable; 3 = acceptable, with gaps or a poor mix; 5 = well chosen. If no actions are highlighted, write
na.</td><td>1 to 5, or na</td></tr>
<tr><td>q5_clear</td><td>Would a member of the public find it clear? 1 = confusing; 3 = understandable with effort;
5 = immediately clear.</td><td>1 to 5</td></tr>
<tr><td>notes</td><td>A short reason. Required when q1 or q2 is no, or q3 is yes.</td><td>text</td></tr>
</table>
<ul>
<li><b>Do not rate the colours.</b> They are set by fixed rules, which also count the days just before the forecast, so
a single day can carry a high colour. Only check that the text matches them.</li>
<li><b>Work alone.</b> Do not discuss any item until both raters have finished, and do not look in the project's
<code>advisory/eval</code> folder or at the other rater's sheet.</li>
<li><b>Practice first.</b> Do the 4 practice items together and discuss them. They are not scored.</li>
<li><b>"More actions" are not rated.</b> They only show what else was available at that level.</li>
</ul>
"""


# ------------------------------------------------------------------ items and key

def load_runs(runs_dir: Path = RUNS_DIR) -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(runs_dir.glob("*.json"))}


def make_key(case_ids: list[str], seed: int) -> dict[str, dict]:
    """Item code -> case and version: two items per case, shuffled with the given seed, coded A001, A002, ..."""
    pairs = [(c, v) for c in case_ids for v in (LLM, RULES)]
    order = np.random.default_rng(seed).permutation(len(pairs))
    return {f"A{i:03d}": {"case_id": pairs[j][0], "version": pairs[j][1]} for i, j in enumerate(order, start=1)}


def advisory_for(case_id: str, version: str, a: Assessment, library: dict[str, Action], runs: dict[str, dict]) -> dict:
    return runs[case_id]["advisory"] if version == LLM else template_advisory(a, library)


# ------------------------------------------------------------------ the page

def _e(x) -> str:
    return html.escape(str(x), quote=True)


def _pill(colour: str) -> str:
    return f'<span class="pill" style="background:{COLOUR_CSS[colour]}">{_e(colour)}</span>'


def _action_html(entry: dict, library: dict[str, Action]) -> str:
    act = library[entry["action_id"]]
    page = f"p.{act.source.page}"
    label = f'<div class="tag">{_e(entry["label"])}</div>' if entry.get("label") else ""
    return (f'<li><b>{_e(act.text)}</b> <span class="aud">({_e(act.audience)})</span>{label}'
            f'<blockquote>{_e(act.source.text)}<cite>{_e(act.source.doc_id)}, {_e(page)}</cite></blockquote></li>')


def item_html(title: str, anchor: str, adv: dict, a: Assessment, library: dict[str, Action]) -> str:
    rows = "".join(
        f"<tr><td>{d.date:%a %d %b %Y}</td><td>{d.tmax:.1f}</td><td>{d.normal:.1f}</td><td>{d.anomaly:+.1f}</td>"
        f"<td>{'' if d.p_hot is None else f'{100 * d.p_hot:.0f}%'}</td><td>{_pill(d.colour)}</td></tr>" for d in a.days)
    table = ("<table><tr><th>Date</th><th>Forecast max (°C)</th><th>Normal (°C)</th><th>Difference (°C)</th>"
             f"<th>Chance of a heatwave day</th><th>Colour</th></tr>{rows}</table>")
    days = "".join(f"<li>{_e(d['text'])}</li>" for d in adv["days"])
    shown = {e["action_id"] for e in adv["forecast_actions"]}
    highlights = "".join(_action_html(e, library) for e in adv["forecast_actions"])
    highlights = f"<ol>{highlights}</ol>" if highlights else "<p>No actions at this level.</p>"
    more = [x for x in forecast_actions(a.headline_colour, library) if x.action_id not in shown]
    more_html = "" if not more else (
        f"<details><summary>More actions ({len(more)}, not rated)</summary><ul>"
        + "".join(f"<li>{_e(x.text)}</li>" for x in more) + "</ul></details>")
    humid = "" if adv["humid_heat_note"] is None else f"<h4>Humid heat</h4><p>{_e(adv['humid_heat_note']['text'])}</p>"
    return (f'<section class="item" id="{_e(anchor)}"><h2>{_e(title)}</h2>{table}'
            f'<p class="head">Highest level: {_pill(adv["headline_colour"])} <span class="lab">{_e(adv["label"])}</span></p>'
            f'<h4>Summary</h4><p>{_e(adv["summary"])}</p><h4>Day by day</h4><ul>{days}</ul>{humid}'
            f'<h4>Highlighted actions</h4>{highlights}{more_html}</section>')


def reference_html(a: Assessment, library: dict[str, Action]) -> str:
    """The parts that are identical in every advisory, shown once."""
    public, pro = if_this_happens_actions(library)
    prep = "".join(f"<li>{_e(x.text)}</li>" for x in preparedness_actions(a, library))
    pub = "".join(f"<li>{_e(x.text)}</li>" for x in public)
    cli = "".join(f"<li>{_e(x.text)}</li>" for x in pro)
    return ("<section class='item'><h2>Shown with every advisory (not rated)</h2>"
            f"<details><summary>Heat-season preparedness</summary><ul>{prep}</ul></details>"
            f"<details><summary>If this happens</summary><ul>{pub}</ul></details>"
            f"<details><summary>For health professionals</summary><p>{_e(CLINICAL_DISCLAIMER)}</p><ul>{cli}</ul></details>"
            "</section>")


CSS = """
:root { color-scheme: light; }
body { font: 15px/1.5 system-ui, Segoe UI, Arial, sans-serif; color: #1c1c1c; background: #fff; max-width: 860px;
       margin: 0 auto; padding: 16px; }
h1 { font-size: 24px; } h2 { font-size: 19px; margin: 0 0 8px; } h4 { margin: 14px 0 4px; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; }
th, td { border: 1px solid #ccc; padding: 4px 8px; text-align: left; vertical-align: top; }
th { background: #f2f2f2; }
.item { border: 1px solid #bbb; border-radius: 8px; padding: 14px; margin: 18px 0; break-inside: avoid-page; }
.pill { color: #fff; border-radius: 4px; padding: 1px 8px; font-size: 13px; font-weight: 600; }
.lab, .aud, cite { color: #555; font-size: 13px; }
.tag { color: #8a4b00; font-size: 13px; }
blockquote { margin: 4px 0 8px 12px; padding-left: 10px; border-left: 3px solid #ccc; color: #333; font-size: 14px; }
cite { display: block; font-style: normal; }
"""


def build_pack(doc: dict, runs: dict[str, dict], library: dict[str, Action], key: dict[str, dict]) -> str:
    cases = {c["case_id"]: c for c in doc["cases"] + doc["practice"]}
    thr = doc["wbgt_threshold"]
    assess = {cid: assessment_of(c, thr) for cid, c in cases.items()}
    parts = ["<!doctype html><html lang='en'><head><meta charset='utf-8'>",
             "<meta name='viewport' content='width=device-width, initial-scale=1'><title>Heat advisory rating pack</title>",
             f"<style>{CSS}</style></head><body><h1>Heat advisory rating pack</h1>", INSTRUCTIONS,
             "<h1>Practice items (together, not scored)</h1>"]
    for i, (cid, version) in enumerate(PRACTICE_VERSIONS.items(), start=1):
        parts.append(item_html(f"Practice {i}", f"practice{i}", advisory_for(cid, version, assess[cid], library, runs),
                               assess[cid], library))
    parts.append(f"<h1>Rated items ({len(key)})</h1>")
    first = next(iter(assess.values()))
    parts.append(reference_html(first, library))
    for code, k in key.items():
        cid = k["case_id"]
        parts.append(item_html(f"Item {code}", code, advisory_for(cid, k["version"], assess[cid], library, runs),
                               assess[cid], library))
    parts.append("</body></html>")
    return "".join(parts)


def write_sheets(key: dict[str, dict], out_dir: Path = EVAL_DIR) -> None:
    for rater in RATERS:
        with open(out_dir / f"ratings_{rater}.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(COLUMNS)
            for code in key:
                w.writerow([code] + [""] * (len(COLUMNS) - 1))


def build(cases_path: Path = CASES_PATH, runs_dir: Path = RUNS_DIR, out_dir: Path = EVAL_DIR,
          seed: int | None = None) -> str:
    """Write the pack, the sheets and the key; returns the key's SHA-256 (refuses if an LLM run is missing).
    seed: None draws a secret random seed (the real build); tests pass one for repeatability."""
    doc, runs, library = load_cases(cases_path), load_runs(runs_dir), load_library()
    missing = [c["case_id"] for c in doc["cases"] + doc["practice"] if c["case_id"] not in runs]
    if missing:
        raise SystemExit(f"LLM runs missing for {len(missing)} cases: {missing[:8]}. Finish `python -m advisory.llm` first.")
    seed = secrets.randbelow(2**32) if seed is None else seed
    key = make_key([c["case_id"] for c in doc["cases"]], seed)
    text = json.dumps({"seed": seed, "items": key}, indent=1) + "\n"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / KEY_PATH.name).write_text(text, encoding="utf-8", newline="\n")
    (out_dir / PACK_PATH.name).write_text(build_pack(doc, runs, library, key), encoding="utf-8")
    write_sheets(key, out_dir)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ analysis

def read_sheet(path: Path, codes: list[str]) -> tuple[pd.DataFrame, list[str], list[str]]:
    """(sheet indexed by item, errors, missing-note warnings). Values are normalised to lower case."""
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    errors, warnings = [], []
    absent = [c for c in COLUMNS if c not in df.columns]
    if absent:
        return df, [f"{path.name}: missing columns {absent}"], warnings
    df["item"] = df["item"].str.strip()
    if sorted(df["item"]) != sorted(codes):
        errors.append(f"{path.name}: the item codes do not match the key")
    df = df.set_index("item")
    for q in (*YES_NO, *SCALES):
        df[q] = df[q].str.strip().str.lower()
    for code, r in df.iterrows():
        for q in YES_NO:
            if r[q] not in ("yes", "no"):
                errors.append(f"{path.name} {code}: {q} must be yes or no, got {r[q]!r}")
        if r["q4_actions"] not in ("1", "2", "3", "4", "5", "na"):
            errors.append(f"{path.name} {code}: q4_actions must be 1-5 or na, got {r['q4_actions']!r}")
        if r["q5_clear"] not in ("1", "2", "3", "4", "5", "na"):
            errors.append(f"{path.name} {code}: q5_clear must be 1-5, got {r['q5_clear']!r}")
        elif r["q5_clear"] == "na":  # deviation 2026-11-08: counted as missing, like na in q4
            warnings.append(f"{path.name} {code}: q5_clear is na (counted as missing)")
        if (r["q1_forecast"] == "no" or r["q2_sources"] == "no" or r["q3_harmful"] == "yes") and not r["notes"].strip():
            warnings.append(f"{path.name} {code}: a note is required")
    return df, errors, warnings


def cohen_kappa(a, b) -> float:
    """Agreement beyond chance for two raters' categorical answers (1 when chance agreement is already total)."""
    a, b = np.asarray(a), np.asarray(b)
    cats = np.unique(np.concatenate([a, b]))
    po = float((a == b).mean())
    pe = float(sum((a == c).mean() * (b == c).mean() for c in cats))
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def weighted_kappa(a, b, lo: int = 1, hi: int = 5) -> float:
    """Quadratic-weighted kappa for ordinal scores in lo..hi."""
    a, b = np.asarray(a, dtype=int), np.asarray(b, dtype=int)
    k = hi - lo + 1
    obs = np.zeros((k, k))
    for x, y in zip(a, b):
        obs[x - lo, y - lo] += 1
    obs /= obs.sum()
    exp = np.outer(obs.sum(axis=1), obs.sum(axis=0))
    w = (np.subtract.outer(np.arange(k), np.arange(k)) ** 2) / (k - 1) ** 2
    denom = (w * exp).sum()
    return 1.0 if denom == 0 else float(1 - (w * obs).sum() / denom)


def mcnemar_p(better: int, worse: int) -> float:
    """Exact two-sided McNemar p from the discordant pairs."""
    n = better + worse
    return 1.0 if n == 0 else float(stats.binomtest(better, n, 0.5).pvalue)


def combine(s1: pd.DataFrame, s2: pd.DataFrame) -> pd.DataFrame:
    """Item scores: q1, q2 true only if both raters say yes; q3 true if either says yes; q4, q5 the two raters' mean
    (NaN when either wrote na)."""
    out = pd.DataFrame(index=s1.index)
    out["q1_forecast"] = (s1["q1_forecast"] == "yes") & (s2.loc[s1.index, "q1_forecast"] == "yes")
    out["q2_sources"] = (s1["q2_sources"] == "yes") & (s2.loc[s1.index, "q2_sources"] == "yes")
    out["q3_harmful"] = (s1["q3_harmful"] == "yes") | (s2.loc[s1.index, "q3_harmful"] == "yes")
    for q in SCALES:
        out[q] = (pd.to_numeric(s1[q], errors="coerce") + pd.to_numeric(s2.loc[s1.index, q], errors="coerce")) / 2
    return out


def _boot_ci(values: np.ndarray, rng: np.random.Generator) -> list[float]:
    """95% bootstrap CI of the mean over cases (NaN cases dropped)."""
    v = values[~np.isnan(values)]
    if v.size == 0:
        return [float("nan"), float("nan")]
    means = v[rng.integers(0, v.size, size=(N_BOOT, v.size))].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def analyse(s1: pd.DataFrame, s2: pd.DataFrame, key: dict[str, dict]) -> dict:
    items = combine(s1, s2)
    meta = pd.DataFrame.from_dict(key, orient="index")
    joined = items.join(meta)
    wide = {q: joined.pivot(index="case_id", columns="version", values=q) for q in (*YES_NO, *SCALES)}
    rng = np.random.default_rng(BOOT_SEED)
    res: dict = {"n_items": int(len(items)), "n_cases": int(meta["case_id"].nunique()), "questions": {}, "agreement": {}}
    for q in (*YES_NO, *SCALES):
        w = wide[q].astype(float)
        llm, rules = w[LLM].to_numpy(), w[RULES].to_numpy()
        row = {"llm": float(np.nanmean(llm)), "rule_based": float(np.nanmean(rules)),
               "llm_ci": _boot_ci(llm, rng), "rule_based_ci": _boot_ci(rules, rng)}
        d = (llm - rules)[~np.isnan(llm - rules)]
        row["difference"] = float(d.mean()) if d.size else float("nan")
        row["difference_ci"] = _boot_ci(llm - rules, rng)
        if q in YES_NO:
            up, down = int((d > 0).sum()), int((d < 0).sum())
            row.update(llm_only=up, rule_based_only=down, p=mcnemar_p(up, down))
        else:
            row["p"] = 1.0 if not np.any(d) else float(stats.wilcoxon(d[d != 0]).pvalue)
        res["questions"][q] = row
    for q in YES_NO:
        res["agreement"][q] = {"kappa": cohen_kappa(s1[q], s2.loc[s1.index, q]),
                               "raw": float((s1[q] == s2.loc[s1.index, q]).mean())}
    for q in SCALES:
        both = (s1[q] != "na") & (s2.loc[s1.index, q] != "na")
        res["agreement"][q] = {"kappa": weighted_kappa(s1.loc[both, q].astype(int), s2.loc[s1.index, q][both].astype(int)),
                               "raw": float((s1.loc[both, q] == s2.loc[s1.index, q][both]).mean())}
    return res


NAMES = {"q1_forecast": "Faithful to the forecast (share of items)", "q2_sources": "Supported by the sources (share)",
         "q3_harmful": "Harmful or misleading (share; lower is better)", "q4_actions": "Actions suit the level (mean, 1-5)",
         "q5_clear": "Clear for the public (mean, 1-5)"}


def write_results(res: dict, runs: dict[str, dict], path: Path = RESULTS_PATH) -> None:
    outcomes = pd.Series([r["outcome"] for r in runs.values()]).value_counts().to_dict()
    L = ["# Blind rating of the heat advisories: results", "",
         "Pre-registered in `context/decisions.md` (2026-10-30). Descriptive and post-test.",
         f"{res['n_cases']} cases, {res['n_items']} items, two raters. An item counts as unfaithful or harmful if either "
         "rater says so; scores 4-5 are the two raters' mean. 95% CIs: bootstrap over cases.", "",
         f"LLM generation outcomes (all cases and practice items): {outcomes}.", "",
         "| Question | LLM (95% CI) | Rule-based (95% CI) | LLM minus rule-based (95% CI) | p |", "|---|---|---|---|---|"]
    for q, r in res["questions"].items():
        L.append(f"| {NAMES[q]} | {r['llm']:.2f} ({r['llm_ci'][0]:.2f} to {r['llm_ci'][1]:.2f}) | {r['rule_based']:.2f} "
                 f"({r['rule_based_ci'][0]:.2f} to {r['rule_based_ci'][1]:.2f}) | {r['difference']:+.2f} "
                 f"({r['difference_ci'][0]:+.2f} to {r['difference_ci'][1]:+.2f}) | {r['p']:.3f} |")
    L += ["", "Paired tests: McNemar (exact) for the yes/no questions, Wilcoxon signed-rank for the scores.", "",
          "## Agreement between the raters", "", "| Question | Kappa | Same answer |", "|---|---|---|"]
    for q, r in res["agreement"].items():
        L.append(f"| {NAMES[q]} | {r['kappa']:.2f} | {100 * r['raw']:.0f}% |")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the rating pack, or analyse the filled sheets.")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--build", action="store_true")
    group.add_argument("--analyse", action="store_true")
    args = ap.parse_args(argv)
    if args.build:
        print("Key SHA-256 (record it in context/decisions.md before sending the pack):", build())
        return 0
    key = json.loads(KEY_PATH.read_text(encoding="utf-8"))["items"]
    sheets, problems = [], []
    for rater in RATERS:
        s, errors, warnings = read_sheet(EVAL_DIR / f"ratings_{rater}.csv", list(key))
        sheets.append(s)
        problems += errors
        for w in warnings:
            print("Warning:", w)
    if problems:
        print("\n".join(problems[:30]))
        return 1
    res = analyse(sheets[0], sheets[1], key)
    write_results(res, load_runs())
    print(RESULTS_PATH.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
