"""
Proposed amendment A7 (`docs/PROPOSED_A7_station_years.md`): the study's four station-reading analyses rerun with
the station Tmax files cut at the last correctly dated year.

The study's code runs unchanged. Only two things are redirected: the station-file paths it reads (to copies without
the rows after the cut year), and the folder it writes to.

The analyses: the H-A window test (`v2.run_window_test`), the graded analysis (`v2.run_graded`), the station checks
(`v2.run_station_checks`) and the alert-gap check (`v2_post.alert_gap`).

Run from repo root:
    python -m v2_fix.station_safe_years --check           (whole files through the same path; must reproduce
                                                           results_v2, to floating-point noise)
    python -m v2_fix.station_safe_years --last-year 2010  -> results_v2_fix/station_to_2010/ and its comparison.md
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import shutil
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = REPO_ROOT / "results_v2"
OUT_ROOT = REPO_ROOT / "results_v2_fix"
SAFDARJUNG_SRC = REPO_ROOT / "sources" / "data" / "safdarjung_daily_tidy.csv"
GHCN_SRC = REPO_ROOT / "sources" / "data" / "ghcn"
GHCN_IDS = ("IN005010600", "IN020040900")
OUTPUTS = ("h_a_delhi.json", "graded_delhi.json", "station_checks.json", "alert_gap.json")
# (module, attribute) the runners read at call time; tests check they still exist
PATCHED = (("v2.run_window_test", "STATION"), ("v2.run_window_test", "RESULTS"), ("v2.run_graded", "STATION"),
           ("v2.run_graded", "RESULTS"), ("v2.run_station_checks", "SAFDARJUNG"), ("v2.run_station_checks", "GHCN"),
           ("v2.run_station_checks", "RESULTS"), ("v2_post.alert_gap", "OUT"))


def year_of(line: str, date_column: int) -> int:
    """The year of a CSV row whose date sits in column date_column (quoted or not). Columns before the date never
    contain commas in these files."""
    return int(line.split(",")[date_column].strip().strip('"')[:4])


def cut_csv(src: Path, dst: Path, last_year: int, date_column: int) -> int:
    """Copy src to dst, keeping the header and the rows dated up to last_year, each kept line byte for byte. Returns
    the number of rows dropped."""
    with src.open("r", encoding="utf-8", newline="") as f:
        lines = f.readlines()
    keep = [lines[0]] + [ln for ln in lines[1:] if ln.strip() and year_of(ln, date_column) <= last_year]
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8", newline="") as f:
        f.writelines(keep)
    return len(lines) - len(keep)


def run_all(last_year: int | None, out_dir: Path) -> dict[str, int]:
    """Run the four analyses with the station files cut at last_year (None: copied whole) into out_dir. Returns the
    rows dropped per station file."""
    last = 9999 if last_year is None else last_year
    work = Path(tempfile.mkdtemp(prefix="v2fix_"))
    try:
        saf, ghcn = work / "safdarjung_daily_tidy.csv", work / "ghcn"
        dropped = {"safdarjung": cut_csv(SAFDARJUNG_SRC, saf, last, 0)}
        for sid in GHCN_IDS:
            dropped[sid] = cut_csv(GHCN_SRC / f"{sid}.csv", ghcn / f"{sid}.csv", last, 1)
        out_dir.mkdir(parents=True, exist_ok=True)
        values = {"STATION": saf, "SAFDARJUNG": saf, "GHCN": ghcn, "RESULTS": out_dir, "OUT": out_dir / "alert_gap.json"}
        modules = {name: importlib.import_module(name) for name, _ in PATCHED}
        saved = [(modules[m], a, getattr(modules[m], a)) for m, a in PATCHED]
        try:
            for m, a in PATCHED:
                setattr(modules[m], a, values[a])
            for name in ("v2.run_window_test", "v2.run_graded", "v2.run_station_checks", "v2_post.alert_gap"):
                if modules[name].main() != 0:
                    raise RuntimeError(f"{name} failed")
        finally:
            for module, a, value in saved:
                setattr(module, a, value)
        return dropped
    finally:
        shutil.rmtree(work, ignore_errors=True)


def flat(x, path: str = ""):
    """(path, value) pairs of a JSON value."""
    if isinstance(x, dict):
        for k, v in x.items():
            yield from flat(v, f"{path}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from flat(v, f"{path}[{i}]")
    else:
        yield path, x


def same_values(a, b, rel: float = 1e-9) -> tuple[bool, float, list[str]]:
    """Whether two JSON values match: the same keys and non-numbers, and numbers equal within rel (relative, with an
    absolute floor of 1e-12; NaN equals NaN). Returns (match, largest numeric difference, paths that differ)."""
    fa, fb = dict(flat(a)), dict(flat(b))
    bad = sorted(set(fa) ^ set(fb))
    worst = 0.0
    for k in set(fa) & set(fb):
        x, y = fa[k], fb[k]
        nums = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (x, y))
        if nums and x != x and y != y:
            continue
        if nums:
            worst = max(worst, abs(x - y))
            if not math.isclose(x, y, rel_tol=rel, abs_tol=1e-12):
                bad.append(k)
        elif x != y:
            bad.append(k)
    return not bad, worst, bad


def check() -> bool:
    """Whole files through the same path: every output must match the committed results_v2 file, byte for byte or
    to floating-point noise (same_values). Her own script, run unchanged on another machine, differs from her
    committed files in the 16th-17th significant digit of a few CI bounds (summation order), hence the tolerance."""
    out = Path(tempfile.mkdtemp(prefix="v2fix_check_"))
    ok = {}
    try:
        run_all(None, out)
        for name in OUTPUTS:
            if (out / name).read_bytes() == (ORIGINAL / name).read_bytes():
                ok[name] = True
                print(f"  {name}: identical", flush=True)
                continue
            match, worst, bad = same_values(json.loads((ORIGINAL / name).read_text(encoding="utf-8")),
                                            json.loads((out / name).read_text(encoding="utf-8")))
            ok[name] = match
            print(f"  {name}: " + (f"same values (largest difference {worst:.1e})" if match else
                                   f"DIFFERENT in {len(bad)} values, e.g. {bad[:3]}"), flush=True)
    finally:
        shutil.rmtree(out, ignore_errors=True)
    return all(ok.values())


# ------------------------------------------------------------------ comparison report


def get(d, *path, default=None):
    for p in path:
        if isinstance(d, dict) and p in d:
            d = d[p]
        elif isinstance(d, list) and isinstance(p, int) and p < len(d):
            d = d[p]
        else:
            return default
    return d


def no_ci(t) -> bool:
    lo, hi = (t.get("ci_low"), t.get("ci_high")) if isinstance(t, dict) else (None, None)
    return lo is None or hi is None or lo != lo or hi != hi


def ci(t) -> str:
    if not isinstance(t, dict) or t.get("estimate") is None:
        return "-"
    if no_ci(t):
        return f"{t['estimate']:+.3f} (no CI, {t.get('n_clusters')} clusters)"
    return f"{t['estimate']:+.3f} [{t['ci_low']:+.3f}, {t['ci_high']:+.3f}]"


def decision(row: dict) -> str:
    """'True'/'False' as registered, or 'no test' when the CI could not be computed (fewer than 5 clusters)."""
    t = row.get("primary_diff")
    return "no test" if isinstance(t, dict) and no_ci(t) else str(row.get("supported"))


def num(x, fmt: str = "{:.3f}") -> str:
    return "-" if x is None else fmt.format(x)


def comparison(orig: dict[str, dict], new: dict[str, dict], last_year: int, dropped: dict[str, int]) -> str:
    L = [f"# Station data cut at {last_year}: the registered numbers and the corrected ones", "",
         "Proposed amendment A7 (`docs/PROPOSED_A7_station_years.md`). The study's code ran unchanged, with station "
         f"Tmax files that stop at {last_year}-12-31 (rows dropped: " + ", ".join(f"{k} {v}" for k, v in dropped.items())
         + "). Registered = `results_v2/`, corrected = this folder. Full detail is in the JSON files here.", "",
         "How to read it:",
         "- \"No CI\" means fewer than 5 year clusters remain, so no decision is possible. It is not a rejection.",
         "- H-A and A1 test on the alert days of 2007-2018. With station data only to the cut year, the test keeps "
         f"only 2007-{last_year}.",
         "- The shares of unsafe or rest hours outside a window are computed over all the development years' alert "
         "days (for example 41 -> 33 clusters for light work), so they keep their CIs.", ""]
    w1 = "share_outside W1 12-4 pm (HAP p.155)"
    L += ["## H-A, Delhi window test (`h_a_delhi.json`)", "",
          "| Work / status | Alert days | Best window minus W1 | Supported | Unsafe hours outside 12-4 pm |", "|---|---|---|---|---|"]
    for key in (k for k in orig["h_a_delhi.json"] if "/" in k):
        o, n = orig["h_a_delhi.json"][key], new["h_a_delhi.json"].get(key, {})
        L.append(f"| {key} | {o.get('alert_days')} -> {n.get('alert_days')} | {ci(o.get('primary_diff'))} -> "
                 f"{ci(n.get('primary_diff'))} | {decision(o)} -> {decision(n)} | "
                 f"{num(get(o, w1, 'estimate'), '{:.2f}')} -> {num(get(n, w1, 'estimate'), '{:.2f}')} |")
    sens_o, sens_n = orig["h_a_delhi.json"].get("sensitivity_alert_population", {}), \
        new["h_a_delhi.json"].get("sensitivity_alert_population", {})
    if isinstance(sens_o, dict) and sens_o:
        L += ["", "Alert-population sensitivities:", "", "| Population | Alert days | Best window minus W1 |", "|---|---|---|"]
        for key, o in sens_o.items():
            n = sens_n.get(key, {}) if isinstance(sens_n, dict) else {}
            L.append(f"| {key} | {get(o, 'alert_days')} -> {get(n, 'alert_days')} | {ci(get(o, 'primary_diff'))} -> "
                     f"{ci(get(n, 'primary_diff'))} |")
    L += ["", "## A1, graded analysis (`graded_delhi.json`, first rows of h_a1)", "",
          "| Work / level / WBGT | Alert days | Best window minus W1 | Rest needed outside 12-4 pm |", "|---|---|---|---|"]
    for i, o in enumerate(get(orig["graded_delhi.json"], "h_a1", default=[])):
        n = get(new["graded_delhi.json"], "h_a1", i, default={})
        share = "share_rest_outside W1 12-4 pm (HAP p.155)"
        L.append(f"| {o.get('work')} / {o.get('level')} / {o.get('wbgt')} | {o.get('alert_days')} -> "
                 f"{n.get('alert_days')} | {ci(o.get('diff_best_minus_W1'))} -> {ci(n.get('diff_best_minus_W1'))} | "
                 f"{num(get(o, share, 'estimate'), '{:.2f}')} -> {num(get(n, share, 'estimate'), '{:.2f}')} |")
    L += ["", "## A5, station checks (`station_checks.json`)", "",
          "| City | Days | ERA5 minus station, in season (°C) |", "|---|---|---|"]
    for city in ("delhi", "ahmedabad", "chennai"):
        o, n = get(orig["station_checks.json"], city, "bias", default={}), get(new["station_checks.json"], city, "bias",
                                                                                default={})
        L.append(f"| {city} | {o.get('days')} -> {n.get('days')} | {num(o.get('mean_diff_era5_minus_station'), '{:+.2f}')}"
                 f" -> {num(n.get('mean_diff_era5_minus_station'), '{:+.2f}')} |")
    L += ["", "| City | Trigger | Station years | Seasons paid, station trigger | Same years, ERA5 trigger |",
          "|---|---|---|---|---|"]
    for city in ("delhi", "ahmedabad"):
        for i, o in enumerate(get(orig["station_checks.json"], city, "insurance_station", default=[])):
            n = get(new["station_checks.json"], city, "insurance_station", i, default={})
            L.append(f"| {city} | {o.get('trigger')} | {o.get('station_years')} -> {n.get('station_years')} | "
                     f"{num(get(o, 'station_trigger_9cell_rest', 'share_seasons_paid'), '{:.2f}')} -> "
                     f"{num(get(n, 'station_trigger_9cell_rest', 'share_seasons_paid'), '{:.2f}')} | "
                     f"{num(get(o, 'era5_per_cell_same_years', 'share_seasons_paid'), '{:.2f}')} -> "
                     f"{num(get(n, 'era5_per_cell_same_years', 'share_seasons_paid'), '{:.2f}')} |")
    L += ["", "## Section 18, alert gap (`alert_gap.json`)", "",
          "| City | Alert days | Any-unsafe-hour days alerted | Rest-heavy days alerted |", "|---|---|---|---|"]
    for city in ("delhi", "ahmedabad", "chennai"):
        o, n = get(orig["alert_gap.json"], city, default={}), get(new["alert_gap.json"], city, default={})
        L.append(f"| {city} | {o.get('alert_days')} -> {n.get('alert_days')} | "
                 f"{num(get(o, 'any_unsafe_hour', 'share_with_alert'))} -> {num(get(n, 'any_unsafe_hour', 'share_with_alert'))}"
                 f" | {num(get(o, 'rest_hours_ge_R', 'share_with_alert'))} -> "
                 f"{num(get(n, 'rest_hours_ge_R', 'share_with_alert'))} |")
    L += ["", "Ahmedabad's alert gap uses the ERA5 rule (no station series), so it does not change.", ""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Proposed A7: station data cut at the last correctly dated year.")
    ap.add_argument("--check", action="store_true", help="whole files; must reproduce results_v2 byte for byte")
    ap.add_argument("--last-year", type=int, default=None)
    args = ap.parse_args(argv)
    if args.check:
        ok = check()
        print("check passed: the driver reproduces the study's results" if ok else "CHECK FAILED", flush=True)
        if not ok:
            return 1
    if args.last_year is not None:
        out = OUT_ROOT / f"station_to_{args.last_year}"
        dropped = run_all(args.last_year, out)
        load = lambda root: {n: json.loads((root / n).read_text(encoding="utf-8")) for n in OUTPUTS}  # noqa: E731
        (out / "comparison.md").write_text(comparison(load(ORIGINAL), load(out), args.last_year, dropped) + "\n",
                                           encoding="utf-8")
        print(f"wrote {out.relative_to(REPO_ROOT).as_posix()}/ (4 JSON files and comparison.md)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
