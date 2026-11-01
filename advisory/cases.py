"""
The evaluation cases for the advisory rating (Phase 6 part 4; decisions.md 2026-10-27 and 2026-10-30, written before
any LLM call).
Post-test and descriptive: built only from the committed 2019-2026 test forecasts; changes no forecasting result.

Per case (one query date q, the first forecast day):
  observed  the 7 days before q: ERA5 Tmax (the test actuals) and the 1980-2018 normal;
  forecast  the 5 days from q: the 10-seed ensemble of C3-hop's Tmax forecast, the normal, the hot-day probability
            (computed exactly as evaluation.test_evaluation does: recalibrated on 2017-2018, Gaussian spread), and
            the 10-seed ensemble of C3-hop's WBGT forecast;
  wbgt_threshold  the WBGT label threshold from 1980-2018.
Sampling: heat-season query dates (Mar 15 - Jul 31), stratified by the rule engine's headline colour (probability
rule): CASES_PER_COLOUR rated cases and PRACTICE_PER_COLOUR practice items per colour, no two within MIN_GAP_DAYS days.
Seeds are tried in order from SEED; the first that fills every colour is used and recorded (feasibility only).

Reading the test period's normals needs the test lock (valid while the frozen code is unchanged); the written
cases.json does not, so the rating can be redone from it.

Run from repo root:  python -m advisory.cases  ->  advisory/eval/cases.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from advisory import rules
from advisory.library import COLOURS
from pipeline.labels_v2 import in_season

REPO_ROOT = Path(__file__).resolve().parents[1]
PRED_TEST = REPO_ROOT / "predictions_test"
CASES_PATH = REPO_ROOT / "advisory" / "eval" / "cases.json"
TMAX_RUN, WBGT_RUN = "A1prime_hw5_C3hop", "A2Lr_hw5_C3hop"
CASES_PER_COLOUR = 13  # decision 2026-10-30: 15 per colour at a 5-day gap is infeasible on 2019-2026
PRACTICE_PER_COLOUR = 1  # 4 practice items, not rated (decision 2026-10-27)
MIN_GAP_DAYS = 3
SEED = 20261027
MAX_SEED_TRIES = 1000


def _load(run: str, suffix: str = "") -> pd.DataFrame:
    return pd.read_parquet(PRED_TEST / run / f"test{suffix}.parquet")


def forecast_table() -> pd.DataFrame:
    """One row per (query_date, lead): ensemble Tmax and WBGT forecasts, the hot-day probability, the actual Tmax."""
    from evaluation.calibration_check import apply_recal, ensemble, fit_recal, probability_frames

    tmax_val, tmax_stop = _load(TMAX_RUN), _load(TMAX_RUN, "_stop")
    coef = fit_recal(tmax_stop)
    pf = probability_frames(ensemble(apply_recal(tmax_val, coef)), ensemble(apply_recal(tmax_stop, coef)), "test", "Tmax")
    t = ensemble(tmax_val)[["query_date", "lead", "target_date", "pred", "actual"]].rename(
        columns={"pred": "tmax", "actual": "tmax_actual"})
    w = ensemble(_load(WBGT_RUN))[["query_date", "lead", "pred"]].rename(columns={"pred": "wbgt"})
    out = t.merge(w, on=["query_date", "lead"], validate="one_to_one")
    out = out.merge(pf[["query_date", "lead", "p"]].rename(columns={"p": "p_hot"}), on=["query_date", "lead"], how="left")
    out["p_hot"] = out["p_hot"].fillna(0.0)  # outside the label season a hot day is impossible (p = 0)
    return out.sort_values(["query_date", "lead"]).reset_index(drop=True)


def normals() -> pd.Series:
    """1980-2018 day-of-year normal Tmax per date (the test fold's climatology; needs the test lock)."""
    from training.folds import fold_daily

    d, _ = fold_daily("test", "t_max", "v2")
    return d["clim_mean_t_max"]


def case_frames(q: pd.Timestamp, fc: pd.DataFrame, actual: pd.Series, normal: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(observed, forecast) frames for rules.assess, for query date q."""
    obs_dates = pd.date_range(q - pd.Timedelta(days=rules.OBSERVED_DAYS), periods=rules.OBSERVED_DAYS, freq="D")
    observed = pd.DataFrame({"date": obs_dates, "tmax": actual.reindex(obs_dates).to_numpy(),
                             "normal": normal.reindex(obs_dates).to_numpy()})
    f = fc[fc["query_date"] == q].sort_values("lead")
    forecast = pd.DataFrame({"date": pd.DatetimeIndex(f["target_date"]), "tmax": f["tmax"].to_numpy(),
                             "normal": normal.reindex(pd.DatetimeIndex(f["target_date"])).to_numpy(),
                             "p_hot": f["p_hot"].to_numpy(), "wbgt": f["wbgt"].to_numpy()})
    return observed, forecast


def sample(headline: pd.Series, per_colour: int = CASES_PER_COLOUR, min_gap: int = MIN_GAP_DAYS,
           seed: int = SEED, avoid: list[pd.Timestamp] | None = None) -> list[pd.Timestamp]:
    """Stratified sample of query dates (index) by headline colour (values): per_colour each, in a random order,
    skipping any date within min_gap days of one already taken (or of a date in `avoid`). Rarer colours are drawn
    first. Raises if a colour cannot be filled."""
    rng = np.random.default_rng(seed)
    counts = headline.value_counts()
    taken: list[pd.Timestamp] = []
    blocked = list(avoid or [])
    for colour in sorted(COLOURS, key=lambda c: counts.get(c, 0)):
        pool = list(headline.index[headline == colour])
        got = 0
        for i in rng.permutation(len(pool)):
            q = pool[i]
            if all(abs((q - t).days) >= min_gap for t in taken + blocked):
                taken.append(q)
                got += 1
                if got == per_colour:
                    break
        if got < per_colour:
            raise ValueError(f"only {got} {colour} cases available with a {min_gap}-day gap (needed {per_colour})")
    return sorted(taken)


def draw(headline: pd.Series, first_seed: int = SEED,
         tries: int = MAX_SEED_TRIES) -> tuple[list[pd.Timestamp], list[pd.Timestamp], int]:
    """(rated dates, practice dates, seed used): seeds are tried in order from first_seed; the first one that fills
    every colour for both the rated cases and the practice items is used (decision 2026-10-30)."""
    for seed in range(first_seed, first_seed + tries):
        try:
            chosen = sample(headline, seed=seed)
            practice = sample(headline.drop(chosen), per_colour=PRACTICE_PER_COLOUR, seed=seed + 1, avoid=chosen)
            return chosen, practice, seed
        except ValueError:
            continue
    raise ValueError(f"no seed in {tries} tries fills {CASES_PER_COLOUR} cases per colour")


def _frame_records(df: pd.DataFrame) -> list[dict]:
    return [{k: (v.strftime("%Y-%m-%d") if isinstance(v, pd.Timestamp) else float(v)) for k, v in r.items()}
            for r in df.to_dict("records")]


def build_cases() -> dict:
    from evaluation.compare_v2 import _wbgt_thresholds

    fc = forecast_table()
    actual = fc.drop_duplicates("target_date").set_index("target_date")["tmax_actual"]
    normal = normals()
    thr = float(_wbgt_thresholds(("test",))["test"])
    first_obs = actual.index.min() + pd.Timedelta(days=rules.OBSERVED_DAYS)
    queries = [q for q in pd.DatetimeIndex(fc["query_date"].unique()) if q >= first_obs and in_season([q])[0]]
    assessed = {}
    for q in queries:
        obs, f = case_frames(q, fc, actual, normal)
        assessed[q] = (obs, f, rules.assess(obs, f, wbgt_threshold=thr))
    headline = pd.Series({q: a.headline_colour for q, (_, _, a) in assessed.items()})
    chosen, practice, seed = draw(headline)

    def record(case_id: str, q: pd.Timestamp) -> dict:
        obs, f, a = assessed[q]
        return {"case_id": case_id, "query_date": q.strftime("%Y-%m-%d"), "headline_colour": a.headline_colour,
                "observed": _frame_records(obs), "forecast": _frame_records(f)}

    return {"description": "Advisory evaluation cases (post-test, descriptive). Built by advisory/cases.py from "
                           "predictions_test/; see context/decisions.md 2026-10-27.",
            "seed_first_tried": SEED, "seed_used": seed, "cases_per_colour": CASES_PER_COLOUR,
            "practice_per_colour": PRACTICE_PER_COLOUR, "min_gap_days": MIN_GAP_DAYS,
            "forecast_rule": "probability", "p_hot_level": rules.P_HOT_LEVEL,
            "wbgt_threshold": thr, "candidates_by_colour": {c: int((headline == c).sum()) for c in COLOURS},
            "cases": [record(f"C{k:02d}", q) for k, q in enumerate(chosen, start=1)],
            "practice": [record(f"P{k}", q) for k, q in enumerate(practice, start=1)]}


def load_cases(path: Path = CASES_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def assessment_of(case: dict, wbgt_threshold: float) -> rules.Assessment:
    """Rebuild a case's rule-engine assessment from the JSON (no lock needed)."""
    obs, f = pd.DataFrame(case["observed"]), pd.DataFrame(case["forecast"])
    obs["date"], f["date"] = pd.to_datetime(obs["date"]), pd.to_datetime(f["date"])
    return rules.assess(obs, f, wbgt_threshold=wbgt_threshold)


def main() -> None:
    doc = build_cases()
    CASES_PATH.parent.mkdir(parents=True, exist_ok=True)
    CASES_PATH.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    by = pd.Series([c["headline_colour"] for c in doc["cases"]]).value_counts().to_dict()
    pr = [c["headline_colour"] for c in doc["practice"]]
    print(f"Wrote {CASES_PATH.relative_to(REPO_ROOT)}: {len(doc['cases'])} cases {by}, practice {pr}; "
          f"candidates {doc['candidates_by_colour']}; WBGT threshold {doc['wbgt_threshold']:.2f}")


if __name__ == "__main__":
    main()
