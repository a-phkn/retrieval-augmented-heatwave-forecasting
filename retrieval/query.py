"""
Query-time retrieval: given a query window (by query_date), returns the
top-K eligible historical analogues, applying THREE independent filters
before deduplication. All three are necessary and none subsumes another --
see the eligibility rule docstring below for why.

Run from repo root for a manual sanity check against a known heatwave date:
    python -m retrieval.query
"""
from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RETRIEVAL_DIR = REPO_ROOT / "retrieval"

# How many nearest neighbors to pull from FAISS before eligibility
# filtering + dedup. Needs to be comfortably larger than K since most of
# the ~17k windows will be ruled ineligible for any given query (wrong
# split, too recent, same episode) -- 500 is generous headroom relative to
# the smallest eligible pool sizes seen in practice (see build_index.py's
# dataset size), cheap to search exhaustively at this data scale regardless.
SEARCH_POOL_SIZE = 500

_index = None
_candidates = None


def _load():
    global _index, _candidates
    if _index is None:
        _index = faiss.read_index(str(RETRIEVAL_DIR / "faiss_index.bin"))
        _candidates = pd.read_parquet(RETRIEVAL_DIR / "candidates.parquet")
    return _index, _candidates


def eligibility_mask(query_row: pd.Series, candidates: pd.DataFrame) -> np.ndarray:
    """
    Three independent eligibility rules, ALL required:

    1. NO FUTURE DATA (chronological leakage): candidate's own 14-day input
       + 5-day forecast must fully finish before the query's forecast
       period begins. Encoded as candidate.query_date <= query's own
       candidate_latest_start (precomputed by prepare_datasets.py,
       query_date - 19 days). This is the SAME rule tests/test_no_leakage.py
       enforces at the dataset level.

    2. SPLIT ELIGIBILITY (methodological, not just chronological): even
       though splits are chronologically contiguous, rule 1 ALONE is not
       enough -- e.g. a late-val query's rule-1 cutoff can fall on an
       EARLIER val date, which rule 1 alone would allow. That would let val
       queries retrieve from other val dates, contaminating what's supposed
       to be a held-out set with retrieval evidence about itself, and
       symmetrically for test. So:
         - train query  -> only 'train' candidates
         - val query    -> only 'train' candidates (never other val dates)
         - test query   -> 'train' or 'val' candidates (never other test dates)
       This must be checked independently of rule 1, not derived from it.

    3. NO SAME EPISODE: a candidate cannot share the query's own
       target_episode_id (when the query itself is part of a heatwave
       episode) -- otherwise the model could retrieve literally adjacent
       days from the SAME ongoing heatwave, which isn't a genuine
       "historical analogue," it's peeking at the answer via a neighboring
       day of the same event.

    Returns a boolean mask over `candidates`, aligned to its row order.
    """
    query_date = query_row["query_date"]
    query_split = query_row["split"]
    query_latest_start = query_row["candidate_latest_start"]
    query_episode = query_row["target_episode_id"]

    # Rule 1: chronological.
    mask = candidates["query_date"] <= query_latest_start

    # Rule 2: split eligibility.
    allowed_splits = {"train": {"train"}, "val": {"train"}, "test": {"train", "val"}}[query_split]
    mask &= candidates["split"].isin(allowed_splits)

    # Rule 3: no same episode (only applies if the query itself has one).
    if pd.notnull(query_episode):
        mask &= (candidates["target_episode_id"] != query_episode) | candidates["target_episode_id"].isnull()

    # A window is never its own candidate (redundant given rule 1 in almost
    # all cases, kept explicit for safety).
    mask &= candidates["query_date"] != query_date

    return mask.to_numpy()


def dedup_max_per_episode(
    ranked_indices: list[int],
    candidates: pd.DataFrame,
    max_per_episode: int = 2,
    min_days_apart: int = 10,
) -> list[int]:
    """Walks a similarity-ranked candidate list and applies TWO
    deduplication rules together:

    1. Max `max_per_episode` candidates from any single formally-qualifying
       episode (target_episode_id not null) -- the roadmap's literal spec.

    2. Minimum `min_days_apart` days between ANY two selected candidates'
       query_dates, regardless of episode membership.

    Rule 2 exists because rule 1 alone is insufficient: only 590 of 17,025
    windows (3.5%) have a non-null target_episode_id at all, so 96.5% of
    candidates are completely exempt from rule 1. In practice this let a
    real query's top-5 come back as 5 consecutive days from the SAME
    non-qualifying 2010 warm spell (shifted by 1-4 days each) -- since
    consecutive-day windows share 13 of 14 input days, they're near-
    duplicate vectors, so the "top 5" was really one analogue repeated five
    times, defeating the purpose of retrieving diverse historical evidence.
    `min_days_apart=10` is a deliberate, moderate choice (less than the
    19-day chronological leakage buffer, comfortably more than the 1-4 day
    shifts observed) -- not swept/tuned, a reasonable next step if pursued
    further.
    """
    episode_counts: dict[float, int] = {}
    selected_dates: list[pd.Timestamp] = []
    kept = []
    for idx in ranked_indices:
        row = candidates.iloc[idx]
        query_date = row["query_date"]

        if any(abs((query_date - d).days) < min_days_apart for d in selected_dates):
            continue

        episode = row["target_episode_id"]
        if pd.notnull(episode):
            count = episode_counts.get(episode, 0)
            if count >= max_per_episode:
                continue
            episode_counts[episode] = count + 1

        kept.append(idx)
        selected_dates.append(query_date)
    return kept


def query_analogues(query_date, k: int = 5, max_per_episode: int = 2) -> pd.DataFrame:
    """Returns a DataFrame of the top-k eligible, deduplicated analogues for
    the given query_date, sorted by similarity (descending), with a
    `similarity` column (cosine similarity, since vectors are L2-normalized
    and the index is IndexFlatIP)."""
    index, candidates = _load()

    query_date = pd.Timestamp(query_date)
    query_rows = candidates.index[candidates["query_date"] == query_date]
    if len(query_rows) == 0:
        raise ValueError(f"query_date={query_date} not found in retrieval/candidates.parquet")
    query_row = candidates.loc[query_rows[0]]

    feat_cols = [c for c in candidates.columns if c.startswith("feat_")]
    query_vec = candidates.loc[[query_rows[0]], feat_cols].to_numpy(dtype=np.float32)

    mask = eligibility_mask(query_row, candidates)
    n_eligible = int(mask.sum())
    if n_eligible == 0:
        return candidates.iloc[0:0]  # empty, same schema

    similarities, faiss_indices = index.search(query_vec, min(SEARCH_POOL_SIZE, len(candidates)))
    similarities, faiss_indices = similarities[0], faiss_indices[0]

    # Keep only eligible results, in similarity-descending order (FAISS
    # already returns results sorted by similarity).
    eligible_ranked = [(idx, sim) for idx, sim in zip(faiss_indices, similarities) if mask[idx]]

    if len(eligible_ranked) < k and n_eligible > len(eligible_ranked):
        # Search pool wasn't big enough to surface all eligible candidates
        # (rare at this dataset size, but don't silently return too few --
        # fall back to an exhaustive scan restricted to the eligible mask).
        eligible_idx = np.where(mask)[0]
        all_sims = (candidates.loc[eligible_idx, feat_cols].to_numpy(dtype=np.float32) @ query_vec.T).ravel()
        order = np.argsort(-all_sims)
        eligible_ranked = [(eligible_idx[i], all_sims[i]) for i in order]

    ranked_indices = [idx for idx, _ in eligible_ranked]
    sims_by_idx = {idx: sim for idx, sim in eligible_ranked}

    deduped = dedup_max_per_episode(ranked_indices, candidates, max_per_episode)[:k]

    result = candidates.iloc[deduped][["query_date", "split", "target_episode_id", "target_is_heatwave"]].copy()
    result["similarity"] = [sims_by_idx[i] for i in deduped]
    return result.reset_index(drop=True)


def _manual_sanity_check():
    """Prints top-5 analogues for a known 2022 Delhi heatwave query date,
    for the roadmap's Step 4 validation: 'manually inspect top-K analogues
    for a known heatwave query date and confirm they look plausible.'"""
    _, candidates = _load()
    heatwave_queries = candidates[candidates["target_is_heatwave"]].sort_values("query_date")
    if len(heatwave_queries) == 0:
        print("No heatwave-labeled query dates found -- check candidates.parquet.")
        return

    # Pick a query well into the test split's known 2022 episodes so results
    # are inspectable against real, well-documented Delhi heatwave dates.
    test_heatwave_queries = heatwave_queries[heatwave_queries["split"] == "test"]
    sample = test_heatwave_queries.iloc[len(test_heatwave_queries) // 2]
    query_date = sample["query_date"]

    print(f"Query date: {query_date.date()} (split={sample['split']}, "
          f"episode_id={sample['target_episode_id']})\n")
    result = query_analogues(query_date, k=5)
    if len(result) == 0:
        print("No eligible analogues found for this query.")
        return
    print(result.to_string(index=False))
    query_row = candidates[candidates["query_date"] == query_date].iloc[0]
    pool_size = int(eligibility_mask(query_row, candidates).sum())
    print(f"\nEligible candidate pool size (before dedup): {pool_size}")


if __name__ == "__main__":
    _manual_sanity_check()
