"""
Precomputes the top-20 eligible, deduplicated analogues for EVERY window
(all 17,025), once, up front -- rather than calling retrieval.query's
query_analogues() one at a time during training (measured at ~60ms/query
via pandas-heavy eligibility filtering, which would mean ~17 minutes just
to precompute, repeated every time this needs re-running).

Retrieval is independent of any model's parameters (the retriever is
frozen/static per Execution_Pipeline.md Step 4 -- Step 5 only trains the
fusion architecture, not the retriever itself), so this only needs to run
ONCE per index build, not per training run.

Precomputes top-20 (not just top-K=5) so future K-sweep ablations (Step 7:
K in {1,3,5,10,20}) can just take the first K rows per query without
re-running retrieval at all.

Uses the exact same eligibility + dedup RULES as retrieval/query.py
(reimplemented here in pure numpy for speed -- logic must be kept in sync
if either changes; a correctness check against query.py's own output for a
sample of dates is run automatically at the end).

Output: retrieval/analogues_top20.parquet, long format:
    query_date, rank (1-20), analogue_query_date, similarity

Run from repo root:
    python -m retrieval.precompute_analogues
"""
from __future__ import annotations

import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RETRIEVAL_DIR = REPO_ROOT / "retrieval"

TOP_K_STORE = 20
SEARCH_POOL_SIZE = 500
MIN_DAYS_APART = 10
MAX_PER_EPISODE = 2


def eligibility_and_dedup_numpy(
    q_idx: int,
    pool_indices: np.ndarray,
    pool_sims: np.ndarray,
    query_dates_ns: np.ndarray,
    splits_code: np.ndarray,
    episode_ids: np.ndarray,
    candidate_latest_start_ns: np.ndarray,
    top_k: int,
) -> list[tuple[int, float]]:
    """Pure-numpy reimplementation of retrieval.query's eligibility_mask +
    dedup_max_per_episode, operating only on the SEARCH_POOL_SIZE candidates
    FAISS already returned for this query (not the full 17,025), which is
    what makes this fast."""
    q_date = query_dates_ns[q_idx]
    q_split = splits_code[q_idx]
    q_latest_start = candidate_latest_start_ns[q_idx]
    q_episode = episode_ids[q_idx]

    allowed = {0: {0}, 1: {0}, 2: {0, 1}}  # 0=train,1=val,2=test -- see split_code_map
    pool_dates = query_dates_ns[pool_indices]
    pool_splits = splits_code[pool_indices]
    pool_episodes = episode_ids[pool_indices]

    mask = pool_dates <= q_latest_start
    mask &= np.isin(pool_splits, list(allowed[q_split]))
    if not np.isnan(q_episode):
        mask &= (pool_episodes != q_episode) | np.isnan(pool_episodes)
    mask &= pool_indices != q_idx

    eligible_order = np.where(mask)[0]  # already similarity-sorted since pool is
    day_us = 86_400_000_000  # one day in microseconds -- candidates.parquet's
                              # datetime columns are datetime64[us], NOT [ns];
                              # using the wrong unit here silently inflated the
                              # min-days-apart threshold ~1000x and caused
                              # almost every candidate past the first to be
                              # wrongly rejected as "too close" -- caught by
                              # this script's own correctness check against
                              # retrieval.query's output, not by inspection.

    selected = []
    selected_dates = []
    episode_counts = {}
    for pos in eligible_order:
        idx = pool_indices[pos]
        date = pool_dates[pos]
        if any(abs(int(date - d)) < MIN_DAYS_APART * day_us for d in selected_dates):
            continue
        ep = pool_episodes[pos]
        if not np.isnan(ep):
            c = episode_counts.get(ep, 0)
            if c >= MAX_PER_EPISODE:
                continue
            episode_counts[ep] = c + 1
        selected.append((idx, float(pool_sims[pos])))
        selected_dates.append(date)
        if len(selected) >= top_k:
            break
    return selected


def main():
    print("Loading candidates + FAISS index...")
    candidates = pd.read_parquet(RETRIEVAL_DIR / "candidates.parquet")
    index = faiss.read_index(str(RETRIEVAL_DIR / "faiss_index.bin"))

    feat_cols = [c for c in candidates.columns if c.startswith("feat_")]
    all_vectors = candidates[feat_cols].to_numpy(dtype=np.float32)

    query_dates_ns = candidates["query_date"].astype("int64").to_numpy()
    candidate_latest_start_ns = candidates["candidate_latest_start"].astype("int64").to_numpy()
    split_code_map = {"train": 0, "val": 1, "test": 2}
    splits_code = candidates["split"].map(split_code_map).to_numpy()
    episode_ids = candidates["target_episode_id"].to_numpy(dtype=np.float64)  # NaN for null

    n = len(candidates)
    print(f"Batch FAISS search: {n} queries x top-{SEARCH_POOL_SIZE} each...")
    t0 = time.time()
    all_sims, all_idx = index.search(all_vectors, SEARCH_POOL_SIZE)
    print(f"  done in {time.time()-t0:.1f}s")

    print("Applying eligibility + dedup per query (numpy, no pandas in the loop)...")
    t0 = time.time()
    rows = []
    for q_idx in range(n):
        selected = eligibility_and_dedup_numpy(
            q_idx, all_idx[q_idx], all_sims[q_idx],
            query_dates_ns, splits_code, episode_ids, candidate_latest_start_ns,
            TOP_K_STORE,
        )
        q_date = candidates.iloc[q_idx]["query_date"]
        for rank, (analogue_idx, sim) in enumerate(selected, start=1):
            rows.append((q_date, rank, candidates.iloc[analogue_idx]["query_date"], sim))
    print(f"  done in {time.time()-t0:.1f}s, {len(rows)} (query, analogue) pairs")

    result = pd.DataFrame(rows, columns=["query_date", "rank", "analogue_query_date", "similarity"])
    result.to_parquet(RETRIEVAL_DIR / "analogues_top20.parquet", index=False)
    print(f"Saved to {RETRIEVAL_DIR / 'analogues_top20.parquet'}")

    # Correctness check: compare against retrieval.query's own (slower,
    # pandas-based) output for a small sample, must match exactly.
    print("\nVerifying against retrieval.query's output (sample of 10 dates)...")
    from retrieval.query import query_analogues
    sample_dates = candidates["query_date"].sample(10, random_state=0).tolist()
    all_match = True
    for qd in sample_dates:
        slow = query_analogues(qd, k=5)["query_date"].tolist()
        fast = result[(result["query_date"] == qd) & (result["rank"] <= 5)].sort_values("rank")["analogue_query_date"].tolist()
        match = slow == fast
        all_match &= match
        if not match:
            print(f"  MISMATCH for {qd}: slow={slow} fast={fast}")
    print("  All matched." if all_match else "  MISMATCHES FOUND -- do not trust precomputed results until fixed.")


if __name__ == "__main__":
    main()
