"""
Builds the retrieval system's two artifacts:
  retrieval/candidates.parquet  -- one row per window (ALL splits), with its
                                    metadata (query_date, split, episode id)
                                    needed for query-time eligibility
                                    filtering, plus its normalized feature
                                    vector (stored for inspection/debugging;
                                    the FAISS index is the actual search
                                    structure).
  retrieval/faiss_index.bin     -- FAISS IndexFlatIP over L2-normalized
                                    feature vectors (inner product ==
                                    cosine similarity). Exact search, no
                                    approximation -- dataset is small
                                    (~17k windows), no need for anything
                                    fancier.

Run from repo root:
    python -m retrieval.build_index
"""
from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from retrieval.features import (
    FEATURE_NAMES,
    apply_normalization,
    build_all_feature_vectors,
    fit_feature_normalization,
    l2_normalize,
    save_normalization_stats,
)
from training.data import _load_daily, _load_windows

REPO_ROOT = Path(__file__).resolve().parents[1]
RETRIEVAL_DIR = REPO_ROOT / "retrieval"


def main():
    print("Loading daily data and forecast windows...")
    daily = _load_daily()
    windows = _load_windows()

    print(f"Computing {len(FEATURE_NAMES)}-dim feature vectors for {len(windows)} windows...")
    X_raw = build_all_feature_vectors(daily, windows)

    print("Fitting normalization stats on TRAIN-split windows only...")
    stats = fit_feature_normalization(X_raw, windows)
    save_normalization_stats(stats)

    X_norm = apply_normalization(X_raw, stats)
    X_unit = l2_normalize(X_norm).astype(np.float32)

    print("Building FAISS IndexFlatIP...")
    index = faiss.IndexFlatIP(X_unit.shape[1])
    index.add(X_unit)
    faiss.write_index(index, str(RETRIEVAL_DIR / "faiss_index.bin"))

    print("Saving candidate metadata + feature vectors...")
    candidates = windows[["query_date", "split", "target_episode_id", "target_is_heatwave", "candidate_latest_start"]].copy()
    candidates["faiss_row"] = np.arange(len(candidates))  # row index into the FAISS index, stable join key
    for i, name in enumerate(FEATURE_NAMES):
        candidates[f"feat_{name}"] = X_unit[:, i]
    candidates.to_parquet(RETRIEVAL_DIR / "candidates.parquet", index=False)

    print(f"\nDone. {len(candidates)} candidates indexed.")
    print(f"  {RETRIEVAL_DIR / 'candidates.parquet'}")
    print(f"  {RETRIEVAL_DIR / 'faiss_index.bin'}")
    print(f"  {RETRIEVAL_DIR / 'feature_normalization_stats.json'}")


if __name__ == "__main__":
    main()
