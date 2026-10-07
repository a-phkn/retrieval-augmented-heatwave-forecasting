"""The analogue lists used by the week-3 retrieval queue must never change.

tests/data/retrieval_fingerprints.json holds SHA-256 fingerprints of every (family, fold,
mode, seed, split) analogue list, recorded on 2026-10-06 before retrieval/fold_retrieval.py
was extended (independent review). This test re-checks a representative subset (all folds
for sim and time; rand seeds 0 and 9 on f1 and f4). The full set (192 lists, ~4 min) is
checked by calling check(all_entries=True)."""
import hashlib
import json

import pytest

from retrieval.fold_retrieval import FoldRetriever
from scripts.make_manifest import REPO_ROOT

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")
FINGERPRINTS = REPO_ROOT / "tests" / "data" / "retrieval_fingerprints.json"


def check(all_entries: bool = False) -> list[str]:
    """Keys whose recomputed fingerprint differs (empty list = unchanged)."""
    entries = json.loads(FINGERPRINTS.read_text(encoding="utf-8"))["entries"]
    retrievers, bad = {}, []
    for key, expected in entries.items():
        target, labels, fold, mode, seed, split = key.split("|")
        seed = None if seed == "None" else int(seed)
        if not all_entries and mode == "rand" and not (fold in ("f1", "f4") and seed in (0, 9)):
            continue
        fr = retrievers.setdefault((target, labels, fold), FoldRetriever(fold, labels, target))
        w = fr.train_w if split == "train" else fr.val_w
        got = hashlib.sha256(fr.retrieve(w["query_date"], mode, 5, seed=seed).idx.tobytes()).hexdigest()
        if got != expected:
            bad.append(key)
    return bad


def test_queue_analogue_lists_are_unchanged():
    assert check() == []
