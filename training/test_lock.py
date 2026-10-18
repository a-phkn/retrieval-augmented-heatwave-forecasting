"""
The test lock (docs/CONFIRMATORY_PROTOCOL.md section 5): the 2019+ test period may be read only after the
protocol has been frozen as the git tag `test-lock`, and only by the code frozen in that tag.

require() passes only if ALL hold:
  1. the tag `test-lock` exists;
  2. it is HEAD or an ancestor of HEAD (later commits, e.g. results, are allowed);
  3. no file under LOCKED_DIRS differs between the tag and HEAD (the code is the frozen code);
  4. no file under LOCKED_DIRS has uncommitted changes.
Otherwise it raises TestLocked and nothing from 2019 on is read. Every reader of 2019+ data calls it first
(training.folds for the "test" fold, training.graph_data, retrieval.fold_retrieval). Development folds never
call it and never read 2019+.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TAG = "test-lock"
LOCKED_DIRS = ("training", "models", "pipeline", "retrieval", "evaluation", "configs")
_unlocked = False  # cached only after a successful check


class TestLocked(RuntimeError):
    """Raised when code tries to read the test period before (or outside) the frozen protocol."""


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)


def require() -> None:
    """Raise TestLocked unless the frozen-protocol conditions above hold."""
    global _unlocked
    if _unlocked:
        return
    if _git("rev-parse", "--verify", "--quiet", f"refs/tags/{TAG}").returncode != 0:
        raise TestLocked(f"the test period (2019+) is locked: tag '{TAG}' does not exist (freeze the protocol first)")
    if _git("merge-base", "--is-ancestor", TAG, "HEAD").returncode != 0:
        raise TestLocked(f"HEAD does not contain the frozen tag '{TAG}'")
    if _git("diff", "--quiet", TAG, "HEAD", "--", *LOCKED_DIRS).returncode != 0:
        raise TestLocked(f"code under {LOCKED_DIRS} changed after '{TAG}'; the test must use the frozen code")
    status = _git("status", "--porcelain", "--", *LOCKED_DIRS)
    if status.returncode != 0 or status.stdout.strip():
        raise TestLocked(f"uncommitted changes under {LOCKED_DIRS}; the test must use the frozen code")
    _unlocked = True
