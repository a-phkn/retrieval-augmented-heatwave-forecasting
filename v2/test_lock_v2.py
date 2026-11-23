"""
Second test lock, for the v2 study (physics-constrained unsafe-hours forecasting; docs/PROTOCOL_V2.md).

The 2019+ test period may be read by v2 code only when ALL hold:
  1. the original lock passes (training.test_lock.require): the frozen v1 code is unchanged;
  2. the tag `test-lock-v2` exists and is HEAD or an ancestor of HEAD;
  3. no file under V2_LOCKED_DIRS differs between that tag and HEAD;
  4. no file under V2_LOCKED_DIRS has uncommitted changes.
Every v2 reader of 2019+ data calls require() first. Development folds never call it.
Order: commit protocol and keep rules -> development folds only -> tag test-lock-v2 -> score the test years once.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from training import test_lock as v1_lock

REPO_ROOT = Path(__file__).resolve().parents[1]
TAG = "test-lock-v2"
V2_LOCKED_DIRS = ("v2", "configs_v2")
_unlocked = False


class TestLockedV2(RuntimeError):
    """Raised when v2 code tries to read the test period before (or outside) the frozen v2 protocol."""


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)


def require() -> None:
    global _unlocked
    if _unlocked:
        return
    v1_lock.require()
    if _git("rev-parse", "--verify", "--quiet", f"refs/tags/{TAG}").returncode != 0:
        raise TestLockedV2(f"the test period (2019+) is locked for v2: tag '{TAG}' does not exist")
    if _git("merge-base", "--is-ancestor", TAG, "HEAD").returncode != 0:
        raise TestLockedV2(f"HEAD does not contain the frozen tag '{TAG}'")
    dirs = [d for d in V2_LOCKED_DIRS if (REPO_ROOT / d).exists()]
    if _git("diff", "--quiet", TAG, "HEAD", "--", *dirs).returncode != 0:
        raise TestLockedV2(f"code under {dirs} changed after '{TAG}'; the test must use the frozen v2 code")
    status = _git("status", "--porcelain", "--", *dirs)
    if status.returncode != 0 or status.stdout.strip():
        raise TestLockedV2(f"uncommitted changes under {dirs}; the test must use the frozen v2 code")
    _unlocked = True
