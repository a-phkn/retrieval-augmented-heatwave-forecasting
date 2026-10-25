"""
Builds / verifies data/TEST_MANIFEST.json: fingerprints of the one-time test run (2026-10-19, tag test-lock)
and of the data it read.

Why: the test period is used once. training/test_run.py skips any model whose test forecasts already exist, so
deleting or overwriting a file in predictions_test/ would silently allow a second "one-time" run, and the
test lock (training/test_lock.py) covers the code folders, not the data folders. tests/test_test_manifest.py
fails if any of these files changes or goes missing.

Covered (every git-tracked file under):
  outputs  predictions_test/, evaluation_test/, registry/test_runs.csv
  inputs   datasets_v2/, splits/, data/external/ (the daily data, window index and 500 hPa extracts)

Hashing is the same as scripts/make_manifest.py (text CRLF->LF; parquet also by content); --check fails on a
content change or a missing file and only reports a bytes-only change.

Run from repo root:
    python -m scripts.make_test_manifest           # (re)write data/TEST_MANIFEST.json (only after a deliberate change)
    python -m scripts.make_test_manifest --check   # verify, non-zero exit on a real change
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone

from scripts.make_manifest import REPO_ROOT, describe, git_provenance, hashing_environment, load_manifest, verify_manifest

TEST_MANIFEST_PATH = REPO_ROOT / "data" / "TEST_MANIFEST.json"
COVERED = ["predictions_test", "evaluation_test", "registry/test_runs.csv", "datasets_v2", "splits", "data/external"]


def covered_files() -> list[str]:
    """Every git-tracked file under COVERED (repo-relative, POSIX), sorted."""
    out = subprocess.run(["git", "ls-files", "--", *COVERED], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    files = sorted(line for line in out.stdout.splitlines() if line.strip())
    if not files:
        raise RuntimeError("no tracked files found under the covered folders")
    return files


def build_test_manifest() -> dict:
    return {
        "description": "Fingerprints of the one-time test run (outputs) and the data it read (inputs). "
        "Text files hashed with CRLF->LF normalisation; parquet files also carry a content hash.",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "test_lock_tag": "test-lock",
        "git": {k: v for k, v in git_provenance().items() if k != "freeze_tag"},
        "hashing_environment": hashing_environment(),
        "covered": COVERED,
        "files": [describe(p) for p in covered_files()],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or verify data/TEST_MANIFEST.json.")
    parser.add_argument("--check", action="store_true", help="verify instead of writing")
    args = parser.parse_args(argv)
    if args.check:
        problems, notes = verify_manifest(load_manifest(TEST_MANIFEST_PATH))
        for line in problems + notes:
            print(line)
        print("Test manifest OK." if not problems else f"{len(problems)} problem(s).")
        return 1 if problems else 0
    manifest = build_test_manifest()
    with open(TEST_MANIFEST_PATH, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"Wrote {TEST_MANIFEST_PATH.relative_to(REPO_ROOT)} ({len(manifest['files'])} files).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
