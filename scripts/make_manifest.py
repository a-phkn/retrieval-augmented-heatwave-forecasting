"""
Builds / verifies data/MANIFEST.json: fingerprints of every frozen v1 artefact
(processed data, datasets, the frozen window index, data spec, normalisation
stats, and the archived v1 results).

Why: plan v5 freezes v1 as the reference everything else is compared against.
If any of these files changes silently (a re-run of prepare_datasets.py, a
re-download, a retrain that rewrites normalisation stats), comparisons against
v1 are no longer valid. tests/test_manifest.py fails when that happens.

Two hashes per parquet file:
  sha256          -- exact bytes. Parquet embeds the writing library's version in
                     its metadata, so a re-write with identical data but a newer
                     pandas/pyarrow changes the bytes.
  content_sha256  -- the data itself (column names, normalised Arrow types,
                     values), independent of file metadata and of the
                     pandas/pyarrow version that wrote or reads it.
--check FAILS on a content change or a missing file, and only REPORTS a
bytes-only change (same data, different writer).

Line endings: the repo uses core.autocrlf, so text files are CRLF on Windows and
LF on Colab/Linux. Text files are hashed with CRLF normalised to LF.

Provenance: the frozen state is anchored by the git tag FREEZE_TAG ("v1-frozen"),
created on the commit that adds this manifest. The manifest records that commit's
parent and whether the working tree was dirty when it was generated.

Run from repo root:
    python -m scripts.make_manifest           # (re)write data/MANIFEST.json
    python -m scripts.make_manifest --check   # verify, non-zero exit on a real change
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Annotated git tag that anchors the frozen reference (a manifest cannot record
# the hash of the commit that contains it).
FREEZE_TAG = "v1-frozen"

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "data" / "MANIFEST.json"

# Files frozen as the v1 reference. Paths are repo-relative, POSIX style.
FROZEN_FILES = [
    # data spec + processed data + datasets (P1 pipeline output)
    "config/data_spec.yaml",
    "data/processed/weather_daily.parquet",
    "data/processing_log.json",
    "datasets/all_daily.parquet",
    "datasets/forecast_windows.parquet",
    "datasets/train.parquet",
    "datasets/val.parquet",
    "datasets/test.parquet",
    "events/event_catalogue.parquet",
    # the single source of windows for every model (read via training.data._load_windows)
    "splits/window_index_v1.parquet",
    # train-only normalisation stats (rewritten -- identically -- by training/retrieval runs)
    "evaluation/normalization_stats.json",
    "retrieval/feature_normalization_stats.json",
    # archived v1 results (copies of evaluation/ and predictions/ as of 2026-10-04)
    "archive_v1/evaluation/baseline_lstm/val_extended_metrics.json",
    "archive_v1/evaluation/baseline_lstm/val_metrics.json",
    "archive_v1/evaluation/figures/01_global_accuracy.png",
    "archive_v1/evaluation/figures/02_stratified_rmse.png",
    "archive_v1/evaluation/figures/03_detection.png",
    "archive_v1/evaluation/figures/04_bias_diagnostic.png",
    "archive_v1/evaluation/hw_sweep/bootstrap_ci_15_vs_20.json",
    "archive_v1/evaluation/hw_sweep/sweep_results.json",
    "archive_v1/evaluation/normalization_stats.json",
    "archive_v1/evaluation/retrieval_augmented/val_extended_metrics.json",
    "archive_v1/evaluation/retrieval_augmented/val_metrics.json",
    "archive_v1/evaluation/sanity_baselines.json",
    "archive_v1/predictions/retrieval_augmented/val_predictions.parquet",
    # run A1: canonical weighted-MSE LSTM (hot_weight=20), retrained 2026-10-04 with the
    # unchanged training/train_lstm.py; reproduces the archived evaluation to 1e-7
    # (docs/REPRODUCIBILITY_LOG.md)
    "models/frozen/lstm_tmax_v1/seed_0/checkpoint.pt",
    "models/frozen/lstm_tmax_v1/seed_1/checkpoint.pt",
    "models/frozen/lstm_tmax_v1/seed_2/checkpoint.pt",
    "models/frozen/lstm_tmax_v1/seed_3/checkpoint.pt",
    "models/frozen/lstm_tmax_v1/seed_4/checkpoint.pt",
]

# Raw inputs: gitignored (275 MB), so fingerprinted as a whole folder and only
# checked where present (skipped on Colab bundles / fresh clones).
RAW_DIR = "data/raw/era5_monthly"

# v1 artefacts that matter but are NOT fingerprinted, and why.
NOT_FROZEN = {
    "models/baseline_lstm/seed_*/checkpoint.pt": "gitignored working copy; the frozen A1 copy is "
    "models/frozen/lstm_tmax_v1/ (fingerprinted above).",
    "models/retrieval_augmented/seed_*/checkpoint.pt": "gitignored; original Colab checkpoints were not "
    "recoverable. RA-v1 is to be retrained with per-seed predictions + analogue provenance (plan v5, Week 1).",
    "retrieval/faiss_index.bin, retrieval/candidates.parquet": "gitignored; regenerate with "
    "`python -m retrieval.build_index` (deterministic from the frozen datasets).",
    "retrieval/analogues_top20.parquet": "gitignored; regenerate with `python -m retrieval.precompute_analogues`.",
    "data/raw/era5_monthly/": "gitignored (275 MB); folder fingerprint in raw_inputs, verified when present.",
}


def raw_folder_fingerprint(rel_dir: str = RAW_DIR) -> dict:
    """Aggregate SHA-256 over every file in the raw folder (sorted relative path +
    file hash), plus the file count. Raw API files are not in git, so bytes are
    hashed as-is."""
    root = REPO_ROOT / rel_dir
    files = sorted(p for p in root.rglob("*") if p.is_file())
    h = hashlib.sha256()
    for p in files:
        h.update(p.relative_to(root).as_posix().encode("utf-8"))
        h.update(hashlib.sha256(p.read_bytes()).hexdigest().encode("ascii"))
    return {"path": rel_dir, "n_files": len(files), "aggregate_sha256": h.hexdigest()}

TEXT_SUFFIXES = {".json", ".yaml", ".yml", ".csv", ".md", ".txt"}
DATE_COLUMNS = ("date", "query_date", "episode_start")


def sha256_of(path: Path) -> str:
    """SHA-256 of a file; text files are hashed with CRLF normalised to LF."""
    data = path.read_bytes()
    if path.suffix.lower() in TEXT_SUFFIXES:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def _normalised_column(col: pa.ChunkedArray) -> pa.ChunkedArray:
    """Remove storage choices that differ between writers but not in meaning:
    large_string vs string (pandas 3 vs 2), timestamp precision (us vs ns),
    dictionary encoding, and integer vs float for numeric columns (pandas turns
    an integer column with nulls into float64 on any read/write round trip)."""
    t = col.type
    if pa.types.is_dictionary(t):
        col, t = col.cast(t.value_type), t.value_type
    if pa.types.is_large_string(t):
        return col.cast(pa.string())
    if pa.types.is_integer(t) or (pa.types.is_floating(t) and t != pa.float64()):
        return col.cast(pa.float64())  # exact for integers below 2**53
    if pa.types.is_timestamp(t) and t.unit != "us":
        return col.cast(pa.timestamp("us", tz=t.tz), safe=True)  # raises if sub-us precision would be lost
    return col


def content_sha256_of_parquet(path: Path) -> str:
    """Hash of the table's data only -- column names, normalised Arrow types and
    values -- independent of parquet metadata and of the pandas/pyarrow version
    that wrote or reads the file. Every stored column is hashed, including a
    pandas index stored as a column (it is data). A default RangeIndex is stored
    only as metadata, so writing with index=True vs False does not matter."""
    table = pq.read_table(path)
    h = hashlib.sha256()
    for name in table.column_names:
        col = _normalised_column(table.column(name))
        h.update(json.dumps([name, str(col.type)]).encode("utf-8"))
        values = col.to_pylist()  # Python scalars: float repr is exact (shortest round-trip)
        h.update(json.dumps(values, default=lambda v: v.isoformat() if hasattr(v, "isoformat") else str(v)).encode("utf-8"))
    return h.hexdigest()


def describe(rel_path: str) -> dict:
    """Hash plus cheap descriptive metadata (rows/columns/date range for parquet)."""
    path = REPO_ROOT / rel_path
    entry: dict = {"path": rel_path, "sha256": sha256_of(path)}
    if path.suffix == ".parquet":
        entry["content_sha256"] = content_sha256_of_parquet(path)
        pf = pq.ParquetFile(path)
        entry["rows"] = pf.metadata.num_rows
        entry["columns"] = pf.schema_arrow.names
        for col in DATE_COLUMNS:
            if col in entry["columns"]:
                values = pq.read_table(path, columns=[col]).column(col).to_pandas()
                entry["date_column"] = col
                entry["date_min"] = str(values.min().date())
                entry["date_max"] = str(values.max().date())
                break
    return entry


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_provenance() -> dict:
    """The commit the manifest was generated on top of (its parent once committed),
    and whether TRACKED files had uncommitted changes at that moment (untracked
    folders such as work-in-progress sources/ are ignored)."""
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "freeze_tag": FREEZE_TAG,
        "parent_commit": _git("rev-parse", "HEAD"),
        "worktree_dirty": None if status is None else bool(status),
    }


def hashing_environment() -> dict:
    import numpy
    import pyarrow

    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "pandas": pd.__version__,
        "pyarrow": pyarrow.__version__,
    }


def build_manifest() -> dict:
    missing = [p for p in FROZEN_FILES if not (REPO_ROOT / p).exists()]
    if missing:
        raise FileNotFoundError(f"Frozen files missing: {missing}")
    return {
        "description": "Fingerprints of the frozen v1 reference (plan v5). Text files hashed "
        "with CRLF->LF normalisation; parquet files also carry a content hash.",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_provenance(),
        "hashing_environment": hashing_environment(),
        "files": [describe(p) for p in FROZEN_FILES],
        "raw_inputs": raw_folder_fingerprint() if (REPO_ROOT / RAW_DIR).exists() else None,
        "not_frozen": NOT_FROZEN,
    }


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def verify_manifest(manifest: dict) -> tuple[list[str], list[str]]:
    """Returns (problems, notes). Problems = missing file or changed content (fail).
    Notes = parquet bytes changed but content identical (writer metadata only)."""
    problems, notes = [], []
    for entry in manifest["files"]:
        path = REPO_ROOT / entry["path"]
        if not path.exists():
            problems.append(f"MISSING          {entry['path']}")
            continue
        if sha256_of(path) == entry["sha256"]:
            continue
        if "content_sha256" in entry and content_sha256_of_parquet(path) == entry["content_sha256"]:
            notes.append(f"BYTES CHANGED    {entry['path']} (content identical; re-written by a different library version)")
        else:
            problems.append(f"CONTENT CHANGED  {entry['path']}")

    raw = manifest.get("raw_inputs")
    if raw:
        if not (REPO_ROOT / raw["path"]).exists():
            notes.append(f"RAW NOT PRESENT  {raw['path']} (skipped; gitignored, expected on Colab/fresh clones)")
        elif raw_folder_fingerprint(raw["path"]) != raw:
            problems.append(f"RAW CHANGED      {raw['path']} (files added, removed or modified)")
    return problems, notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or verify data/MANIFEST.json.")
    parser.add_argument("--check", action="store_true", help="verify instead of writing")
    args = parser.parse_args(argv)

    if args.check:
        problems, notes = verify_manifest(load_manifest())
        for line in problems + notes:
            print(line)
        print("Manifest OK." if not problems else f"{len(problems)} problem(s).")
        return 1 if problems else 0

    manifest = build_manifest()
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"Wrote {MANIFEST_PATH.relative_to(REPO_ROOT)} ({len(manifest['files'])} files).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
