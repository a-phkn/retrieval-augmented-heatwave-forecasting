"""
Guards the frozen v1 reference (plan v5, Phase 0).

If one of these fails, a frozen file changed. Do NOT just regenerate the
manifest: find out what rewrote the file first. Regenerate only for a deliberate,
documented change to the frozen set.

The module is skipped when data/MANIFEST.json is absent (e.g. a Colab bundle
that only ships training code and datasets).
"""
import shutil

import pandas as pd
import pytest

from scripts.make_manifest import (
    FROZEN_FILES,
    MANIFEST_PATH,
    REPO_ROOT,
    content_sha256_of_parquet,
    load_manifest,
    sha256_of,
    verify_manifest,
)

pytestmark = pytest.mark.skipif(not MANIFEST_PATH.exists(), reason="data/MANIFEST.json not present")


def test_manifest_matches_files():
    problems, _ = verify_manifest(load_manifest())
    assert not problems, "Frozen v1 files changed:\n" + "\n".join(problems)


def test_manifest_covers_frozen_list():
    listed = {e["path"] for e in load_manifest()["files"]}
    assert listed == set(FROZEN_FILES)


def test_window_index_is_exact_copy_of_forecast_windows():
    assert sha256_of(REPO_ROOT / "splits/window_index_v1.parquet") == sha256_of(
        REPO_ROOT / "datasets/forecast_windows.parquet"
    )


def test_training_code_reads_the_frozen_window_index(monkeypatch, tmp_path):
    import training.data as td
    import training.retrieval_data as trd

    assert td.WINDOW_INDEX_PATH == REPO_ROOT / "splits" / "window_index_v1.parquet"
    assert len(td._load_windows()) == 17025

    # Point the index at a sentinel: every window reader must go through it.
    sentinel = tmp_path / "sentinel_window_index.parquet"
    monkeypatch.setattr(td, "WINDOW_INDEX_PATH", sentinel)
    for loader in (td._load_windows, lambda: trd.build_split_arrays_with_analogues("val")):
        with pytest.raises(FileNotFoundError, match="sentinel_window_index"):
            loader()


def test_no_direct_reads_of_forecast_windows_in_code():
    offenders = []
    for folder in ("training", "retrieval", "models", "evaluation"):
        for py in (REPO_ROOT / folder).glob("*.py"):
            for i, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
                if "read_parquet" in line and "forecast_windows" in line:
                    offenders.append(f"{py.relative_to(REPO_ROOT)}:{i}")
    assert not offenders, f"Read windows via training.data._load_windows(): {offenders}"


def test_window_index_split_sizes():
    windows = pd.read_parquet(REPO_ROOT / "splits/window_index_v1.parquet")
    assert len(windows) == 17025
    assert windows["split"].value_counts().to_dict() == {"train": 13131, "val": 1092, "test": 2802}
    assert windows["query_date"].is_unique


def test_crlf_and_lf_text_hash_identically(tmp_path):
    lf, crlf = tmp_path / "a.json", tmp_path / "b.json"
    lf.write_bytes(b'{\n  "x": 1\n}\n')
    crlf.write_bytes(b'{\r\n  "x": 1\r\n}\r\n')
    assert sha256_of(lf) == sha256_of(crlf)


def test_parquet_content_hash_ignores_writer_metadata(tmp_path):
    src = REPO_ROOT / "events/event_catalogue.parquet"
    copy = tmp_path / "copy.parquet"
    df = pd.read_parquet(src)
    df.to_parquet(copy, index=False)  # same data, re-written by this environment
    assert content_sha256_of_parquet(copy) == content_sha256_of_parquet(src)
    df.iloc[0, df.columns.get_loc("duration_days")] += 1  # a real data change
    df.to_parquet(copy, index=False)
    assert content_sha256_of_parquet(copy) != content_sha256_of_parquet(src)


def test_parquet_content_hash_is_pandas_version_independent(tmp_path):
    """pandas 2 (Colab) writes object strings and datetime64[ns]; pandas 3 writes
    str and datetime64[us]. Same data must give the same content hash."""
    src = REPO_ROOT / "splits/window_index_v1.parquet"
    df = pd.read_parquet(src)
    legacy = df.copy()
    for col in legacy.columns:
        if pd.api.types.is_datetime64_any_dtype(legacy[col]):
            legacy[col] = legacy[col].astype("datetime64[ns]")
        elif pd.api.types.is_string_dtype(legacy[col]):
            legacy[col] = legacy[col].astype(object)
    out = tmp_path / "legacy.parquet"
    legacy.to_parquet(out, index=False)
    assert content_sha256_of_parquet(out) == content_sha256_of_parquet(src)


def test_content_hash_covers_data_stored_as_index(tmp_path):
    """A frozen file re-written with set_index(...) must still have its index data hashed."""
    df = pd.read_parquet(REPO_ROOT / "splits/window_index_v1.parquet")
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    df.set_index("query_date").to_parquet(a)
    changed = df.copy()
    changed.loc[0, "query_date"] = pd.Timestamp("1900-01-01")
    changed.set_index("query_date").to_parquet(b)
    assert content_sha256_of_parquet(a) != content_sha256_of_parquet(b)


def test_verify_distinguishes_bytes_only_from_content_change(tmp_path, monkeypatch):
    import scripts.make_manifest as mm

    rel = "events/event_catalogue.parquet"
    fake_root = tmp_path
    (fake_root / "events").mkdir()
    shutil.copy(REPO_ROOT / rel, fake_root / rel)
    manifest = {"files": [mm.describe(rel)]}
    monkeypatch.setattr(mm, "REPO_ROOT", fake_root)

    df = pd.read_parquet(fake_root / rel)
    df.to_parquet(fake_root / rel, index=False, compression="gzip")  # bytes differ, data same
    problems, notes = mm.verify_manifest(manifest)
    assert not problems and len(notes) == 1

    df.iloc[0, df.columns.get_loc("duration_days")] += 1
    df.to_parquet(fake_root / rel, index=False)
    problems, _ = mm.verify_manifest(manifest)
    assert problems and problems[0].startswith("CONTENT CHANGED")
