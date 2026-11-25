"""
P3-7 (context/decisions_phase3.md): station records rebuilt from NOAA's Integrated Surface Database (ISD).

This part downloads the yearly raw files and writes their manifest. The parsing, the census and the daily series
(P3-7 steps 1-4) are added to this module next.

- Source: https://www.ncei.noaa.gov/pub/data/noaa/<YEAR>/<USAF>-<WBAN>-<YEAR>.gz, ISD's raw archive. A made-up
  station number there returns 404; the "global-hourly access" path returns an HTML page for any number, so it is not
  used.
- Stations, 1980-2025:
  - Safdarjung 421820-99999 and Palam 421810-99999 (Delhi);
  - Ahmedabad 426470-99999 and Chennai 432790-99999 (added by the P3-7 deviation of 2026-11-25).
- Storage: files go to data/raw/noaa_isd/<USAF>-<WBAN>/ (gitignored, like every raw download).
- A reply is kept only if it is gzip data.
- The manifest, followup/data/isd_manifest.json (committed), gives each file's URL, size and SHA-256. It also lists
  the two GHCN files received from the experiments-v2 study.
- A file already on disk with its manifest hash is not fetched again, so an interrupted run resumes.

Run from repo root:  python -m followup.isd_rebuild --download
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw" / "noaa_isd"
MANIFEST = REPO_ROOT / "followup" / "data" / "isd_manifest.json"
BASE_URL = "https://www.ncei.noaa.gov/pub/data/noaa"
STATIONS = {"safdarjung": "421820-99999", "palam": "421810-99999",
            "ahmedabad": "426470-99999", "chennai": "432790-99999"}
YEARS = range(1980, 2026)
GZIP_MAGIC = b"\x1f\x8b"
# The GHCN-Daily files the experiments-v2 study used, received from it on 2026-11-25 (P3-7 part B), kept here.
GHCN_DIR = REPO_ROOT / "data" / "raw" / "ghcn_experiments_v2"
GHCN_RECEIVED = {
    "IN005010600": ("Ahmedabad", "0f09273e57ed7a74b9315309511eb6412605508e0e936115a60894498fbccdf8"),
    "IN020040900": ("Chennai, Minambakkam", "e2f6e651c8aca0332b99a6097a5b75e4a388b694b8cda994c6c133f878feb381"),
}


def url(station: str, year: int) -> str:
    """The raw-archive URL of one station-year."""
    return f"{BASE_URL}/{year}/{station}-{year}.gz"


def raw_path(station: str, year: int, root: Path = RAW_DIR) -> Path:
    """Where one station-year is stored."""
    return root / station / f"{station}-{year}.gz"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)  # atomic on the same filesystem


def fetch(session, u: str, retries: int = 4, sleep=time.sleep) -> bytes | None:
    """The file's bytes, or None if the archive has no such file (404). Timeouts, connection errors, 429 and server
    errors are retried; a 200 reply that is not gzip data (an HTML page) raises ValueError."""
    attempts = 0
    while True:
        try:
            r = session.get(u, timeout=120)
        except requests.RequestException:
            attempts += 1
            if attempts >= retries:
                raise
            sleep(10 * attempts)
            continue
        if r.status_code == 404:
            return None
        if r.status_code == 429 or r.status_code >= 500:
            attempts += 1
            if attempts >= retries:
                r.raise_for_status()
            sleep(10 * attempts)
            continue
        r.raise_for_status()
        if not r.content.startswith(GZIP_MAGIC):
            raise ValueError(f"{u}: the reply is not gzip data")
        return r.content


def _ghcn_entries(root: Path = GHCN_DIR) -> dict:
    """The received GHCN files with their expected hash, and whether the copy on disk matches it."""
    out = {}
    for sid, (name, expected) in GHCN_RECEIVED.items():
        p = root / f"{sid}.csv"
        out[sid] = {"name": name, "sha256": expected, "received": "2026-11-25, from the experiments-v2 study",
                    "on_disk_matches": p.exists() and sha256(p.read_bytes()) == expected}
    return out


def download(session, stations: dict[str, str] = STATIONS, years=YEARS, root: Path = RAW_DIR,
             manifest_path: Path = MANIFEST, ghcn_root: Path = GHCN_DIR, pause: float = 0.5, sleep=time.sleep) -> dict:
    """Fetch every station-year not on disk with its manifest hash, then write the manifest (also when stopped
    part-way, so a re-run resumes). A year the archive does not have is recorded as missing. Returns the manifest."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    files = manifest.setdefault("isd_files", {})
    try:
        for name, st in stations.items():
            for y in years:
                key, p = f"{st}-{y}", raw_path(st, y, root)
                if p.exists() and files.get(key, {}).get("sha256") == sha256(p.read_bytes()):
                    continue
                data = fetch(session, url(st, y), sleep=sleep)
                if data is None:
                    files[key] = {"station": name, "url": url(st, y), "missing": True}
                else:
                    _write_atomic(p, data)
                    files[key] = {"station": name, "url": url(st, y), "bytes": len(data), "sha256": sha256(data)}
                sleep(pause)
    finally:
        manifest["source"] = BASE_URL + "/<YEAR>/<USAF>-<WBAN>-<YEAR>.gz"
        manifest["ghcn_received"] = _ghcn_entries(ghcn_root)
        manifest["isd_files"] = dict(sorted(files.items()))
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="P3-7: station records rebuilt from NOAA ISD.")
    ap.add_argument("--download", action="store_true", help="fetch the yearly raw files and write the manifest")
    args = ap.parse_args(argv)
    if args.download:
        with requests.Session() as s:
            m = download(s)
        got = [v for v in m["isd_files"].values() if not v.get("missing")]
        print(f"{len(got)} files on disk ({sum(v['bytes'] for v in got) / 1e6:.1f} MB), "
              f"{sum(1 for v in m['isd_files'].values() if v.get('missing'))} missing in the archive; "
              f"manifest: {MANIFEST.relative_to(REPO_ROOT).as_posix()}", flush=True)


if __name__ == "__main__":
    main()
