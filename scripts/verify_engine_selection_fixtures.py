#!/usr/bin/env python3
"""Verify the tracked content matrix used by the auto-engine study."""

from __future__ import annotations

import csv
import hashlib
import logging
from collections import defaultdict
from pathlib import Path

from PIL import Image

log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
_FIXTURE_ROOT = _ROOT / "data" / "evaluations" / "engine-selection"
_MANIFEST = _FIXTURE_ROOT / "content-manifest.csv"
_CARRIER_MANIFEST = _FIXTURE_ROOT / "carrier-manifest.csv"
_EXPECTED_PROVIDERS = {"meta", "openai"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    errors: list[str] = []
    pairs: dict[str, list[dict[str, str]]] = defaultdict(list)
    paths: set[Path] = set()
    hashes: set[str] = set()

    with _MANIFEST.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    with _CARRIER_MANIFEST.open(newline="", encoding="utf-8") as stream:
        carrier_rows = list(csv.DictReader(stream))

    for row in rows:
        pair_id = row["pair_id"]
        relative_path = Path(row["file"])
        path = _FIXTURE_ROOT / relative_path
        pairs[pair_id].append(row)

        if relative_path in paths:
            errors.append(f"duplicate path: {relative_path}")
        paths.add(relative_path)

        expected_hash = row["sha256"]
        if expected_hash in hashes:
            errors.append(f"duplicate file hash: {expected_hash}")
        hashes.add(expected_hash)

        if not path.is_file():
            errors.append(f"missing file: {relative_path}")
            continue
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            errors.append(f"hash mismatch for {relative_path}: expected {expected_hash}, got {actual_hash}")

        with Image.open(path) as image:
            actual_size = image.size
            actual_format = image.format
        expected_size = (int(row["width"]), int(row["height"]))
        if actual_size != expected_size:
            errors.append(f"size mismatch for {relative_path}: expected {expected_size}, got {actual_size}")
        if actual_format != "PNG":
            errors.append(f"format mismatch for {relative_path}: expected PNG, got {actual_format}")

    for pair_id, pair_rows in sorted(pairs.items()):
        providers = {row["provider"] for row in pair_rows}
        prompts = {row["prompt"] for row in pair_rows}
        strata = {row["content_stratum"] for row in pair_rows}
        if providers != _EXPECTED_PROVIDERS:
            errors.append(f"pair {pair_id} providers: expected {_EXPECTED_PROVIDERS}, got {providers}")
        if len(prompts) != 1:
            errors.append(f"pair {pair_id} has mismatched prompts")
        if len(strata) != 1:
            errors.append(f"pair {pair_id} has mismatched content strata")

    expected_pair_ids = {f"{index:02d}" for index in range(19)}
    if set(pairs) != expected_pair_ids:
        errors.append(f"pair ids: expected {sorted(expected_pair_ids)}, got {sorted(pairs)}")

    committed_files = {
        path.relative_to(_FIXTURE_ROOT) for path in (_FIXTURE_ROOT / "originals").glob("**/*") if path.is_file()
    }
    unlisted = committed_files - paths
    if unlisted:
        errors.append(f"unlisted files: {sorted(str(path) for path in unlisted)}")

    carrier_providers: set[str] = set()
    for row in carrier_rows:
        relative_path = Path(row["file"])
        path = (_FIXTURE_ROOT / relative_path).resolve()
        carrier_providers.add(row["provider"])
        if not path.is_relative_to(_ROOT / "data"):
            errors.append(f"carrier outside data/: {relative_path}")
            continue
        if not path.is_file():
            errors.append(f"missing carrier: {relative_path}")
            continue
        actual_hash = _sha256(path)
        if actual_hash != row["sha256"]:
            errors.append(f"carrier hash mismatch for {relative_path}: expected {row['sha256']}, got {actual_hash}")
    if carrier_providers != {"google", "meta", "openai"}:
        errors.append(f"carrier providers: expected google/meta/openai; got {sorted(carrier_providers)}")

    if errors:
        for error in errors:
            log.error("%s", error)
        return 1

    log.info(
        "Verified %s content files in %s matched pairs and %s canonical carriers",
        len(rows),
        len(pairs),
        len(carrier_rows),
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
