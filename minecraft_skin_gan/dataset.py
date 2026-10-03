"""Deterministic skin datasets and explicitly reversible, byte-checked curation.

Preparation preserves decoded RGBA bytes and positional NPZ keys. Its grouped
split is a new policy; the untouched legacy script retains historical splitting.
"""

from __future__ import annotations

import hashlib
import io
import itertools
import json
import math
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any

import imagehash
import numpy as np
from PIL import Image, UnidentifiedImageError

from sort_skins import should_filter_skin

type Report = dict[str, Any]
MAX_IMAGE_BYTES = 16 * 1024 * 1024


def _json(path: Path, value: Report) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def _read_image(path: Path) -> tuple[bytes, Image.Image, str]:
    if path.is_symlink():
        raise ValueError("symlink images are not supported")
    if path.stat().st_size > MAX_IMAGE_BYTES:
        raise ValueError(f"image exceeds {MAX_IMAGE_BYTES} byte limit")
    data = path.read_bytes()
    with Image.open(io.BytesIO(data)) as image:
        if image.format != "PNG":
            raise ValueError("expected PNG content")
        if image.size != (64, 64):
            raise ValueError("expected 64x64 skin")
        image.load()
        mode = image.mode
        rgba = image.convert("RGBA")
    return data, rgba, mode


def _inventory(source: Path) -> list[Report]:
    if not source.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {source}")
    records = []
    paths = sorted(
        (path for path in source.rglob("*") if path.suffix.lower() == ".png"),
        key=lambda path: path.relative_to(source).as_posix(),
    )
    for path in paths:
        record: Report = {"path": path.relative_to(source).as_posix()}
        try:
            data, rgba, mode = _read_image(path)
            record.update(
                byte_sha256=hashlib.sha256(data).hexdigest(),
                rgba_sha256=hashlib.sha256(rgba.tobytes()).hexdigest(),
                shape=[64, 64, 4],
                mode=mode,
                perceptual_hash=str(imagehash.phash(rgba)),
            )
        except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as error:
            record["error"] = f"{type(error).__name__}: {error}"
            if path.is_file() and not path.is_symlink():
                record["byte_sha256"] = _sha256(path)
        records.append(record)
    return records


class _HashTree:
    """BK-tree over distinct 64-bit hashes, avoiding a corpus-wide all-pairs scan."""

    def __init__(self, value: int):
        self.value = value
        self.children: dict[int, _HashTree] = {}

    def insert(self, value: int) -> None:
        node = self
        while (distance := (node.value ^ value).bit_count()) in node.children:
            node = node.children[distance]
        if distance:
            node.children[distance] = _HashTree(value)

    def search(self, value: int, radius: int) -> list[tuple[int, int]]:
        result = []
        pending: list[_HashTree] = [self]
        while pending:
            node = pending.pop()
            distance = (node.value ^ value).bit_count()
            if distance <= radius:
                result.append((node.value, distance))
            pending.extend(
                child
                for edge, child in node.children.items()
                if distance - radius <= edge <= distance + radius
            )
        return sorted(result)


def _near_duplicates(records: list[Report], distance: int, limit: int) -> Report:
    # One representative per exact group: near matches never change the split.
    groups = {record["rgba_sha256"]: record for record in reversed(records)}
    buckets: dict[int, dict[str, list[str]]] = {}
    for record in groups.values():
        key = int(record["perceptual_hash"], 16)
        bucket = buckets.setdefault(key, {"train": [], "validation": []})
        bucket[record["split"]].append(record["path"])
    keys = sorted(buckets)
    tree = _HashTree(keys[0])
    for key in keys[1:]:
        tree.insert(key)
    candidates: list[Report] = []
    count = 0
    for key in keys:
        neighbors = [(key, 0)] if distance == 0 else tree.search(key, distance)
        for other, actual_distance in neighbors:
            if other < key:
                continue
            pairs = [(buckets[key]["train"], buckets[other]["validation"])]
            if other != key:
                pairs.append((buckets[other]["train"], buckets[key]["validation"]))
            for training, validation in pairs:
                count += len(training) * len(validation)
                for train_path, validation_path in itertools.islice(
                    itertools.product(training, validation), max(0, limit - len(candidates))
                ):
                    candidates.append(
                        {
                            "train": train_path,
                            "validation": validation_path,
                            "distance": actual_distance,
                        }
                    )
    return {
        "algorithm": "ImageHash.phash/64-bit/RGB-derived",
        "distance": distance,
        "candidate_limit": limit,
        "candidate_count": count,
        "truncated": count > len(candidates),
        "candidates": candidates,
        "policy": "report-only; alpha is ignored by pHash; exact grouping includes alpha",
    }


def prepare_dataset(
    source_directory: Path | str,
    output_directory: Path | str,
    *,
    seed: int = 1976,
    validation_fraction: float = 0.2,
    provenance: Report | None = None,
    near_duplicate_distance: int = 0,
    candidate_limit: int = 1000,
) -> Report:
    """Create an atomic v1 manifest and NPZ with a seeded exact-group split.

    The validation fraction targets group counts, not file counts; duplicated
    groups can make the achieved file fraction differ. Invalid files are logged
    and excluded. Output must not exist. Array writing uses disk-backed buffers.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if (
        isinstance(near_duplicate_distance, bool)
        or not isinstance(near_duplicate_distance, int)
        or not 0 <= near_duplicate_distance <= 64
        or isinstance(candidate_limit, bool)
        or not isinstance(candidate_limit, int)
        or candidate_limit < 1
    ):
        raise ValueError("near_duplicate_distance must be 0..64 and candidate_limit positive")
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(output)
    source = Path(source_directory).resolve()
    records = _inventory(source)
    valid = [record for record in records if "error" not in record]
    group_keys = sorted({record["rgba_sha256"] for record in valid})
    if len(group_keys) < 2:
        raise ValueError("at least two valid exact-content groups are required")
    random.Random(seed).shuffle(group_keys)
    validation_count = min(
        len(group_keys) - 1, max(1, math.ceil(len(group_keys) * validation_fraction))
    )
    validation_groups = set(group_keys[:validation_count])
    for record in valid:
        record["split"] = "validation" if record["rgba_sha256"] in validation_groups else "train"
    manifest: Report = {
        "schema_version": 1,
        "policy": "sorted-relative-path/exact-rgba-groups/python-random-v1",
        "seed": seed,
        "validation_fraction": validation_fraction,
        "provenance": provenance or {},
        "files": records,
        "counts": {
            "valid": len(valid),
            "invalid": len(records) - len(valid),
            "exact_groups": len(group_keys),
            "train": sum(record["split"] == "train" for record in valid),
            "validation": sum(record["split"] == "validation" for record in valid),
        },
        "near_duplicates": _near_duplicates(valid, near_duplicate_distance, candidate_limit),
    }
    manifest["dataset_sha256"] = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest["fingerprint_contract"] = {
        "dataset_sha256": "semantic file records and split membership",
        "archive_sha256": "exact train_test.npz bytes",
    }
    manifest["exposure"] = "development; no unexposed release-test claim"
    # Validate metadata serialization before creating any outputs.
    json.dumps(manifest, allow_nan=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=output.parent) as directory:
        staging = Path(directory) / "dataset"
        staging.mkdir()
        arrays = []
        for split in ("train", "validation"):
            rows = [record for record in valid if record["split"] == split]
            array = np.lib.format.open_memmap(
                staging / f"{split}.npy", mode="w+", dtype=np.uint8, shape=(len(rows), 64, 64, 4)
            )
            for index, record in enumerate(rows):
                data, rgba, _ = _read_image(source / record["path"])
                if hashlib.sha256(data).hexdigest() != record["byte_sha256"]:
                    raise ValueError(f"source changed during preparation: {record['path']}")
                array[index] = np.asarray(rgba)
            array.flush()
            arrays.append(array)
        np.savez(staging / "train_test.npz", *arrays)
        del array, arrays
        manifest["archive_sha256"] = _sha256(staging / "train_test.npz")
        for split in ("train", "validation"):
            (staging / f"{split}.npy").unlink()
        _json(staging / "manifest.json", manifest)
        output.mkdir()
        try:
            staging.replace(output)
        except BaseException:
            output.rmdir()
            raise
    return manifest


def plan_curation(source_directory: Path | str, *, policy: str = "exact") -> Report:
    """Inspect without mutation and propose exact duplicates or legacy head filtering."""
    if policy not in {"exact", "head-filter"}:
        raise ValueError("policy must be exact or head-filter")
    source = Path(source_directory).resolve()
    records = _inventory(source)
    keepers: dict[str, str] = {}
    moves = []
    for record in records:
        if "error" in record:
            continue
        reason = None
        keeper = None
        if policy == "exact":
            keeper = keepers.get(record["rgba_sha256"])
            if keeper is not None:
                reason = "exact-rgba-duplicate"
            else:
                keepers[record["rgba_sha256"]] = record["path"]
        else:
            _, rgba, _ = _read_image(source / record["path"])
            if should_filter_skin(np.asarray(rgba)):
                reason = "head-filter"
        if reason:
            moves.append({**record, "reason": reason, "keeper": keeper})
    return {
        "schema_version": 1,
        "source_directory": str(source),
        "policy": policy,
        "moves": moves,
        "errors": [record for record in records if "error" in record],
    }


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _safe_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("curation paths must be nonempty relative paths without traversal")
    result = root / path
    if result.resolve() != result.absolute():
        raise ValueError("curation relative paths must not traverse symlinks")
    return result


def _move_file(source: Path, destination: Path, expected_sha256: str | None = None) -> None:
    """Move across filesystems without replacing any existing destination."""
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents shutil.move's overwrite behavior. A copy is
    # verified before removing the source, making cross-filesystem moves safe.
    created = False
    original_hash = _sha256(source)
    if expected_sha256 is not None and original_hash != expected_sha256:
        raise ValueError("source changed since curation inspection")
    try:
        with destination.open("xb") as writer:
            created = True
            with source.open("rb") as reader:
                shutil.copyfileobj(reader, writer)
        if original_hash != _sha256(source) or original_hash != _sha256(destination):
            raise ValueError("source changed while moving")
        source.unlink()
    except BaseException:
        if created and destination.is_file():
            destination.unlink()
        raise


def _validate_plan(plan: Report) -> list[Report]:
    if (
        not isinstance(plan, dict)
        or plan.get("schema_version") != 1
        or not isinstance(plan.get("moves"), list)
    ):
        raise ValueError("unsupported curation manifest schema")
    moves = plan["moves"]
    seen = set()
    for move in moves:
        if not isinstance(move, dict) or not isinstance(move.get("path"), str):
            raise ValueError("invalid curation move")
        if not isinstance(move.get("byte_sha256"), str) or len(move["byte_sha256"]) != 64:
            raise ValueError("invalid curation byte hash")
        if move["path"] in seen:
            raise ValueError("duplicate curation move path")
        seen.add(move["path"])
    if (
        not isinstance(plan.get("source_directory"), str)
        or not Path(plan["source_directory"]).is_absolute()
    ):
        raise ValueError("curation source_directory must be an absolute path")
    return moves


def apply_curation(plan: Report, quarantine_directory: Path | str) -> Path:
    """Apply a inspected plan to a new quarantine; roll back partial failures."""
    moves = _validate_plan(plan)
    source = Path(plan["source_directory"]).resolve()
    quarantine = Path(quarantine_directory).absolute()
    if quarantine.exists() or quarantine.is_symlink():
        raise FileExistsError(quarantine)
    if quarantine == source or source in quarantine.parents:
        raise ValueError("quarantine must be outside the source dataset")
    pairs = []
    for move in moves:
        original = _safe_path(source, move["path"])
        if _sha256(original) != move["byte_sha256"]:
            raise ValueError(f"source changed since inspection: {move['path']}")
        if move.get("keeper"):
            keeper = _safe_path(source, move["keeper"])
            _, rgba, _ = _read_image(keeper)
            if hashlib.sha256(rgba.tobytes()).hexdigest() != move["rgba_sha256"]:
                raise ValueError(f"keeper changed since inspection: {move['keeper']}")
        pairs.append(
            (original, _safe_path(quarantine / "files", move["path"]), move["byte_sha256"])
        )
    quarantine.mkdir(parents=True)
    manifest = {**plan, "status": "applying", "quarantine_directory": str(quarantine)}
    manifest_path = quarantine / "manifest.json"
    completed = []
    try:
        _json(manifest_path, manifest)
        for original, destination, expected_hash in pairs:
            _move_file(original, destination, expected_hash)
            completed.append((original, destination))
        manifest["status"] = "applied"
        _json(manifest_path, manifest)
    except BaseException:
        for original, destination in reversed(completed):
            _move_file(destination, original)
        shutil.rmtree(quarantine)
        raise
    return manifest_path


def undo_curation(manifest_path: Path | str) -> int:
    """Restore every quarantined byte after collision/integrity preflight."""
    path = Path(manifest_path)
    with path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    moves = _validate_plan(manifest)
    if manifest.get("status") != "applied":
        raise ValueError("curation is not applied or has already been undone")
    source = Path(manifest["source_directory"]).resolve()
    quarantine = Path(manifest["quarantine_directory"]).resolve()
    if path.resolve() != quarantine / "manifest.json":
        raise ValueError("manifest does not belong to its quarantine directory")
    pairs = []
    for move in moves:
        original = _safe_path(source, move["path"])
        destination = _safe_path(quarantine / "files", move["path"])
        if original.exists():
            raise FileExistsError(original)
        if _sha256(destination) != move["byte_sha256"]:
            raise ValueError(f"quarantine changed: {move['path']}")
        pairs.append((original, destination, move["byte_sha256"]))
    restored = []
    try:
        for original, destination, expected_hash in pairs:
            _move_file(destination, original, expected_hash)
            restored.append((original, destination))
        manifest["status"] = "undone"
        _json(path, manifest)
    except BaseException:
        for original, destination in reversed(restored):
            _move_file(original, destination)
        raise
    return len(restored)
