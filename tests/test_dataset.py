"""Versioned preparation and reversible curation contracts on disposable images."""

import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from minecraft_skin_gan.dataset import (
    apply_curation,
    plan_curation,
    prepare_dataset,
    undo_curation,
)


def skin(path, value=0, alpha=255):
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.full((64, 64, 4), value, np.uint8)
    pixels[:, :, 3] = alpha
    Image.fromarray(pixels).save(path)
    return pixels


def corpus(root):
    for index in range(5):
        skin(root / f"{index}.png", index * 30)
    skin(root / "duplicate.png", 0)
    skin(root / "alpha.png", 0, 0)


def test_preparation_is_deterministic_and_preserves_rgba_and_groups(tmp_path):
    source = tmp_path / "source"
    corpus(source)
    first = prepare_dataset(source, tmp_path / "first", provenance={"origin": "fixture"})
    second = prepare_dataset(source, tmp_path / "second", provenance={"origin": "fixture"})
    assert first == second
    assert first["schema_version"] == 1
    assert (
        first["archive_sha256"]
        == hashlib.sha256((tmp_path / "first/train_test.npz").read_bytes()).hexdigest()
    )
    assert first["exposure"].startswith("development")
    assert first["provenance"] == {"origin": "fixture"}
    records = {row["path"]: row for row in first["files"]}
    assert records["0.png"]["split"] == records["duplicate.png"]["split"]
    assert records["0.png"]["rgba_sha256"] != records["alpha.png"]["rgba_sha256"]
    assert (
        records["0.png"]["byte_sha256"]
        == hashlib.sha256((source / "0.png").read_bytes()).hexdigest()
    )
    with np.load(tmp_path / "first/train_test.npz", allow_pickle=False) as arrays:
        assert arrays.files == ["arr_0", "arr_1"]
        assert sum(len(arrays[key]) for key in arrays.files) == 7
        assert arrays["arr_0"].dtype == np.uint8
        for key, split in (("arr_0", "train"), ("arr_1", "validation")):
            paths = [row["path"] for row in first["files"] if row.get("split") == split]
            for pixel_array, path in zip(arrays[key], paths, strict=True):
                with Image.open(source / path) as image:
                    np.testing.assert_array_equal(pixel_array, np.asarray(image.convert("RGBA")))
    assert first["counts"]["exact_groups"] == 6
    assert first["near_duplicates"]["candidates"]
    assert first["near_duplicates"]["distance"] == 0


def test_reports_invalid_inputs_and_original_mode(tmp_path):
    source = tmp_path / "source"
    corpus(source)
    (source / "broken.png").write_bytes(b"invalid image")
    Image.new("RGB", (32, 64)).save(source / "small.png")
    Image.new("RGB", (64, 64), "red").save(source / "rgb.png")
    manifest = prepare_dataset(source, tmp_path / "prepared")
    records = {row["path"]: row for row in manifest["files"]}
    assert "error" in records["broken.png"]
    assert "64" in records["small.png"]["error"]
    assert records["rgb.png"]["mode"] == "RGB"
    assert records["rgb.png"]["shape"] == [64, 64, 4]
    assert manifest["counts"]["invalid"] == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"validation_fraction": 0},
        {"validation_fraction": 1},
        {"near_duplicate_distance": -1},
        {"candidate_limit": 0},
    ],
)
def test_bad_configuration_leaves_no_outputs(tmp_path, kwargs):
    source = tmp_path / "source"
    corpus(source)
    with pytest.raises(ValueError):
        prepare_dataset(source, tmp_path / "output", **kwargs)
    assert not (tmp_path / "output").exists()


def test_empty_or_one_group_cannot_create_honest_split(tmp_path):
    tmp_path.joinpath("source").mkdir()
    with pytest.raises(ValueError, match="groups"):
        prepare_dataset(tmp_path / "source", tmp_path / "out")
    skin(tmp_path / "source/a.png")
    skin(tmp_path / "source/b.png")
    with pytest.raises(ValueError, match="groups"):
        prepare_dataset(tmp_path / "source", tmp_path / "out")


def test_output_collision_preserves_existing(tmp_path):
    corpus(tmp_path / "source")
    (tmp_path / "output").mkdir()
    sentinel = tmp_path / "output/keep"
    sentinel.write_text("owned", encoding="utf-8")
    with pytest.raises(FileExistsError):
        prepare_dataset(tmp_path / "source", tmp_path / "output")
    assert sentinel.read_text() == "owned"


def test_curation_inspection_apply_undo_preserves_bytes(tmp_path):
    source = tmp_path / "source"
    corpus(source)
    before = {path.name: path.read_bytes() for path in source.iterdir()}
    plan = plan_curation(source)
    assert len(plan["moves"]) == 1
    assert plan["moves"][0]["path"] == "duplicate.png"
    assert plan["moves"][0]["keeper"] == "0.png"
    assert {path.name: path.read_bytes() for path in source.iterdir()} == before
    manifest_path = apply_curation(plan, tmp_path / "quarantine")
    assert not (source / "duplicate.png").exists()
    assert undo_curation(manifest_path) == 1
    assert {path.name: path.read_bytes() for path in source.iterdir()} == before
    assert json.loads(manifest_path.read_text())["status"] == "undone"
    with pytest.raises(ValueError, match="undone"):
        undo_curation(manifest_path)


def test_named_head_filter_and_invalid_policy(tmp_path):
    source = tmp_path / "source"
    skin(source / "plain.png")
    assert plan_curation(source, policy="head-filter")["moves"][0]["reason"] == "head-filter"
    with pytest.raises(ValueError, match="policy"):
        plan_curation(source, policy="unrecognized")


def test_changed_source_and_quarantine_collision_fail_before_move(tmp_path):
    source = tmp_path / "source"
    corpus(source)
    plan = plan_curation(source)
    skin(source / "duplicate.png", 42)
    with pytest.raises(ValueError, match="changed"):
        apply_curation(plan, tmp_path / "quarantine")
    assert not (tmp_path / "quarantine").exists()
    plan = plan_curation(source, policy="head-filter")
    (tmp_path / "quarantine").mkdir()
    with pytest.raises(FileExistsError):
        apply_curation(plan, tmp_path / "quarantine")
    assert all((source / move["path"]).exists() for move in plan["moves"])


def test_undo_rejects_original_collision_and_changed_quarantine(tmp_path):
    source = tmp_path / "source"
    corpus(source)
    manifest_path = apply_curation(plan_curation(source), tmp_path / "quarantine")
    skin(source / "duplicate.png", 66)
    with pytest.raises(FileExistsError):
        undo_curation(manifest_path)
    (source / "duplicate.png").unlink()
    skin(tmp_path / "quarantine/files/duplicate.png", 99)
    with pytest.raises(ValueError, match="changed"):
        undo_curation(manifest_path)
    assert not (source / "duplicate.png").exists()


def test_apply_rolls_back_partial_failure(tmp_path, monkeypatch):
    import minecraft_skin_gan.dataset as dataset

    source = tmp_path / "source"
    skin(source / "a.png")
    skin(source / "b.png", 30)
    original = {path.name: path.read_bytes() for path in source.iterdir()}
    real_move = dataset._move_file
    calls = 0

    def fail_second(source_path, target_path, expected_hash=None):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected move failure")
        real_move(source_path, target_path, expected_hash)

    monkeypatch.setattr(dataset, "_move_file", fail_second)
    with pytest.raises(OSError, match="injected"):
        apply_curation(plan_curation(source, policy="head-filter"), tmp_path / "quarantine")
    assert {path.name: path.read_bytes() for path in source.iterdir()} == original
    assert not (tmp_path / "quarantine").exists()


def test_rejects_traversal_and_symlink_inputs(tmp_path):
    source = tmp_path / "source"
    corpus(source)
    (source / "linked.png").symlink_to(source / "0.png")
    report = plan_curation(source)
    assert any(row["path"] == "linked.png" for row in report["errors"])
    report["moves"][0]["path"] = "../outside.png"
    with pytest.raises(ValueError, match="relative"):
        apply_curation(report, tmp_path / "quarantine")


def test_near_candidates_bound_reporting_without_changing_split(tmp_path):
    source = tmp_path / "source"
    corpus(source)
    rng = np.random.default_rng(123)
    for index in range(4):
        pixels = rng.integers(0, 256, (64, 64, 4), dtype=np.uint8)
        Image.fromarray(pixels).save(source / f"noise_{index}.png")
    exact = prepare_dataset(source, tmp_path / "exact")
    nearby = prepare_dataset(
        source, tmp_path / "nearby", near_duplicate_distance=64, candidate_limit=1
    )
    assert nearby["files"] == exact["files"]
    report = nearby["near_duplicates"]
    assert report["truncated"]
    assert len(report["candidates"]) == 1
    assert report["candidate_count"] > 1
    assert report["candidate_count"] > exact["near_duplicates"]["candidate_count"]
    tight = prepare_dataset(source, tmp_path / "tight", near_duplicate_distance=1)
    assert (
        tight["near_duplicates"]["candidate_count"] == exact["near_duplicates"]["candidate_count"]
    )


def test_order_independent_roots_and_nested_names(tmp_path):
    for name in ("z.PNG", "nested/b.png", "a.png"):
        skin(tmp_path / "first" / name, len(name))
    for name in ("a.png", "nested/b.png", "z.PNG"):
        skin(tmp_path / "second" / name, len(name))
    assert prepare_dataset(tmp_path / "first", tmp_path / "out1") == prepare_dataset(
        tmp_path / "second", tmp_path / "out2"
    )


def test_invalid_content_and_missing_directory_are_reported(tmp_path, monkeypatch):
    import minecraft_skin_gan.dataset as dataset

    with pytest.raises(FileNotFoundError):
        plan_curation(tmp_path / "missing")
    source = tmp_path / "source"
    skin(source / "large.png")
    Image.new("RGBA", (64, 64)).save(source / "not_png.png", format="BMP")
    monkeypatch.setattr(dataset, "MAX_IMAGE_BYTES", 100)
    report = plan_curation(source)
    assert all("byte limit" in row["error"] for row in report["errors"])
    monkeypatch.setattr(dataset, "MAX_IMAGE_BYTES", 100_000)
    report = plan_curation(source)
    assert "PNG content" in report["errors"][0]["error"]


def test_head_filter_keeps_variable_rgb_and_reports_errors(tmp_path):
    source = tmp_path / "source"
    skin(source / "keep.png")
    pixels = np.indices((64, 64)).sum(axis=0).astype(np.uint8) * 4
    rgba = np.stack((pixels, pixels, pixels, np.full_like(pixels, 255)), axis=-1)
    Image.fromarray(rgba).save(source / "keep.png")
    (source / "broken.png").write_bytes(b"broken")
    report = plan_curation(source, policy="head-filter")
    assert report["moves"] == []
    assert len(report["errors"]) == 1


def test_keeper_changes_and_nested_quarantine_are_rejected(tmp_path):
    source = tmp_path / "source"
    corpus(source)
    report = plan_curation(source)
    with pytest.raises(ValueError, match="outside"):
        apply_curation(report, source / "quarantine")
    skin(source / "0.png", 20)
    with pytest.raises(ValueError, match="keeper changed"):
        apply_curation(report, tmp_path / "quarantine")


@pytest.mark.parametrize(
    "plan",
    [
        {"schema_version": 2, "moves": []},
        {"schema_version": 1, "moves": []},
        {"schema_version": 1, "moves": [None]},
        {"schema_version": 1, "moves": [{"path": "a.png", "byte_sha256": "short"}]},
        {"schema_version": 1, "moves": [{"path": "a.png", "byte_sha256": "0" * 64}] * 2},
    ],
)
def test_rejects_malformed_curation_manifests(tmp_path, plan):
    with pytest.raises(ValueError):
        apply_curation(plan, tmp_path / "quarantine")


def test_undo_partial_failure_rolls_back_to_quarantine(tmp_path, monkeypatch):
    import minecraft_skin_gan.dataset as dataset

    source = tmp_path / "source"
    skin(source / "a.png")
    skin(source / "b.png", 30)
    manifest = apply_curation(plan_curation(source, policy="head-filter"), tmp_path / "quarantine")
    real_move = dataset._move_file
    calls = 0

    def fail_second(original, destination, expected_hash=None):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected undo failure")
        real_move(original, destination, expected_hash)

    monkeypatch.setattr(dataset, "_move_file", fail_second)
    with pytest.raises(OSError, match="undo failure"):
        undo_curation(manifest)
    assert not list(source.glob("*.png"))
    assert len(list(tmp_path.glob("quarantine/files/*.png"))) == 2
    assert json.loads(manifest.read_text())["status"] == "applied"
    monkeypatch.setattr(dataset, "_move_file", real_move)
    assert undo_curation(manifest) == 2


def test_move_collision_and_write_failure_preserve_owned_files(tmp_path, monkeypatch):
    import minecraft_skin_gan.dataset as dataset

    source = tmp_path / "source.png"
    skin(source)
    destination = tmp_path / "destination.png"
    destination.write_bytes(b"owned")
    with pytest.raises(FileExistsError):
        dataset._move_file(source, destination)
    assert destination.read_bytes() == b"owned"
    destination.unlink()

    def partial_copy(reader, writer):
        writer.write(b"partial")
        raise OSError("injected copy failure")

    monkeypatch.setattr(dataset.shutil, "copyfileobj", partial_copy)
    original_bytes = source.read_bytes()
    with pytest.raises(OSError, match="copy failure"):
        dataset._move_file(source, destination)
    assert source.read_bytes() == original_bytes
    assert not destination.exists()


def test_preparation_detects_mid_read_changes_and_cleans_staging(tmp_path, monkeypatch):
    import minecraft_skin_gan.dataset as dataset

    source = tmp_path / "source"
    corpus(source)
    real_read = dataset._read_image
    reads = {}

    def modify_second_read(path):
        reads[path] = reads.get(path, 0) + 1
        if reads[path] == 2:
            skin(path, 111)
        return real_read(path)

    monkeypatch.setattr(dataset, "_read_image", modify_second_read)
    with pytest.raises(ValueError, match="changed during"):
        prepare_dataset(source, tmp_path / "output")
    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob(".prepare-*"))


def test_manifest_location_and_symlink_paths_are_rejected(tmp_path):
    source = tmp_path / "source"
    corpus(source)
    plan = plan_curation(source)
    (source / "alias").symlink_to(source, target_is_directory=True)
    plan["moves"][0]["path"] = "alias/duplicate.png"
    with pytest.raises(ValueError, match="symlinks"):
        apply_curation(plan, tmp_path / "quarantine")
    manifest = apply_curation(plan_curation(source), tmp_path / "quarantine")
    copied = tmp_path / "copied_manifest.json"
    copied.write_bytes(manifest.read_bytes())
    with pytest.raises(ValueError, match="does not belong"):
        undo_curation(copied)
