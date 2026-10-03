"""Review packet isolation, identity, rendering profiles and CLI contracts."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from minecraft_skin_gan.cli import main
from minecraft_skin_gan.review import create_review


@pytest.fixture
def batch(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for index in range(3):
        pixels = np.full((64, 64, 4), [index * 70, 23, 54, 128], dtype=np.uint8)
        Image.fromarray(pixels).save(source / f"GAN-variant-<script>-{index}.png")
    return source


def test_anonymous_packet_preserves_bytes_profiles_and_seed(batch, tmp_path):
    before = {p.name: p.read_bytes() for p in batch.glob("*.png")}
    a = create_review(batch, tmp_path / "a", model_type="slim", alpha_mode="opaque-base", seed=9)
    b = create_review(batch, tmp_path / "b", model_type="slim", alpha_mode="opaque-base", seed=9)
    assert a.read_bytes() == b.read_bytes()
    manifest = json.loads((a.parent / "review.json").read_text())
    mapping = json.loads((a.parent / "candidate-map.json").read_text())
    assert manifest["model_type"] == "slim"
    assert manifest["alpha_mode"] == "opaque-base"
    assert len(manifest["files"]) == 3
    for row in mapping:
        copied = (a.parent / (row["candidate"] + ".png")).read_bytes()
        assert copied == before[row["source_name"]]
        assert hashlib.sha256(copied).hexdigest() == row["sha256"]
        assert row["source_name"] not in a.read_text()
        assert row["source_name"] not in (a.parent / (row["candidate"] + ".html")).read_text()
    assert {p.name: p.read_bytes() for p in batch.glob("*.png")} == before
    independent = {k: v for k, v in manifest.items() if k != "packet_id"}
    assert (
        manifest["packet_id"]
        == hashlib.sha256(
            json.dumps(independent, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    other = create_review(batch, tmp_path / "c", alpha_mode="original", seed=9)
    assert (
        json.loads((other.parent / "review.json").read_text())["packet_id"] != manifest["packet_id"]
    )


@pytest.mark.parametrize(
    "kwargs", [{"model_type": "bad"}, {"alpha_mode": "bad"}, {"seed": True}, {"seed": "3"}]
)
def test_invalid_profile_or_seed_publishes_nothing(batch, tmp_path, kwargs):
    with pytest.raises(ValueError):
        create_review(batch, tmp_path / "review", **kwargs)
    assert not (tmp_path / "review").exists()


def test_collision_keeps_existing_output(batch, tmp_path):
    output = tmp_path / "review"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        create_review(batch, output)
    assert marker.read_text() == "keep"


@pytest.mark.parametrize("kind", ["empty", "many", "symlink", "wrong-size", "bad-bytes", "large"])
def test_bad_inputs_do_not_publish(tmp_path, kind):
    source = tmp_path / "source"
    source.mkdir()
    if kind == "many":
        for i in range(257):
            (source / f"{i}.png").touch()
    elif kind == "symlink":
        (source / "skin.png").symlink_to(tmp_path / "missing")
    elif kind == "wrong-size":
        Image.new("RGBA", (32, 32)).save(source / "skin.png")
    elif kind == "bad-bytes":
        (source / "skin.png").write_bytes(b"not png")
    elif kind == "large":
        (source / "skin.png").write_bytes(b"x" * (4 * 1024 * 1024 + 1))
    with pytest.raises((ValueError, OSError)):
        create_review(source, tmp_path / "output")
    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob(".review-*"))


def test_review_cli(batch, tmp_path, capsys):
    assert (
        main(
            ["review", str(batch), str(tmp_path / "review"), "--model-type", "slim", "--seed", "11"]
        )
        == 0
    )
    assert Path(json.loads(capsys.readouterr().out)).name == "index.html"
    assert main(["review", str(batch), str(tmp_path / "review")]) == 2


def test_publication_failure_removes_empty_destination(batch, tmp_path, monkeypatch):
    output = tmp_path / "review"
    original_replace = Path.replace

    def fail_publish(self, target):
        if Path(target) == output:
            raise OSError("publication failed")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_publish)
    with pytest.raises(OSError, match="publication failed"):
        create_review(batch, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".review-*"))


def test_animated_png_is_rejected(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    first = Image.new("RGBA", (64, 64), (255, 0, 0, 255))
    second = Image.new("RGBA", (64, 64), (0, 255, 0, 255))
    first.save(source / "animated.png", save_all=True, append_images=[second], duration=100, loop=0)
    with pytest.raises(ValueError, match="single-frame"):
        create_review(source, tmp_path / "review")
    assert not (tmp_path / "review").exists()
