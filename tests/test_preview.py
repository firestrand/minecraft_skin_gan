"""Offline preview embeds real face textures and explicit layer/model controls."""

import base64
import io
import json
import re

import numpy as np
import pytest
from PIL import Image

from minecraft_skin_gan.preview import create_preview


def make_skin(path):
    y, x = np.indices((64, 64))
    pixels = np.stack((x * 4, y * 4, np.zeros_like(x), np.full_like(x, 255)), -1).astype(np.uint8)
    pixels[8:16, 40:48, 3] = 0
    Image.fromarray(pixels).save(path)
    return pixels


def test_html_embeds_face_rgba_and_rotation_layer_controls(tmp_path):
    source = tmp_path / "source.png"
    make_skin(source)
    output = create_preview(source, tmp_path / "preview.html", model_type="slim")
    html = output.read_text(encoding="utf-8")
    assert "Front" in html and "Back" in html and "Rotate" in html
    assert 'id="base"' in html and 'id="overlay"' in html
    assert "@keyframes turn" in html and "preserve-3d" in html
    assert "<script src" not in html and "https://" not in html
    metadata = json.loads(
        re.search(r'<script type="application/json" id="layout">(.*?)</script>', html).group(1)
    )
    assert metadata["model_type"] == "slim"
    assert metadata["layout"]["right_arm"]["base"]["front"] == [44, 20, 3, 12]
    uri = re.search(r'data-face="head/base/front"[^>]*data:image/png;base64,([^\)]+)', html).group(
        1
    )
    with Image.open(io.BytesIO(base64.b64decode(uri))) as image:
        assert image.size == (8, 8) and image.mode == "RGBA"
        assert image.getpixel((0, 0)) == (32, 32, 0, 255)
    overlay = re.search(
        r'data-face="head/overlay/front"[^>]*data:image/png;base64,([^\)]+)', html
    ).group(1)
    with Image.open(io.BytesIO(base64.b64decode(overlay))) as image:
        assert image.getpixel((0, 0))[3] == 0


def test_preview_refuses_collision_and_invalid_png_or_model(tmp_path):
    source = tmp_path / "source.png"
    make_skin(source)
    output = tmp_path / "preview.html"
    output.write_text("owned", encoding="utf-8")
    with pytest.raises(FileExistsError):
        create_preview(source, output)
    assert output.read_text() == "owned"
    with pytest.raises(ValueError, match="model"):
        create_preview(source, tmp_path / "invalid.html", model_type="unknown")
    Image.new("RGBA", (32, 64)).save(source)
    with pytest.raises(ValueError, match="64"):
        create_preview(source, tmp_path / "invalid.html")
    source.write_bytes(b"invalid")
    with pytest.raises(ValueError, match="PNG"):
        create_preview(source, tmp_path / "invalid.html")
    assert not (tmp_path / "invalid.html").exists()


def test_rgb_png_converts_explicitly_to_rgba_and_names_are_not_embedded(tmp_path):
    source = tmp_path / '<img onerror="bad">.png'
    Image.new("RGB", (64, 64), "red").save(source)
    output = create_preview(source, tmp_path / "nested/preview.html")
    html = output.read_text()
    assert "onerror" not in html
    assert html.count("data-face=") == 72


def test_bounded_input_and_failed_publication_leave_no_temporary_files(tmp_path, monkeypatch):
    from pathlib import Path

    import minecraft_skin_gan.preview as preview

    source = tmp_path / "skin.png"
    make_skin(source)
    monkeypatch.setattr(preview, "_MAX_IMAGE_BYTES", 1)
    with pytest.raises(ValueError, match="input limit"):
        create_preview(source, tmp_path / "preview.html")
    monkeypatch.setattr(preview, "_MAX_IMAGE_BYTES", 4 * 1024 * 1024)

    def collision(output, _temporary):
        output.write_text("concurrent user content", encoding="utf-8")
        raise FileExistsError(output)

    monkeypatch.setattr(Path, "hardlink_to", collision)
    with pytest.raises(FileExistsError):
        create_preview(source, tmp_path / "preview.html")
    assert (tmp_path / "preview.html").read_text() == "concurrent user content"
    assert not list(tmp_path.glob(".preview-*"))


@pytest.mark.parametrize("model_type", ["classic", "slim"])
def test_opaque_base_changes_only_embedded_base_alpha(tmp_path, model_type):
    import hashlib

    from minecraft_skin_gan.skin_layout import face_pixels

    y, x = np.indices((64, 64))
    alpha = np.array([0, 128, 255], np.uint8)[(x + y) % 3]
    pixels = np.stack((x * 4, y * 4, (x + y) % 256, alpha), axis=-1).astype(np.uint8)
    source = tmp_path / "source.png"
    Image.fromarray(pixels).save(source)
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    outputs = {
        mode: create_preview(
            source, tmp_path / f"{mode}.html", model_type=model_type, alpha_mode=mode
        ).read_text()
        for mode in ("original", "opaque-base")
    }
    for mode, html in outputs.items():
        metadata = json.loads(
            re.search(r'<script type="application/json" id="layout">(.*?)</script>', html).group(1)
        )
        assert metadata["alpha_mode"] == mode
        assert metadata["game_acceptance"] == "unverified"
        assert f"alpha mode: {mode}" in html
        assert "game-format acceptance is unverified" in html
        textures = re.findall(r'data-face="([^\"]+)"[^>]*data:image/png;base64,([^\)]+)', html)
        assert len(textures) == 72
        for label, uri in textures:
            region, layer, face = label.split("/")
            expected = face_pixels(pixels, region, layer, face, model_type=model_type)
            with Image.open(io.BytesIO(base64.b64decode(uri))) as image:
                actual = np.asarray(image)
            np.testing.assert_array_equal(actual[:, :, :3], expected[:, :, :3])
            if mode == "opaque-base" and layer == "base":
                np.testing.assert_array_equal(actual[:, :, 3], np.full(expected.shape[:2], 255))
            else:
                np.testing.assert_array_equal(actual, expected)
    default = create_preview(source, tmp_path / "default.html", model_type=model_type)
    assert default.read_text() == outputs["original"]
    assert "Base faces render opaque" in outputs["opaque-base"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_hash
    with Image.open(source) as image:
        np.testing.assert_array_equal(np.asarray(image), pixels)


@pytest.mark.parametrize("mode", ["auto", "", None, False, ["opaque-base"]])
def test_invalid_alpha_mode_fails_before_io_and_writes(tmp_path, mode):
    output = tmp_path / "new-directory/preview.html"
    with pytest.raises(ValueError, match="alpha_mode"):
        create_preview(tmp_path / "missing.png", output, alpha_mode=mode)
    assert not output.parent.exists()


def test_opaque_base_refuses_output_collision(tmp_path):
    source = tmp_path / "source.png"
    make_skin(source)
    output = tmp_path / "preview.html"
    output.write_text("user owned", encoding="utf-8")
    with pytest.raises(FileExistsError):
        create_preview(source, output, alpha_mode="opaque-base")
    assert output.read_text() == "user owned"
