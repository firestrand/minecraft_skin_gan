"""Creator exports preserve selected pixels and identify postprocessing."""

import json
import zipfile

import numpy as np
import pytest
from PIL import Image

from minecraft_skin_gan.creator import create_gallery, export_favorites, recolor_skin


def test_palette_changes_visible_color_but_preserves_alpha_and_hidden_bytes(tmp_path):
    pixels = np.zeros((64, 64, 4), dtype=np.uint8)
    pixels[0, 0] = [240, 20, 30, 123]
    pixels[0, 1] = [200, 20, 0, 0]
    with Image.fromarray(pixels) as image:
        image.save(tmp_path / "source.png")
    result = recolor_skin(tmp_path / "source.png", tmp_path / "colored.png", ["#ff0000", "#000000"])
    with Image.open(result) as image:
        actual = np.asarray(image)
        np.testing.assert_array_equal(actual[0, 0], [255, 0, 0, 123])
        np.testing.assert_array_equal(actual[0, 1], pixels[0, 1])
        np.testing.assert_array_equal(actual[..., 3], pixels[..., 3])
    with pytest.raises(FileExistsError):
        recolor_skin(tmp_path / "source.png", result, ["#000000"])
    with pytest.raises(ValueError):
        recolor_skin(tmp_path / "source.png", tmp_path / "invalid.png", ["not-a-color"])


def test_gallery_and_export_favorites_preserve_png_bytes(tmp_path):
    source = tmp_path / "batch"
    source.mkdir()
    for index in range(2):
        with Image.fromarray(np.full((64, 64, 4), index * 255, dtype=np.uint8)) as image:
            image.save(source / f"skin_{index:04d}.png")
    (source / "generation.json").write_text(json.dumps({"seed": 42, "sampler": "kde"}))
    gallery = create_gallery(source, tmp_path / "gallery")
    assert "image/png;base64" in gallery.read_text()
    assert "Favorites" in gallery.read_text()
    assert (gallery.parent / "skin_0000.png.preview.html").exists()
    assert json.loads((gallery.parent / "gallery.json").read_text())["generation"]["seed"] == 42
    output = export_favorites(gallery.parent, ["skin_0001.png"], tmp_path / "chosen.zip")
    with zipfile.ZipFile(output) as archive:
        assert archive.read("skin_0001.png") == (source / "skin_0001.png").read_bytes()
        metadata = json.loads(archive.read("selection.json"))
        assert metadata["selected"] == ["skin_0001.png"]
        assert metadata["generation"]["seed"] == 42
    with pytest.raises(ValueError):
        export_favorites(gallery.parent, ["../skin_0000.png"], tmp_path / "bad.zip")
    with pytest.raises(FileExistsError):
        export_favorites(gallery.parent, ["skin_0000.png"], output)


def test_gallery_requires_images_and_export_detects_changes(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError):
        create_gallery(empty, tmp_path / "gallery")
    with Image.fromarray(np.zeros((64, 64, 4), dtype=np.uint8)) as image:
        image.save(empty / "skin.png")
    gallery = create_gallery(empty, tmp_path / "gallery")
    with pytest.raises(FileExistsError):
        create_gallery(empty, gallery.parent)
    with Image.fromarray(np.ones((64, 64, 4), dtype=np.uint8)) as image:
        image.save(gallery.parent / "skin.png")
    with pytest.raises(ValueError, match="changed"):
        export_favorites(gallery.parent, ["skin.png"], tmp_path / "changed.zip")
    assert not (tmp_path / "changed.zip").exists()
    with pytest.raises(ValueError):
        export_favorites(gallery.parent, [], tmp_path / "none.zip")
    with pytest.raises(ValueError):
        export_favorites(gallery.parent, ["skin.png", "skin.png"], tmp_path / "duplicates.zip")


@pytest.mark.parametrize(
    "name", ["skin#one.png", "skin?one.png", "skin%one.png", "javascript:skin.png"]
)
def test_gallery_download_links_are_relative_encoded_urls(tmp_path, name):
    from urllib.parse import quote

    source = tmp_path / "skins"
    source.mkdir()
    with Image.fromarray(np.zeros((64, 64, 4), dtype=np.uint8)) as image:
        image.save(source / name)
    gallery = create_gallery(source, tmp_path / "gallery")
    assert f'href="./{quote(name, safe="")}"' in gallery.read_text()
    output = export_favorites(gallery.parent, [name], tmp_path / "selection.zip")
    with zipfile.ZipFile(output) as archive:
        assert archive.read(name) == (source / name).read_bytes()
