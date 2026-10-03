"""Installed CLI contract: explicit inputs and actionable failures."""

import json

import pytest

from minecraft_skin_gan.cli import main


@pytest.mark.parametrize("status,exit_code,key", [(404, 0, "skipped"), (403, 1, "failed")])
def test_acquisition_cli_manifest_and_failure_status(
    tmp_path, monkeypatch, capsys, status, exit_code, key
):
    from unittest.mock import Mock

    from minecraft_skin_gan import acquisition

    response = Mock(status_code=status)
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    get = Mock(return_value=response)
    monkeypatch.setattr(acquisition.requests, "get", get)
    output = tmp_path / "acquired"
    assert (
        main(
            [
                "acquire",
                str(output),
                "--url",
                "https://example.test/skin/{}",
                "--start",
                "3",
                "--stop",
                "4",
                "--provenance",
                "mock transport",
                "--max-attempts",
                "1",
                "--max-workers",
                "1",
            ]
        )
        == exit_code
    )
    result = json.loads(capsys.readouterr().out)
    assert result["counts"][key] == 1
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["records"][0]["id"] == 3
    assert manifest["provenance"] == {"description": "mock transport"}
    assert get.call_args.args == ("https://example.test/skin/3",)
    assert list(output.glob("*.png")) == []
    response.__exit__.assert_called_once()


def test_acquisition_cli_rejects_unbounded_range_without_output(tmp_path, capsys):
    assert (
        main(
            [
                "acquire",
                str(tmp_path / "output"),
                "--url",
                "https://example.test/skin/{}",
                "--start",
                "0",
                "--stop",
                str(2**100),
                "--provenance",
                "mock transport",
            ]
        )
        == 2
    )
    assert "IDs require" in capsys.readouterr().err
    assert not (tmp_path / "output").exists()


def test_cli_help_and_required_command(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    assert "prepare" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exit_info:
        main([])
    assert exit_info.value.code == 2


def test_creator_cli_opaque_base_profile_keeps_png_bytes(tmp_path, capsys):
    import re
    import zipfile

    from PIL import Image

    source = tmp_path / "skins"
    source.mkdir()
    with Image.new("RGBA", (64, 64), (30, 50, 70, 0)) as image:
        image.save(source / "skin.png")
    original = (source / "skin.png").read_bytes()
    preview = tmp_path / "preview.html"
    assert (
        main(["preview", str(source / "skin.png"), str(preview), "--alpha-mode", "opaque-base"])
        == 0
    )
    capsys.readouterr()
    metadata = json.loads(
        re.search(
            r'<script type="application/json" id="layout">(.*?)</script>', preview.read_text()
        ).group(1)
    )
    assert metadata["alpha_mode"] == "opaque-base"
    gallery = tmp_path / "gallery"
    assert main(["gallery", str(source), str(gallery), "--alpha-mode", "opaque-base"]) == 0
    capsys.readouterr()
    assert json.loads((gallery / "gallery.json").read_text())["preview_alpha_mode"] == "opaque-base"
    assert (gallery / "skin.png").read_bytes() == original
    assert (source / "skin.png").read_bytes() == original
    favorites = tmp_path / "favorites.json"
    favorites.write_text('["skin.png"]')
    exported = tmp_path / "selected.zip"
    assert main(["export", str(gallery), str(favorites), str(exported)]) == 0
    capsys.readouterr()
    with zipfile.ZipFile(exported) as archive:
        assert archive.read("skin.png") == original
        assert json.loads(archive.read("selection.json"))["preview_alpha_mode"] == "opaque-base"
    from minecraft_skin_gan.creator import create_gallery

    with pytest.raises(ValueError, match="alpha_mode"):
        create_gallery(source, tmp_path / "invalid-gallery", alpha_mode="invalid")
    assert not (tmp_path / "invalid-gallery").exists()


def test_prepare_cli_uses_explicit_paths_and_seed(tmp_path, monkeypatch, capsys):
    import minecraft_skin_gan.dataset as dataset

    observed = {}

    def prepare(source, output, **options):
        observed.update(source=source, output=output, **options)
        return {"schema": "test", "seed": options["seed"]}

    monkeypatch.setattr(dataset, "prepare_dataset", prepare)
    assert main(["prepare", str(tmp_path), str(tmp_path / "out"), "--seed", "42"]) == 0
    assert observed["seed"] == 42
    assert json.loads(capsys.readouterr().out)["seed"] == 42


def test_generate_missing_bundle_is_actionable(tmp_path, capsys):
    assert main(["generate", str(tmp_path / "missing"), str(tmp_path / "out")]) == 2
    assert "error:" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_prepare_and_reversible_curation_work_from_arbitrary_cwd(tmp_path, monkeypatch, capsys):
    import numpy as np
    from PIL import Image

    source = tmp_path / "source"
    source.mkdir()
    for name, color in (("a", 0), ("b", 255), ("duplicate", 0)):
        with Image.fromarray(np.full((64, 64, 4), color, dtype=np.uint8)) as image:
            image.save(source / f"{name}.png")
    original = {path.name: path.read_bytes() for path in source.glob("*.png")}
    monkeypatch.chdir(tmp_path)
    assert (
        main(["prepare", str(source), str(tmp_path / "dataset"), "--provenance", "Test fixture"])
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["counts"]["valid"] == 3
    assert report["counts"]["exact_groups"] == 2
    assert report["provenance"]["description"] == "Test fixture"
    assert main(["curate", str(source)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert len(plan["moves"]) == 1
    assert {path.name: path.read_bytes() for path in source.glob("*.png")} == original
    assert main(["curate", str(source), "--apply", str(tmp_path / "quarantine")]) == 0
    manifest = json.loads(capsys.readouterr().out)
    assert len(list(source.glob("*.png"))) == 2
    assert main(["undo", manifest]) == 0
    assert json.loads(capsys.readouterr().out)["restored"] == 1
    assert {path.name: path.read_bytes() for path in source.glob("*.png")} == original


def test_bundle_generate_and_evaluate_cli_real_models(tmp_path, capsys):
    import numpy as np

    from generate_skin import keras

    inputs = keras.Input(shape=(2,))
    color = keras.layers.Dense(4, kernel_initializer="zeros", bias_initializer="zeros")(inputs)
    decoder = keras.Model(inputs, keras.layers.RepeatVector(4096)(color))
    # Bundle supports flattened or image-shaped decoders, not a flat pixel list.
    decoder = keras.Model(inputs, keras.layers.Flatten()(decoder(inputs)))
    decoder.save(tmp_path / "decoder.keras")
    np.savez(tmp_path / "codes.npz", codes=np.zeros((2, 2)))
    assert (
        main(
            [
                "bundle",
                str(tmp_path / "decoder.keras"),
                str(tmp_path / "codes.npz"),
                str(tmp_path / "bundle"),
                "--dataset-fingerprint",
                "b" * 64,
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert (
        main(
            [
                "generate",
                str(tmp_path / "bundle"),
                str(tmp_path / "skins"),
                "--count",
                "2",
                "--seed",
                "1",
            ]
        )
        == 0
    )
    outputs = json.loads(capsys.readouterr().out)
    assert len(outputs) == 2
    data = np.zeros((2, 64, 64, 4), dtype=np.uint8)
    np.savez(tmp_path / "data.npz", data, data)
    assert (
        main(
            [
                "evaluate",
                str(tmp_path / "bundle"),
                str(tmp_path / "data.npz"),
                str(tmp_path / "evaluation"),
                "--count",
                "2",
            ]
        )
        == 0
    )
    from pathlib import Path

    report = json.loads(Path(json.loads(capsys.readouterr().out)).read_text())
    assert report["generated"]["exact_duplicate_count"] == 1
    assert "reconstruction" not in report
    assert main(["generate", str(tmp_path / "bundle"), str(tmp_path / "skins")]) == 2
    assert "error:" in capsys.readouterr().err


def test_creator_cli_preview_favorites_palette_and_preserve(tmp_path, capsys):
    from pathlib import Path

    import numpy as np
    from PIL import Image

    source = tmp_path / "skins"
    source.mkdir()
    original = np.zeros((64, 64, 4), dtype=np.uint8)
    candidate = np.full_like(original, 255)
    for name, pixels in (("original", original), ("candidate", candidate)):
        with Image.fromarray(pixels) as image:
            image.save(source / f"{name}.png")
    assert (
        main(
            [
                "preview",
                str(source / "original.png"),
                str(tmp_path / "preview.html"),
                "--model-type",
                "slim",
            ]
        )
        == 0
    )
    assert "slim" in (tmp_path / "preview.html").read_text()
    assert main(["gallery", str(source), str(tmp_path / "gallery")]) == 0
    selection = tmp_path / "favorites.json"
    selection.write_text(json.dumps(["original.png"]))
    assert (
        main(["export", str(tmp_path / "gallery"), str(selection), str(tmp_path / "selected.zip")])
        == 0
    )
    assert (tmp_path / "selected.zip").is_file()
    assert (
        main(
            [
                "palette",
                str(source / "candidate.png"),
                str(tmp_path / "palette.png"),
                "--colors",
                "#ff0000",
            ]
        )
        == 0
    )
    with Image.open(tmp_path / "palette.png") as image:
        assert image.getpixel((10, 10)) == (255, 0, 0, 255)
    assert (
        main(
            [
                "preserve",
                str(source / "original.png"),
                str(source / "candidate.png"),
                str(tmp_path / "preserved.png"),
                "--regions",
                "head",
            ]
        )
        == 0
    )
    with Image.open(tmp_path / "preserved.png") as image:
        assert image.getpixel((10, 10)) == (0, 0, 0, 0)
        assert image.getpixel((10, 30)) == (255, 255, 255, 255)
    capsys.readouterr()
    selection.write_text(json.dumps({"wrong": "schema"}))
    assert (
        main(["export", str(tmp_path / "gallery"), str(selection), str(tmp_path / "bad.zip")]) == 2
    )
    assert "JSON list" in capsys.readouterr().err
    assert not Path(tmp_path / "bad.zip").exists()


def test_cpu_cli_training_real_legacy_architecture(tmp_path, capsys):
    import numpy as np

    training = np.full((2, 64, 64, 4), 128, dtype=np.uint8)
    training[1, :, :, 0] = 255
    data = tmp_path / "data.npz"
    np.savez(data, training, training)
    output = tmp_path / "run"
    assert (
        main(
            [
                "train",
                str(data),
                str(output),
                "--encoded-dim",
                "2",
                "--batch-size",
                "2",
                "--ae-epochs",
                "1",
                "--discriminator-epochs",
                "1",
                "--gan-steps",
                "1",
                "--checkpoint-interval",
                "1",
            ]
        )
        == 0
    )
    report = json.loads((output / "metrics.json").read_text())
    assert report["completed"] == {"ae": 1, "discriminator": 1, "gan": 1}
    assert np.isfinite(report["validation_mse"])
    assert (output / "bundle" / "metadata.json").is_file()
    capsys.readouterr()
