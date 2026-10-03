"""Generation contracts: uniform latents and exact RGBA PNG conversion."""

import runpy
import shutil
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

import generate_skin


def test_generation_preserves_pixel_formula_and_names(tmp_path, monkeypatch):
    pixels = np.tile([-0.1, 0.5, 0.1, 1.2], (2, 64 * 64, 1))
    decoder = Mock()
    decoder.predict.return_value = pixels.reshape(2, -1)
    loader = Mock(return_value=decoder)
    monkeypatch.setattr(generate_skin, "load_decoder", loader)
    latent = np.full((2, 128), 0.25)
    random = Mock(return_value=latent)
    monkeypatch.setattr(generate_skin.np.random, "rand", random)

    paths = generate_skin.generate_skins("decoder.keras", tmp_path / "output", count=2)

    loader.assert_called_once_with("decoder.keras")
    random.assert_called_once_with(2, 128)
    np.testing.assert_array_equal(decoder.predict.call_args.args[0], latent)
    assert paths == [tmp_path / "output" / "gae_0.png", tmp_path / "output" / "gae_1.png"]
    for path in paths:
        with Image.open(path) as image:
            assert image.size == (64, 64)
            assert image.mode == "RGBA"
            assert image.getpixel((0, 0)) == (0, 128, 26, 255)
            assert image.getpixel((63, 63)) == (0, 128, 26, 255)


def test_generation_defaults_and_main(monkeypatch, tmp_path):
    decoder = Mock()
    decoder.predict.return_value = np.zeros((100, 64 * 64 * 4))
    monkeypatch.setattr(generate_skin, "load_decoder", Mock(return_value=decoder))
    monkeypatch.chdir(tmp_path)
    generate_skin.main()
    assert decoder.predict.call_args.args[0].shape == (100, 128)
    assert (tmp_path / "images/results/gae_99.png").exists()


def test_malformed_decoded_shape_fails(tmp_path, monkeypatch):
    decoder = Mock()
    decoder.predict.return_value = np.zeros((1, 7))
    monkeypatch.setattr(generate_skin, "load_decoder", Mock(return_value=decoder))
    with pytest.raises(ValueError, match="reshape"):
        generate_skin.generate_skins("decoder.keras", tmp_path, count=1)


@pytest.mark.parametrize("count", [0, -1])
def test_invalid_generation_count(count, tmp_path):
    with pytest.raises(ValueError, match="positive"):
        generate_skin.generate_skins("decoder.keras", tmp_path, count=count)


@pytest.mark.parametrize("suffix", [".keras", ".mdl"])
def test_real_decoder_save_load_generation_round_trip(tmp_path, suffix):
    keras = generate_skin.keras
    inputs = keras.Input(shape=(128,))
    channels = keras.layers.Dense(
        4,
        kernel_initializer="zeros",
        bias_initializer=keras.initializers.Constant([-0.1, 0.5, 0.1, 1.2]),
    )(inputs)
    pixels = keras.layers.RepeatVector(64 * 64)(channels)
    decoder = keras.Model(inputs, keras.layers.Flatten()(pixels))
    path = tmp_path / f"decoder{suffix}"
    if suffix == ".mdl":
        hdf5_path = tmp_path / "original.h5"
        decoder.save(hdf5_path)
        shutil.copyfile(hdf5_path, path)
    else:
        decoder.save(path)

    paths = generate_skin.generate_skins(path, tmp_path / "results", count=2)

    assert len(paths) == 2
    for output in paths:
        with Image.open(output) as image:
            assert image.mode == "RGBA"
            assert image.size == (64, 64)
            np.testing.assert_array_equal(
                np.asarray(image), np.tile([0, 128, 26, 255], (64, 64, 1))
            )


def test_default_decoder_falls_back_to_legacy_hdf5(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "models/decoder.mdl"
    path.parent.mkdir()
    with generate_skin.h5py.File(path, "w") as artifact:
        artifact["test"] = 1
    decoder = Mock()
    copied_paths = []

    def load(path, *, compile):
        copied_paths.append(path)
        assert path.suffix == ".h5"
        assert path.read_bytes() == (tmp_path / "models/decoder.mdl").read_bytes()
        assert compile is False
        return decoder

    monkeypatch.setattr(generate_skin.keras.models, "load_model", load)
    assert generate_skin.load_decoder() is decoder
    assert not copied_paths[0].exists()
    assert path.exists()


def test_missing_default_decoder_has_actionable_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="models/decoder.keras"):
        generate_skin.load_decoder()


def test_saved_model_directory_is_explicitly_unsupported(tmp_path):
    with pytest.raises(ValueError, match="original compatible Keras"):
        generate_skin.load_decoder(tmp_path)


def test_non_hdf5_legacy_model_is_explicitly_unsupported(tmp_path):
    path = tmp_path / "decoder.mdl"
    path.write_bytes(b"legacy unsupported model")
    with pytest.raises(ValueError, match="not HDF5"):
        generate_skin.load_decoder(path)


def test_invalid_modern_model_propagates_loader_failure(tmp_path):
    path = tmp_path / "decoder.keras"
    path.write_bytes(b"not a keras archive")
    with pytest.raises(ValueError, match="File not found"):
        generate_skin.load_decoder(path)


def test_script_entrypoint_and_import_safety(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "models/decoder.keras"
    path.parent.mkdir()
    path.touch()
    decoder = Mock()
    decoder.predict.return_value = np.zeros((1, 64 * 64 * 4))
    loader = Mock(return_value=decoder)
    monkeypatch.setattr(generate_skin.keras.models, "load_model", loader)
    runpy.run_path(generate_skin.__file__, run_name="import_test")
    loader.assert_not_called()

    runpy.run_path(generate_skin.__file__, run_name="__main__")

    loader.assert_called_once_with(path.relative_to(tmp_path), compile=False)
    assert decoder.predict.call_args.args[0].shape == (100, 128)
    with Image.open(tmp_path / "images/results/gae_0.png") as image:
        assert image.getpixel((0, 0)) == (0, 0, 0, 0)
