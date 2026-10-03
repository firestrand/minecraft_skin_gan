"""Persisted generation reproduces trained samplers and rejects corrupt bundles."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from generate_skin import keras
from minecraft_skin_gan.bundle import create_bundle, generate_bundle, load_bundle


@pytest.fixture
def sources(tmp_path):
    inputs = keras.Input(shape=(2,))
    channels = keras.layers.Dense(
        4,
        kernel_initializer=keras.initializers.Constant(0.1),
        bias_initializer=keras.initializers.Constant(0.5),
    )(inputs)
    model = keras.Model(inputs, keras.layers.Flatten()(keras.layers.RepeatVector(4096)(channels)))
    decoder = tmp_path / "source.keras"
    model.save(decoder)
    codes = tmp_path / "source.npz"
    np.savez(codes, codes=np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0]]), bandwidth=0.2)
    return decoder, codes


def make_bundle(sources, tmp_path):
    return create_bundle(*sources, tmp_path / "bundle", dataset_fingerprint="a" * 64, bandwidth=0.2)


def test_real_fresh_process_seeded_roundtrip(sources, tmp_path):
    bundle = make_bundle(sources, tmp_path)
    loaded = load_bundle(bundle)
    np.testing.assert_allclose(loaded.decoder.predict(np.array([[0.25, 0.25]]), verbose=0), 0.55)
    first = generate_bundle(bundle, tmp_path / "first", count=3, seed=7)
    code = "from minecraft_skin_gan.bundle import generate_bundle; import sys; generate_bundle(sys.argv[1], sys.argv[2], count=3, seed=7)"
    subprocess.run(
        [sys.executable, "-c", code, str(bundle), str(tmp_path / "second")],
        check=True,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
    )
    for path in first:
        assert path.read_bytes() == (tmp_path / "second" / path.name).read_bytes()
        with Image.open(path) as image:
            assert image.mode == "RGBA" and image.size == (64, 64)
    report = json.loads((tmp_path / "first" / "generation.json").read_text())
    assert report["seed"] == 7 and report["sampler"] == "kde" and report["count"] == 3
    assert loaded.metadata["dataset_fingerprint"] == "a" * 64


@pytest.mark.parametrize("sampler", ["empirical", "legacy_uniform"])
def test_sampling_modes_are_seeded(sources, tmp_path, sampler):
    bundle = make_bundle(sources, tmp_path)
    one = generate_bundle(bundle, tmp_path / "one", count=4, seed=21, sampler=sampler)
    two = generate_bundle(bundle, tmp_path / "two", count=4, seed=21, sampler=sampler)
    assert [p.read_bytes() for p in one] == [p.read_bytes() for p in two]
    if sampler == "empirical":
        values = set()
        for path in one:
            with Image.open(path) as image:
                values.add(image.getpixel((0, 0))[0])
        assert values <= {76, 128, 178}


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "v999"),
        ("decoder", "../source.keras"),
        ("latent_dimension", 99),
        ("normalization", [-1, 1]),
    ],
)
def test_invalid_metadata_rejected(sources, tmp_path, field, value):
    bundle = make_bundle(sources, tmp_path)
    metadata_path = bundle / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata[field] = value
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        generate_bundle(bundle, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_corruption_and_collision_do_not_overwrite(sources, tmp_path):
    bundle = make_bundle(sources, tmp_path)
    with pytest.raises(FileExistsError):
        make_bundle(sources, tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep"
    marker.write_text("original")
    with pytest.raises(FileExistsError):
        generate_bundle(bundle, output)
    assert marker.read_text() == "original"
    (bundle / "codes.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="fingerprint"):
        load_bundle(bundle)


@pytest.mark.parametrize("codes", [np.array([[np.nan, 0]]), np.zeros((0, 2)), np.zeros((3, 4))])
def test_invalid_codes_fail_without_output(sources, tmp_path, codes):
    np.savez(sources[1], codes=codes)
    with pytest.raises(ValueError):
        make_bundle(sources, tmp_path)
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize(
    "kwargs", [{"count": 0}, {"bandwidth": -1}, {"sampler": "bogus"}, {"seed": -1}]
)
def test_invalid_generation_options(sources, tmp_path, kwargs):
    bundle = make_bundle(sources, tmp_path)
    with pytest.raises(ValueError):
        generate_bundle(bundle, tmp_path / "output", **kwargs)
    assert not (tmp_path / "output").exists()


def test_nonfinite_decoder_output_rolls_back(sources, tmp_path, monkeypatch):
    bundle = make_bundle(sources, tmp_path)
    loaded = load_bundle(bundle)
    monkeypatch.setattr(loaded.decoder, "predict", lambda *_a, **_kw: np.full((1, 16384), np.nan))
    monkeypatch.setattr("minecraft_skin_gan.bundle.load_bundle", lambda _path: loaded)
    with pytest.raises(ValueError, match="finite"):
        generate_bundle(bundle, tmp_path / "output", count=1)
    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob(".output-*"))


def test_kde_sampling_parity_and_bandwidth_override(sources, tmp_path, monkeypatch):
    from sklearn.neighbors import KernelDensity

    bundle = make_bundle(sources, tmp_path)
    loaded = load_bundle(bundle)
    actual_latents = []
    original = loaded.decoder.predict

    def capture(latents, **kwargs):
        actual_latents.append(latents.copy())
        return original(latents, **kwargs)

    monkeypatch.setattr(loaded.decoder, "predict", capture)
    monkeypatch.setattr("minecraft_skin_gan.bundle.load_bundle", lambda _path: loaded)
    generate_bundle(bundle, tmp_path / "output", count=5, seed=23, bandwidth=0.4)
    expected = KernelDensity(bandwidth=0.4).fit(loaded.codes).sample(5, random_state=23)
    np.testing.assert_array_equal(actual_latents[0], expected)
    report = json.loads((tmp_path / "output" / "generation.json").read_text())
    assert report["bandwidth"] == 0.4


def test_bounded_batches_and_shape_failure(sources, tmp_path, monkeypatch):
    bundle = make_bundle(sources, tmp_path)
    loaded = load_bundle(bundle)
    sizes = []

    def capture(latents, **_kwargs):
        sizes.append(len(latents))
        return np.zeros((len(latents), 16384))

    monkeypatch.setattr(loaded.decoder, "predict", capture)
    monkeypatch.setattr("minecraft_skin_gan.bundle.load_bundle", lambda _path: loaded)
    assert len(generate_bundle(bundle, tmp_path / "output", count=65)) == 65
    assert sizes == [64, 1]
    monkeypatch.setattr(loaded.decoder, "predict", lambda *_a, **_kw: np.zeros((1, 9)))
    with pytest.raises(ValueError, match="shape"):
        generate_bundle(bundle, tmp_path / "bad_output", count=1)
    assert not (tmp_path / "bad_output").exists()


def test_bundle_has_valid_fingerprints(sources, tmp_path):
    bundle = make_bundle(sources, tmp_path)
    loaded = load_bundle(bundle)
    for artifact in ("decoder", "codes"):
        digest = (
            __import__("hashlib")
            .sha256((bundle / loaded.metadata[artifact]).read_bytes())
            .hexdigest()
        )
        assert loaded.metadata[f"{artifact}_sha256"] == digest
    with pytest.raises(ValueError, match="fingerprint"):
        create_bundle(*sources, tmp_path / "invalid", dataset_fingerprint="unknown")
    assert not (tmp_path / "invalid").exists()


def test_symlink_artifact_rejected(sources, tmp_path):
    bundle = make_bundle(sources, tmp_path)
    (bundle / "decoder.keras").unlink()
    (bundle / "decoder.keras").symlink_to(sources[0])
    with pytest.raises(ValueError, match="path"):
        load_bundle(bundle)


def test_public_latent_probes_match_generated_batches(sources, tmp_path, monkeypatch):
    from minecraft_skin_gan.bundle import sample_latents

    bundle = make_bundle(sources, tmp_path)
    loaded = load_bundle(bundle)
    probes = sample_latents(loaded, count=65, seed=15)
    actual = []

    def capture(latents, **_kwargs):
        actual.append(latents.copy())
        return np.zeros((len(latents), 16384))

    monkeypatch.setattr(loaded.decoder, "predict", capture)
    monkeypatch.setattr("minecraft_skin_gan.bundle.load_bundle", lambda _path: loaded)
    generate_bundle(bundle, tmp_path / "output", count=65, seed=15)
    np.testing.assert_array_equal(probes, np.concatenate(actual))
    for kwargs in ({"count": 0}, {"seed": -1}, {"sampler": "invalid"}):
        with pytest.raises(ValueError):
            sample_latents(loaded, **kwargs)


@pytest.mark.parametrize("value", [None, "unknown", "z" * 64])
def test_loaded_dataset_fingerprint_is_required(sources, tmp_path, value):
    bundle = make_bundle(sources, tmp_path)
    metadata_path = bundle / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["dataset_fingerprint"] = value
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="fingerprint"):
        load_bundle(bundle)
