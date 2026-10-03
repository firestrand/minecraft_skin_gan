"""Evaluation checks use fixed pixel contracts, not model-quality assumptions."""

import numpy as np
import pytest

from minecraft_skin_gan.evaluation import image_metrics, nearest_examples, reconstruction_metrics


def test_reconstruction_ignores_invisible_rgb_but_measures_alpha():
    original = np.zeros((2, 64, 64, 4), dtype=np.float32)
    original[:, 0, 0] = [1, 0, 0, 1]
    reconstructed = original.copy()
    reconstructed[:, 1, 1, :3] = 1
    result = reconstruction_metrics(original, reconstructed)
    assert result["visible_rgb_mse"] == 0
    assert result["alpha_mae"] == 0
    assert result["rgba_mse"] > 0
    reconstructed[:, 0, 0, 0] = 0
    assert reconstruction_metrics(original, reconstructed)["visible_rgb_mse"] == pytest.approx(
        1 / 3
    )


def test_duplicate_report_and_nearest_examples_preserve_indices():
    black = np.zeros((64, 64, 4), dtype=np.float32)
    white = np.ones_like(black)
    samples = np.stack([black, black, white])
    metrics = image_metrics(samples)
    assert metrics["exact_duplicate_count"] == 1
    assert metrics["unique_count"] == 2
    assert metrics["mean_pairwise_rgba_mse"] == pytest.approx(2 / 3)
    distances = nearest_examples(samples, np.stack([white, black]), batch_size=1)
    assert [row["training_index"] for row in distances] == [1, 1, 0]
    assert all(row["rgba_mse"] == 0 for row in distances)


@pytest.mark.parametrize(
    "bad",
    [
        np.empty((0, 64, 64, 4)),
        np.zeros((2, 4)),
        np.full((1, 64, 64, 4), np.nan),
        np.full((1, 64, 64, 4), 2),
    ],
)
def test_invalid_pixel_data_is_rejected(bad):
    with pytest.raises(ValueError):
        image_metrics(bad)


def test_mismatched_reconstruction_and_empty_neighbors_rejected():
    images = np.zeros((1, 64, 64, 4))
    with pytest.raises(ValueError, match="same shape"):
        reconstruction_metrics(images, np.zeros((2, 64, 64, 4)))
    with pytest.raises(ValueError):
        nearest_examples(images, images, batch_size=0)


def test_development_report_real_bundle_and_optional_reconstruction(tmp_path):
    import json

    from generate_skin import keras
    from minecraft_skin_gan.bundle import create_bundle, file_fingerprint
    from minecraft_skin_gan.evaluation import evaluate_bundle

    inputs = keras.Input(shape=(2,))
    pixels = keras.layers.Dense(
        4, kernel_initializer="zeros", bias_initializer=keras.initializers.Constant(0.5)
    )(inputs)
    decoder = keras.Model(
        inputs, keras.layers.Reshape((64, 64, 4))(keras.layers.RepeatVector(4096)(pixels))
    )
    decoder.save(tmp_path / "decoder.keras")
    np.savez(tmp_path / "codes.npz", codes=np.zeros((2, 2)))
    bundle = create_bundle(
        tmp_path / "decoder.keras",
        tmp_path / "codes.npz",
        tmp_path / "bundle",
        dataset_fingerprint="a" * 64,
    )
    original = np.full((2, 64, 64, 4), 128, dtype=np.uint8)
    np.savez(tmp_path / "data.npz", original, original)
    auto_inputs = keras.Input(shape=(64, 64, 4))
    autoencoder = keras.Model(auto_inputs, keras.layers.Activation("linear")(auto_inputs))
    autoencoder.save(tmp_path / "autoencoder.keras")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {"exposure": "development", "archive_sha256": file_fingerprint(tmp_path / "data.npz")}
        )
    )
    path = evaluate_bundle(
        bundle,
        tmp_path / "data.npz",
        tmp_path / "report",
        count=2,
        autoencoder_path=tmp_path / "autoencoder.keras",
        model_type="slim",
    )
    report = json.loads(path.read_text())
    assert report["gate"] == "development"
    assert report["human_quality"] == "unverified"
    assert len(report["skin_diagnostics"]) == 2
    assert report["skin_diagnostics"][0]["model_type"] == "slim"
    assert report["skin_diagnostics"][0]["alpha"]["base"]["texel_count"] == 1568
    assert report["skin_diagnostics"][0]["game_acceptance"] == "unverified"
    assert report["generated"]["unique_count"] == 1
    assert report["reconstruction"]["rgba_mse"] == 0
    assert report["reconstruction_count"] == 2
    assert report["autoencoder_sha256"] == file_fingerprint(tmp_path / "autoencoder.keras")
    assert report["exact_cross_split_groups"] == 1
    assert report["training_archive_match"] is False
    assert (path.parent / "generated.png").is_file()
    assert (path.parent / "reconstructed.png").is_file()
    with pytest.raises(FileExistsError):
        evaluate_bundle(bundle, tmp_path / "data.npz", path.parent)
    with pytest.raises(ValueError, match="model_type"):
        evaluate_bundle(bundle, tmp_path / "data.npz", tmp_path / "bad-model", model_type="auto")
    assert not (tmp_path / "bad-model").exists()
    with pytest.raises(ValueError, match="between"):
        evaluate_bundle(bundle, tmp_path / "data.npz", tmp_path / "invalid", count=65)
    (tmp_path / "manifest.json").write_text(json.dumps({"archive_sha256": "wrong"}))
    with pytest.raises(ValueError, match="manifest"):
        evaluate_bundle(bundle, tmp_path / "data.npz", tmp_path / "wrong-manifest", count=1)
    assert not (tmp_path / "wrong-manifest").exists()
    (tmp_path / "manifest.json").unlink()
    np.save(tmp_path / "array.npy", original)
    with pytest.raises(ValueError, match="NPZ"):
        evaluate_bundle(bundle, tmp_path / "array.npy", tmp_path / "array-out", count=1)
    np.savez(tmp_path / "missing.npz", wrong=original)
    with pytest.raises(ValueError, match="arr_0"):
        evaluate_bundle(bundle, tmp_path / "missing.npz", tmp_path / "missing-out", count=1)
    np.savez(tmp_path / "float.npz", original.astype(float), original.astype(float))
    with pytest.raises(ValueError, match="uint8"):
        evaluate_bundle(bundle, tmp_path / "float.npz", tmp_path / "float-out", count=1)


def test_evaluation_save_failure_leaves_no_output(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from PIL import Image

    import minecraft_skin_gan.bundle as bundle
    from minecraft_skin_gan.evaluation import evaluate_bundle

    generated = np.zeros((1, 64, 64, 4), dtype=np.float32)
    fake = SimpleNamespace(
        metadata={"dataset_fingerprint": "a" * 64},
        decoder=SimpleNamespace(predict=lambda *a, **k: generated),
    )
    monkeypatch.setattr(bundle, "load_bundle", lambda path: fake)
    monkeypatch.setattr(bundle, "sample_latents", lambda *a, **k: np.zeros((1, 2)))
    np.savez(tmp_path / "data.npz", generated.astype(np.uint8), generated.astype(np.uint8))

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(Image.Image, "save", fail)
    with pytest.raises(OSError, match="disk full"):
        evaluate_bundle(tmp_path, tmp_path / "data.npz", tmp_path / "report", count=1)
    assert not (tmp_path / "report").exists()
    assert not list(tmp_path.glob(".evaluation-*"))


def test_empty_alpha_and_single_image_have_defined_metrics():
    images = np.zeros((1, 64, 64, 4), dtype=np.float32)
    assert reconstruction_metrics(images, images)["visible_rgb_mse"] == 0
    assert image_metrics(images)["pair_count"] == 0


def test_bounded_uint8_neighbors_match_normalized_float_path():
    raw = np.full((3, 64, 64, 4), 128, dtype=np.uint8)
    raw[0] = 0
    samples = raw[1:].astype(np.float32) / 255
    expected = nearest_examples(samples, raw.astype(np.float32) / 255, batch_size=1)
    assert nearest_examples(samples, raw, batch_size=2) == expected
    with pytest.raises(ValueError):
        nearest_examples(samples, np.zeros((1, 2), dtype=np.uint8))
