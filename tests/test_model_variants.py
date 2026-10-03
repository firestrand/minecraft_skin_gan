"""Spatial architecture and visible-pixel objectives keep decoder contracts."""

import numpy as np
import pytest

from generate_skin import keras
from minecraft_skin_gan.models import create_conv_models, visible_rgba_loss


def test_visible_rgba_loss_ignores_hidden_rgb_and_penalizes_alpha():
    truth = np.array([[[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0]]]], dtype=np.float32)
    predicted = np.array([[[[0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]]], dtype=np.float32)
    # Visible RGB squared error sums to 1 over three visible channels; alpha agrees.
    assert float(visible_rgba_loss(truth, predicted)) == pytest.approx(1 / 3)
    predicted[0, 0, 0, 3] = 1.0
    # False opaque prediction is penalized independently of hidden RGB.
    assert float(visible_rgba_loss(truth, predicted)) == pytest.approx(1 / 3 + 1 / 2)


def test_fully_transparent_loss_is_finite_and_partial_alpha_weights_rgb():
    truth = np.zeros((1, 1, 2, 4), dtype=np.float32)
    predicted = np.ones_like(truth)
    assert float(visible_rgba_loss(truth, predicted)) == pytest.approx(1.0)
    truth[0, 0, 0] = [1.0, 0.0, 0.0, 0.5]
    truth[0, 0, 1] = [1.0, 0.0, 0.0, 1.0]
    predicted[:] = truth
    predicted[0, 0, 0, 0] = 0.0
    assert float(visible_rgba_loss(truth, predicted)) == pytest.approx(0.5 / 4.5)


def test_conv_real_updates_freeze_and_native_decoder_roundtrip(tmp_path):
    from minecraft_skin_gan.training import TrainingConfig, _compile

    keras.utils.set_random_seed(11)
    models = create_conv_models(encoded_dim=4)
    config = TrainingConfig(
        tmp_path / "unused.npz",
        tmp_path / "unused",
        architecture="conv",
        encoded_dim=4,
        reconstruction_loss="visible_rgba",
    )
    _compile(models, config)
    images = np.random.default_rng(2).random((2, 64, 64, 4)).astype(np.float32)
    before = [value.copy() for value in models.encoder.get_weights()]
    assert np.isfinite(models.autoencoder.train_on_batch(images, images))
    assert any(
        not np.array_equal(a, b) for a, b in zip(before, models.encoder.get_weights(), strict=True)
    )
    models.discriminator.trainable = True
    models.discriminator.train_on_batch(images, np.ones((2, 1)), return_dict=True)
    models.discriminator.trainable = False
    frozen = [value.copy() for value in models.discriminator.get_weights()]
    decoder_before = [value.copy() for value in models.decoder.get_weights()]
    latent = np.ones((2, 4), dtype=np.float32)
    metrics = models.decoder_discriminator.train_on_batch(latent, np.ones((2, 1)), return_dict=True)
    assert np.isfinite(metrics["loss"])
    for old, new in zip(frozen, models.discriminator.get_weights(), strict=True):
        np.testing.assert_array_equal(old, new)
    assert any(
        not np.array_equal(a, b)
        for a, b in zip(decoder_before, models.decoder.get_weights(), strict=True)
    )
    probe = models.decoder.predict(latent, verbose=0)
    assert probe.shape == (2, 64, 64, 4)
    assert np.isfinite(probe).all() and probe.min() >= 0 and probe.max() <= 1
    path = tmp_path / "decoder.keras"
    models.decoder.save(path)
    loaded = keras.models.load_model(path, compile=False, safe_mode=True)
    np.testing.assert_allclose(loaded.predict(latent, verbose=0), probe, rtol=1e-6, atol=1e-7)
    models.autoencoder.save(tmp_path / "autoencoder.keras")
    restored = keras.models.load_model(tmp_path / "autoencoder.keras", safe_mode=True)
    assert np.isfinite(restored.evaluate(images, images, verbose=0))
    assert models.autoencoder.count_params() < 5_000_000


def test_conv_invalid_latent_dimension():
    for invalid in (0, -1, True):
        with pytest.raises(ValueError):
            create_conv_models(encoded_dim=invalid)
