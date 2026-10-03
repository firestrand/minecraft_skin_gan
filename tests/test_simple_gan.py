"""Regression tests for numerical behavior, actual Keras updates and orchestration."""

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from sklearn.neighbors import KernelDensity

import simple_gan as gan

keras = gan.keras


@pytest.fixture(scope="module")
def model():
    keras.utils.set_random_seed(1976)
    ann = gan.GAE(img_shape=(2, 2), encoded_dim=2)
    ann.kde = KernelDensity(bandwidth=3.16).fit(np.array([[0.0, 0.0], [1.0, 1.0]]))
    return ann


def changed(before, model):
    return any(not np.array_equal(a, b) for a, b in zip(before, model.get_weights(), strict=True))


def test_architecture_prediction_and_optimizer_ownership(model):
    assert [
        layer.units for layer in model.encoder.layers if isinstance(layer, keras.layers.Dense)
    ] == [1000, 1000, 2]
    assert [
        layer.units for layer in model.decoder.layers if isinstance(layer, keras.layers.Dense)
    ] == [1000, 1000, 4]
    assert [
        layer.units for layer in model.discriminator.layers if isinstance(layer, keras.layers.Dense)
    ] == [1000, 1000, 1]
    assert float(model.optimizer.learning_rate.numpy()) == pytest.approx(0.001)
    assert float(model.optimizer_discriminator.learning_rate.numpy()) == pytest.approx(0.00001)
    assert model.autoencoder.optimizer is not model.discriminator.optimizer
    images = model.generate(3)
    assert images.shape == (3, 2, 2)
    assert np.all((images >= 0) & (images <= 1))


def test_real_updates_and_generator_freezes_discriminator(model):
    images = np.array([[[0.1, 0.9], [0.2, 0.8]], [[0.9, 0.1], [0.8, 0.2]]], dtype=np.float32)
    before_encoder = model.encoder.get_weights()
    before_decoder = model.decoder.get_weights()
    model.autoencoder.fit(images, images, epochs=1, batch_size=2, verbose=0)
    assert changed(before_encoder, model.encoder)
    assert changed(before_decoder, model.decoder)
    before_discriminator = model.discriminator.get_weights()
    model.discriminator.trainable = True
    model.discriminator.train_on_batch(images, np.ones((2, 1)))
    assert changed(before_discriminator, model.discriminator)
    model.discriminator.trainable = False
    before_discriminator = model.discriminator.get_weights()
    before_decoder = model.decoder.get_weights()
    before_encoder = model.encoder.get_weights()
    model.decoder_discriminator.train_on_batch(np.array([[1.0, 2.0], [-1.0, 1.0]]), np.ones((2, 1)))
    assert changed(before_decoder, model.decoder)
    assert not changed(before_discriminator, model.discriminator)
    assert not changed(before_encoder, model.encoder)
    model.trainGAN(images, epochs=2, batch_size=2)
    assert not model.discriminator.trainable


def test_real_serialization_and_checkpoint(model, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("models").mkdir()
    images = np.ones((2, 2, 2), dtype=np.float32)
    expected = model.autoencoder.predict(images, verbose=0)
    model.autoencoder.save("models/autoencoder.keras")
    restored = keras.models.load_model("models/autoencoder.keras")
    np.testing.assert_allclose(restored.predict(images, verbose=0), expected, atol=1e-7)
    model.autoencoder.fit(
        images, images, epochs=1, batch_size=2, callbacks=gan._callbacks("autoencoder"), verbose=0
    )
    assert Path("models/weights_autoencoder.01.weights.h5").is_file()
    expected = model.autoencoder.get_weights()
    model.autoencoder.set_weights([np.zeros_like(x) for x in expected])
    assert gan._resume_weights(model.autoencoder, "autoencoder") == 1
    for actual, saved in zip(model.autoencoder.get_weights(), expected, strict=True):
        np.testing.assert_array_equal(actual, saved)
    model.autoencoder.save("models/weights_autoencoder.02.hdf5")
    assert gan._resume_weights(model.autoencoder, "autoencoder") == 2


def test_nearest_and_likelihood():
    images = np.array([[[1, 2], [3, 4]], [[4, 3], [2, 1]], [[1, 2], [3, 4]]], dtype=float)
    np.testing.assert_array_equal(gan.findNearest(images, images[0] + 0.1), images[0])
    with pytest.raises(ValueError):
        gan.findNearest(np.empty((0, 2, 2)), images[0])
    generated = np.arange(12).reshape(6, 2)
    test = np.array([[3.0, 4.0], [5.0, 6.0]])
    expected = KernelDensity(bandwidth=1.0).fit(generated).score_samples(test).mean()
    assert gan.approximateLogLiklihood(generated, test, [1.0]) == pytest.approx(expected)
    assert np.isfinite(gan.approximateLogLiklihood(generated, test))
    assert gan.GAE.mean_log_likelihood(test) is None


def test_checkpoint_numerical_order_legacy_and_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = Mock()
    assert gan._resume_weights(model, "autoencoder") == 0
    Path("models").mkdir()
    for name in (
        "weights_autoencoder.9.weights.h5",
        "weights_mnist_autoencoder.100.hdf5",
        "weights_autoencoder.bad.h5",
        "weights_autoencoder.99.txt",
    ):
        Path("models", name).touch()
    assert gan._resume_weights(model, "autoencoder") == 100
    model.load_weights.assert_called_once_with(Path("models/weights_mnist_autoencoder.100.hdf5"))


def harness():
    ann = object.__new__(gan.GAE)
    ann.img_shape = (2, 2)
    ann.autoencoder = Mock()
    ann.encoder = Mock()
    ann.encoder.predict.return_value = np.array([[0.0, 0.0], [1.0, 1.0]])
    ann.discriminator = Mock()
    ann.discriminator.train_on_batch.return_value = [0.5, 0.5]
    ann.decoder_discriminator = Mock()
    ann.decoder_discriminator.train_on_batch.return_value = [0.5, 0.5]
    ann.decoder = Mock()
    ann.decoder.predict.return_value = np.zeros((2, 2, 2))
    ann.generate = Mock(side_effect=lambda n: np.zeros((n, 2, 2)))
    ann.generateAndPlot = Mock()
    return ann


@pytest.mark.parametrize("completed", [0, 1, 2])
def test_training_orchestration(completed, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ann = harness()
    monkeypatch.setattr(gan, "_resume_weights", Mock(return_value=completed))
    images = np.ones((2, 2, 2))
    ann.train(images, batch_size=2, epochs=2)
    if completed < 2:
        ae_call = ann.autoencoder.fit.call_args
        np.testing.assert_array_equal(ae_call.args[0], images)
        np.testing.assert_array_equal(ae_call.args[1], images)
        assert ae_call.kwargs["initial_epoch"] == completed
        discr_call = ann.discriminator.fit.call_args
        np.testing.assert_array_equal(
            discr_call.args[0], np.concatenate([images, np.zeros_like(images)])
        )
        np.testing.assert_array_equal(discr_call.args[1], [[1], [1], [0], [0]])
        assert discr_call.kwargs["initial_epoch"] == completed
    else:
        ann.autoencoder.fit.assert_not_called()
        ann.discriminator.fit.assert_not_called()
    assert ann.kde.bandwidth == 3.16
    assert not ann.discriminator.trainable
    assert ann.generateAndPlot.call_count == 2
    assert ann.discriminator.train_on_batch.call_count == 2
    for call in ann.discriminator.train_on_batch.call_args_list:
        assert call.args[0].shape == (1, 2, 2)
    np.testing.assert_array_equal(ann.discriminator.train_on_batch.call_args_list[0].args[1], [[1]])
    np.testing.assert_array_equal(ann.discriminator.train_on_batch.call_args_list[1].args[1], [[0]])
    np.testing.assert_array_equal(
        ann.decoder_discriminator.train_on_batch.call_args.args[1], [[1], [1]]
    )


@pytest.mark.parametrize("phase", ["fit", "batch"])
def test_discriminator_restored_on_error(phase, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ann = harness()
    ann.kde = KernelDensity().fit(np.zeros((2, 2)))
    getattr(
        ann.discriminator, "fit" if phase == "fit" else "train_on_batch"
    ).side_effect = RuntimeError("failure")
    with pytest.raises(RuntimeError, match="failure"):
        if phase == "fit":
            ann.train(np.ones((2, 2, 2)), batch_size=2, epochs=1)
        else:
            ann.trainGAN(np.ones((2, 2, 2)), epochs=1, batch_size=2)
    assert not ann.discriminator.trainable


def test_plotting_and_cleanup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gan.plt, "show", Mock())
    ann = harness()
    # Invoke class method because harness substitutes expensive plot orchestration.
    gan.GAE.generateAndPlot(ann, np.ones((2, 2, 2)), n=2, fileName="grid.png")
    assert Path("grid.png").is_file()
    assert not gan.plt.get_fignums()
    ann.decoder.predict.return_value = np.zeros((1, 2, 2))
    ann.imagegrid(3)
    assert ann.decoder.predict.call_count == 100
    np.testing.assert_array_equal(ann.decoder.predict.call_args_list[0].args[0], [[-2.5, -2.5]])
    np.testing.assert_array_equal(ann.decoder.predict.call_args_list[-1].args[0], [[2.0, 2.0]])
    assert Path("3.png").is_file()
    assert not gan.plt.get_fignums()
    ann.generate.side_effect = RuntimeError("generation failure")
    with pytest.raises(RuntimeError):
        gan.GAE.generateAndPlot(ann, np.ones((2, 2, 2)))
    assert not gan.plt.get_fignums()


def test_main_with_local_dataset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gan.plt, "show", Mock())
    Path("images").mkdir()
    pixels = np.full((11, 64, 64, 4), 128, dtype=np.uint8)
    np.savez("images/train_test.npz", pixels, pixels)
    ann = harness()
    ann.encoder.predict.return_value = np.zeros((11, 128))
    ann.autoencoder.predict.return_value = np.zeros((11, 64, 64, 4))
    ann.train = Mock()
    constructor = Mock(return_value=ann)
    monkeypatch.setattr(gan, "GAE", constructor)
    gan.main()
    constructor.assert_called_once_with(img_shape=(64, 64, 4), encoded_dim=128)
    np.testing.assert_allclose(ann.train.call_args.args[0], 128 / 255)
    assert ann.train.call_args.kwargs == {"epochs": 50}
    ann.autoencoder.save.assert_called_once_with("models/autoencoder.keras")
    ann.decoder.save.assert_called_once_with("models/decoder.keras")
    assert Path("images/results/mnist_gae_00.jpg").is_file()
    assert not gan.plt.get_fignums()


def test_real_train_and_resume(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ann = gan.GAE(img_shape=(2, 2), encoded_dim=2)
    monkeypatch.setattr(ann, "generateAndPlot", Mock())
    images = np.array([[[0.1, 0.9], [0.3, 0.7]], [[0.9, 0.1], [0.7, 0.3]]], dtype=np.float32)
    ann.train(images, epochs=1, batch_size=2)
    assert Path("models/weights_autoencoder.01.weights.h5").is_file()
    assert Path("models/weights_discriminator.01.weights.h5").is_file()
    autoencoder_steps = int(ann.optimizer.iterations.numpy())
    discriminator_steps = int(ann._discriminator_optimizer.iterations.numpy())
    assert autoencoder_steps == 1
    assert discriminator_steps == 4  # two initialization batches, two GAN batches
    ann.train(images, epochs=1, batch_size=2)
    assert int(ann.optimizer.iterations.numpy()) == autoencoder_steps
    assert int(ann._discriminator_optimizer.iterations.numpy()) == 4
    # Restoring a weight checkpoint also restores its optimizer state (two
    # initialization batches); the resumed GAN takes two fresh updates.
    assert not ann.discriminator.trainable


def test_discriminator_initialization_advances_and_seed_reproduces():
    keras.utils.set_random_seed(1976)
    first = gan.GAE.get_discriminator_model((2, 2))
    second = gan.GAE.get_discriminator_model((2, 2))
    first_dense = [layer for layer in first.layers if isinstance(layer, keras.layers.Dense)]
    # Equal-width hidden biases must be independent Gaussian draws, not copies.
    assert not np.array_equal(first_dense[0].bias.numpy(), first_dense[1].bias.numpy())
    for first_weight, second_weight in zip(first.get_weights(), second.get_weights(), strict=True):
        assert not np.array_equal(first_weight, second_weight)
    for layer in first_dense:
        assert layer.kernel_initializer.mean == 0.0
        assert layer.kernel_initializer.stddev == 0.01
        assert layer.bias_initializer.mean == 0.0
        assert layer.bias_initializer.stddev == 0.01
    keras.utils.set_random_seed(1976)
    repeated = gan.GAE.get_discriminator_model((2, 2))
    for first_weight, repeated_weight in zip(
        first.get_weights(), repeated.get_weights(), strict=True
    ):
        np.testing.assert_array_equal(first_weight, repeated_weight)
