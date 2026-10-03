"""Maintained training phase budgets, isolated outputs and honest continuation."""

import json

import numpy as np
import pytest

from generate_skin import keras
from minecraft_skin_gan import training


@pytest.fixture
def tiny_data(tmp_path):
    path = tmp_path / "dataset.npz"
    rng = np.random.default_rng(3)
    np.savez(
        path,
        arr_0=rng.integers(0, 256, (4, 64, 64, 4), dtype=np.uint8),
        arr_1=rng.integers(0, 256, (2, 64, 64, 4), dtype=np.uint8),
    )
    return path


def tiny_factory(config):
    from types import SimpleNamespace

    images = keras.Input(shape=(64, 64, 4))
    codes = keras.layers.Dense(config.encoded_dim)(keras.layers.GlobalAveragePooling2D()(images))
    encoder = keras.Model(images, codes)
    latent = keras.Input(shape=(config.encoded_dim,))
    decoded = keras.layers.Dense(4, activation="sigmoid")(latent)
    decoder = keras.Model(
        latent, keras.layers.Reshape((64, 64, 4))(keras.layers.RepeatVector(4096)(decoded))
    )
    discriminator = keras.Sequential(
        [
            keras.Input(shape=(64, 64, 4)),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    autoencoder = keras.Model(images, decoder(encoder(images)))
    adversary = keras.Model(latent, discriminator(decoder(latent)))
    return SimpleNamespace(
        encoder=encoder,
        decoder=decoder,
        discriminator=discriminator,
        autoencoder=autoencoder,
        decoder_discriminator=adversary,
    )


def test_real_training_budgets_and_complete_resume(tiny_data, tmp_path, monkeypatch):
    monkeypatch.setattr(training, "_create_models", tiny_factory)
    config = training.TrainingConfig(
        tiny_data,
        tmp_path / "run",
        encoded_dim=2,
        ae_epochs=1,
        discriminator_epochs=1,
        gan_steps=2,
        batch_size=2,
    )
    report = training.run_training(config)
    metrics = json.loads(report.read_text())
    assert metrics["completed"] == {"ae": 1, "discriminator": 1, "gan": 2}
    assert metrics["validation_mse"] >= 0
    assert metrics["host_peak_bytes"] > 0
    assert metrics["parameter_counts"]["autoencoder"] > 0
    environment = json.loads((config.run_path / "configuration.json").read_text())
    assert environment["package_versions"]["keras"] == keras.__version__
    assert environment["device_details"][0]["platform"] == "cpu"
    assert environment["system"]["machine"]
    assert (config.run_path / "bundle/metadata.json").exists()
    assert training.run_training(config, resume=True) == report
    changed = dict(environment["package_versions"], keras="0.0.0")
    monkeypatch.setattr(training, "_package_versions", lambda: changed)
    with pytest.raises(ValueError, match="Resolved package versions"):
        training.run_training(config, resume=True)
    with pytest.raises(FileExistsError):
        training.run_training(config)


@pytest.mark.parametrize("phase", ["ae", "discriminator", "gan"])
def test_interruption_resume_matches_uninterrupted(tiny_data, tmp_path, monkeypatch, phase):
    monkeypatch.setattr(training, "_create_models", tiny_factory)
    common = dict(
        encoded_dim=2,
        ae_epochs=2,
        discriminator_epochs=2,
        gan_steps=2,
        batch_size=2,
        checkpoint_interval=1,
    )
    reference = training.TrainingConfig(tiny_data, tmp_path / "reference", **common)
    training.run_training(reference)
    interrupted = training.TrainingConfig(tiny_data, tmp_path / "interrupted", **common)
    original = training._checkpoint

    def stop_after_first_gan(*args, **kwargs):
        original(*args, **kwargs)
        if args[2]["completed"][phase] == 1:
            raise RuntimeError("interrupted")

    monkeypatch.setattr(training, "_checkpoint", stop_after_first_gan)
    with pytest.raises(RuntimeError, match="interrupted"):
        training.run_training(interrupted)
    monkeypatch.setattr(training, "_checkpoint", original)
    training.run_training(interrupted, resume=True)
    one = keras.models.load_model(reference.run_path / "models/decoder.keras", compile=False)
    two = keras.models.load_model(interrupted.run_path / "models/decoder.keras", compile=False)
    for before, after in zip(one.get_weights(), two.get_weights(), strict=True):
        np.testing.assert_allclose(before, after, rtol=1e-6, atol=1e-7)


def test_resume_mismatch_rejected_before_models(tiny_data, tmp_path, monkeypatch):
    monkeypatch.setattr(training, "_create_models", tiny_factory)
    config = training.TrainingConfig(
        tiny_data, tmp_path / "run", encoded_dim=2, ae_epochs=0, discriminator_epochs=0, gan_steps=0
    )
    training.run_training(config)
    changed = training.TrainingConfig(
        tiny_data, config.run_path, encoded_dim=3, ae_epochs=0, discriminator_epochs=0, gan_steps=0
    )
    monkeypatch.setattr(
        training, "_create_models", lambda _cfg: pytest.fail("must validate before construction")
    )
    with pytest.raises(ValueError, match="configuration"):
        training.run_training(changed, resume=True)
    tiny_data.write_bytes(b"changed")
    with pytest.raises(ValueError, match="fingerprint"):
        training.run_training(config, resume=True)


@pytest.mark.parametrize(
    "options",
    [
        {"batch_size": 1},
        {"gan_steps": -1},
        {"device": "invalid"},
        {"bandwidth": 0},
        {"ae_learning_rate": -1},
    ],
)
def test_invalid_configuration(tiny_data, tmp_path, options):
    with pytest.raises(ValueError):
        training.run_training(training.TrainingConfig(tiny_data, tmp_path / "run", **options))
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("kind", ["missing", "npy", "shape", "dtype"])
def test_invalid_dataset_contract(tiny_data, tmp_path, kind):
    if kind == "missing":
        np.savez(tiny_data, other=np.zeros(1))
    elif kind == "npy":
        with tiny_data.open("wb") as source:
            np.save(source, np.zeros(1))
    else:
        values = np.zeros(
            (2, 64, 64, 3) if kind == "shape" else (2, 64, 64, 4),
            dtype=np.uint8 if kind == "shape" else np.float32,
        )
        np.savez(tiny_data, arr_0=values, arr_1=values)
    with pytest.raises(ValueError, match="Dataset"):
        training.run_training(training.TrainingConfig(tiny_data, tmp_path / "run"))
    assert not (tmp_path / "run").exists()


def test_checkpoint_retention_and_fingerprint(tiny_data, tmp_path, monkeypatch):
    monkeypatch.setattr(training, "_create_models", tiny_factory)
    config = training.TrainingConfig(
        tiny_data,
        tmp_path / "run",
        encoded_dim=2,
        ae_epochs=1,
        discriminator_epochs=1,
        gan_steps=1,
        batch_size=2,
    )
    training.run_training(config)
    snapshots = list((config.run_path / "checkpoints").iterdir())
    assert len(snapshots) == 1
    (config.run_path / "metrics.json").unlink()
    (snapshots[0] / "state.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="fingerprint"):
        training.run_training(config, resume=True)


def test_interrupted_final_report_reuses_verified_bundle(tiny_data, tmp_path, monkeypatch):
    monkeypatch.setattr(training, "_create_models", tiny_factory)
    config = training.TrainingConfig(
        tiny_data, tmp_path / "run", encoded_dim=2, ae_epochs=0, discriminator_epochs=0, gan_steps=0
    )
    original = training._write_json

    def interrupt(path, data):
        if path.name == "metrics.json":
            raise RuntimeError("final report interrupted")
        original(path, data)

    monkeypatch.setattr(training, "_write_json", interrupt)
    with pytest.raises(RuntimeError):
        training.run_training(config)
    monkeypatch.setattr(training, "_write_json", original)
    assert training.run_training(config, resume=True).exists()


@pytest.mark.parametrize("phase", ["ae", "discriminator", "gan"])
def test_fresh_process_resume(tiny_data, tmp_path, monkeypatch, phase):
    import os
    import subprocess
    import sys
    from pathlib import Path

    monkeypatch.setattr(training, "_create_models", tiny_factory)
    config = training.TrainingConfig(
        tiny_data,
        tmp_path / "run",
        encoded_dim=2,
        ae_epochs=2,
        discriminator_epochs=2,
        gan_steps=2,
        batch_size=2,
        checkpoint_interval=1,
    )
    from dataclasses import replace

    reference = replace(config, run_path=tmp_path / "reference")
    training.run_training(reference)
    original = training._checkpoint

    def interrupt(*args, **kwargs):
        original(*args, **kwargs)
        if args[2]["completed"][phase] == 1:
            raise RuntimeError("interrupted")

    monkeypatch.setattr(training, "_checkpoint", interrupt)
    with pytest.raises(RuntimeError):
        training.run_training(config)
    code = """import importlib.util, json, sys
from pathlib import Path
from minecraft_skin_gan import training
spec = importlib.util.spec_from_file_location('fixture_models', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
training._create_models = module.tiny_factory
config = json.loads((Path(sys.argv[2])/'configuration.json').read_text())['config']
training.run_training(training.TrainingConfig(**config), resume=True)
"""
    subprocess.run(
        [sys.executable, "-c", code, str(Path(__file__).resolve()), str(config.run_path)],
        check=True,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
    )
    report = json.loads((config.run_path / "metrics.json").read_text())
    assert report["completed"]["gan"] == 2
    assert len(list((config.run_path / "checkpoints").iterdir())) == 1
    actual_snapshot = next((config.run_path / "checkpoints").iterdir())
    expected_snapshot = next((reference.run_path / "checkpoints").iterdir())
    with (
        np.load(actual_snapshot / "state.npz") as actual,
        np.load(expected_snapshot / "state.npz") as expected,
    ):
        assert set(actual.files) == set(expected.files)
        for key in actual.files:
            np.testing.assert_allclose(
                actual[key], expected[key], rtol=1e-6, atol=1e-7, err_msg=key
            )
    actual_state = json.loads((actual_snapshot / "state.json").read_text())
    expected_state = json.loads((expected_snapshot / "state.json").read_text())
    assert actual_state["rng"] == expected_state["rng"]
    assert actual_state["completed"] == expected_state["completed"]
    assert actual_state["best_validation"] == pytest.approx(
        expected_state["best_validation"], rel=1e-6
    )
    for name in ("encoder", "decoder", "discriminator"):
        actual = keras.models.load_model(config.run_path / f"models/{name}.keras", compile=False)
        expected = keras.models.load_model(
            reference.run_path / f"models/{name}.keras", compile=False
        )
        for before, after in zip(actual.get_weights(), expected.get_weights(), strict=True):
            np.testing.assert_allclose(before, after, rtol=1e-6, atol=1e-7)


def test_warm_start_weights_and_reset_progress(tiny_data, tmp_path, monkeypatch):
    monkeypatch.setattr(training, "_create_models", tiny_factory)
    first = training.TrainingConfig(
        tiny_data,
        tmp_path / "first",
        encoded_dim=2,
        ae_epochs=1,
        discriminator_epochs=0,
        gan_steps=0,
        batch_size=2,
    )
    training.run_training(first)
    second = training.TrainingConfig(
        tiny_data,
        tmp_path / "second",
        encoded_dim=2,
        seed=4,
        ae_epochs=0,
        discriminator_epochs=0,
        gan_steps=0,
    )
    training.run_training(second, warm_start=first.run_path)
    metadata = json.loads((second.run_path / "configuration.json").read_text())
    assert set(metadata["warm_start"]) == {"encoder", "decoder", "discriminator"}
    report = json.loads((second.run_path / "metrics.json").read_text())
    assert report["completed"] == {"ae": 0, "discriminator": 0, "gan": 0}
    before = keras.models.load_model(first.run_path / "models/decoder.keras", compile=False)
    after = keras.models.load_model(second.run_path / "models/decoder.keras", compile=False)
    for one, two in zip(before.get_weights(), after.get_weights(), strict=True):
        np.testing.assert_array_equal(one, two)
    with pytest.raises(ValueError, match="mutually exclusive"):
        training.run_training(second, resume=True, warm_start=first.run_path)
    with pytest.raises(ValueError, match="Warm start"):
        training.run_training(
            training.TrainingConfig(tiny_data, tmp_path / "bad"), warm_start=tmp_path / "absent"
        )


def test_generator_updates_freeze_discriminator(tiny_data, tmp_path, monkeypatch):
    calls = []

    def factory(config):
        models = tiny_factory(config)
        original = models.decoder_discriminator.train_on_batch

        def checked(*args, **kwargs):
            before = [weight.copy() for weight in models.discriminator.get_weights()]
            result = original(*args, **kwargs)
            for old, new in zip(before, models.discriminator.get_weights(), strict=True):
                np.testing.assert_array_equal(old, new)
            calls.append(True)
            return result

        models.decoder_discriminator.train_on_batch = checked
        return models

    monkeypatch.setattr(training, "_create_models", factory)
    training.run_training(
        training.TrainingConfig(
            tiny_data,
            tmp_path / "run",
            encoded_dim=2,
            ae_epochs=0,
            discriminator_epochs=0,
            gan_steps=1,
            batch_size=2,
        )
    )
    assert calls == [True]


def test_best_validation_weights_publish_with_checkpoint(tiny_data, tmp_path, monkeypatch):
    from dataclasses import replace

    monkeypatch.setattr(training, "_create_models", tiny_factory)
    config = training.TrainingConfig(
        tiny_data,
        tmp_path / "run",
        encoded_dim=2,
        ae_epochs=3,
        discriminator_epochs=0,
        gan_steps=0,
        batch_size=2,
    )
    reference = replace(config, run_path=tmp_path / "reference")
    training.run_training(reference)
    original = training._write_json
    calls = 0

    def interrupt_pointer(path, payload):
        nonlocal calls
        if path.name == "checkpoint.json":
            calls += 1
            if calls == 2:
                raise RuntimeError("crash before checkpoint cursor publication")
        original(path, payload)

    monkeypatch.setattr(training, "_write_json", interrupt_pointer)
    with pytest.raises(RuntimeError, match="cursor"):
        training.run_training(config)
    assert not (config.run_path / "best_ae.weights.h5").exists()
    # A legacy, corrupt standalone best file cannot influence new continuation.
    (config.run_path / "best_ae.weights.h5").write_bytes(b"corrupt legacy file")
    monkeypatch.setattr(training, "_write_json", original)
    training.run_training(config, resume=True)
    for name in ("encoder", "decoder"):
        before = keras.models.load_model(reference.run_path / f"models/{name}.keras", compile=False)
        after = keras.models.load_model(config.run_path / f"models/{name}.keras", compile=False)
        for left, right in zip(before.get_weights(), after.get_weights(), strict=True):
            np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-7)


def test_discriminator_epoch_loss_is_sample_weighted(tiny_data, tmp_path, monkeypatch):
    with np.load(tiny_data) as dataset:
        train, validation = dataset["arr_0"][:3], dataset["arr_1"]
    np.savez(tiny_data, arr_0=train, arr_1=validation)
    values = iter([1.0, 3.0, 10.0, 14.0])

    def factory(config):
        models = tiny_factory(config)
        original = models.discriminator.train_on_batch

        def checked(*args, **kwargs):
            original(*args, **kwargs)
            return {"loss": next(values), "accuracy": 0.5}

        models.discriminator.train_on_batch = checked
        return models

    monkeypatch.setattr(training, "_create_models", factory)
    report = training.run_training(
        training.TrainingConfig(
            tiny_data,
            tmp_path / "run",
            encoded_dim=2,
            ae_epochs=0,
            discriminator_epochs=2,
            gan_steps=0,
            batch_size=2,
        )
    )
    history = json.loads(report.read_text())["history"]
    assert [epoch["loss"] for epoch in history] == pytest.approx([5 / 3, 34 / 3])
    assert all(epoch["loss_aggregation"] == "sample_weighted_epoch_mean" for epoch in history)


def test_checkpoint_selection_and_rng_metadata_are_fingerprinted(tiny_data, tmp_path, monkeypatch):
    monkeypatch.setattr(training, "_create_models", tiny_factory)
    config = training.TrainingConfig(
        tiny_data,
        tmp_path / "run",
        encoded_dim=2,
        ae_epochs=1,
        discriminator_epochs=0,
        gan_steps=0,
        batch_size=2,
    )
    training.run_training(config)
    (config.run_path / "metrics.json").unlink()
    snapshot = next((config.run_path / "checkpoints").iterdir())
    metadata = json.loads((snapshot / "state.json").read_text())
    metadata["best_validation"] = 100.0
    (snapshot / "state.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="state fingerprint"):
        training.run_training(config, resume=True)


def test_runner_handles_precompiled_legacy_accuracy_metrics(tiny_data, tmp_path, monkeypatch):
    def factory(config):
        models = tiny_factory(config)
        models.discriminator.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
        models.discriminator.trainable = False
        models.decoder_discriminator.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
        return models

    monkeypatch.setattr(training, "_create_models", factory)
    path = training.run_training(
        training.TrainingConfig(
            tiny_data,
            tmp_path / "run",
            encoded_dim=2,
            ae_epochs=0,
            discriminator_epochs=1,
            gan_steps=1,
            batch_size=2,
        )
    )
    history = json.loads(path.read_text())["history"]
    assert np.isfinite(history[0]["loss"])
    assert np.isfinite(history[1]["discriminator_loss"])
    assert np.isfinite(history[1]["generator_loss"])


def test_actual_legacy_gae_recompiles_and_updates_with_named_metrics(tiny_data, tmp_path):
    from simple_gan import GAE

    keras.utils.set_random_seed(4)
    models = GAE(img_shape=(2, 2, 4), encoded_dim=2)
    config = training.TrainingConfig(tiny_data, tmp_path / "unused", encoded_dim=2)
    training._compile(models, config)
    real = np.full((2, 2, 2, 4), 0.6, dtype=np.float32)
    models.discriminator.trainable = True
    discriminator_metrics = models.discriminator.train_on_batch(
        real, np.ones((2, 1)), return_dict=True
    )
    models.discriminator.trainable = False
    frozen = [value.copy() for value in models.discriminator.get_weights()]
    before = [value.copy() for value in models.decoder.get_weights()]
    generator_metrics = models.decoder_discriminator.train_on_batch(
        np.full((2, 2), 0.4), np.ones((2, 1)), return_dict=True
    )
    assert set(discriminator_metrics) == {"loss", "accuracy"}
    assert set(generator_metrics) == {"loss", "accuracy"}
    assert all(np.isfinite(value) for value in discriminator_metrics.values())
    assert all(np.isfinite(value) for value in generator_metrics.values())
    for old, new in zip(frozen, models.discriminator.get_weights(), strict=True):
        np.testing.assert_array_equal(old, new)
    assert any(
        not np.array_equal(old, new)
        for old, new in zip(before, models.decoder.get_weights(), strict=True)
    )


def test_non_jax_backend_fails_before_artifacts(tiny_data, tmp_path, monkeypatch):
    monkeypatch.setattr(keras.backend, "backend", lambda: "tensorflow")
    with pytest.raises(ValueError, match="JAX backend"):
        training.run_training(training.TrainingConfig(tiny_data, tmp_path / "run"))
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    "options", [{"architecture": "unknown"}, {"reconstruction_loss": "unknown"}]
)
def test_invalid_model_variant_rejected(tiny_data, tmp_path, options):
    with pytest.raises(ValueError):
        training.run_training(training.TrainingConfig(tiny_data, tmp_path / "run", **options))
    assert not (tmp_path / "run").exists()


def test_conv_runner_exports_reproducible_native_bundle(tiny_data, tmp_path):
    from minecraft_skin_gan.bundle import generate_bundle, load_bundle

    config = training.TrainingConfig(
        tiny_data,
        tmp_path / "run",
        encoded_dim=2,
        architecture="conv",
        reconstruction_loss="visible_rgba",
        ae_epochs=1,
        discriminator_epochs=1,
        gan_steps=1,
        batch_size=2,
    )
    report = training.run_training(config)
    assert json.loads(report.read_text())["completed"] == {"ae": 1, "discriminator": 1, "gan": 1}
    configuration = json.loads((config.run_path / "configuration.json").read_text())["config"]
    assert configuration["architecture"] == "conv"
    assert configuration["reconstruction_loss"] == "visible_rgba"
    assert load_bundle(config.run_path / "bundle").decoder.name == "conv_decoder_v1"
    outputs = generate_bundle(config.run_path / "bundle", tmp_path / "skins", count=2, seed=7)
    assert len(outputs) == 2 and all(path.is_file() for path in outputs)
    assert training.run_training(config, resume=True) == report


def test_older_dense_configuration_migrates_only_default_variant(tiny_data, tmp_path, monkeypatch):
    from dataclasses import replace

    monkeypatch.setattr(training, "_create_models", tiny_factory)
    config = training.TrainingConfig(
        tiny_data, tmp_path / "run", encoded_dim=2, ae_epochs=0, discriminator_epochs=0, gan_steps=0
    )
    report = training.run_training(config)
    path = config.run_path / "configuration.json"
    metadata = json.loads(path.read_text())
    metadata["config"].pop("architecture")
    metadata["config"].pop("reconstruction_loss")
    path.write_text(json.dumps(metadata))
    assert training.run_training(config, resume=True) == report
    for options in ({"architecture": "conv"}, {"reconstruction_loss": "visible_rgba"}):
        with pytest.raises(ValueError, match="configuration"):
            training.run_training(replace(config, **options), resume=True)


def test_execution_policy_fingerprints_flags_without_recording_contents(
    tiny_data, tmp_path, monkeypatch
):
    import hashlib

    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_autotune_level=0")
    monkeypatch.setattr(training, "_create_models", tiny_factory)
    config = training.TrainingConfig(
        tiny_data, tmp_path / "run", encoded_dim=2, ae_epochs=0, discriminator_epochs=0, gan_steps=0
    )
    training.run_training(config)
    text = (config.run_path / "configuration.json").read_text()
    policy = json.loads(text)["execution_policy"]
    assert policy["xla_flags_sha256"] == hashlib.sha256(b"--xla_gpu_autotune_level=0").hexdigest()
    assert "--xla_gpu_autotune_level=0" not in text
    assert policy["jax_enable_x64"] is training.jax.config.jax_enable_x64
    assert (
        policy["jax_default_matmul_precision"] == training.jax.config.jax_default_matmul_precision
    )


@pytest.mark.parametrize("changed", ["flags", "x64", "matmul"])
def test_resume_rejects_changed_execution_policy_before_model_or_data_loading(
    tiny_data, tmp_path, monkeypatch, changed
):
    from contextlib import nullcontext

    monkeypatch.delenv("XLA_FLAGS", raising=False)
    monkeypatch.setattr(training, "_create_models", tiny_factory)
    config = training.TrainingConfig(
        tiny_data,
        tmp_path / "run",
        encoded_dim=2,
        ae_epochs=1,
        discriminator_epochs=0,
        gan_steps=0,
        batch_size=2,
    )
    training.run_training(config)
    paths = sorted(path for path in config.run_path.rglob("*") if path.is_file())
    before = {path: path.read_bytes() for path in paths}

    def forbidden(*args):
        raise AssertionError("Changed policy must fail before data/model loading")

    monkeypatch.setattr(training, "_data", forbidden)
    monkeypatch.setattr(training, "_create_models", forbidden)
    if changed == "flags":
        monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_autotune_level=0")
        context = nullcontext()
    elif changed == "x64":
        context = training.jax.enable_x64(not training.jax.config.jax_enable_x64)
    else:
        precision = (
            "highest"
            if training.jax.config.jax_default_matmul_precision != "highest"
            else "default"
        )
        context = training.jax.default_matmul_precision(precision)
    with context, pytest.raises(ValueError, match="execution policy"):
        training.run_training(config, resume=True)
    assert {path: path.read_bytes() for path in paths} == before


def test_older_configuration_retains_narrower_execution_policy_checks(
    tiny_data, tmp_path, monkeypatch
):
    monkeypatch.setattr(training, "_create_models", tiny_factory)
    config = training.TrainingConfig(
        tiny_data, tmp_path / "run", encoded_dim=2, ae_epochs=0, discriminator_epochs=0, gan_steps=0
    )
    metrics = training.run_training(config)
    path = config.run_path / "configuration.json"
    payload = json.loads(path.read_text())
    payload.pop("execution_policy", None)
    path.write_text(json.dumps(payload))
    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_autotune_level=0")
    assert training.run_training(config, resume=True) == metrics
