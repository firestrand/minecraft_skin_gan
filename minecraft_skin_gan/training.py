"""Isolated, checkpointed training with explicit budgets and validation selection."""

import hashlib
import json
import math
import os
import platform
import re
import resource
import shutil
import time
from dataclasses import asdict, dataclass
from importlib.metadata import distributions
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, TypedDict
from uuid import uuid4

import jax
import numpy as np
from sklearn.neighbors import KernelDensity

from generate_skin import keras
from minecraft_skin_gan.bundle import create_bundle, file_fingerprint, load_bundle
from minecraft_skin_gan.models import create_conv_models, visible_rgba_loss
from simple_gan import GAE


class Completed(TypedDict):
    ae: int
    discriminator: int
    gan: int


class TrainingState(TypedDict):
    completed: Completed
    history: list[dict[str, Any]]
    best_validation: float | None


@dataclass(frozen=True)
class TrainingConfig:
    data_path: Path
    run_path: Path
    seed: int = 1976
    encoded_dim: int = 128
    ae_epochs: int = 10
    discriminator_epochs: int = 10
    gan_steps: int = 830
    batch_size: int = 128
    device: str = "cpu"
    bandwidth: float = 3.16
    ae_learning_rate: float = 0.001
    discriminator_learning_rate: float = 0.001
    generator_learning_rate: float = 0.00001
    discriminator_updates: int = 1
    checkpoint_interval: int = 50
    architecture: str = "dense"
    reconstruction_loss: str = "rgba_mse"


def _write_json(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _configuration(config: TrainingConfig) -> dict[str, Any]:
    value = asdict(config)
    value["data_path"] = str(Path(config.data_path).resolve())
    value["run_path"] = str(Path(config.run_path).resolve())
    return value


def _package_versions() -> dict[str, str]:
    """Record resolved distributions, without paths or environment secrets."""
    return dict(
        sorted(
            (distribution.metadata["Name"].lower(), distribution.version)
            for distribution in distributions()
        )
    )


def _execution_policy() -> dict[str, Any]:
    """Fingerprint compiler flags without retaining their potentially private contents."""
    return {
        "xla_flags_sha256": hashlib.sha256(os.environ.get("XLA_FLAGS", "").encode()).hexdigest(),
        "jax_enable_x64": jax.config.jax_enable_x64,
        "jax_default_matmul_precision": jax.config.jax_default_matmul_precision,
    }


def _validate(config: TrainingConfig) -> None:
    if config.architecture not in ("dense", "conv"):
        raise ValueError("Architecture must be dense or conv")
    if config.reconstruction_loss not in ("rgba_mse", "visible_rgba"):
        raise ValueError("Reconstruction loss must be rgba_mse or visible_rgba")
    for name in ("encoded_dim", "batch_size", "discriminator_updates", "checkpoint_interval"):
        value = getattr(config, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < (2 if name == "batch_size" else 1)
        ):
            raise ValueError(f"{name} must be a positive integer (batch size >=2)")
    for name in ("ae_epochs", "discriminator_epochs", "gan_steps", "seed"):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if config.seed >= 2**32:
        raise ValueError("Seed exceeds 32-bit range")
    for name in (
        "bandwidth",
        "ae_learning_rate",
        "discriminator_learning_rate",
        "generator_learning_rate",
    ):
        value = getattr(config, name)
        if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if config.device not in ("cpu", "gpu", "auto"):
        raise ValueError("Device must be cpu, gpu, or auto")


def _create_models(config: TrainingConfig) -> Any:
    if config.architecture == "conv":
        return create_conv_models(encoded_dim=config.encoded_dim)
    return GAE(img_shape=(64, 64, 4), encoded_dim=config.encoded_dim)


def _compile(models: Any, config: TrainingConfig) -> None:
    models.autoencoder.compile(
        optimizer=keras.optimizers.Adam(config.ae_learning_rate),
        loss="mse" if config.reconstruction_loss == "rgba_mse" else visible_rgba_loss,
    )
    models.discriminator.trainable = True
    models.discriminator.compile(
        optimizer=keras.optimizers.Adam(config.discriminator_learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    models.discriminator.trainable = False
    models.decoder_discriminator.compile(
        optimizer=keras.optimizers.Adam(config.generator_learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )


def _data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    loaded = np.load(path, allow_pickle=False)
    if not isinstance(loaded, np.lib.npyio.NpzFile):
        raise ValueError("Dataset requires an NPZ archive")
    with loaded as source:
        if not {"arr_0", "arr_1"}.issubset(source.files):
            raise ValueError("Dataset requires NPZ arr_0 and arr_1 arrays")
        train, validation = source["arr_0"], source["arr_1"]
    for images in (train, validation):
        if images.ndim != 4 or images.shape[1:] != (64, 64, 4) or not len(images):
            raise ValueError("Dataset requires nonempty 64x64 RGBA training and validation images")
        if images.dtype != np.uint8:
            raise ValueError("Dataset must preserve the uint8 arr_0/arr_1 contract")
    return train, validation


def _normalized(images: np.ndarray) -> np.ndarray:
    return images.astype(np.float32) / 255.0


def _predict(model: Any, images: np.ndarray, batch_size: int) -> np.ndarray:
    return np.concatenate(
        [
            model.predict(_normalized(images[start : start + batch_size]), verbose=0)
            for start in range(0, len(images), batch_size)
        ]
    )


def _memory(device: Any) -> dict[str, Any]:
    return {
        "host_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if platform.system() == "Darwin" else 1024),
        "device_memory_stats": device.memory_stats() or {},
        "memory_method": "process lifetime host high-water RSS; device allocator counters if exposed",
    }


def _validation(models: Any, images: np.ndarray, batch_size: int) -> float:
    total = 0.0
    for start in range(0, len(images), batch_size):
        batch = _normalized(images[start : start + batch_size])
        decoded = models.autoencoder.predict(batch, verbose=0)
        total += float(np.square(decoded - batch).sum(dtype=np.float64))
    return total / images.size


def _optimizer_models(models: Any):
    return [
        ("ae", models.autoencoder),
        ("discriminator", models.discriminator),
        ("gan", models.decoder_discriminator),
    ]


def _checkpoint(
    run: Path, models: Any, state: TrainingState, random: np.random.RandomState
) -> None:
    """Publish complete shared weights, optimizer variables and local RNG atomically."""
    snapshots = run / "checkpoints"
    snapshots.mkdir(exist_ok=True)
    index = (
        state["completed"]["ae"] + state["completed"]["discriminator"] + state["completed"]["gan"]
    )
    destination = snapshots / f"{index:08d}-{uuid4().hex}"
    arrays = {}
    for name in ("encoder", "decoder", "discriminator"):
        for number, weights in enumerate(getattr(models, name).get_weights()):
            arrays[f"{name}_{number}"] = weights
    if models._best_ae_weights is not None:
        for number, weights in enumerate(models._best_ae_weights):
            arrays[f"best_ae_{number}"] = weights
    for name, model in _optimizer_models(models):
        for number, variable in enumerate(model.optimizer.variables):
            arrays[f"optimizer_{name}_{number}"] = np.asarray(variable)
    rng = random.get_state()
    saved = {
        "schema": "minecraft-skin-gan.checkpoint/v2",
        **state,
        "rng": [rng[0], rng[1].tolist(), rng[2], rng[3], rng[4]],
    }
    with TemporaryDirectory(prefix=".checkpoint-", dir=snapshots) as temporary:
        staging = Path(temporary) / "snapshot"
        staging.mkdir()
        np.savez(staging / "state.npz", **arrays)
        saved["weights_sha256"] = file_fingerprint(staging / "state.npz")
        _write_json(staging / "state.json", saved)
        if destination.exists():
            raise FileExistsError(destination)
        staging.replace(destination)
    pointer_path = run / "checkpoint.json"
    previous = (
        json.loads(pointer_path.read_text(encoding="utf-8"))["snapshot"]
        if pointer_path.exists()
        else None
    )
    _write_json(
        pointer_path,
        {
            "snapshot": destination.name,
            "state_sha256": file_fingerprint(destination / "state.json"),
        },
    )
    if previous is not None and previous != destination.name:
        previous_path = snapshots / previous
        if (
            isinstance(previous, str)
            and re.fullmatch(r"[0-9]{8}-[0-9a-f]{32}", previous) is not None
            and not previous_path.is_symlink()
        ):
            shutil.rmtree(previous_path)


def _restore_unchecked(run: Path, models: Any, random: np.random.RandomState) -> TrainingState:
    reference = json.loads((run / "checkpoint.json").read_text(encoding="utf-8"))
    pointer = reference["snapshot"]
    if not isinstance(pointer, str) or re.fullmatch(r"[0-9]{8}-[0-9a-f]{32}", pointer) is None:
        raise ValueError("Invalid checkpoint path")
    snapshot = run / "checkpoints" / pointer
    if file_fingerprint(snapshot / "state.json") != reference.get("state_sha256"):
        raise ValueError("Checkpoint state fingerprint mismatch")
    state = json.loads((snapshot / "state.json").read_text(encoding="utf-8"))
    if state.pop("schema", None) != "minecraft-skin-gan.checkpoint/v2":
        raise ValueError("Unsupported checkpoint schema; older runs require explicit warm start")
    if file_fingerprint(snapshot / "state.npz") != state.pop("weights_sha256"):
        raise ValueError("Checkpoint weights fingerprint mismatch")
    with np.load(snapshot / "state.npz", allow_pickle=False) as arrays:
        for name in ("encoder", "decoder", "discriminator"):
            model = getattr(models, name)
            model.set_weights([arrays[f"{name}_{i}"] for i in range(len(model.get_weights()))])
        if state["best_validation"] is not None:
            models._best_ae_weights = [
                arrays[f"best_ae_{i}"] for i in range(len(models.autoencoder.get_weights()))
            ]
            if any(
                saved.shape != current.shape
                for saved, current in zip(
                    models._best_ae_weights, models.autoencoder.get_weights(), strict=True
                )
            ):
                raise ValueError("Checkpoint best autoencoder weights are incompatible")
        for name, model in _optimizer_models(models):
            model.trainable = True
            if name == "gan":
                models.discriminator.trainable = False
            model.optimizer.build(model.trainable_variables)
            count = sum(key.startswith(f"optimizer_{name}_") for key in arrays.files)
            if count > 2:
                if count != len(model.optimizer.variables):
                    raise ValueError("Checkpoint optimizer variables are incompatible")
                for number, variable in enumerate(model.optimizer.variables):
                    variable.assign(arrays[f"optimizer_{name}_{number}"])
            else:
                for number in range(count):
                    model.optimizer.variables[number].assign(arrays[f"optimizer_{name}_{number}"])
    models.discriminator.trainable = False
    rng = state.pop("rng")
    random.set_state((rng[0], np.asarray(rng[1], dtype=np.uint32), rng[2], rng[3], rng[4]))
    return state


def _restore(run: Path, models: Any, random: np.random.RandomState) -> TrainingState:
    try:
        state = _restore_unchecked(run, models, random)
        completed = state["completed"]
        if set(completed) != {"ae", "discriminator", "gan"} or any(
            type(value) is not int or value < 0 for value in completed.values()
        ):
            raise ValueError("Checkpoint phase counters are invalid")
        if not isinstance(state["history"], list) or any(
            not isinstance(item, dict) for item in state["history"]
        ):
            raise ValueError("Checkpoint history is invalid")
        best = state["best_validation"]
        if best is not None and (
            not isinstance(best, (float, int)) or not math.isfinite(best) or best < 0
        ):
            raise ValueError("Checkpoint validation state is invalid")
        return state
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError("Checkpoint schema is invalid") from error


def _warm_sources(path: Path) -> dict[str, Path]:
    source = path / "models" if (path / "models").is_dir() else path
    files = {name: source / f"{name}.keras" for name in ("encoder", "decoder", "discriminator")}
    if any(not filename.is_file() for filename in files.values()):
        raise ValueError(
            "Warm start requires trusted encoder, decoder and discriminator .keras models"
        )
    return files


def _warm_weights(models: Any, files: dict[str, Path]) -> None:
    for name, path in files.items():
        source = keras.models.load_model(path, compile=False)
        weights = source.get_weights()
        target = getattr(models, name)
        if [weight.shape for weight in weights] != [
            weight.shape for weight in target.get_weights()
        ]:
            raise ValueError("Warm-start model weights do not match configured architecture")
        target.set_weights(weights)


def run_training(
    config: TrainingConfig, *, resume: bool = False, warm_start: Path | None = None
) -> Path:
    """Train explicit phases; resume only matching, fully checkpointed local runs."""
    if resume and warm_start is not None:
        raise ValueError("Resume and warm start are mutually exclusive")
    _validate(config)
    if keras.backend.backend() != "jax":
        raise ValueError("Maintained training requires the Keras JAX backend")
    run, data = Path(config.run_path), Path(config.data_path)
    fingerprint = file_fingerprint(data)
    resolved = _configuration(config)
    if resume:
        original = json.loads((run / "configuration.json").read_text(encoding="utf-8"))
        if original["dataset_fingerprint"] != fingerprint:
            raise ValueError("Dataset fingerprint differs from checkpoint")
        # Additive migration for earlier maintained dense runs: absence of these
        # fields unambiguously meant the historical dense/all-RGBA objective.
        original_config = dict(original["config"])
        original_config.setdefault("architecture", "dense")
        original_config.setdefault("reconstruction_loss", "rgba_mse")
        if original_config != resolved:
            raise ValueError("Training configuration differs from checkpoint")
        if (
            original.get("python") != platform.python_version()
            or original.get("keras") != keras.__version__
            or original.get("jax") != jax.__version__
        ):
            raise ValueError("Training environment differs from checkpoint")
        if "package_versions" in original and original["package_versions"] != _package_versions():
            raise ValueError("Resolved package versions differ from checkpoint")
        if "execution_policy" in original and original["execution_policy"] != _execution_policy():
            raise ValueError("Training execution policy differs from checkpoint")
        if (run / "metrics.json").exists():
            finished = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
            expected = {
                "ae": config.ae_epochs,
                "discriminator": config.discriminator_epochs,
                "gan": config.gan_steps,
            }
            if finished.get("completed") != expected:
                raise ValueError("Completed run has incompatible phase counters")
            existing = load_bundle(run / "bundle")
            if existing.metadata["dataset_fingerprint"] != fingerprint:
                raise ValueError("Completed bundle dataset fingerprint mismatch")
            return run / "metrics.json"
    elif run.exists() or run.is_symlink():
        raise FileExistsError(run)
    warm_files = _warm_sources(Path(warm_start)) if warm_start is not None else {}
    try:
        devices = jax.devices() if config.device == "auto" else jax.devices(config.device)
    except RuntimeError as error:
        raise ValueError(f"Requested {config.device} device is unavailable") from error
    matching = [
        device for device in devices if config.device == "auto" or device.platform == config.device
    ]
    if not matching:
        raise ValueError(f"Requested {config.device} device is unavailable")
    train, validation = _data(data)
    run.mkdir(parents=True, exist_ok=resume)
    if not resume:
        _write_json(
            run / "configuration.json",
            {
                "schema": "minecraft-skin-gan.training/v1",
                "config": resolved,
                "dataset_fingerprint": fingerprint,
                "python": platform.python_version(),
                "keras": keras.__version__,
                "jax": jax.__version__,
                "devices": [str(device) for device in matching],
                "device_details": [
                    {"platform": device.platform, "kind": device.device_kind} for device in matching
                ],
                "system": {
                    "os": platform.system(),
                    "release": platform.release(),
                    "machine": platform.machine(),
                },
                "package_versions": _package_versions(),
                "execution_policy": _execution_policy(),
                "validation_exposure": "development; previously exposed corpus",
                "warm_start": {
                    name: {"source": str(path.resolve()), "sha256": file_fingerprint(path)}
                    for name, path in warm_files.items()
                },
            },
        )
    started = time.perf_counter()
    random = np.random.RandomState(config.seed)
    with jax.default_device(matching[0]):
        keras.utils.set_random_seed(config.seed)
        models = _create_models(config)
        _compile(models, config)
        models._best_ae_weights = None
        if warm_files:
            _warm_weights(models, warm_files)
        state: TrainingState = {
            "completed": {"ae": 0, "discriminator": 0, "gan": 0},
            "history": [],
            "best_validation": None,
        }
        if resume:
            state = _restore(run, models, random)
            if (
                state["completed"]["ae"] > config.ae_epochs
                or state["completed"]["discriminator"] > config.discriminator_epochs
                or state["completed"]["gan"] > config.gan_steps
            ):
                raise ValueError("Checkpoint phase counters exceed configured budgets")
        for epoch in range(state["completed"]["ae"], config.ae_epochs):
            phase_start = time.perf_counter()
            indices = random.permutation(len(train))
            for start in range(0, len(train), config.batch_size):
                batch = _normalized(train[indices[start : start + config.batch_size]])
                models.autoencoder.train_on_batch(batch, batch)
            score = _validation(models, validation, config.batch_size)
            if state["best_validation"] is None or score < state["best_validation"]:
                state["best_validation"] = score
                models._best_ae_weights = models.autoencoder.get_weights()
            state["completed"]["ae"] = epoch + 1
            state["history"].append(
                {
                    "phase": "ae",
                    "epoch": epoch + 1,
                    "validation_mse": score,
                    "seconds": time.perf_counter() - phase_start,
                }
            )
            if epoch + 1 == config.ae_epochs:
                models.autoencoder.set_weights(models._best_ae_weights)
            state["history"][-1].update(_memory(matching[0]))
            _checkpoint(run, models, state, random)
        codes = _predict(models.encoder, train, config.batch_size)
        kde = KernelDensity(bandwidth=config.bandwidth).fit(codes)
        for epoch in range(state["completed"]["discriminator"], config.discriminator_epochs):
            phase_start = time.perf_counter()
            models.discriminator.trainable = True
            weighted_loss = 0.0
            for start in range(0, len(train), config.batch_size):
                real = _normalized(train[start : start + config.batch_size])
                fake = models.decoder.predict(kde.sample(len(real), random_state=random), verbose=0)
                loss = models.discriminator.train_on_batch(
                    np.concatenate([real, fake]),
                    np.concatenate([np.ones((len(real), 1)), np.zeros((len(real), 1))]),
                    return_dict=True,
                )
                weighted_loss += float(loss["loss"]) * len(real)
            models.discriminator.trainable = False
            state["completed"]["discriminator"] = epoch + 1
            state["history"].append(
                {
                    "phase": "discriminator",
                    "epoch": epoch + 1,
                    "loss": weighted_loss / len(train),
                    "loss_aggregation": "sample_weighted_epoch_mean",
                    "seconds": time.perf_counter() - phase_start,
                }
            )
            state["history"][-1].update(_memory(matching[0]))
            _checkpoint(run, models, state, random)
        for step in range(state["completed"]["gan"], config.gan_steps):
            phase_start = time.perf_counter()
            half = max(1, config.batch_size // 2)
            for _ in range(config.discriminator_updates):
                real = _normalized(train[random.randint(len(train), size=half)])
                fake = models.decoder.predict(kde.sample(half, random_state=random), verbose=0)
                models.discriminator.trainable = True
                d_loss = models.discriminator.train_on_batch(
                    np.concatenate([real, fake]),
                    np.concatenate([np.ones((half, 1)), np.zeros((half, 1))]),
                    return_dict=True,
                )
                models.discriminator.trainable = False
            g_loss = models.decoder_discriminator.train_on_batch(
                kde.sample(config.batch_size, random_state=random),
                np.ones((config.batch_size, 1)),
                return_dict=True,
            )
            state["completed"]["gan"] = step + 1
            state["history"].append(
                {
                    "phase": "gan",
                    "step": step + 1,
                    "discriminator_loss": float(d_loss["loss"]),
                    "generator_loss": float(g_loss["loss"]),
                    "seconds": time.perf_counter() - phase_start,
                }
            )
            state["history"][-1].update(_memory(matching[0]))
            if (step + 1) % config.checkpoint_interval == 0 or step + 1 == config.gan_steps:
                _checkpoint(run, models, state, random)
        model_path = run / "models"
        model_path.mkdir(exist_ok=True)
        for name in ("encoder", "decoder", "autoencoder", "discriminator"):
            getattr(models, name).save(model_path / f"{name}.keras")
        np.savez_compressed(run / "latent_kde.npz", codes=codes, bandwidth=config.bandwidth)
        if not (run / "checkpoint.json").exists():
            _checkpoint(run, models, state, random)
        bundle_path = run / "bundle"
        if bundle_path.exists():
            existing = load_bundle(bundle_path)
            if existing.metadata["dataset_fingerprint"] != fingerprint or not np.array_equal(
                existing.codes, codes
            ):
                raise ValueError("Existing bundle differs from resumed training")
            for before, after in zip(
                existing.decoder.get_weights(), models.decoder.get_weights(), strict=True
            ):
                if not np.array_equal(before, after):
                    raise ValueError("Existing bundle decoder differs from resumed training")
        else:
            create_bundle(
                model_path / "decoder.keras",
                run / "latent_kde.npz",
                bundle_path,
                dataset_fingerprint=fingerprint,
                bandwidth=config.bandwidth,
            )
        stats = matching[0].memory_stats() or {}
        report = {
            **state,
            "schema": "minecraft-skin-gan.training-metrics/v1",
            "validation_mse": _validation(models, validation, config.batch_size),
            "training_images": len(train),
            "validation_images": len(validation),
            "elapsed_seconds_this_process": time.perf_counter() - started,
            "host_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if platform.system() == "Darwin" else 1024),
            "device_memory_stats": stats,
            "memory_method": "process lifetime host high-water RSS; device allocator counters if exposed",
            "resume_contract": "shared model weights, optimizer variables, completed phase counters and local sampler RNG; pinned environment only; uncheckpointed updates replay",
            "gan_checkpoint_interval": config.checkpoint_interval,
            "parameter_counts": {
                name: getattr(models, name).count_params()
                for name in ("encoder", "decoder", "autoencoder", "discriminator")
            },
        }
        _write_json(run / "metrics.json", report)
    return run / "metrics.json"
