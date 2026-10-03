"""Bounded, sequential development experiments with explicit shared cohorts.

Sampling ablations reuse a trained bundle. Reports describe pixel metrics and
resource budgets; they make no human-quality, originality or adoption claim.
"""

import hashlib
import json
import math
import platform
import re
import tempfile
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields, replace
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from generate_skin import keras, load_decoder
from minecraft_skin_gan.bundle import file_fingerprint, load_bundle, sample_latents
from minecraft_skin_gan.evaluation import image_metrics, nearest_examples, reconstruction_metrics
from minecraft_skin_gan.training import (
    TrainingConfig,
    _configuration,
    _data,
    _validate,
    run_training,
)


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    training: TrainingConfig | None = None
    bundle_path: Path | None = None
    source_run: str | None = None
    sampler: str = "kde"
    bandwidth: float | None = None
    sample_count: int = 16
    sample_seed: int = 1976


@dataclass(frozen=True)
class ExperimentPlan:
    data_path: Path
    output_path: Path
    configs: tuple[ExperimentConfig, ...]


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _nonfinite(value: str) -> None:
    raise ValueError(f"Nonfinite JSON value: {value}")


def _json_path(value: Any, parent: Path) -> Path:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError("Plan paths must be nonempty strings")
    path = Path(value)
    return path if path.is_absolute() else parent / path


def load_experiment_plan(path: Path | str) -> ExperimentPlan:
    """Read strict v1 JSON; resolve relative paths beside the plan, without writes.

    Training data/run paths and all phase budgets are mandatory. Other declared
    TrainingConfig fields retain their defaults. Unknown and duplicate keys,
    invalid field types, nonfinite values and malformed references are rejected.
    """
    source = Path(path).resolve()
    raw = json.loads(
        source.read_text(encoding="utf-8"),
        object_pairs_hook=_json_object,
        parse_constant=_nonfinite,
    )
    expected = {"schema", "data_path", "output_path", "configs"}
    if (
        not isinstance(raw, dict)
        or set(raw) != expected
        or raw["schema"] != "minecraft-skin-gan.experiment-plan/v1"
    ):
        raise ValueError(
            "Plan requires schema minecraft-skin-gan.experiment-plan/v1, data_path, output_path and configs"
        )
    data = _json_path(raw["data_path"], source.parent).resolve()
    output = _json_path(raw["output_path"], source.parent)
    if not isinstance(raw["configs"], list) or not 1 <= len(raw["configs"]) <= 64:
        raise ValueError("Plan configs must be a list of between 1 and 64 entries")
    configs = []
    names = set()
    allowed = {field.name for field in fields(ExperimentConfig)}
    for entry in raw["configs"]:
        if not isinstance(entry, dict) or set(entry) - allowed or "name" not in entry:
            raise ValueError("Experiment entry has missing name or unknown fields")
        name = entry["name"]
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", name)
            or name in names
        ):
            raise ValueError("Experiment names must be unique safe directory names")
        names.add(name)
        values = dict(entry)
        if values.get("training") is not None:
            training = values["training"]
            required = {"data_path", "run_path", "ae_epochs", "discriminator_epochs", "gan_steps"}
            declared = {field.name: field for field in fields(TrainingConfig)}
            if (
                not isinstance(training, dict)
                or set(training) - declared.keys()
                or not required <= training.keys()
            ):
                raise ValueError(
                    "Training requires explicit data_path, run_path and all phase budgets; unknown fields are rejected"
                )
            parameters = dict(training)
            for key, value in training.items():
                if key in {"data_path", "run_path"}:
                    parameters[key] = _json_path(value, source.parent)
                else:
                    default = declared[key].default
                    valid = (
                        isinstance(value, (int, float))
                        if isinstance(default, float)
                        else isinstance(value, type(default))
                    )
                    if not valid or isinstance(value, bool):
                        raise ValueError(f"Invalid training field type: {key}")
            config = TrainingConfig(**parameters)
            _validate(config)
            if Path(config.data_path).resolve() != data:
                raise ValueError("Training data_path must match plan data_path")
            values["training"] = config
        if values.get("bundle_path") is not None:
            values["bundle_path"] = _json_path(values["bundle_path"], source.parent)
        if values.get("source_run") is not None and not isinstance(values["source_run"], str):
            raise ValueError("source_run must be a training candidate name")
        candidate = ExperimentConfig(**values)
        _validate_sampling(candidate)
        if (
            sum(
                item is not None
                for item in (candidate.training, candidate.bundle_path, candidate.source_run)
            )
            != 1
        ):
            raise ValueError("Each experiment requires exactly one source")
        configs.append(candidate)
    sources = {candidate.name for candidate in configs if candidate.training is not None}
    first = configs[0]
    for candidate in configs:
        if candidate.source_run is not None and candidate.source_run not in sources:
            raise ValueError("source_run must name a training candidate")
        if (
            candidate.sample_count != first.sample_count
            or candidate.sample_seed != first.sample_seed
        ):
            raise ValueError("All experiments must share sample count and sample seed")
    return ExperimentPlan(data, output, tuple(configs))


def _validate_sampling(candidate: ExperimentConfig) -> None:
    if (
        isinstance(candidate.sample_count, bool)
        or not isinstance(candidate.sample_count, int)
        or not 1 <= candidate.sample_count <= 64
    ):
        raise ValueError("Sample count must be between 1 and 64")
    if (
        isinstance(candidate.sample_seed, bool)
        or not isinstance(candidate.sample_seed, int)
        or not 0 <= candidate.sample_seed < 2**32
    ):
        raise ValueError("Sample seed must be a 32-bit nonnegative integer")
    if not isinstance(candidate.sampler, str) or candidate.sampler not in {
        "kde",
        "empirical",
        "legacy_uniform",
    }:
        raise ValueError("Unknown experiment sampler")
    if candidate.bandwidth is not None and (
        isinstance(candidate.bandwidth, bool)
        or not isinstance(candidate.bandwidth, (int, float))
        or not math.isfinite(candidate.bandwidth)
        or candidate.bandwidth <= 0
    ):
        raise ValueError("Bandwidth must be finite and positive")


def _spec(config: ExperimentConfig) -> dict[str, Any]:
    value = asdict(config)
    value["training"] = _configuration(config.training) if config.training is not None else None
    value["bundle_path"] = (
        str(Path(config.bundle_path).resolve()) if config.bundle_path is not None else None
    )
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def controlled_matrix(
    base: TrainingConfig,
    *,
    seeds: Sequence[int] = (1976,),
    gan_steps: Sequence[int] = (0, 830),
    discriminator_updates: Sequence[int] = (1, 2),
    bandwidths: Sequence[float] = (1.0, 3.16),
    include_empirical: bool = True,
    architectures: Sequence[str] = (),
    reconstruction_losses: Sequence[str] = (),
) -> list[ExperimentConfig]:
    """Build independent one-factor contrasts relative to each seed's baseline.

    Bandwidth contrasts affect inference only, leaving the trained model fixed.
    Architecture/loss contrasts require the corresponding TrainingConfig fields.
    """
    result = []
    for seed in seeds:
        name = f"seed-{seed}-baseline"
        reference = replace(base, seed=seed, run_path=Path(base.run_path) / name)
        result.append(ExperimentConfig(name, training=reference))
        variations: list[tuple[str, int | str]] = [("gan_steps", value) for value in gan_steps]
        variations += [("discriminator_updates", value) for value in discriminator_updates]
        variations += [("architecture", value) for value in architectures]
        variations += [("reconstruction_loss", value) for value in reconstruction_losses]
        for field, value in variations:
            if getattr(reference, field) == value:
                continue
            label = f"seed-{seed}-{field}-{value}"
            variant = replace(reference, **{field: value, "run_path": Path(base.run_path) / label})
            result.append(ExperimentConfig(label, training=variant))
        for index, bandwidth in enumerate(bandwidths):
            if bandwidth != base.bandwidth:
                result.append(
                    ExperimentConfig(
                        f"seed-{seed}-bandwidth-{index}", source_run=name, bandwidth=bandwidth
                    )
                )
        if include_empirical:
            result.append(
                ExperimentConfig(f"seed-{seed}-empirical", source_run=name, sampler="empirical")
            )
    return result


def _preflight(
    data: Path, output: Path, configs: Sequence[ExperimentConfig], resume: bool
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, dict[str, Any]]:
    if not 1 <= len(configs) <= 64:
        raise ValueError("Experiment matrix requires between 1 and 64 candidates")
    if output.exists() and not resume or output.is_symlink():
        raise FileExistsError(output)
    fingerprint = file_fingerprint(data)
    train, held = _data(data)
    first = configs[0]
    names: set[str] = set()
    runs: set[Path] = set()
    bundles: dict[str, Any] = {}
    for candidate in configs:
        if (
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", candidate.name)
            or candidate.name in names
        ):
            raise ValueError("Experiment names must be unique safe directory names")
        names.add(candidate.name)
        if (
            sum(
                value is not None
                for value in (candidate.training, candidate.bundle_path, candidate.source_run)
            )
            != 1
        ):
            raise ValueError("Each experiment requires exactly one training, bundle or source_run")
        if (
            candidate.sample_count != first.sample_count
            or candidate.sample_seed != first.sample_seed
        ):
            raise ValueError("All experiments must share sample count and sample seed")
        _validate_sampling(candidate)
        if candidate.training is not None:
            config = candidate.training
            _validate(config)
            if Path(config.data_path).resolve() != data:
                raise ValueError("All training variants must use the same dataset archive")
            if (
                max(config.ae_epochs, config.discriminator_epochs) > 10000
                or config.gan_steps > 1000000
                or config.discriminator_updates > 100
            ):
                raise ValueError("Training budget exceeds bounded experiment limits")
            run = Path(config.run_path).resolve()
            if (
                any(run == other or run in other.parents or other in run.parents for other in runs)
                or run == output
                or run in output.parents
                or output in run.parents
            ):
                raise ValueError(
                    "Training run paths must be distinct and separate from report output"
                )
            runs.add(run)
            if run.exists() or Path(config.run_path).is_symlink():
                if not resume or Path(config.run_path).is_symlink():
                    raise FileExistsError(run)
                original = json.loads((run / "configuration.json").read_text())
                original_config = original.get("config")
                if not isinstance(original_config, dict):
                    raise ValueError("Existing training run configuration is invalid")
                original_config = dict(original_config)
                original_config.setdefault("architecture", "dense")
                original_config.setdefault("reconstruction_loss", "rgba_mse")
                if (
                    original_config != _configuration(config)
                    or original.get("dataset_fingerprint") != fingerprint
                ):
                    raise ValueError("Existing training run configuration or dataset differs")
        elif candidate.bundle_path is not None:
            bundle = load_bundle(candidate.bundle_path)
            if bundle.metadata["dataset_fingerprint"] != fingerprint:
                raise ValueError("Existing bundle dataset fingerprint differs")
            bundles[candidate.name] = bundle
    sources = {candidate.name: candidate for candidate in configs}
    for candidate in configs:
        if candidate.source_run is not None:
            source = sources.get(candidate.source_run)
            if source is None or source.training is None:
                raise ValueError("source_run must name a training candidate")
    manifest = data.parent / "manifest.json"
    provenance = {}
    if manifest.exists():
        provenance = json.loads(manifest.read_text())
        if provenance.get("archive_sha256") != fingerprint:
            raise ValueError("Adjacent manifest does not match dataset archive")
    indices = list(range(min(first.sample_count, len(held))))
    identity = {
        "schema": "minecraft-skin-gan.experiments/v1",
        "dataset_sha256": fingerprint,
        "held_out_indices": indices,
        "cohort_sha256": hashlib.sha256(held[indices].tobytes()).hexdigest(),
        "configs": [_spec(candidate) for candidate in configs],
        "python": platform.python_version(),
        "keras": keras.__version__,
        "jax": version("jax"),
        "numpy": np.__version__,
        "scikit_learn": version("scikit-learn"),
    }
    if output.exists():
        previous = json.loads((output / "plan.json").read_text())
        if previous != identity:
            raise ValueError("Existing experiment plan differs from requested matrix")
        for candidate in configs:
            directory = output / candidate.name
            if directory.is_symlink() or directory.exists() and not directory.is_dir():
                raise FileExistsError(directory)
            if directory.exists():
                metrics = json.loads((directory / "metrics.json").read_text())
                if metrics.get("config") != _spec(candidate):
                    raise ValueError("Completed experiment configuration differs")
    return (
        identity,
        train,
        held[indices].astype(np.float32) / 255,
        {"bundles": bundles, "provenance": provenance},
    )


def run_experiments(
    data_path: Path | str,
    output_path: Path | str,
    configs: Sequence[ExperimentConfig],
    *,
    resume: bool = False,
) -> Path:
    """Preflight the entire matrix, then run isolated sequential development checks.

    An existing directory requires resume and an identical plan. Completed
    training uses the maintained runner's identity checks; partial training uses
    its documented checkpoint semantics. Sampling never retrains a source run.
    """
    if Path(output_path).is_symlink():
        raise FileExistsError(output_path)
    data, output = Path(data_path).resolve(), Path(output_path).resolve()
    identity, train, held, preflight = _preflight(data, output, configs, resume)
    output.mkdir(parents=True, exist_ok=resume)
    _write(output / "plan.json", identity)
    bundles = preflight["bundles"]
    training_results: dict[str, dict[str, Any]] = {}
    # Resolve training first, regardless of where sampling references appear.
    for candidate in configs:
        if candidate.training is None:
            continue
        run = Path(candidate.training.run_path)
        metrics = run_training(candidate.training, resume=resume and run.exists())
        bundle = load_bundle(run / "bundle")
        if bundle.metadata["dataset_fingerprint"] != identity["dataset_sha256"]:
            raise ValueError("Trained bundle dataset fingerprint differs")
        bundles[candidate.name] = bundle
        autoencoder = load_decoder(run / "models/autoencoder.keras")
        reconstructed = np.asarray(autoencoder.predict(held, verbose=0))
        training_results[candidate.name] = {
            "training_metrics_path": str(metrics.resolve()),
            "training_metrics": json.loads(metrics.read_text()),
            "reconstruction": reconstruction_metrics(held, reconstructed),
            "autoencoder_sha256": file_fingerprint(run / "models/autoencoder.keras"),
        }
    results = []
    for candidate in configs:
        started = time.perf_counter()
        source = candidate.source_run or candidate.name
        bundle = bundles[source]
        latents = sample_latents(
            bundle,
            count=candidate.sample_count,
            seed=candidate.sample_seed,
            sampler=candidate.sampler,
            bandwidth=candidate.bandwidth,
        )
        generated = np.asarray(bundle.decoder.predict(latents, verbose=0)).reshape(
            candidate.sample_count, 64, 64, 4
        )
        result = {
            "name": candidate.name,
            "config": _spec(candidate),
            "bundle": bundle.metadata,
            "generated": image_metrics(generated),
            "nearest_training": nearest_examples(generated, train),
            "evaluation_seconds": time.perf_counter() - started,
            **training_results.get(source, {}),
        }
        directory = output / candidate.name
        if directory.exists():
            if not resume:
                raise FileExistsError(directory)
            existing = json.loads((directory / "metrics.json").read_text())
            if existing["config"] != result["config"] or existing["bundle"] != result["bundle"]:
                raise ValueError("Completed experiment bundle or config differs")
            if not np.array_equal(
                np.load(directory / "generated.npy", allow_pickle=False), generated
            ):
                raise ValueError("Completed experiment generated output differs")
            result = existing
        else:
            with tempfile.TemporaryDirectory(prefix=".experiment-", dir=output) as temporary:
                staging = Path(temporary) / candidate.name
                staging.mkdir()
                np.save(staging / "latents.npy", latents)
                np.save(staging / "generated.npy", generated)
                _write(staging / "metrics.json", result)
                # Reserve without replacing another writer's output.
                directory.mkdir()
                try:
                    staging.replace(directory)
                except BaseException:
                    directory.rmdir()
                    raise
        results.append(result)
    if file_fingerprint(data) != identity["dataset_sha256"]:
        raise ValueError("Dataset changed during experiment execution")
    _write(
        output / "experiments.json",
        {
            **identity,
            "dataset_manifest": preflight["provenance"],
            "training_images": len(train),
            "reconstruction_count": len(held),
            "nearest_neighbor_reference": "all arr_0 training examples; exact pixel RGBA MSE",
            "gate": "development; previously exposed corpus",
            "human_quality": "unverified; metrics do not establish adoption or originality",
            "resource_budget": "explicit per-config epochs/steps; sequential execution; no wall-clock guarantee",
            "results": results,
        },
    )
    return output / "experiments.json"
