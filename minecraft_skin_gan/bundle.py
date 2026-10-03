"""Versioned, validated decoder and latent-sampler artifacts.

Bundles contain trusted local Keras models. Fingerprints detect modification;
these artifacts are not a format for executing untrusted downloaded models.
"""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from PIL import Image
from sklearn.neighbors import KernelDensity

from generate_skin import load_decoder

SCHEMA = "minecraft-skin-gan.bundle/v1"


def file_fingerprint(path: str | Path) -> str:
    """Hash a file without loading its contents into memory."""
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


@dataclass(frozen=True)
class GenerationBundle:
    path: Path
    metadata: dict[str, Any]
    codes: np.ndarray
    decoder: Any


def _bandwidth(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError("Bandwidth must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError("Bandwidth must be a finite positive number")
    return result


def _shape(decoder: Any) -> int:
    shape = decoder.input_shape
    output = decoder.output_shape
    if not isinstance(shape, tuple) or len(shape) != 2 or not isinstance(shape[1], int):
        raise ValueError("Decoder must accept a two-dimensional latent batch")
    if output not in ((None, 16384), (None, 64, 64, 4)):
        raise ValueError("Decoder must produce 64x64 RGBA skins")
    return shape[1]


def _codes(path: Path, dimension: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as source:
        if "codes" not in source:
            raise ValueError("Sampler archive requires codes")
        codes = source["codes"]
    if codes.ndim != 2 or codes.shape[0] == 0 or codes.shape[1] != dimension:
        raise ValueError("Latent codes must be nonempty and match decoder dimension")
    if codes.dtype.kind not in "fiu" or not np.isfinite(codes).all():
        raise ValueError("Latent codes must be finite numeric values")
    codes.setflags(write=False)
    return codes


def _json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def _publish(staging: Path, destination: Path) -> None:
    # mkdir reserves the final name exclusively. A successful rename publishes
    # every artifact together; failure removes only the empty reservation we own.
    destination.mkdir()
    try:
        staging.replace(destination)
    except BaseException:
        destination.rmdir()
        raise


def create_bundle(
    decoder_path: str | Path,
    codes_path: str | Path,
    output_directory: str | Path,
    *,
    dataset_fingerprint: str,
    bandwidth: float = 3.16,
) -> Path:
    """Create a new immutable bundle; refuse an existing destination."""
    bandwidth = _bandwidth(bandwidth)
    if (
        not isinstance(dataset_fingerprint, str)
        or len(dataset_fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in dataset_fingerprint)
    ):
        raise ValueError("Dataset fingerprint must be a SHA256 hexadecimal digest")
    output = Path(output_directory)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    decoder = load_decoder(decoder_path)
    dimension = _shape(decoder)
    codes = _codes(Path(codes_path), dimension)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir()
        decoder.save(staging / "decoder.keras")
        np.savez_compressed(staging / "codes.npz", codes=codes, bandwidth=bandwidth)
        metadata = {
            "schema": SCHEMA,
            "decoder": "decoder.keras",
            "codes": "codes.npz",
            "decoder_sha256": file_fingerprint(staging / "decoder.keras"),
            "codes_sha256": file_fingerprint(staging / "codes.npz"),
            "dataset_fingerprint": dataset_fingerprint,
            "latent_dimension": dimension,
            "image_shape": [64, 64, 4],
            "normalization": [0, 1],
            "bandwidth": bandwidth,
            "sampler": "gaussian_kde",
        }
        _json(staging / "metadata.json", metadata)
        _publish(staging, output)
    return output


def load_bundle(path: str | Path) -> GenerationBundle:
    """Validate fingerprints and contracts before loading a trusted decoder."""
    root = Path(path)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("schema") != SCHEMA:
        raise ValueError("Unsupported generation bundle schema")
    fingerprint = metadata.get("dataset_fingerprint")
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
    ):
        raise ValueError("Dataset fingerprint must be a SHA256 hexadecimal digest")
    if metadata.get("image_shape") != [64, 64, 4] or metadata.get("normalization") != [0, 1]:
        raise ValueError("Unsupported image shape or normalization")
    if metadata.get("sampler") != "gaussian_kde":
        raise ValueError("Unsupported persisted sampler")
    _bandwidth(metadata.get("bandwidth"))
    for key, name in (("decoder", "decoder.keras"), ("codes", "codes.npz")):
        artifact = root / name
        if metadata.get(key) != name or artifact.is_symlink():
            raise ValueError("Invalid bundle artifact path")
        if file_fingerprint(artifact) != metadata.get(f"{key}_sha256"):
            raise ValueError(f"{key} fingerprint mismatch")
    decoder = load_decoder(root / "decoder.keras")
    dimension = _shape(decoder)
    if (
        type(metadata.get("latent_dimension")) is not int
        or metadata["latent_dimension"] != dimension
    ):
        raise ValueError("Metadata latent dimension does not match decoder")
    codes = _codes(root / "codes.npz", dimension)
    return GenerationBundle(root, metadata, codes, decoder)


def generate_bundle(
    bundle_path: str | Path,
    output_directory: str | Path,
    *,
    count: int = 16,
    seed: int = 1976,
    bandwidth: float | None = None,
    sampler: str = "kde",
) -> list[Path]:
    """Generate seeded samples atomically in a new directory, in bounded batches."""
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("Count must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an integer in [0, 2**32)")
    if sampler not in {"kde", "empirical", "legacy_uniform"}:
        raise ValueError("Unknown sampler")
    if bandwidth is not None:
        bandwidth = _bandwidth(bandwidth)
    output = Path(output_directory)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    bundle = load_bundle(bundle_path)
    selected_bandwidth = bundle.metadata["bandwidth"] if bandwidth is None else bandwidth
    output.parent.mkdir(parents=True, exist_ok=True)
    names = []
    with TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary) / "generated"
        staging.mkdir()
        start = 0
        for latent in _latent_batches(bundle, count, seed, selected_bandwidth, sampler):
            size = len(latent)
            images = np.asarray(bundle.decoder.predict(latent, verbose=0))
            if images.shape not in ((size, 16384), (size, 64, 64, 4)):
                raise ValueError("Decoded image shape is incompatible")
            if not np.isfinite(images).all():
                raise ValueError("Decoded images must be finite")
            pixels = (np.clip(images.reshape(size, 64, 64, 4), 0, 1) * 255).round().astype(np.uint8)
            for index, image in enumerate(pixels, start=start):
                name = f"skin_{index + 1:04d}.png"
                with Image.fromarray(image) as skin:
                    skin.save(staging / name)
                names.append(name)
            start += size
        _json(
            staging / "generation.json",
            {
                "schema": "minecraft-skin-gan.generation/v1",
                "bundle_metadata_sha256": file_fingerprint(bundle.path / "metadata.json"),
                "dataset_fingerprint": bundle.metadata["dataset_fingerprint"],
                "seed": seed,
                "count": count,
                "sampler": sampler,
                "bandwidth": selected_bandwidth if sampler == "kde" else None,
                "prediction_batch_size": 64,
            },
        )
        _publish(staging, output)
    return [output / name for name in names]


def _latent_batches(
    bundle: GenerationBundle, count: int, seed: int, bandwidth: float, sampler: str
):
    random = np.random.RandomState(seed)
    kde = KernelDensity(bandwidth=bandwidth).fit(bundle.codes) if sampler == "kde" else None
    for start in range(0, count, 64):
        size = min(64, count - start)
        if kde is not None:
            yield kde.sample(size, random_state=random)
        elif sampler == "empirical":
            yield bundle.codes[random.randint(len(bundle.codes), size=size)]
        else:
            yield random.rand(size, bundle.metadata["latent_dimension"])


def sample_latents(
    bundle: GenerationBundle,
    *,
    count: int = 16,
    seed: int = 1976,
    bandwidth: float | None = None,
    sampler: str = "kde",
) -> np.ndarray:
    """Return the exact seeded latent probes used by generation (64-item batches)."""
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("Count must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an integer in [0, 2**32)")
    if sampler not in {"kde", "empirical", "legacy_uniform"}:
        raise ValueError("Unknown sampler")
    selected = _bandwidth(bundle.metadata["bandwidth"] if bandwidth is None else bandwidth)
    return np.concatenate(list(_latent_batches(bundle, count, seed, selected, sampler)))
