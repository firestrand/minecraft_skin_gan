"""Development metrics for normalized RGBA atlases, separate from human quality."""

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

type Pixels = NDArray[np.floating[Any]]


def _pixels(images: Pixels) -> Pixels:
    array = np.asarray(images)
    if array.ndim != 4 or array.shape[1:] != (64, 64, 4) or len(array) == 0:
        raise ValueError("Images must be a nonempty (N, 64, 64, 4) array")
    if not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
        raise ValueError("Images must have finite normalized values in [0, 1]")
    return array


def reconstruction_metrics(original: Pixels, reconstructed: Pixels) -> dict[str, float]:
    """Measure color at original visible pixels; report alpha separately."""
    original, reconstructed = _pixels(original), _pixels(reconstructed)
    if original.shape != reconstructed.shape:
        raise ValueError("Original and reconstructed images must have the same shape")
    alpha = original[..., 3:4]
    color_error = np.square(original[..., :3] - reconstructed[..., :3])
    weight = float(np.sum(alpha, dtype=np.float64) * 3)
    return {
        "rgba_mse": float(np.mean(np.square(original - reconstructed), dtype=np.float64)),
        "visible_rgb_mse": float(np.sum(color_error * alpha, dtype=np.float64) / weight)
        if weight
        else 0.0,
        "alpha_mae": float(
            np.mean(np.abs(original[..., 3] - reconstructed[..., 3]), dtype=np.float64)
        ),
        "visible_rgb_weight": weight,
    }


def image_metrics(images: Pixels) -> dict[str, float | int]:
    """Summarize finite output, quantized duplicates, alpha and batch variety."""
    images = _pixels(images)
    hashes = {
        hashlib.sha256((image * 255).round().astype(np.uint8).tobytes()).hexdigest()
        for image in images
    }
    total = 0.0
    pairs = 0
    for index, image in enumerate(images):
        for other in images[index + 1 :]:
            total += float(np.mean(np.square(image - other), dtype=np.float64))
            pairs += 1
    return {
        "count": len(images),
        "unique_count": len(hashes),
        "exact_duplicate_count": len(images) - len(hashes),
        "mean_pairwise_rgba_mse": total / pairs if pairs else 0.0,
        "pair_count": pairs,
        "mean_alpha": float(np.mean(images[..., 3], dtype=np.float64)),
        "intermediate_alpha_fraction": float(np.mean((images[..., 3] > 0) & (images[..., 3] < 1))),
    }


def nearest_examples(
    samples: Pixels, training: Pixels | NDArray[np.uint8], *, batch_size: int = 32
) -> list[dict[str, float | int]]:
    """Exact pixel-MSE neighbors in bounded chunks; first index breaks ties."""
    samples = _pixels(samples)
    training = np.asarray(training)
    if training.dtype == np.uint8:
        if training.ndim != 4 or training.shape[1:] != (64, 64, 4) or not len(training):
            raise ValueError("Training images must be a nonempty (N, 64, 64, 4) array")
    else:
        training = _pixels(training)
    if batch_size < 1:
        raise ValueError("Neighbor batch size must be positive")
    result = []
    for sample in samples:
        best_index, best_distance = 0, float("inf")
        for start in range(0, len(training), batch_size):
            chunk = training[start : start + batch_size]
            if chunk.dtype == np.uint8:
                chunk = chunk.astype(np.float32) / 255
            distances = np.mean(
                np.square(chunk - sample),
                axis=(1, 2, 3),
                dtype=np.float64,
            )
            index = int(np.argmin(distances))
            if distances[index] < best_distance:
                best_index, best_distance = start + index, float(distances[index])
        result.append({"training_index": best_index, "rgba_mse": best_distance})
    return result


def evaluate_bundle(
    bundle_path: Path | str,
    data_path: Path | str,
    output_directory: Path | str,
    *,
    seed: int = 1976,
    count: int = 16,
    autoencoder_path: Path | str | None = None,
    model_type: str = "classic",
) -> Path:
    """Write comparable development metrics and sheets; never claim a release gate."""
    from PIL import Image

    from minecraft_skin_gan.bundle import file_fingerprint, load_bundle, sample_latents
    from minecraft_skin_gan.skin_checks import layer_masks, skin_diagnostics

    layer_masks(model_type)
    if not 1 <= count <= 64:
        raise ValueError("Evaluation count must be between 1 and 64")
    output = Path(output_directory)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    bundle = load_bundle(bundle_path)
    archive = np.load(data_path, allow_pickle=False)
    if not isinstance(archive, np.lib.npyio.NpzFile):
        raise ValueError("Evaluation dataset must be an NPZ archive")
    with archive as data:
        if not {"arr_0", "arr_1"}.issubset(data.files):
            raise ValueError("Evaluation dataset requires arr_0 and arr_1")
        training, validation = data["arr_0"], data["arr_1"]
        if training.dtype != np.uint8 or validation.dtype != np.uint8:
            raise ValueError("Evaluation dataset must contain uint8 RGBA pixels")
    for images in (training, validation):
        if images.ndim != 4 or images.shape[1:] != (64, 64, 4) or not len(images):
            raise ValueError("Evaluation requires nonempty (N, 64, 64, 4) splits")
    held_out = validation[:count].astype(np.float32) / 255
    _pixels(held_out)
    # Bundle sampling owns the precise seed/distribution contract.
    latents = sample_latents(bundle, count=count, seed=seed)
    generated = np.asarray(bundle.decoder.predict(latents, verbose=0)).reshape(count, 64, 64, 4)
    data_fingerprint = file_fingerprint(data_path)
    training_hashes = {hashlib.sha256(image.tobytes()).hexdigest() for image in training}
    validation_hashes = {hashlib.sha256(image.tobytes()).hexdigest() for image in validation}
    report: dict[str, Any] = {
        "schema": "minecraft-skin-gan.evaluation/v1",
        "gate": "development",
        "release_gate": "unverified: requires genuinely unexposed disjoint test data and an agreed rubric",
        "seed": seed,
        "evaluation_data_sha256": data_fingerprint,
        "training_archive_match": data_fingerprint == bundle.metadata["dataset_fingerprint"],
        "exact_cross_split_groups": len(training_hashes & validation_hashes),
        "exposure": "development; no unexposed release-test claim",
        "bundle": bundle.metadata,
        "generated": image_metrics(generated),
        "nearest_training": nearest_examples(generated, training),
        "training_count": len(training),
        "held_out_count": len(validation),
        "sheet_held_out_indices": list(range(min(count, len(held_out)))),
        "skin_format": "renderer-profile diagnostics; game acceptance remains unverified",
        "skin_diagnostics": [
            skin_diagnostics(np.rint(pixels * 255).astype(np.uint8), model_type=model_type)
            for pixels in generated
        ],
        "human_quality": "unverified",
    }
    manifest_path = Path(data_path).parent / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("archive_sha256") != data_fingerprint:
            raise ValueError("Adjacent manifest does not match the evaluation archive SHA256")
        report["dataset_manifest"] = {
            key: manifest.get(key)
            for key in (
                "schema_version",
                "dataset_sha256",
                "archive_sha256",
                "fingerprint_contract",
                "exposure",
                "counts",
                "near_duplicates",
                "provenance",
            )
        }
    reconstructed = None
    if autoencoder_path is not None:
        from generate_skin import load_decoder

        autoencoder = load_decoder(autoencoder_path)
        report["autoencoder_sha256"] = file_fingerprint(autoencoder_path)
        reconstructed = np.asarray(autoencoder.predict(held_out[:count], verbose=0))
        report["reconstruction"] = reconstruction_metrics(held_out[:count], reconstructed)
        report["reconstruction_count"] = len(held_out[:count])
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".evaluation-", dir=output.parent) as directory:
        staging = Path(directory) / "report"
        staging.mkdir()
        for name, images in (
            ("generated", generated),
            ("held_out", held_out[:count]),
            ("reconstructed", reconstructed),
        ):
            if images is None:
                continue
            columns = min(4, len(images))
            with Image.new(
                "RGBA", (columns * 256, ((len(images) + columns - 1) // columns) * 256), "#dddddd"
            ) as sheet:
                for index, image in enumerate(images):
                    with (
                        Image.fromarray((image * 255).round().astype(np.uint8)) as tile,
                        tile.resize((256, 256), Image.Resampling.NEAREST) as enlarged,
                    ):
                        sheet.alpha_composite(
                            enlarged, ((index % columns) * 256, (index // columns) * 256)
                        )
                with sheet.convert("RGB") as rgb:
                    rgb.save(staging / f"{name}.png")
        (staging / "evaluation.json").write_text(serialized)
        output.mkdir()
        try:
            staging.replace(output)
        except BaseException:
            output.rmdir()
            raise
    return output / "evaluation.json"
