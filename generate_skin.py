"""Generate Minecraft RGBA skins from a saved GAE decoder."""

import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

os.environ.setdefault("KERAS_BACKEND", "jax")

import h5py
import keras
import numpy as np
from PIL import Image


def load_decoder(model_path: str | Path = "models/decoder.keras") -> keras.Model:
    """Load a modern decoder or a legacy HDF5 model stored with a .mdl suffix.

    Only load trusted local model artifacts. Legacy TensorFlow SavedModel and
    PlaidML-specific artifacts need conversion in their original environment.
    """
    path = Path(model_path)
    if path == Path("models/decoder.keras") and not path.exists():
        legacy_path = path.with_suffix(".mdl")
        if legacy_path.exists():
            path = legacy_path
    if path.is_dir():
        raise ValueError(
            "Legacy SavedModel directories are unsupported by Keras 3. "
            "Load this decoder in its original compatible Keras environment and "
            "export a full HDF5 .h5 model, then load and save it as decoder.keras."
        )
    if not path.is_file():
        raise FileNotFoundError(f"Decoder model not found: {path}")
    if path.suffix == ".mdl":
        if not h5py.is_hdf5(path):
            raise ValueError(
                "Legacy .mdl decoder is not HDF5. Export a full HDF5 .h5 model "
                "using its original compatible Keras/backend environment."
            )
        # Keras 3 dispatches legacy HDF5 loading by extension rather than signature.
        with TemporaryDirectory(prefix="minecraft-decoder-") as temporary:
            compatible_path = Path(temporary) / "decoder.h5"
            shutil.copyfile(path, compatible_path)
            return keras.models.load_model(compatible_path, compile=False)
    return keras.models.load_model(path, compile=False)


def generate_skins(
    model_path: str | Path = "models/decoder.keras",
    output_directory: str | Path = "images/results",
    *,
    count: int = 100,
) -> list[Path]:
    """Decode uniform 128-dimensional samples and save clipped, rounded RGBA PNGs."""
    if count <= 0:
        raise ValueError("Skin count must be positive")
    decoder = load_decoder(model_path)
    latent = np.random.rand(count, 128)
    images = decoder.predict(latent)
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, image in enumerate(images):
        pixels = (np.clip(image.reshape((64, 64, 4)), 0.0, 1.0) * 255).round()
        path = output_path / f"gae_{index}.png"
        with Image.fromarray(pixels.astype(np.uint8)) as skin:
            skin.save(path)
        paths.append(path)
    return paths


def main() -> None:
    """Generate 100 skins using the original script defaults."""
    generate_skins()


if __name__ == "__main__":
    main()
