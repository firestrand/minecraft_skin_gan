"""Move skins with insufficient head colour variation to the other directory."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image


def should_filter_skin(skin_array: NDArray[np.uint8]) -> bool:
    """Apply the historical strict <10 threshold to each RGB head channel."""
    head_array = skin_array[8:16, 0:32, :]
    std_count = sum(np.std(head_array[:, :, channel : channel + 1]) < 10 for channel in range(3))
    return bool(std_count > 1)


def main(source_dir: Path | str = "images/skins", other_dir: Path | str = "images/other") -> None:
    """Move matching PNGs, leaving their bytes unchanged."""
    for file in Path(source_dir).glob("*.png"):
        with Image.open(file) as skin_image:
            skin_array = np.asarray(skin_image.convert("RGBA"))
        if should_filter_skin(skin_array):
            destination = Path(other_dir)
            destination.mkdir(parents=True, exist_ok=True)
            file.rename(destination / file.name)
    print("done")


if __name__ == "__main__":
    main()
