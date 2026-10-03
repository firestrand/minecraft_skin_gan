"""Convert PNG skins to the historical reproducible train/test NPZ archive."""

from pathlib import Path

import numpy as np
from PIL import Image
from progress.bar import Bar
from sklearn.model_selection import train_test_split


def main(
    source_dir: Path | str = "images/skins", output_path: Path | str = "images/train_test.npz"
) -> None:
    """Preserve RGBA bytes, positional NPZ keys and the 80/20 seed-1976 split."""
    file_list = list(Path(source_dir).glob("*.png"))
    skin_arrays = []
    with Bar("Processing", max=len(file_list)) as bar:
        for file in file_list:
            with Image.open(file) as skin_image:
                skin_arrays.append(np.asarray(skin_image.convert("RGBA")))
            bar.next()

    stacked = np.stack(skin_arrays)
    x_train, x_test = train_test_split(stacked, test_size=0.2, random_state=1976)
    np.savez(Path(output_path), x_train, x_test)


if __name__ == "__main__":
    main()
