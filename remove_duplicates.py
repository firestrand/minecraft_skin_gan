"""Remove repeated perceptual hashes, retaining the first matching PNG."""

from pathlib import Path

import imagehash
from PIL import Image
from progress.bar import Bar


def main(source_dir: Path | str = "images") -> None:
    """Preserve the original RGBA ImageHash pHash duplicate criterion."""
    file_list = list(Path(source_dir).glob("*.png"))
    image_hashes: set[str] = set()
    with Bar("Processing", max=len(file_list)) as bar:
        for file in file_list:
            with Image.open(file) as image:
                image_hash = str(imagehash.phash(image.convert("RGBA")))
            if image_hash in image_hashes:
                file.unlink()
            else:
                image_hashes.add(image_hash)
            bar.next()


if __name__ == "__main__":
    main()
