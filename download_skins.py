"""Download skins for later classification as standard, slim or legacy images."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import batched
from pathlib import Path

import requests
from progress.bar import Bar

DEFAULT_SKIN_URL = "https://www.minecraftskins.com/skin/download/{}"


def download_skin(
    skin_url: str,
    skin_id: int,
    output_dir: Path | str = "images/skins",
    *,
    timeout: float = 30.0,
) -> bool:
    """Save response bytes on HTTP 200; return False for other HTTP statuses.

    Transport and filesystem errors propagate to the caller. The request and
    file are closed even if writing fails.
    """
    with requests.get(skin_url.format(skin_id), stream=True, timeout=timeout) as response:
        if response.status_code == 200:
            destination = Path(output_dir)
            destination.mkdir(parents=True, exist_ok=True)
            with (destination / f"{skin_id}.png").open("wb") as file:
                file.write(response.content)
            return True
    return False


def main(
    skin_url: str = DEFAULT_SKIN_URL,
    start: int = 15148205,
    stop: int = 15206629,
    output_dir: Path | str = "images/skins",
    *,
    max_workers: int = 20,
    batch_size: int = 100,
    timeout: float = 30.0,
) -> None:
    """Download the half-open ID range with bounded concurrent submissions."""
    with (
        Bar("Processing", max=max(0, stop - start)) as bar,
        ThreadPoolExecutor(max_workers=max_workers) as executor,
    ):
        for skin_ids in batched(range(start, stop), batch_size, strict=False):
            futures = [
                executor.submit(download_skin, skin_url, skin_id, output_dir, timeout=timeout)
                for skin_id in skin_ids
            ]
            for future in as_completed(futures):
                if future.result():
                    bar.next()


if __name__ == "__main__":
    main()
