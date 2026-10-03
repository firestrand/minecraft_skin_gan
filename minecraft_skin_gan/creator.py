"""Local gallery, exact favorite exports, and explicit palette postprocessing."""

import base64
import hashlib
import html
import json
import re
import tempfile
import zipfile
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import quote

import numpy as np
from PIL import Image


def _skin(path: Path) -> np.ndarray:
    if path.is_symlink() or path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("Skin must be a regular local image below 4 MiB")
    with Image.open(path) as image:
        if image.format != "PNG" or image.size != (64, 64):
            raise ValueError("Skin must be a 64x64 PNG")
        return np.asarray(image.convert("RGBA")).copy()


def _publish_file(staging: Path, output: Path) -> None:
    # Staging shares the destination filesystem. A hard link publishes complete
    # bytes atomically and refuses any existing file without an empty-file window.
    output.hardlink_to(staging)


def recolor_skin(source: Path | str, output: Path | str, palette: Sequence[str]) -> Path:
    """Nearest RGB palette postprocessing preserves alpha and invisible RGB bytes."""
    destination = Path(output)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if not 1 <= len(palette) <= 256 or any(
        re.fullmatch(r"#[0-9a-fA-F]{6}", color) is None for color in palette
    ):
        raise ValueError("Palette requires 1..256 #RRGGBB colors")
    pixels = _skin(Path(source))
    colors = np.asarray(
        [[int(color[index : index + 2], 16) for index in (1, 3, 5)] for color in palette],
        dtype=np.int32,
    )
    visible = pixels[..., 3] > 0
    rgb = pixels[..., :3][visible].astype(np.int32)
    nearest = np.argmin(np.square(rgb[:, None, :] - colors[None, :, :]).sum(axis=2), axis=1)
    pixels[..., :3][visible] = colors[nearest].astype(np.uint8)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".palette-", dir=destination.parent) as directory:
        staging = Path(directory) / "skin.png"
        with Image.fromarray(pixels) as image:
            image.save(staging)
        _publish_file(staging, destination)
    return destination


def preserve_regions(
    original: Path | str,
    candidate: Path | str,
    output: Path | str,
    regions: Sequence[str],
    *,
    model_type: str = "classic",
) -> Path:
    """Composite explicit regions from an original skin into a candidate."""
    from minecraft_skin_gan.skin_layout import composite_locked

    destination = Path(output)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    pixels = composite_locked(
        _skin(Path(original)), _skin(Path(candidate)), regions, model_type=model_type
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".preserve-", dir=destination.parent) as directory:
        staging = Path(directory) / "skin.png"
        with Image.fromarray(pixels) as image:
            image.save(staging)
        _publish_file(staging, destination)
    return destination


def create_gallery(
    source_directory: Path | str,
    output_directory: Path | str,
    *,
    model_type: str = "classic",
    alpha_mode: str = "original",
) -> Path:
    """Copy a batch into an offline gallery with portable favorite selection JSON."""
    source, output = Path(source_directory), Path(output_directory)
    from minecraft_skin_gan.preview import create_preview
    from minecraft_skin_gan.skin_layout import atlas_layout

    atlas_layout(model_type)
    if alpha_mode not in ("original", "opaque-base"):
        raise ValueError("alpha_mode must be original or opaque-base")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    paths = sorted(source.glob("*.png"))
    if not 1 <= len(paths) <= 256:
        raise ValueError("Gallery requires 1..256 PNGs; choose a smaller review batch")
    generation_path = source / "generation.json"
    generation = json.loads(generation_path.read_text()) if generation_path.exists() else {}
    if not isinstance(generation, dict):
        raise ValueError("Generation metadata must be an object")
    files = []
    cards = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".gallery-", dir=output.parent) as directory:
        staging = Path(directory) / "gallery"
        staging.mkdir()
        for path in paths:
            _skin(path)
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            (staging / path.name).write_bytes(data)
            preview_name = path.name + ".preview.html"
            create_preview(
                staging / path.name,
                staging / preview_name,
                model_type=model_type,
                alpha_mode=alpha_mode,
            )
            files.append({"name": path.name, "sha256": digest})
            encoded = base64.b64encode(data).decode("ascii")
            name = html.escape(path.name, quote=True)
            url = html.escape("./" + quote(path.name, safe=""), quote=True)
            preview_url = html.escape("./" + quote(preview_name, safe=""), quote=True)
            cards.append(
                f'<article><img alt="{name}" src="data:image/png;base64,{encoded}"><p>{name}</p><label><input type="checkbox" value="{name}"> Favorite</label> <a download="{name}" href="data:image/png;base64,{encoded}">Download PNG</a> <a href="{url}">Open PNG</a> <a href="{preview_url}">3D preview</a></article>'
            )
        metadata = {
            "schema": "minecraft-skin-gan.gallery/v1",
            "model_type": model_type,
            "preview_alpha_mode": alpha_mode,
            "generation": generation,
            "files": files,
        }
        (staging / "gallery.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False) + "\n"
        )
        page = (
            """<!doctype html><html lang="en"><meta charset="utf-8"><title>Skin gallery</title>
<style>body{font:16px system-ui;background:#171b22;color:#eee;margin:24px}main{display:flex;flex-wrap:wrap;gap:16px}article{padding:16px;background:#262d37}img{width:256px;height:256px;image-rendering:pixelated;background:repeating-conic-gradient(#ccc 0% 25%,#eee 0% 50%) 0/16px 16px}a{color:#8bd}button{padding:12px;margin:12px 0}</style>
<h1>Skin gallery</h1><p>Compare candidates, mark favorites, and download the original PNGs. Save favorites.json to export the exact selection with the skin-gan export command.</p><button id="save">Save Favorites</button><main>"""
            + "".join(cards)
            + """</main>
<script>document.getElementById('save').addEventListener('click',()=>{const selected=Array.from(document.querySelectorAll('input:checked'),x=>x.value);const url=URL.createObjectURL(new Blob([JSON.stringify(selected,null,2)],{type:'application/json'}));const link=document.createElement('a');link.href=url;link.download='favorites.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});</script></html>"""
        )
        (staging / "index.html").write_text(page)
        output.mkdir()
        try:
            staging.replace(output)
        except BaseException:
            output.rmdir()
            raise
    return output / "index.html"


def export_favorites(
    gallery_directory: Path | str, selected: Sequence[str], output_path: Path | str
) -> Path:
    """Export checked original PNG bytes and generation/selection provenance."""
    root, output = Path(gallery_directory), Path(output_path)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if (
        isinstance(selected, str)
        or not selected
        or any(not isinstance(name, str) for name in selected)
        or len(set(selected)) != len(selected)
    ):
        raise ValueError("Choose a nonempty selection without duplicates")
    metadata = json.loads((root / "gallery.json").read_text())
    if not isinstance(metadata, dict) or metadata.get("schema") != "minecraft-skin-gan.gallery/v1":
        raise ValueError("Unsupported gallery metadata")
    files = metadata.get("files")
    if not isinstance(files, list) or any(not isinstance(item, dict) for item in files):
        raise ValueError("Invalid gallery files")
    expected = {item.get("name"): item.get("sha256") for item in files}
    for name in selected:
        if not isinstance(name, str) or Path(name).name != name or name not in expected:
            raise ValueError("Selection must name images in this gallery")
        path = root / name
        _skin(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected[name]:
            raise ValueError(f"Gallery image changed: {name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".favorites-", dir=output.parent) as directory:
        staging = Path(directory) / "selection.zip"
        with zipfile.ZipFile(staging, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in selected:
                data = (root / name).read_bytes()
                if hashlib.sha256(data).hexdigest() != expected[name]:
                    raise ValueError(f"Gallery image changed during export: {name}")
                archive.writestr(name, data)
            archive.writestr(
                "selection.json",
                json.dumps(
                    {
                        "schema": "minecraft-skin-gan.selection/v1",
                        "selected": list(selected),
                        "generation": metadata.get("generation", {}),
                        "model_type": metadata.get("model_type"),
                        "preview_alpha_mode": metadata.get("preview_alpha_mode", "original"),
                        "files": [item for item in files if item["name"] in selected],
                    },
                    indent=2,
                    allow_nan=False,
                ),
            )
        _publish_file(staging, output)
    return output
