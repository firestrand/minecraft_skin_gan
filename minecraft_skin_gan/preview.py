"""Offline HTML/CSS 3D skin preview with verified cuboid UVs and explicit layers.

This local preview preserves supplied alpha by default, including base texels.
Optional opaque-base rendering changes embedded base alpha only, never source
pixels. Neither mode certifies game acceptance or full shader equivalence.
Geometry follows skinview3d's body-part dimensions; no external code is loaded.
"""

import base64
import io
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from minecraft_skin_gan.skin_layout import atlas_layout, face_pixels

_MAX_IMAGE_BYTES = 4 * 1024 * 1024
_STYLE = """
body{font:16px system-ui;background:#20252d;color:#eee;margin:24px;max-width:900px}
label{display:inline-block;padding:8px;cursor:pointer}input{margin-left:12px}
.viewport{height:440px;position:relative;perspective:900px;background:repeating-conic-gradient(#39404b 0% 25%,#313742 0% 50%) 0/32px 32px}
.character{width:128px;height:256px;position:absolute;left:calc(50% - 64px);top:80px;transform-style:preserve-3d;transform:rotateY(0deg)}
.part{position:absolute;transform-style:preserve-3d}.face{position:absolute;left:50%;top:50%;backface-visibility:hidden;image-rendering:pixelated;background-size:100% 100%;background-repeat:no-repeat}
#back:checked~.viewport .character{transform:rotateY(180deg)}
#rotate:checked~.viewport .character{animation:turn 12s linear infinite}
#base:not(:checked)~.viewport .base{display:none}
#overlay:not(:checked)~.viewport .overlay{display:none}
@keyframes turn{from{transform:rotateY(0deg)}to{transform:rotateY(360deg)}}
@media(prefers-reduced-motion:reduce){#rotate:checked~.viewport .character{animation:none;transform:rotateY(35deg)}}
"""


def _texture(pixels: np.ndarray) -> str:
    stream = io.BytesIO()
    with Image.fromarray(pixels) as image:
        image.save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def _parts(pixels: np.ndarray, model_type: str, alpha_mode: str) -> str:
    arm = 3 if model_type == "slim" else 4
    geometry = {
        "head": ((0, 4), (8, 8, 8)),
        "body": ((0, -6), (8, 12, 4)),
        "right_arm": ((-(4 + arm / 2), -6), (arm, 12, 4)),
        "left_arm": (((4 + arm / 2), -6), (arm, 12, 4)),
        "right_leg": ((-2, -18), (4, 12, 4)),
        "left_leg": ((2, -18), (4, 12, 4)),
    }
    parts = []
    for region, ((x, y), dimensions) in geometry.items():
        for layer in ("base", "overlay"):
            expansion = 1 if region == "head" else 0.5
            width, height, depth = [
                (dimension + (expansion if layer == "overlay" else 0)) * 8
                for dimension in dimensions
            ]
            left, top = (8 + x) * 8 - width / 2, (8 - y) * 8 - height / 2
            parts.append(
                f'<div class="part {layer}" data-region="{region}" '
                f'style="left:{left:g}px;top:{top:g}px;width:{width:g}px;height:{height:g}px">'
            )
            faces = {
                "front": (width, height, "rotateY(0deg)", depth / 2),
                "back": (width, height, "rotateY(180deg)", depth / 2),
                "right": (depth, height, "rotateY(90deg)", width / 2),
                "left": (depth, height, "rotateY(-90deg)", width / 2),
                "top": (width, depth, "rotateX(90deg)", height / 2),
                "bottom": (width, depth, "rotateX(-90deg)", height / 2),
            }
            for face, (face_width, face_height, rotation, offset) in faces.items():
                texels = face_pixels(pixels, region, layer, face, model_type=model_type)
                if alpha_mode == "opaque-base" and layer == "base":
                    texels[:, :, 3] = 255
                texture = _texture(texels)
                parts.append(
                    f'<div class="face" data-face="{region}/{layer}/{face}" '
                    f'style="width:{face_width:g}px;height:{face_height:g}px;'
                    f"transform:translate(-50%,-50%) {rotation} translateZ({offset:g}px);"
                    f'background-image:url({texture})"></div>'
                )
            parts.append("</div>")
    return "".join(parts)


def create_preview(
    skin_path: Path | str,
    output_path: Path | str,
    *,
    model_type: str = "classic",
    alpha_mode: str = "original",
) -> Path:
    """Write an offline preview with explicit alpha rendering; refuse collisions.

    original preserves every face's RGBA bytes. opaque-base renders only base
    face alpha as 255, retaining RGB and overlay alpha. Source pixels are never
    edited; this option is a rendering profile, not skin repair or certification.
    """
    if not isinstance(alpha_mode, str) or alpha_mode not in {"original", "opaque-base"}:
        raise ValueError("alpha_mode must be original or opaque-base")
    layout = atlas_layout(model_type)
    output = Path(output_path)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if Path(skin_path).stat().st_size > _MAX_IMAGE_BYTES:
        raise ValueError("preview PNG must be below the 4 MiB input limit")
    try:
        with Image.open(skin_path) as image:
            if image.format != "PNG" or image.size != (64, 64):
                raise ValueError("preview requires a 64x64 PNG skin")
            image.load()
            pixels = np.asarray(image.convert("RGBA"))
    except (UnidentifiedImageError, Image.DecompressionBombError) as error:
        raise ValueError("preview requires a valid 64x64 PNG skin") from error
    metadata = json.dumps(
        {
            "schema": "minecraft-skin-gan.preview/v1",
            "model_type": model_type,
            "layout": layout,
            "alpha_mode": alpha_mode,
            "game_acceptance": "unverified",
        },
        separators=(",", ":"),
    )
    alpha_description = (
        "Original alpha is preserved"
        if alpha_mode == "original"
        else "Base faces render opaque; original RGB and overlay alpha are preserved"
    )
    html = (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Skin preview</title><style>{_STYLE}</style><body>"
        f"<h1>Skin preview · {model_type}</h1>"
        f"<p>Local atlas preview · alpha mode: {alpha_mode}. {alpha_description}; "
        "game-format acceptance is unverified.</p>"
        '<input type="radio" name="view" id="front" checked><label for="front">Front</label>'
        '<input type="radio" name="view" id="back"><label for="back">Back</label>'
        '<input type="checkbox" id="rotate"><label for="rotate">Rotate</label>'
        '<input type="checkbox" id="base" checked><label for="base">Base layer</label>'
        '<input type="checkbox" id="overlay" checked><label for="overlay">Overlay layer</label>'
        f'<div class="viewport" aria-label="3D skin preview"><div class="character">{_parts(pixels, model_type, alpha_mode)}</div></div>'
        f'<script type="application/json" id="layout">{metadata}</script></body></html>\n'
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".preview-", dir=output.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(html)
        # Exclusive hard-link publication is atomic and refuses existing files.
        output.hardlink_to(temporary_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output
