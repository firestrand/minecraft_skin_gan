"""Explicit classic/slim UV regions, independent of model generation.

Atlas offsets and dimensions verified against the renderer's primary source:
https://github.com/bs-community/skinview3d/blob/84906e983a2cf325f33f515b9a71c871f799054b/src/model.ts
(`setUVs`, `SkinObject`); face orientation is defined by Three.js BoxGeometry:
https://github.com/mrdoob/three.js/blob/9b4a2ac29c63ccb43fd51c5661f2f873ac2c39b8/src/geometries/BoxGeometry.js
The bottom-face UV row order is reversed by skinview3d. Coordinates describe
used texels only; they do not imply opacity, originality, or game acceptance.
"""

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

type Pixels = NDArray[np.uint8]
type Layout = dict[str, dict[str, dict[str, list[int]]]]


def _rectangles(u: int, v: int, width: int, height: int, depth: int) -> dict[str, list[int]]:
    return {
        "top": [u + depth, v, width, depth],
        "bottom": [u + width + depth, v, width, depth],
        "left": [u, v + depth, depth, height],
        "front": [u + depth, v + depth, width, height],
        "right": [u + width + depth, v + depth, depth, height],
        "back": [u + width + 2 * depth, v + depth, width, height],
    }


def atlas_layout(model_type: str = "classic") -> Layout:
    """Return fresh UV rectangles [x, y, width, height] for each body part/layer."""
    if model_type not in {"classic", "slim"}:
        raise ValueError("model_type must be classic or slim; automatic guessing is unsupported")
    arm_width = 3 if model_type == "slim" else 4
    parts = {
        "head": ((0, 0), (32, 0), (8, 8, 8)),
        "body": ((16, 16), (16, 32), (8, 12, 4)),
        "right_arm": ((40, 16), (40, 32), (arm_width, 12, 4)),
        "left_arm": ((32, 48), (48, 48), (arm_width, 12, 4)),
        "right_leg": ((0, 16), (0, 32), (4, 12, 4)),
        "left_leg": ((16, 48), (0, 48), (4, 12, 4)),
    }
    return {
        part: {
            "base": _rectangles(*base, *dimensions),
            "overlay": _rectangles(*overlay, *dimensions),
        }
        for part, (base, overlay, dimensions) in parts.items()
    }


def _pixels(pixels: Pixels) -> Pixels:
    array = np.asarray(pixels)
    if array.shape != (64, 64, 4) or array.dtype != np.uint8:
        raise ValueError("skin pixels must be a 64x64 RGBA uint8 array")
    return array


def face_pixels(
    pixels: Pixels,
    region: str,
    layer: str,
    face: str,
    *,
    model_type: str = "classic",
) -> Pixels:
    """Copy texels oriented for a CSS/Three.js face (bottom flips vertically)."""
    array = _pixels(pixels)
    try:
        x, y, width, height = atlas_layout(model_type)[region][layer][face]
    except KeyError as error:
        raise ValueError("unknown skin region, layer, or face") from error
    result = array[y : y + height, x : x + width]
    if face == "bottom":
        result = result[::-1]
    return result.copy()


def region_masks(model_type: str = "classic") -> dict[str, NDArray[np.bool_]]:
    """Return disjoint 64x64 masks covering each region's base and overlay texels."""
    result = {}
    for region, layers in atlas_layout(model_type).items():
        mask = np.zeros((64, 64), dtype=np.bool_)
        for faces in layers.values():
            for x, y, width, height in faces.values():
                mask[y : y + height, x : x + width] = True
        result[region] = mask
    return result


def composite_locked(
    original: Pixels,
    candidate: Pixels,
    regions: Iterable[str],
    *,
    model_type: str = "classic",
) -> Pixels:
    """Copy a candidate, retaining exact original RGBA bytes in locked regions.

    This is output compositing, not a claim that a model understands regions or
    that seams are repaired. Unused atlas texels retain candidate bytes.
    """
    original, candidate = _pixels(original), _pixels(candidate)
    masks = region_masks(model_type)
    locked = np.zeros((64, 64), np.bool_)
    for region in regions:
        if region not in masks:
            raise ValueError(f"unknown skin region: {region}")
        locked |= masks[region]
    result = candidate.copy()
    result[locked] = original[locked]
    return result
