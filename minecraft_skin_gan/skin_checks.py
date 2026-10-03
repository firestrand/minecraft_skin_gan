"""Renderer-profile alpha and cuboid-edge diagnostics, without changing pixels.

See docs/skin-format.md for pinned primary sources and metric limitations.
"""

import numpy as np
from numpy.typing import NDArray

from minecraft_skin_gan.skin_layout import Pixels, atlas_layout, face_pixels

# Edge order follows the 3D axes in Three.js BoxGeometry and skinview3d setUVs.
# Bottom faces are already vertically flipped by face_pixels.
_SEAMS = (
    ("front", "left", "left", "right", False),
    ("front", "right", "right", "left", False),
    ("back", "left", "right", "right", False),
    ("back", "right", "left", "left", False),
    ("top", "bottom", "front", "top", False),
    ("top", "top", "back", "top", True),
    ("top", "left", "left", "top", False),
    ("top", "right", "right", "top", True),
    ("bottom", "top", "front", "bottom", False),
    ("bottom", "bottom", "back", "bottom", True),
    ("bottom", "left", "left", "bottom", True),
    ("bottom", "right", "right", "bottom", False),
)


def layer_masks(model_type: str = "classic") -> dict[str, NDArray[np.bool_]]:
    """Return disjoint masks of the model's used base and overlay atlas texels."""
    masks = {layer: np.zeros((64, 64), dtype=np.bool_) for layer in ("base", "overlay")}
    for layers in atlas_layout(model_type).values():
        for layer, faces in layers.items():
            for x, y, width, height in faces.values():
                masks[layer][y : y + height, x : x + width] = True
    return masks


def _edge(pixels: Pixels, edge: str) -> Pixels:
    if edge == "top":
        return pixels[0]
    if edge == "bottom":
        return pixels[-1]
    if edge == "left":
        return pixels[:, 0]
    return pixels[:, -1]


def _alpha_summary(alpha: Pixels) -> dict[str, int | float]:
    return {
        "texel_count": int(alpha.size),
        "opaque_count": int(np.count_nonzero(alpha == 255)),
        "transparent_count": int(np.count_nonzero(alpha == 0)),
        "partial_alpha_count": int(np.count_nonzero((alpha > 0) & (alpha < 255))),
        "mean_alpha": float(np.mean(alpha, dtype=np.float64) / 255),
    }


def skin_diagnostics(pixels: Pixels, *, model_type: str = "classic") -> dict[str, object]:
    """Describe a 64x64 RGBA uint8 skin under an explicit renderer UV profile.

    Nonopaque base texels and seam discontinuities are observations, not
    rejection criteria. Edge RGB error is weighted by the minimum paired alpha;
    a wholly invisible edge has None RGB error. Metrics normalize bytes to 0..1.
    Texels at corners participate in multiple edges. Inputs remain unchanged.
    """
    array = np.asarray(pixels)
    if array.shape != (64, 64, 4) or array.dtype != np.uint8:
        raise ValueError("skin pixels must be a 64x64 RGBA uint8 array")
    layout = atlas_layout(model_type)
    masks = layer_masks(model_type)
    alpha = {layer: _alpha_summary(array[:, :, 3][mask]) for layer, mask in masks.items()}
    unused = ~(masks["base"] | masks["overlay"])
    edges: list[dict[str, object]] = []
    total_pairs, total_alpha_error, total_weight, total_rgb_error = 0, 0.0, 0.0, 0.0
    for region, layers in layout.items():
        for layer in layers:
            faces = {
                face: face_pixels(array, region, layer, face, model_type=model_type)
                for face in layers[layer]
            }
            for face_a, edge_a, face_b, edge_b, reverse in _SEAMS:
                a, b = _edge(faces[face_a], edge_a), _edge(faces[face_b], edge_b)
                if reverse:
                    b = b[::-1]
                a, b = a.astype(np.float64) / 255, b.astype(np.float64) / 255
                weight = np.minimum(a[:, 3], b[:, 3])
                weight_sum = float(weight.sum())
                rgb_error = float((np.abs(a[:, :3] - b[:, :3]) * weight[:, None]).sum())
                alpha_error = float(np.abs(a[:, 3] - b[:, 3]).sum())
                pairs = len(a)
                edges.append(
                    {
                        "region": region,
                        "layer": layer,
                        "edge_a": f"{face_a}.{edge_a}",
                        "edge_b": f"{face_b}.{edge_b}",
                        "edge_b_reversed": reverse,
                        "pair_count": pairs,
                        "visible_pair_weight": weight_sum,
                        "alpha_mae": alpha_error / pairs,
                        "visible_rgb_mae": rgb_error / (3 * weight_sum) if weight_sum else None,
                    }
                )
                total_pairs += pairs
                total_alpha_error += alpha_error
                total_weight += weight_sum
                total_rgb_error += rgb_error
    return {
        "schema_version": 1,
        "profile": "skinview3d-84906e98-cuboid-v1",
        "model_type": model_type,
        "rgba_contract": "64x64 uint8",
        "game_acceptance": "unverified",
        "alpha": alpha,
        "unused_texel_count": int(unused.sum()),
        "base_all_opaque": alpha["base"]["opaque_count"] == alpha["base"]["texel_count"],
        "seams": {
            "interpretation": "descriptive cuboid-edge discontinuity; no pass/fail threshold",
            "pair_count": total_pairs,
            "visible_pair_weight": total_weight,
            "alpha_mae": total_alpha_error / total_pairs,
            "visible_rgb_mae": total_rgb_error / (3 * total_weight) if total_weight else None,
            "edges": edges,
        },
    }
