"""Independent geometric fixtures for the renderer diagnostic contract."""

import json

import numpy as np
import pytest

from minecraft_skin_gan.skin_checks import layer_masks, skin_diagnostics
from minecraft_skin_gan.skin_layout import atlas_layout, region_masks


@pytest.mark.parametrize("model_type, count", [("classic", 1632), ("slim", 1568)])
def test_alpha_masks_cover_only_used_texels(model_type, count):
    masks = layer_masks(model_type)
    assert masks["base"].sum() == masks["overlay"].sum() == count
    assert not np.any(masks["base"] & masks["overlay"])
    np.testing.assert_array_equal(
        masks["base"] | masks["overlay"],
        np.logical_or.reduce(list(region_masks(model_type).values())),
    )
    assert not masks["base"][0, 0]
    assert masks["base"][8, 8]
    assert masks["overlay"][8, 40]
    pixels = np.zeros((64, 64, 4), np.uint8)
    pixels[masks["base"], 3] = 255
    pixels[8, 40, 3] = 128
    # Unused alpha is irrelevant to both used-layer summaries.
    pixels[0, 0, 3] = 255
    original = pixels.copy()
    report = skin_diagnostics(pixels, model_type=model_type)
    assert report["base_all_opaque"]
    assert report["unused_texel_count"] == 4096 - 2 * count
    assert report["alpha"]["base"]["opaque_count"] == count
    assert report["alpha"]["overlay"]["partial_alpha_count"] == 1
    assert report["alpha"]["overlay"]["transparent_count"] == count - 1
    assert report["game_acceptance"] == "unverified"
    json.dumps(report, allow_nan=False)
    np.testing.assert_array_equal(pixels, original)


def _geometric_fixture(model_type):
    """Color each face by world coordinates; shared edge endpoints must agree.

    Arrays here are raw atlas orientation. Bottom raw rows run z- to z+,
    unlike oriented bottom face_pixels (z+ to z-).
    """
    pixels = np.zeros((64, 64, 4), np.uint8)
    for layers in atlas_layout(model_type).values():
        for faces in layers.values():
            for face, (x, y, width, height) in faces.items():
                columns = np.linspace(0, 255, width).round().astype(np.uint8)
                rows = np.linspace(0, 255, height).round().astype(np.uint8)
                patch = pixels[y : y + height, x : x + width]
                patch[:, :, 3] = 255
                if face in {"front", "back"}:
                    patch[:, :, 0] = columns if face == "front" else columns[::-1]
                    patch[:, :, 1] = rows[::-1, None]
                    patch[:, :, 2] = 255 if face == "front" else 0
                elif face in {"left", "right"}:
                    patch[:, :, 0] = 0 if face == "left" else 255
                    patch[:, :, 1] = rows[::-1, None]
                    patch[:, :, 2] = columns if face == "left" else columns[::-1]
                else:
                    patch[:, :, 0] = columns
                    patch[:, :, 1] = 255 if face == "top" else 0
                    patch[:, :, 2] = rows[:, None]
    return pixels


@pytest.mark.parametrize("model_type", ["classic", "slim"])
def test_all_twelve_geometric_edges_pair_matching_endpoints(model_type):
    report = skin_diagnostics(_geometric_fixture(model_type), model_type=model_type)
    seams = report["seams"]
    assert len(seams["edges"]) == 144  # Six parts, two layers, twelve cuboid edges.
    assert seams["alpha_mae"] == 0
    assert seams["visible_rgb_mae"] == 0
    assert all(edge["visible_rgb_mae"] == 0 for edge in seams["edges"])
    assert {edge["edge_b_reversed"] for edge in seams["edges"]} == {False, True}


def test_discontinuity_measured_without_rejection_or_unsigned_overflow():
    pixels = _geometric_fixture("classic")
    # Head front-left shares the head left-right, excluding corner effects here.
    pixels[9:15, 8, :3] = 0
    pixels[9:15, 7, :3] = 255
    report = skin_diagnostics(pixels)
    edge = next(
        edge
        for edge in report["seams"]["edges"]
        if edge["region"] == "head" and edge["layer"] == "base" and edge["edge_a"] == "front.left"
    )
    assert edge["visible_rgb_mae"] == pytest.approx(6 / 8)
    assert edge["alpha_mae"] == 0
    assert report["base_all_opaque"]


def test_hidden_rgb_is_excluded_and_nonopaque_base_is_descriptive():
    pixels = np.full((64, 64, 4), 255, np.uint8)
    pixels[:, :, 3] = 0
    pixels[8:16, 8:16, :3] = 0
    report = skin_diagnostics(pixels)
    assert not report["base_all_opaque"]
    assert report["seams"]["visible_rgb_mae"] is None
    assert report["seams"]["visible_pair_weight"] == 0
    pixels[8:16, 8:16, 3] = 128
    report = skin_diagnostics(pixels)
    edge = next(
        edge
        for edge in report["seams"]["edges"]
        if edge["region"] == "head" and edge["layer"] == "base" and edge["edge_a"] == "front.left"
    )
    assert edge["alpha_mae"] == pytest.approx(128 / 255)
    assert edge["visible_rgb_mae"] is None
    assert report["alpha"]["base"]["partial_alpha_count"] == 64


@pytest.mark.parametrize(
    "pixels",
    [
        np.zeros((32, 64, 4), np.uint8),
        np.zeros((64, 64, 3), np.uint8),
        np.zeros((64, 64, 4), np.float32),
        np.zeros((64, 64, 4), np.int16),
    ],
)
def test_invalid_rgba_contract_rejected(pixels):
    with pytest.raises(ValueError, match="RGBA uint8"):
        skin_diagnostics(pixels)


def test_model_type_requires_explicit_supported_choice():
    with pytest.raises(ValueError, match="model_type"):
        skin_diagnostics(np.zeros((64, 64, 4), np.uint8), model_type="auto")


def test_partial_visible_edge_uses_minimum_alpha_weight():
    pixels = np.full((64, 64, 4), 255, np.uint8)
    pixels[8:16, 8:16, :3] = 0
    pixels[8:16, 8:16, 3] = 128
    report = skin_diagnostics(pixels)
    edge = next(
        edge
        for edge in report["seams"]["edges"]
        if edge["region"] == "head" and edge["layer"] == "base" and edge["edge_a"] == "front.left"
    )
    assert edge["pair_count"] == 8
    assert edge["visible_pair_weight"] == pytest.approx(8 * 128 / 255)
    assert edge["visible_rgb_mae"] == 1
    assert edge["alpha_mae"] == pytest.approx(127 / 255)
    # Four affected head edges, all eight pixels long. The aggregate normalizes
    # by all paired visible weights, rather than by 144 equally weighted edges.
    seams = report["seams"]
    assert seams["visible_rgb_mae"] == pytest.approx(
        32 * (128 / 255) / (seams["pair_count"] - 32 + 32 * 128 / 255)
    )


def _geometry_edge_pairs():
    """Derive adjacency from BoxGeometry's six buildPlane calls, not _SEAMS.

    Coordinates use a unit cube; dimensions affect edge lengths but not which
    corner positions coincide. Rows here are oriented face_pixels rows, so the
    bottom's source UV reversal has already been removed.
    """
    planes = {
        "right": (2, 1, 0, -1, -1, 1),
        "left": (2, 1, 0, 1, -1, -1),
        "top": (0, 2, 1, 1, 1, 1),
        "bottom": (0, 2, 1, 1, -1, -1),
        "front": (0, 1, 2, 1, -1, 1),
        "back": (0, 1, 2, -1, -1, -1),
    }
    endpoints = {}
    for face, (u, v, w, u_direction, v_direction, normal) in planes.items():
        corners = np.empty((2, 2, 3), dtype=int)
        for row in range(2):
            for column in range(2):
                corners[row, column, u] = (2 * column - 1) * u_direction
                corners[row, column, v] = (2 * row - 1) * v_direction
                corners[row, column, w] = normal
        for edge, coordinates in {
            "top": corners[0],
            "bottom": corners[-1],
            "left": corners[:, 0],
            "right": corners[:, -1],
        }.items():
            endpoints[f"{face}.{edge}"] = tuple(tuple(point) for point in coordinates)
    groups = {}
    for label, positions in endpoints.items():
        groups.setdefault(frozenset(positions), []).append((label, positions))
    assert len(groups) == 12
    assert all(len(group) == 2 for group in groups.values())
    expected = {}
    for group in groups.values():
        (label_a, positions_a), (label_b, positions_b) = group
        assert positions_a == positions_b or positions_a == positions_b[::-1]
        expected[frozenset((label_a, label_b))] = positions_a == positions_b[::-1]
    return expected


@pytest.mark.parametrize("model_type", ["classic", "slim"])
def test_edge_inventory_and_reversals_match_independent_cuboid_geometry(model_type):
    expected = _geometry_edge_pairs()
    report = skin_diagnostics(_geometric_fixture(model_type), model_type=model_type)
    for region in ("head", "body", "right_arm", "left_arm", "right_leg", "left_leg"):
        for layer in ("base", "overlay"):
            edges = [
                edge
                for edge in report["seams"]["edges"]
                if edge["region"] == region and edge["layer"] == layer
            ]
            observed = {
                frozenset((edge["edge_a"], edge["edge_b"])): edge["edge_b_reversed"]
                for edge in edges
            }
            assert len(edges) == len(observed) == 12
            assert observed == expected
