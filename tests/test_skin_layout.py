"""Published Minecraft atlas coordinates, including nonuniform UV orientation."""

import numpy as np
import pytest

from minecraft_skin_gan.skin_layout import (
    atlas_layout,
    composite_locked,
    face_pixels,
    region_masks,
)


def test_published_head_and_limbs_uv_rectangles():
    layout = atlas_layout()
    assert layout["head"]["base"]["front"] == [8, 8, 8, 8]
    assert layout["head"]["base"]["back"] == [24, 8, 8, 8]
    assert layout["head"]["overlay"]["top"] == [40, 0, 8, 8]
    assert layout["body"]["base"]["front"] == [20, 20, 8, 12]
    assert layout["right_arm"]["base"]["front"] == [44, 20, 4, 12]
    assert layout["left_arm"]["base"]["front"] == [36, 52, 4, 12]
    assert layout["left_arm"]["overlay"]["back"] == [60, 52, 4, 12]
    assert layout["right_leg"]["overlay"]["front"] == [4, 36, 4, 12]
    assert layout["left_leg"]["base"]["front"] == [20, 52, 4, 12]
    assert layout["left_leg"]["overlay"]["front"] == [4, 52, 4, 12]


def test_slim_uv_width_and_shift_follow_published_mapping():
    layout = atlas_layout("slim")
    assert layout["right_arm"]["base"]["front"] == [44, 20, 3, 12]
    assert layout["right_arm"]["base"]["right"] == [47, 20, 4, 12]
    assert layout["left_arm"]["base"]["back"] == [43, 52, 3, 12]
    assert layout["left_arm"]["overlay"]["bottom"] == [55, 48, 3, 4]
    with pytest.raises(ValueError):
        atlas_layout("auto")


def test_nonuniform_face_corners_and_bottom_vertical_flip():
    y, x = np.indices((64, 64))
    pixels = np.stack([x, y, (x + y) % 256, np.full_like(x, 255)], axis=-1).astype(np.uint8)
    front = face_pixels(pixels, "head", "base", "front")
    assert front[0, 0].tolist() == [8, 8, 16, 255]
    assert front[-1, -1].tolist() == [15, 15, 30, 255]
    bottom = face_pixels(pixels, "head", "base", "bottom")
    assert bottom[0, 0].tolist() == [16, 7, 23, 255]
    assert bottom[-1, -1].tolist() == [23, 0, 23, 255]
    back = face_pixels(pixels, "head", "base", "back")
    assert back[0, 0].tolist() == [24, 8, 32, 255]
    with pytest.raises(ValueError, match="region"):
        face_pixels(pixels, "unknown", "base", "front")


def test_region_masks_cover_only_used_texels_and_keep_layers_together():
    masks = region_masks()
    assert masks["head"][8, 8] and masks["head"][8, 40]
    assert not masks["head"][0, 0]
    assert masks["head"].sum() == 768
    assert masks["body"].sum() == 704
    assert masks["left_arm"].sum() == 448
    assert not np.any(sum(mask.astype(int) for mask in masks.values()) > 1)
    assert region_masks("slim")["left_arm"].sum() == 384


def test_locked_regions_are_byte_exact_and_inputs_unmodified():
    rng = np.random.default_rng(42)
    original = rng.integers(0, 256, (64, 64, 4), np.uint8)
    candidate = rng.integers(0, 256, (64, 64, 4), np.uint8)
    original_before, candidate_before = original.copy(), candidate.copy()
    result = composite_locked(original, candidate, ["head", "left_arm"], model_type="slim")
    masks = region_masks("slim")
    locked = masks["head"] | masks["left_arm"]
    np.testing.assert_array_equal(result[locked], original[locked])
    np.testing.assert_array_equal(result[~locked], candidate[~locked])
    np.testing.assert_array_equal(original, original_before)
    np.testing.assert_array_equal(candidate, candidate_before)
    assert result is not candidate
    np.testing.assert_array_equal(composite_locked(original, candidate, []), candidate)
    with pytest.raises(ValueError, match="region"):
        composite_locked(original, candidate, ["cape"])


@pytest.mark.parametrize(
    "pixels",
    [np.zeros((32, 64, 4), np.uint8), np.zeros((64, 64, 4)), np.zeros((64, 64, 3), np.uint8)],
)
def test_invalid_pixel_contract_rejected(pixels):
    with pytest.raises(ValueError):
        composite_locked(pixels, pixels, ["head"])
