"""
Unit tests for alpha compositing, feathering, and encoding
"""

import numpy as np
import cv2
from src.compositing import feather_mask, composite_garment, encode_image


def test_feather_mask_smoothes_edges():
    # Binary square mask
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:70, 30:70] = 255

    feathered = feather_mask(mask, feather_radius=5)
    assert feathered.shape == mask.shape
    # Border pixels that were 0 should now have intermediate gradients (0 < val < 255)
    border_val = feathered[28, 50]
    assert 0 <= border_val < 255
    # Center should remain high
    assert feathered[50, 50] > 200


def test_composite_garment_preserves_background_outside_alpha():
    user = np.full((100, 100, 3), 100, dtype=np.uint8)
    garment = np.full((100, 100, 3), 200, dtype=np.uint8)
    alpha = np.zeros((100, 100), dtype=np.uint8)
    # Garment only in center
    alpha[40:60, 40:60] = 255

    result = composite_garment(user, garment, alpha, feather_radius=1, match_lighting=False)
    # Corner should be unchanged user pixel
    assert np.all(result[0, 0] == [100, 100, 100])
    # Center should be garment pixel
    assert np.all(result[50, 50] == [200, 200, 200])


def test_encode_image_png_and_jpeg():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    png_bytes = encode_image(img, format_ext="png")
    assert png_bytes.startswith(b"\x89PNG")

    jpg_bytes = encode_image(img, format_ext="jpg")
    assert jpg_bytes.startswith(b"\xff\xd8")


def test_composite_garment_protects_face_and_hands():
    user = np.full((120, 100, 3), 120, dtype=np.uint8)
    garment = np.full((120, 100, 3), 220, dtype=np.uint8)
    # Garment alpha covers entire top-to-bottom
    alpha = np.full((120, 100), 255, dtype=np.uint8)

    # User head at y=0..30
    head_bbox = (20, 0, 60, 30)

    # User hands skin detected at waist y=90..110
    skin_mask = np.zeros((120, 100), dtype=np.uint8)
    skin_mask[90:110, 40:60] = 255

    result = composite_garment(
        user_bgr=user,
        garment_bgr=garment,
        garment_alpha=alpha,
        feather_radius=1,
        match_lighting=False,
        head_bbox=head_bbox,
        skin_mask=skin_mask,
        category="Tops"
    )

    # Face region (e.g. y=15, x=50) should preserve original user pixels
    assert np.all(result[15, 50] == [120, 120, 120])
    # Hands region (e.g. y=100, x=50) should preserve user skin (not fully covered by garment)
    # Result in hands region should be close to user (120) rather than garment (220)
    assert result[100, 50, 0] < 150


def test_composite_does_not_replace_full_person_or_background():
    user = np.full((80, 80, 3), 40, dtype=np.uint8)
    garment = np.full((80, 80, 3), 250, dtype=np.uint8)
    alpha = np.full((80, 80), 255, dtype=np.uint8)
    person = np.zeros((80, 80), dtype=np.uint8)
    person[10:70, 25:55] = 255

    result = composite_garment(
        user, garment, alpha,
        feather_radius=0,
        match_lighting=False,
        person_mask=person,
        head_bbox=(30, 5, 20, 16),
    )
    # Background corner stays user pixel
    assert np.all(result[0, 0] == [40, 40, 40])
    # Face box stays user pixel
    assert np.all(result[10, 40] == [40, 40, 40])
    # Torso inside person mask is garment
    assert int(result[40, 40, 0]) > 200

