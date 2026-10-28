"""
Unit tests for garment transformation and category-based alignment
"""

import numpy as np
import cv2
from src.preprocessing import decode_image_safely
from src.transform import (
    extract_garment_alpha,
    compute_target_box,
    transform_and_align_garment,
    measure_garment_geometry,
    normalize_category,
)


def test_extract_garment_alpha_on_white_background(synthetic_garment_image_bytes):
    garment_bgr = decode_image_safely(synthetic_garment_image_bytes)
    bgr, alpha = extract_garment_alpha(garment_bgr)

    assert bgr.shape[:2] == garment_bgr.shape[:2]
    assert alpha.shape == garment_bgr.shape[:2]
    # Background corners should be masked out (0 or near 0)
    assert alpha[0, 0] == 0
    # Center garment should be opaque
    assert alpha[100, 100] == 255


def test_extract_garment_alpha_on_rgba(synthetic_rgba_garment_bytes):
    garment_rgba = decode_image_safely(synthetic_rgba_garment_bytes)
    bgr, alpha = extract_garment_alpha(garment_rgba)

    assert bgr.shape[:2] == (150, 150)
    assert alpha.shape == (150, 150)
    assert alpha[0, 0] == 0
    assert alpha[50, 50] == 255


def test_compute_target_box_categories():
    subj = (50, 50, 200, 400)
    torso = (70, 100, 160, 150)
    shape = (500, 300)

    # Tops should match torso
    tx, ty, tw, th = compute_target_box("Tops", subj, torso, shape)
    assert (tx, ty, tw, th) == torso

    # Outerwear should be wider
    ox, oy, ow, oh = compute_target_box("Outerwear", subj, torso, shape)
    assert ow > tw
    assert oh > th

    # Bottoms should start below waist
    bx, by, bw, bh = compute_target_box("Bottoms", subj, torso, shape)
    assert by >= ty + th // 2


def test_transform_and_align_garment(synthetic_garment_image_bytes):
    garment_img = decode_image_safely(synthetic_garment_image_bytes)
    bgr, alpha = extract_garment_alpha(garment_img)
    target_shape = (500, 300)
    target_box = (50, 100, 150, 180)

    aligned_bgr, aligned_alpha = transform_and_align_garment(
        garment_bgr=bgr,
        garment_alpha=alpha,
        target_shape=target_shape,
        target_box=target_box,
        user_scale=1.0,
        offset_x=10,
        offset_y=-10
    )

    assert aligned_bgr.shape == (500, 300, 3)
    assert aligned_alpha.shape == (500, 300)
    assert np.count_nonzero(aligned_alpha) > 0


def test_extract_garment_model_suppression():
    # Synthetic photo of a model wearing a dark garment with a narrow head at top
    img = np.ones((200, 200, 3), dtype=np.uint8) * 250  # white studio background
    # Narrow head/hat in upper 30% (y=10..60, width 40px)
    img[10:60, 80:120] = [60, 60, 60]
    # Wide garment shoulders below y=60 (y=60..180, width 140px)
    img[60:180, 30:170] = [30, 80, 40]

    bgr, alpha = extract_garment_alpha(img, suppress_model=True)
    # Mask should have suppressed the upper head region above y=60
    assert alpha[20, 100] == 0
    # Fabric body should be preserved
    assert alpha[100, 100] > 0


def test_bottoms_waist_boundary_enforcement(synthetic_garment_image_bytes):
    garment_img = decode_image_safely(synthetic_garment_image_bytes)
    bgr, alpha = extract_garment_alpha(garment_img)
    target_shape = (600, 400)
    # Target box starts at waist y=280
    target_box = (80, 280, 240, 280)

    aligned_bgr, aligned_alpha = transform_and_align_garment(
        garment_bgr=bgr,
        garment_alpha=alpha,
        target_shape=target_shape,
        target_box=target_box,
        category="Jeans"
    )

    # Any area strictly above the waist (y < 280) MUST have zero alpha
    assert np.count_nonzero(aligned_alpha[:278, :]) == 0
    # Lower body should have garment fabric
    assert np.count_nonzero(aligned_alpha[280:, :]) > 0


def test_outerwear_lapel_inner_shirt_preservation(synthetic_garment_image_bytes):
    garment_img = decode_image_safely(synthetic_garment_image_bytes)
    bgr, alpha = extract_garment_alpha(garment_img)
    target_shape = (600, 400)
    target_box = (60, 120, 280, 220)

    aligned_bgr, aligned_alpha = transform_and_align_garment(
        garment_bgr=bgr,
        garment_alpha=alpha,
        target_shape=target_shape,
        target_box=target_box,
        category="Outerwear"
    )

    # Outerwear should be placed
    assert np.count_nonzero(aligned_alpha) > 0
    # At the center collar opening, alpha is attenuated so inner shirt shows through
    center_x = 60 + 280 // 2
    top_y = 120
    # Center lapel is significantly less opaque than outer shoulder fabric
    assert aligned_alpha[top_y + 10, center_x] < aligned_alpha[top_y + 10, center_x - 80]


def test_normalize_category_mapping():
    assert normalize_category("T-Shirts") == "tee"
    assert normalize_category("Shirts") == "shirt"
    assert normalize_category("Outerwear") == "outerwear"
    assert normalize_category("Jeans") == "bottoms"
    assert normalize_category("Dresses") == "dress"
    assert normalize_category("mystery-item") == "other"


def test_measure_garment_geometry_uses_body_not_sleeves():
    alpha = np.zeros((200, 220), dtype=np.uint8)
    # Narrow body
    alpha[40:180, 70:150] = 255
    # Wide sleeves
    alpha[40:90, 10:70] = 255
    alpha[40:90, 150:210] = 255
    geom = measure_garment_geometry(alpha)
    assert geom["has_sleeves"] is True
    assert geom["body_width"] < 100
    assert geom["body_width"] < geom["crop"][2] * 0.55


def test_transform_preserves_aspect_ratio(synthetic_rgba_garment_bytes):
    from src.preprocessing import decode_image_safely
    img = decode_image_safely(synthetic_rgba_garment_bytes)
    bgr, alpha = extract_garment_alpha(img)
    ys, xs = np.where(alpha > 15)
    src_aspect = (np.max(xs) - np.min(xs) + 1) / float(np.max(ys) - np.min(ys) + 1)

    aligned_bgr, aligned_alpha = transform_and_align_garment(
        bgr, alpha, (400, 300), (80, 80, 100, 140), category="T-Shirts"
    )
    ys2, xs2 = np.where(aligned_alpha > 15)
    dst_aspect = (np.max(xs2) - np.min(xs2) + 1) / float(np.max(ys2) - np.min(ys2) + 1)
    assert abs(dst_aspect - src_aspect) < 0.12


def test_transform_does_not_cover_full_canvas(synthetic_garment_image_bytes):
    garment_img = decode_image_safely(synthetic_garment_image_bytes)
    bgr, alpha = extract_garment_alpha(garment_img)
    aligned_bgr, aligned_alpha = transform_and_align_garment(
        bgr, alpha, (500, 400), (120, 100, 90, 130), category="Outerwear"
    )
    coverage = np.count_nonzero(aligned_alpha > 20) / float(500 * 400)
    assert coverage < 0.45
    # Corners remain empty
    assert aligned_alpha[0, 0] == 0
    assert aligned_alpha[0, 399] == 0


def test_jacket_scale_tracks_torso_width(synthetic_garment_image_bytes):
    garment_img = decode_image_safely(synthetic_garment_image_bytes)
    bgr, alpha = extract_garment_alpha(garment_img)
    torso_w = 90
    _, aligned_alpha = transform_and_align_garment(
        bgr, alpha, (500, 360), (135, 90, torso_w, 140), category="Jacket"
    )
    ys, xs = np.where(aligned_alpha > 20)
    placed_w = int(np.max(xs) - np.min(xs) + 1)
    # Jacket may include modest ease/sleeves but must not balloon to 2× torso.
    assert placed_w < torso_w * 2.1
    assert placed_w > torso_w * 0.55

