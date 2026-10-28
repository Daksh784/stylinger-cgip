"""
Unit tests for segmentation, contour detection, and torso estimation
"""

import numpy as np
import cv2
from src.preprocessing import decode_image_safely
from src.segmentation import (
    segment_foreground_threshold,
    segment_foreground_grabcut,
    extract_body_regions
)


def test_segment_foreground_threshold(synthetic_user_image_bytes):
    img = decode_image_safely(synthetic_user_image_bytes)
    mask = segment_foreground_threshold(img)
    assert mask.shape == img.shape[:2]
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 255})
    # Foreground must have non-zero pixels
    assert np.count_nonzero(mask) > 0


def test_extract_body_regions_extracts_plausible_boxes(synthetic_user_image_bytes):
    img = decode_image_safely(synthetic_user_image_bytes)
    result = extract_body_regions(img, method="threshold")

    assert "subject_bbox" in result
    assert "torso_bbox" in result
    assert "centroid" in result

    sx, sy, sw, sh = result["subject_bbox"]
    tx, ty, tw, th = result["torso_bbox"]
    cx, cy = result["centroid"]

    # Torso must be contained vertically within the subject
    assert tx >= 0
    assert ty >= sy
    assert ty + th <= img.shape[0]
    assert tw > 0 and th > 0

    # Centroid should be within subject bounds
    assert sx <= cx <= sx + sw
    assert sy <= cy <= sy + sh


def test_detect_head_and_neck_landmarks():
    from src.segmentation import detect_head_and_neck
    # Synthetic image with skin tone block in upper region
    img = np.zeros((400, 300, 3), dtype=np.uint8)
    # Background: dark studio wall
    img[:] = [50, 50, 50]
    # Head / face in upper region (y=40..120, x=120..180) with warm skin tone BGR
    # BGR approx [120, 150, 200] corresponds to Cr~155, Cb~105, H~15
    img[40:120, 120:180] = [120, 150, 200]

    landmarks = detect_head_and_neck(img)
    assert landmarks is not None
    assert "head_bbox" in landmarks
    assert "neck_point" in landmarks
    hx, hy, hw, hh = landmarks["head_bbox"]
    assert hy <= 50
    assert hh >= 60
    assert landmarks["neck_point"][1] >= 110


def test_shoulder_landmarks_and_proportions(synthetic_user_image_bytes):
    img = decode_image_safely(synthetic_user_image_bytes)
    result = extract_body_regions(img, method="threshold")

    assert "left_shoulder" in result
    assert "right_shoulder" in result
    lx, ly = result["left_shoulder"]
    rx, ry = result["right_shoulder"]

    assert lx < rx
    assert ly >= result["neck_point"][1]
    # Sizing profile contains landmarks dictionary
    sizing = result.get("sizing_profile")
    assert sizing is not None
    assert "landmarks" in sizing
    assert "left_shoulder" in sizing["landmarks"]
    assert sizing.get("size_match_supported") is False
    assert sizing.get("confidence_score", 0) == 0


def test_shoulder_width_ignores_outstretched_arms(synthetic_person_arms_out_bytes):
    img = decode_image_safely(synthetic_person_arms_out_bytes)
    result = extract_body_regions(img, method="threshold")
    sx, sy, sw, sh = result["subject_bbox"]
    tx, ty, tw, th = result["torso_bbox"]
    # Silhouette includes raised arms; torso must stay much narrower.
    assert tw < sw * 0.55
    assert tw < 160


def test_output_dimensions_match_input_aspect(synthetic_user_image_bytes):
    img = decode_image_safely(synthetic_user_image_bytes)
    result = extract_body_regions(img, method="threshold")
    assert result["image_size"] == (img.shape[1], img.shape[0])
    tx, ty, tw, th = result["torso_bbox"]
    assert ty + th <= img.shape[0]
