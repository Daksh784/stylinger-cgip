"""
Unit tests for preprocessing operations
"""

import numpy as np
import cv2
from src.preprocessing import (
    normalize_dimensions,
    reduce_noise,
    adjust_contrast_brightness,
    to_hsv
)


def test_normalize_dimensions_downsamples_large_image():
    # Large 2000x1200 image
    large = np.zeros((2000, 1200, 3), dtype=np.uint8)
    resized, scale = normalize_dimensions(large, max_dimension=1000)
    assert resized.shape == (1000, 600, 3)
    assert scale == 0.5


def test_normalize_dimensions_leaves_small_image_unchanged():
    small = np.zeros((400, 300, 3), dtype=np.uint8)
    resized, scale = normalize_dimensions(small, max_dimension=800)
    assert resized.shape == (400, 300, 3)
    assert scale == 1.0


def test_reduce_noise_gaussian_and_median():
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    gauss = reduce_noise(img, method="gaussian", ksize=3)
    assert gauss.shape == img.shape

    med = reduce_noise(img, method="median", ksize=3)
    assert med.shape == img.shape


def test_clahe_contrast_enhancement():
    # Dark image
    img = np.full((100, 100, 3), 50, dtype=np.uint8)
    enhanced = adjust_contrast_brightness(img, clip_limit=2.0)
    assert enhanced.shape == img.shape
    assert enhanced.dtype == np.uint8


def test_to_hsv_conversion():
    img = np.full((50, 50, 3), (255, 0, 0), dtype=np.uint8)  # Blue in BGR
    hsv = to_hsv(img)
    assert hsv.shape == (50, 50, 3)
    # Blue hue in OpenCV HSV is ~120
    assert 110 <= hsv[0, 0, 0] <= 130
