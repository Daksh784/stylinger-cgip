"""
Unit tests for safe image decoding and validation
"""

import pytest
import numpy as np
from src.preprocessing import decode_image_safely, ImageValidationError


def test_empty_bytes_raises_error():
    with pytest.raises(ImageValidationError, match="Empty image payload"):
        decode_image_safely(b"")


def test_corrupted_data_raises_error():
    with pytest.raises(ImageValidationError, match="Corrupted or invalid"):
        decode_image_safely(b"not-an-image-data-string-at-all")


def test_invalid_extension_raises_error():
    # Construct a valid GIF file (not allowed, only JPEG/PNG allowed)
    gif_header = b"GIF89a\x01\x00\x01\x00\x80\x00\x00\xff\xff\xff\x00\x00\x00!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;"
    with pytest.raises(ImageValidationError, match="Unsupported image format"):
        decode_image_safely(gif_header)


def test_valid_image_decodes_successfully(synthetic_user_image_bytes):
    img = decode_image_safely(synthetic_user_image_bytes)
    assert isinstance(img, np.ndarray)
    assert img.shape == (500, 300, 3)
    assert img.dtype == np.uint8


def test_validate_garment_asset_transparent_png(synthetic_rgba_garment_bytes):
    from src.transform import validate_garment_asset
    img = decode_image_safely(synthetic_rgba_garment_bytes)
    result = validate_garment_asset(img)
    assert result["valid"] is True
    assert result["asset_type"] == "transparent_png"
    assert result["warning"] is None


def test_validate_garment_asset_studio_flatlay(synthetic_garment_image_bytes):
    from src.transform import validate_garment_asset
    img = decode_image_safely(synthetic_garment_image_bytes)
    result = validate_garment_asset(img)
    assert result["valid"] is True
    assert result["asset_type"] == "studio_flatlay"


def test_validate_garment_asset_model_detection():
    from src.transform import validate_garment_asset
    # Create an image of a model: white background with a human face in the upper third
    model_img = np.full((300, 300, 3), 250, dtype=np.uint8)
    # Human skin face in upper third (y=20..90, x=110..190)
    # BGR [120, 150, 200] corresponds to Cr~155, Cb~105, H~15
    model_img[20:90, 110:190] = [120, 150, 200]
    # Dark jacket body below (y=90..250, x=70..230)
    model_img[90:250, 70:230] = [40, 40, 50]

    result = validate_garment_asset(model_img)
    assert result["asset_type"] == "model_detected"
    assert result["warning"] is not None
    assert "model" in result["warning"].lower()

    rejected = validate_garment_asset(model_img, strict_model_rejection=True)
    assert rejected["valid"] is False
    assert rejected["asset_type"] == "model_detected"

