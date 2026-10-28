"""End-to-end pipeline tests: dimensions, masks, invalid assets, category behavior."""

import numpy as np
import cv2
from src.pipeline import execute_tryon
from src.preprocessing import decode_image_safely


def _decode_data_url(data_url: str) -> np.ndarray:
    import base64
    raw = base64.b64decode(data_url.split("base64,")[1])
    return decode_image_safely(raw)


def test_pipeline_preserves_aspect_and_registers_original(
    synthetic_user_image_bytes, synthetic_rgba_garment_bytes
):
    result = execute_tryon(
        synthetic_user_image_bytes,
        synthetic_rgba_garment_bytes,
        category="T-Shirts",
        enhance_user_contrast=False,
        match_lighting=False,
        feather_radius=1,
    )
    assert result["success"] is True
    out = _decode_data_url(result["image_base64"])
    orig = _decode_data_url(result["original_base64"])
    assert out.shape == orig.shape
    assert result["dimensions"]["width"] == orig.shape[1]
    assert result["dimensions"]["height"] == orig.shape[0]
    assert abs(out.shape[1] / out.shape[0] - 300 / 500) < 0.01


def test_pipeline_flags_model_garment(synthetic_user_image_bytes, synthetic_model_garment_bytes):
    result = execute_tryon(
        synthetic_user_image_bytes,
        synthetic_model_garment_bytes,
        category="Outerwear",
    )
    assert result["asset_validation"]["asset_type"] == "model_detected"
    assert result["warnings"]
    assert result["preview_reliable"] is False


def test_pipeline_transparent_asset_ok(synthetic_user_image_bytes, synthetic_rgba_garment_bytes):
    result = execute_tryon(
        synthetic_user_image_bytes,
        synthetic_rgba_garment_bytes,
        category="Shirts",
    )
    assert result["asset_validation"]["asset_type"] == "transparent_png"
    out = _decode_data_url(result["image_base64"])
    orig = _decode_data_url(result["original_base64"])
    # Background corner should stay close to the registered original
    assert np.mean(np.abs(out[2, 2].astype(np.int16) - orig[2, 2].astype(np.int16))) < 12


def test_pipeline_unknown_category_warns(synthetic_user_image_bytes, synthetic_rgba_garment_bytes):
    result = execute_tryon(
        synthetic_user_image_bytes,
        synthetic_rgba_garment_bytes,
        category="Accessory",
    )
    assert result["garment_kind"] == "other"
    assert any("category" in w.lower() for w in result["warnings"])


def test_jeans_do_not_cover_upper_torso(synthetic_user_image_bytes, synthetic_rgba_garment_bytes):
    jeans = np.zeros((180, 120, 4), dtype=np.uint8)
    cv2.rectangle(jeans, (20, 10), (55, 170), (180, 90, 30, 255), -1)
    cv2.rectangle(jeans, (65, 10), (100, 170), (180, 90, 30, 255), -1)
    _, buf = cv2.imencode(".png", jeans)
    result = execute_tryon(synthetic_user_image_bytes, buf.tobytes(), category="Jeans")
    out = _decode_data_url(result["image_base64"])
    orig = _decode_data_url(result["original_base64"])
    # Face / upper chest row should match original, not denim
    assert np.mean(np.abs(out[80].astype(np.int16) - orig[80].astype(np.int16))) < 8
