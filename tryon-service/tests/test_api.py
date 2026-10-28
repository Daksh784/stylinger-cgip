"""
Unit tests for Flask API endpoints and error handling
"""

import io
import pytest
from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health_check(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert "capabilities" in data


def test_tryon_missing_images_returns_400(client):
    response = client.post("/api/tryon/process", data={})
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False
    assert "Missing user photo" in data["error"]


def test_tryon_corrupted_image_returns_400(client, synthetic_garment_image_bytes):
    response = client.post(
        "/api/tryon/process",
        data={
            "user_image": (io.BytesIO(b"garbage-bytes"), "user.jpg"),
            "garment_image": (io.BytesIO(synthetic_garment_image_bytes), "garment.png")
        },
        content_type="multipart/form-data"
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False


def test_tryon_successful_processing(client, synthetic_user_image_bytes, synthetic_garment_image_bytes):
    response = client.post(
        "/api/tryon/process",
        data={
            "user_image": (io.BytesIO(synthetic_user_image_bytes), "user.png"),
            "garment_image": (io.BytesIO(synthetic_garment_image_bytes), "garment.png"),
            "category": "Tops",
            "scale": "1.0",
            "offset_x": "0",
            "offset_y": "0"
        },
        content_type="multipart/form-data"
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert "image_base64" in data
    assert data["image_base64"].startswith("data:image/png;base64,")
    assert "dimensions" in data
    assert "telemetry" in data
    assert "asset_validation" in data
    assert "left_shoulder" in data
    assert "original_base64" in data
    assert data["dimensions"]["width"] == 300
    assert data["dimensions"]["height"] == 500
    assert data["method"] == "approximate_2d_compositing"
    assert data["sizing_profile"]["size_match_supported"] is False


def test_segment_preview_endpoint(client, synthetic_user_image_bytes):
    response = client.post(
        "/api/tryon/segment-preview",
        data={
            "user_image": (io.BytesIO(synthetic_user_image_bytes), "user.png")
        },
        content_type="multipart/form-data"
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert "overlay_base64" in data
    assert "mask_base64" in data
    assert "torso_bbox" in data


def test_numpy_json_serialization(client):
    """Verifies that numpy bools, integers, floats, and arrays serialize cleanly to JSON."""
    import numpy as np
    from app import sanitize_for_json
    from flask import jsonify

    payload = {
        "bool_val": np.bool_(True),
        "int_val": np.int64(123),
        "float_val": np.float32(45.67),
        "arr_val": np.array([1, 2, 3])
    }
    sanitized = sanitize_for_json(payload)
    assert sanitized["bool_val"] is True
    assert type(sanitized["bool_val"]) is bool
    assert sanitized["int_val"] == 123
    assert type(sanitized["int_val"]) is int
    assert isinstance(sanitized["float_val"], float)
    assert sanitized["arr_val"] == [1, 2, 3]

    with app.app_context():
        res = jsonify(payload)
        assert res.status_code == 200
        json_data = res.get_json()
        assert json_data["bool_val"] is True
        assert json_data["int_val"] == 123

