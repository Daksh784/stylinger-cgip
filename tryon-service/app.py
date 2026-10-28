"""
Stylinger Virtual Fashion Try-On Service
Flask Application Entrypoint

Classical Computer Graphics & Image Processing Microservice
"""

import os
import base64
import requests
from io import BytesIO
from typing import Tuple, Optional, Any
from flask import Flask, request, jsonify
from flask.json.provider import DefaultJSONProvider
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge
import numpy as np

from src.pipeline import execute_tryon
from src.preprocessing import ImageValidationError, decode_image_safely, normalize_dimensions
from src.segmentation import extract_body_regions
from src.compositing import encode_image
import cv2

class NumpyJSONProvider(DefaultJSONProvider):
    """
    Extends Flask DefaultJSONProvider to seamlessly serialize NumPy scalars,
    including numpy.bool, numpy.integer, numpy.floating, and numpy.ndarray.
    """
    def default(self, o):
        if isinstance(o, (np.bool_, bool)) or type(o).__name__ == "bool":
            return bool(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def sanitize_for_json(obj: Any) -> Any:
    """
    Recursively converts NumPy booleans, numbers, and arrays into native Python JSON types.
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(x) for x in obj]
    elif isinstance(obj, (np.bool_, bool)) or type(obj).__name__ == "bool":
        return bool(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


app = Flask(__name__)
app.json_provider_class = NumpyJSONProvider
app.json = NumpyJSONProvider(app)

# Request limit: 16 MB max upload
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

# Enable CORS for local development (Vite frontend on port 5173 / 8080)
CORS(
    app,
    resources={r"/api/*": {"origins": ["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:8080", "http://127.0.0.1:8080"]}},
    supports_credentials=True
)


@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(e):
    return jsonify({
        "error": "Payload too large. Maximum file upload size is 16MB.",
        "success": False
    }), 413


@app.errorhandler(Exception)
def handle_unexpected_error(e):
    return jsonify({
        "error": f"An unexpected processing error occurred: {str(e)}",
        "success": False
    }), 500


def extract_bytes_from_input(file_key: str, data_key: str, url_key: str) -> Optional[bytes]:
    """
    Safely retrieves image bytes from multipart form file, base64 payload, or external URL.
    Enforces in-memory extraction with strict size verification.
    """
    # 1. Direct file upload
    if file_key in request.files:
        f = request.files[file_key]
        if f and f.filename:
            content = f.read()
            if content:
                return content

    # 2. Base64 payload in form or JSON
    raw_b64 = None
    if request.is_json:
        raw_b64 = request.json.get(data_key)
    elif request.form.get(data_key):
        raw_b64 = request.form.get(data_key)

    if raw_b64 and isinstance(raw_b64, str):
        # Strip data URI prefix if present
        if "base64," in raw_b64:
            raw_b64 = raw_b64.split("base64,")[1]
        try:
            return base64.b64decode(raw_b64)
        except Exception as err:
            raise ImageValidationError(f"Invalid base64 payload for {data_key}: {str(err)}")

    # 3. Safe external image URL (for catalog products)
    img_url = None
    if request.is_json:
        img_url = request.json.get(url_key)
    elif request.form.get(url_key):
        img_url = request.form.get(url_key)

    if img_url and isinstance(img_url, str):
        if not (img_url.startswith("http://") or img_url.startswith("https://")):
            raise ImageValidationError(f"Invalid URL protocol for {url_key}.")
        try:
            resp = requests.get(img_url, timeout=7, headers={"User-Agent": "StylingerTryOn/1.0"})
            if resp.status_code == 200 and len(resp.content) > 0:
                if len(resp.content) > 16 * 1024 * 1024:
                    raise ImageValidationError("Remote image exceeds 16MB limit.")
                return resp.content
            raise ImageValidationError(f"Failed to fetch image from URL (HTTP {resp.status_code}).")
        except Exception as e:
            raise ImageValidationError(f"Error downloading remote image: {str(e)}")

    return None


@app.route("/api/health", methods=["GET"])
@app.route("/api/tryon/health", methods=["GET"])
def health_check():
    """Service status and computer vision capabilities."""
    return jsonify({
        "status": "healthy",
        "service": "Stylinger Virtual Try-On Engine",
        "engine": "Classical Computer Graphics & Image Processing (OpenCV + NumPy)",
        "opencv_version": cv2.__version__,
        "capabilities": [
            "approximate_2d_compositing",
            "contour_body_estimation",
            "grabcut_segmentation",
            "otsu_thresholding",
            "affine_geometric_transform",
            "edge_feathering",
            "alpha_blending",
            "clahe_luminance_harmonization"
        ]
    }), 200


@app.route("/api/tryon/segment-preview", methods=["POST"])
def preview_segmentation():
    """
    Returns visual mask and contour overlay of the detected human subject.
    Useful for inspecting segmentation quality and bounding box placement.
    """
    try:
        user_bytes = extract_bytes_from_input("user_image", "user_image_base64", "user_image_url")
        if not user_bytes:
            return jsonify({"error": "Missing user photo in request.", "success": False}), 400

        user_img = decode_image_safely(user_bytes)
        user_norm, _ = normalize_dimensions(user_img, max_dimension=800)
        if user_norm.shape[2] == 4:
            user_norm = user_norm[:, :, :3]

        body_info = extract_body_regions(user_norm, method="grabcut")
        mask = body_info["subject_mask"]
        sx, sy, sw, sh = body_info["subject_bbox"]
        tx, ty, tw, th = body_info["torso_bbox"]
        cx, cy = body_info["centroid"]
        lb_x, lb_y, lb_w, lb_h = body_info.get("lower_body_bbox", (0, 0, 0, 0))

        # Render overlay on user image
        overlay = user_norm.copy()
        # Draw torso estimate in emerald green
        cv2.rectangle(overlay, (tx, ty), (tx + tw, ty + th), (80, 200, 120), 2)
        # Draw shoulder span line and acromion points in bright emerald
        ls = body_info.get("left_shoulder")
        rs = body_info.get("right_shoulder")
        if ls and rs:
            cv2.line(overlay, ls, rs, (40, 240, 140), 2)
            cv2.circle(overlay, ls, 5, (40, 240, 140), -1)
            cv2.circle(overlay, rs, 5, (40, 240, 140), -1)
        # Draw collar anchor
        np_pt = body_info.get("neck_point")
        if np_pt:
            cv2.circle(overlay, np_pt, 5, (255, 200, 0), -1)
        # Draw lower body in blue/cyan if detected
        if body_info.get("has_lower_body", False) and lb_w > 0 and lb_h > 0:
            cv2.rectangle(overlay, (lb_x, lb_y), (lb_x + lb_w, lb_y + lb_h), (255, 180, 50), 2)
        # Draw subject box in white/slate
        cv2.rectangle(overlay, (sx, sy), (sx + sw, sy + sh), (200, 200, 200), 1)
        # Draw centroid
        cv2.circle(overlay, (cx, cy), 5, (0, 0, 255), -1)

        overlay_bytes = encode_image(overlay, format_ext="png")
        mask_bytes = encode_image(mask, format_ext="png")

        return jsonify(sanitize_for_json({
            "success": True,
            "overlay_base64": f"data:image/png;base64,{base64.b64encode(overlay_bytes).decode('ascii')}",
            "mask_base64": f"data:image/png;base64,{base64.b64encode(mask_bytes).decode('ascii')}",
            "torso_bbox": {"x": tx, "y": ty, "width": tw, "height": th},
            "lower_body_bbox": {"x": lb_x, "y": lb_y, "width": lb_w, "height": lb_h},
            "has_lower_body": bool(body_info.get("has_lower_body", False)),
            "subject_bbox": {"x": sx, "y": sy, "width": sw, "height": sh},
            "centroid": {"x": cx, "y": cy},
            "left_shoulder": {"x": ls[0], "y": ls[1]} if ls else None,
            "right_shoulder": {"x": rs[0], "y": rs[1]} if rs else None,
            "neck_point": {"x": np_pt[0], "y": np_pt[1]} if np_pt else None,
            "sizing_profile": body_info.get("sizing_profile")
        })), 200

    except ImageValidationError as ve:
        return jsonify({"error": str(ve), "success": False}), 400
    except Exception as e:
        return jsonify({"error": f"Segmentation preview failed: {str(e)}", "success": False}), 500


@app.route("/api/tryon/process", methods=["POST"])
def process_tryon():
    """
    Main Virtual Try-On endpoint.
    Processes user photo and garment image using classical image processing pipeline.
    """
    try:
        user_bytes = extract_bytes_from_input("user_image", "user_image_base64", "user_image_url")
        garment_bytes = extract_bytes_from_input("garment_image", "garment_image_base64", "garment_image_url")

        if not user_bytes:
            return jsonify({"error": "Missing user photo in request.", "success": False}), 400
        if not garment_bytes:
            return jsonify({"error": "Missing garment image in request.", "success": False}), 400

        # Read optional placement parameters with safe boundaries
        def get_param(name, default, cast_fn, min_val, max_val):
            val = None
            if request.is_json:
                val = request.json.get(name)
            elif request.form.get(name) is not None:
                val = request.form.get(name)
            if val is None:
                return default
            try:
                parsed = cast_fn(val)
                return max(min_val, min(parsed, max_val))
            except Exception:
                return default

        category = "Tops"
        if request.is_json and request.json.get("category"):
            category = str(request.json.get("category"))
        elif request.form.get("category"):
            category = str(request.form.get("category"))

        scale = get_param("scale", 1.0, float, 0.4, 2.5)
        offset_x = get_param("offset_x", 0, int, -500, 500)
        offset_y = get_param("offset_y", 0, int, -500, 500)
        rotation_deg = get_param("rotation", 0.0, float, -45.0, 45.0)
        feather_radius = get_param("feather_radius", 3, int, 0, 25)
        opacity = get_param("opacity", 1.0, float, 0.1, 1.0)

        match_lighting = True
        if request.is_json and "match_lighting" in request.json:
            match_lighting = bool(request.json["match_lighting"])
        elif request.form.get("match_lighting") is not None:
            match_lighting = request.form.get("match_lighting").lower() in ["true", "1", "yes"]

        # Run pipeline
        result = execute_tryon(
            user_image_bytes=user_bytes,
            garment_image_bytes=garment_bytes,
            category=category,
            scale=scale,
            offset_x=offset_x,
            offset_y=offset_y,
            rotation_deg=rotation_deg,
            feather_radius=feather_radius,
            match_lighting=match_lighting,
            opacity=opacity
        )

        return jsonify(sanitize_for_json(result)), 200

    except ImageValidationError as ve:
        return jsonify({"error": str(ve), "success": False}), 400
    except Exception as e:
        return jsonify({"error": f"Virtual try-on processing failed: {str(e)}", "success": False}), 500


if __name__ == "__main__":
    port = int(os.environ.get("TRYON_PORT", 5001))
    print(f"[*] Starting Stylinger Virtual Try-On Service on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
