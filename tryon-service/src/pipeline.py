"""
End-to-End Image Processing Pipeline
Stylinger Virtual Fashion Try-On

Orchestrates preprocessing, segmentation, geometric transformation, and compositing.
Produces an approximate 2D overlay — not a 3D garment simulation.
"""

import time
import base64
from typing import Dict, Any
import numpy as np

from .preprocessing import (
    decode_image_safely,
    normalize_dimensions,
    reduce_noise,
    adjust_contrast_brightness,
)
from .segmentation import extract_body_regions
from .transform import (
    validate_garment_asset,
    extract_garment_alpha,
    compute_target_box,
    transform_and_align_garment,
    normalize_category,
)
from .compositing import composite_garment, encode_image


def execute_tryon(
    user_image_bytes: bytes,
    garment_image_bytes: bytes,
    category: str = "Tops",
    scale: float = 1.0,
    offset_x: int = 0,
    offset_y: int = 0,
    rotation_deg: float = 0.0,
    feather_radius: int = 3,
    match_lighting: bool = True,
    opacity: float = 1.0,
    enhance_user_contrast: bool = True,
    segmentation_method: str = "grabcut",
    max_dimension: int = 1024
) -> Dict[str, Any]:
    """
    Executes the modular 2D Virtual Try-On pipeline.
    All operations are in-memory with zero disk persistence of user photos.
    """
    start_time = time.perf_counter()
    telemetry: Dict[str, Any] = {}

    t0 = time.perf_counter()
    user_img = decode_image_safely(user_image_bytes)
    garment_img = decode_image_safely(garment_image_bytes)
    telemetry["decode_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    t1 = time.perf_counter()
    user_norm, _scale_ratio = normalize_dimensions(user_img, max_dimension=max_dimension)
    if len(user_norm.shape) == 3 and user_norm.shape[2] == 4:
        user_norm = user_norm[:, :, :3]

    user_denoised = reduce_noise(user_norm, method="gaussian", ksize=3, sigma=0.8)
    if enhance_user_contrast:
        user_preprocessed = adjust_contrast_brightness(user_denoised, clip_limit=1.6)
    else:
        user_preprocessed = user_denoised
    telemetry["preprocess_ms"] = round((time.perf_counter() - t1) * 1000, 2)

    t2 = time.perf_counter()
    body_info = extract_body_regions(user_preprocessed, method=segmentation_method)
    subject_bbox = body_info["subject_bbox"]
    torso_bbox = body_info["torso_bbox"]
    centroid = body_info["centroid"]
    telemetry["segmentation_ms"] = round((time.perf_counter() - t2) * 1000, 2)

    t3 = time.perf_counter()
    asset_validation = validate_garment_asset(garment_img)
    garment_bgr, garment_alpha = extract_garment_alpha(garment_img)
    kind = normalize_category(category)
    target_box = compute_target_box(
        category=category,
        subject_bbox=subject_bbox,
        torso_bbox=torso_bbox,
        image_shape=user_preprocessed.shape[:2],
        lower_body_bbox=body_info.get("lower_body_bbox"),
        waist_y=body_info.get("waist_y"),
        hip_bbox=body_info.get("hip_bbox"),
        ankle_y=body_info.get("ankle_y"),
        knee_y=body_info.get("knee_y"),
    )

    aligned_garment, aligned_alpha = transform_and_align_garment(
        garment_bgr=garment_bgr,
        garment_alpha=garment_alpha,
        target_shape=user_preprocessed.shape[:2],
        target_box=target_box,
        user_scale=scale,
        offset_x=offset_x,
        offset_y=offset_y,
        rotation_deg=rotation_deg,
        category=category,
        body_info=body_info,
    )
    telemetry["transform_ms"] = round((time.perf_counter() - t3) * 1000, 2)

    t4 = time.perf_counter()
    result_bgr = composite_garment(
        user_bgr=user_preprocessed,
        garment_bgr=aligned_garment,
        garment_alpha=aligned_alpha,
        feather_radius=feather_radius,
        match_lighting=match_lighting,
        opacity=opacity,
        head_bbox=body_info.get("head_bbox"),
        skin_mask=body_info.get("skin_mask"),
        person_mask=body_info.get("subject_mask"),
        category=category,
    )
    telemetry["composite_ms"] = round((time.perf_counter() - t4) * 1000, 2)

    out_png_bytes = encode_image(result_bgr, format_ext="png")
    out_b64 = base64.b64encode(out_png_bytes).decode("ascii")
    original_png = encode_image(user_preprocessed, format_ext="png")
    original_b64 = base64.b64encode(original_png).decode("ascii")

    total_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
    telemetry["total_time_ms"] = total_time_ms

    warnings = []
    if asset_validation.get("warning"):
        warnings.append(asset_validation["warning"])

    if not body_info.get("landmarks_reliable", False):
        warnings.append(
            "Body landmarks are approximate (contour-based, not pose estimation). "
            "Use the scale and offset sliders if the garment does not sit on the shoulders."
        )

    if kind == "other":
        warnings.append(
            "Garment category was not recognized. Applied a conservative upper-body overlay. "
            "Select T-shirt, shirt, jacket, jeans, or dress for a better region."
        )

    if kind == "bottoms" and not body_info.get("has_lower_body", True):
        warnings.append(
            "Lower body appears cut off or not visible. "
            "Upload a full-body photograph showing hips to feet, or use the offset sliders."
        )

    opaque = np.count_nonzero(aligned_alpha > 20)
    if opaque < 80:
        warnings.append("Garment mask is nearly empty after alignment. The preview is not reliable.")

    preview_reliable = bool(
        asset_validation.get("asset_type") in ("transparent_png", "studio_flatlay")
        and bool(body_info.get("landmarks_reliable", False))
        and bool(opaque >= 80)
    )

    def _xy(pt):
        if not pt:
            return None
        return {"x": int(pt[0]), "y": int(pt[1])}

    def _box(b):
        if not b:
            return None
        return {"x": int(b[0]), "y": int(b[1]), "width": int(b[2]), "height": int(b[3])}

    return {
        "success": True,
        "image_base64": f"data:image/png;base64,{out_b64}",
        "original_base64": f"data:image/png;base64,{original_b64}",
        "method": "approximate_2d_compositing",
        "preview_reliable": bool(preview_reliable),
        "garment_kind": str(kind),
        "dimensions": {
            "width": int(user_preprocessed.shape[1]),
            "height": int(user_preprocessed.shape[0])
        },
        "subject_bbox": _box(subject_bbox),
        "torso_bbox": _box(torso_bbox),
        "centroid": {"x": int(centroid[0]), "y": int(centroid[1])},
        "head_bbox": _box(body_info.get("head_bbox")),
        "neck_point": _xy(body_info.get("neck_point")),
        "left_shoulder": _xy(body_info.get("left_shoulder")),
        "right_shoulder": _xy(body_info.get("right_shoulder")),
        "has_lower_body": bool(body_info.get("has_lower_body", True)),
        "lower_body_bbox": _box(body_info.get("lower_body_bbox")),
        "sizing_profile": body_info.get("sizing_profile"),
        "asset_validation": asset_validation,
        "warnings": warnings,
        "telemetry": telemetry
    }
