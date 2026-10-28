"""
Garment Transformation & Alignment Module
Stylinger Virtual Fashion Try-On

Handles:
- Garment asset validation (alpha, studio backdrop, model/mannequin photos)
- Background extraction and model suppression
- Category-aware target regions from body geometry
- Aspect-ratio preserving scale using garment *body* width, not sleeve span
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import cv2


def normalize_category(category: str) -> str:
    cat = (category or "Tops").lower().replace("-", " ").replace("_", " ")
    if any(k in cat for k in ["jean", "trouser", "pant", "bottom", "skirt", "short"]):
        return "bottoms"
    if any(k in cat for k in ["outerwear", "jacket", "coat", "blazer", "hoodie", "parka"]):
        return "outerwear"
    if "dress" in cat:
        return "dress"
    if any(k in cat for k in ["t shirt", "tshirt", "tee"]):
        return "tee"
    if any(k in cat for k in ["formal", "blouse", "button", "polo"]) or "shirt" in cat:
        return "shirt"
    if any(k in cat for k in ["knit", "top"]):
        return "tee"
    return "other"


def validate_garment_asset(
    garment_img: np.ndarray,
    strict_model_rejection: bool = False
) -> Dict[str, Any]:
    """
    Validates whether the garment asset is suitable for 2D compositing.
    Catalog photos of a model wearing the item are flagged; they are not
    treated as isolated garment cutouts.
    """
    if garment_img is None or garment_img.size == 0:
        return {
            "valid": False,
            "asset_type": "invalid",
            "message": "Garment image payload is empty or corrupted.",
            "warning": "Garment image is empty or unreadable."
        }

    h, w = garment_img.shape[:2]

    if len(garment_img.shape) == 3 and garment_img.shape[2] == 4:
        alpha = garment_img[:, :, 3]
        transparent_pixels = np.count_nonzero(alpha < 240)
        opaque_pixels = np.count_nonzero(alpha > 30)
        if transparent_pixels > int(h * w * 0.05) and opaque_pixels > int(h * w * 0.05):
            bgr = garment_img[:, :, :3]
            if _has_model_face(bgr):
                msg = (
                    "Transparent image still contains a visible face or mannequin. "
                    "Use a garment-only PNG (no model) for reliable try-on."
                )
                if strict_model_rejection:
                    return {"valid": False, "asset_type": "model_detected", "message": msg, "warning": msg}
                return {
                    "valid": True,
                    "asset_type": "model_detected",
                    "message": "Model detected in transparent asset.",
                    "warning": msg
                }
            return {
                "valid": True,
                "asset_type": "transparent_png",
                "message": "Transparent garment asset.",
                "warning": None
            }

    bgr = garment_img[:, :, :3] if len(garment_img.shape) == 3 else cv2.cvtColor(garment_img, cv2.COLOR_GRAY2BGR)

    border_samples = np.vstack([
        bgr[0:max(int(h * 0.04), 3), :].reshape(-1, 3),
        bgr[h - max(int(h * 0.04), 3):, :].reshape(-1, 3),
        bgr[:, 0:max(int(w * 0.04), 3)].reshape(-1, 3),
        bgr[:, w - max(int(w * 0.04), 3):].reshape(-1, 3)
    ])
    border_std = np.std(border_samples.astype(np.float32), axis=0)
    mean_std = float(np.mean(border_std))

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    border_hsv_samples = np.vstack([
        hsv[0:max(int(h * 0.04), 3), :].reshape(-1, 3),
        hsv[h - max(int(h * 0.04), 3):, :].reshape(-1, 3),
        hsv[:, 0:max(int(w * 0.04), 3)].reshape(-1, 3),
        hsv[:, w - max(int(w * 0.04), 3):].reshape(-1, 3)
    ])
    border_mean_v = float(np.mean(border_hsv_samples[:, 2]))

    has_model_face = _has_model_face(bgr)

    if has_model_face:
        msg = (
            "This product photo shows a model or mannequin wearing the garment. "
            "It cannot be treated as an isolated clothing cutout. "
            "Upload a flat-lay or transparent PNG of the garment only, "
            "or use the manual scale/offset sliders after a best-effort overlay."
        )
        if strict_model_rejection:
            return {
                "valid": False,
                "asset_type": "model_detected",
                "message": msg,
                "warning": msg
            }
        return {
            "valid": True,
            "asset_type": "model_detected",
            "message": "Model/mannequin photo — overlay is approximate.",
            "warning": msg
        }

    if mean_std > 52.0 and border_mean_v < 180:
        return {
            "valid": True,
            "asset_type": "complex_background",
            "message": "Complex background detected.",
            "warning": (
                "Garment photo has a complex background. "
                "Background removal is heuristic and may leave fringes. "
                "Prefer an isolated flat-lay or transparent PNG."
            )
        }

    return {
        "valid": True,
        "asset_type": "studio_flatlay",
        "message": "Clean studio flat-lay detected.",
        "warning": None
    }


def _has_model_face(bgr: np.ndarray) -> bool:
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    skin_ycrcb = cv2.inRange(ycrcb, np.array([20, 133, 77], dtype=np.uint8), np.array([255, 175, 127], dtype=np.uint8))
    skin_hsv = cv2.inRange(hsv, np.array([0, 25, 40], dtype=np.uint8), np.array([30, 200, 255], dtype=np.uint8))
    skin = cv2.bitwise_and(skin_ycrcb, skin_hsv)
    skin_upper = skin[:int(h * 0.40), :]
    skin_contours, _ = cv2.findContours(skin_upper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in skin_contours:
        area = cv2.contourArea(c)
        if area > (h * w * 0.015):
            bx, by, bw, bh = cv2.boundingRect(c)
            aspect = bw / float(max(bh, 1))
            if 0.5 <= aspect <= 1.8 and by < int(h * 0.32):
                return True
    return False


def extract_garment_alpha(
    garment_img: np.ndarray,
    tolerance: int = 24,
    suppress_model: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ensures the garment has a BGR image plus alpha mask.
    Existing alpha is preserved (light cleanup only). RGB catalog photos
    are isolated from a sampled studio backdrop. Model skin is subtracted
    when present; remaining pixels may still include a worn garment silhouette.
    """
    if len(garment_img.shape) == 3 and garment_img.shape[2] == 4:
        bgr = garment_img[:, :, :3]
        alpha = garment_img[:, :, 3].copy()
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel_close)
        alpha = cv2.erode(alpha, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        return bgr, alpha

    bgr = garment_img[:, :, :3] if len(garment_img.shape) == 3 else cv2.cvtColor(garment_img, cv2.COLOR_GRAY2BGR)
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    border_samples = np.vstack([
        bgr[0:max(int(h * 0.04), 3), :].reshape(-1, 3),
        bgr[h - max(int(h * 0.04), 3):, :].reshape(-1, 3),
        bgr[:, 0:max(int(w * 0.04), 3)].reshape(-1, 3),
        bgr[:, w - max(int(w * 0.04), 3):].reshape(-1, 3)
    ])
    avg_bg = np.mean(border_samples, axis=0)

    color_diff = np.linalg.norm(bgr.astype(np.float32) - avg_bg.astype(np.float32), axis=2)
    bg_mask = (color_diff < tolerance * 1.8).astype(np.uint8) * 255
    white_bg_mask = cv2.inRange(hsv, np.array([0, 0, 215], dtype=np.uint8), np.array([180, 45, 255], dtype=np.uint8))
    combined_bg = cv2.bitwise_or(bg_mask, white_bg_mask)
    garment_mask = cv2.bitwise_not(combined_bg)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    garment_mask = cv2.morphologyEx(garment_mask, cv2.MORPH_CLOSE, kernel_close)

    if suppress_model:
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        skin_ycrcb = cv2.inRange(ycrcb, np.array([0, 133, 77], dtype=np.uint8), np.array([255, 173, 127], dtype=np.uint8))
        skin_hsv = cv2.inRange(hsv, np.array([0, 20, 40], dtype=np.uint8), np.array([30, 200, 255], dtype=np.uint8))
        skin = cv2.bitwise_and(skin_ycrcb, skin_hsv)

        skin_upper = skin[:int(h * 0.45), :]
        skin_ys, _ = np.where(skin_upper > 0)

        if len(skin_ys) > 40:
            neck_y = int(np.percentile(skin_ys, 90))
            skin_xs = np.where(skin_upper > 0)[1]
            if len(skin_xs) > 0:
                h_min_x = max(0, int(np.percentile(skin_xs, 5)) - 14)
                h_max_x = min(w, int(np.percentile(skin_xs, 95)) + 14)
                garment_mask[:neck_y, h_min_x:h_max_x] = 0
            else:
                garment_mask[:neck_y, int(w * 0.32):int(w * 0.68)] = 0
        else:
            for y in range(0, int(h * 0.25)):
                row = np.where(garment_mask[y, :] > 0)[0]
                if len(row) > 0 and (row[-1] - row[0]) < 40 and abs(((row[0] + row[-1]) // 2) - w // 2) < int(w * 0.20):
                    garment_mask[y, :] = 0

        skin_dilated = cv2.dilate(skin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        garment_mask[skin_dilated > 0] = 0

    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    garment_mask = cv2.erode(garment_mask, kernel_erode, iterations=1)
    return bgr, garment_mask


def measure_garment_geometry(garment_alpha: np.ndarray) -> Dict[str, Any]:
    """
    Measures the isolated garment mask.

    Scale is driven by the *body/chest* width (median width of the mid-torso
    band), not sleeve tip-to-tip span. A T-shaped mask (sleeves) is detected
    when the upper band is substantially wider than the body band.
    """
    ys, xs = np.where(garment_alpha > 15)
    if len(ys) == 0:
        h, w = garment_alpha.shape[:2]
        return {
            "crop": (0, 0, w, h),
            "body_width": max(w, 1),
            "shoulder_width": max(w, 1),
            "collar_cx": w // 2,
            "has_sleeves": False,
            "aspect": w / float(max(h, 1)),
        }

    ymin, ymax = int(np.min(ys)), int(np.max(ys))
    xmin, xmax = int(np.min(xs)), int(np.max(xs))
    crop_h = max(ymax - ymin + 1, 1)
    crop_w = max(xmax - xmin + 1, 1)
    cropped = garment_alpha[ymin:ymax + 1, xmin:xmax + 1]

    widths = []
    for y in range(crop_h):
        row = np.where(cropped[y, :] > 15)[0]
        if len(row) > 0:
            widths.append((y, int(row[0]), int(row[-1]), int(row[-1] - row[0])))

    def band_width(y0_frac, y1_frac, percentile=50):
        y0 = int(crop_h * y0_frac)
        y1 = int(crop_h * y1_frac)
        band = [t[3] for t in widths if y0 <= t[0] <= y1]
        if not band:
            return crop_w
        return int(np.percentile(band, percentile))

    body_w = band_width(0.32, 0.68, 50)
    upper_w = band_width(0.08, 0.22, 70)
    waist_w = band_width(0.02, 0.12, 50)

    has_sleeves = upper_w > body_w * 1.22
    # When sleeves exist, body width is the correct chest match.
    # Upper 70th percentile includes hanging sleeves and must not drive scale.
    shoulder_w = body_w if has_sleeves else min(upper_w, body_w)

    top_rows = cropped[:max(int(crop_h * 0.12), 2), :]
    top_xs = np.where(top_rows > 15)[1]
    collar_cx = int(np.mean(top_xs)) if len(top_xs) > 0 else crop_w // 2

    return {
        "crop": (xmin, ymin, crop_w, crop_h),
        "body_width": max(int(body_w), 8),
        "shoulder_width": max(int(shoulder_w), 8),
        "waist_width": max(int(waist_w), 8),
        "collar_cx": int(collar_cx),
        "has_sleeves": bool(has_sleeves),
        "aspect": crop_w / float(crop_h),
    }


def compute_target_box(
    category: str,
    subject_bbox: Tuple[int, int, int, int],
    torso_bbox: Tuple[int, int, int, int],
    image_shape: Tuple[int, int],
    lower_body_bbox: Optional[Tuple[int, int, int, int]] = None,
    waist_y: Optional[int] = None,
    hip_bbox: Optional[Tuple[int, int, int, int]] = None,
    ankle_y: Optional[int] = None,
    knee_y: Optional[int] = None,
) -> Tuple[int, int, int, int]:
    """
    Target clothing region from category + body boxes.

    Outerwear ease is applied once here (small drape beyond the shoulder
    line). Transform must not multiply a second ease factor onto this width.
    """
    h_img, w_img = image_shape
    sx, sy, sw, sh = subject_bbox
    tx, ty, tw, th = torso_bbox
    kind = normalize_category(category)

    if kind == "bottoms":
        if lower_body_bbox and lower_body_bbox[3] > 30:
            lx, ly, lw, lh = lower_body_bbox
            if hip_bbox and hip_bbox[2] > 10:
                hx, hy, hw, hh = hip_bbox
                target_w = max(hw, int(lw * 0.92))
                target_x = hx + (hw - target_w) // 2
            else:
                target_x, target_w = lx, lw
            target_y = ly if waist_y is None else int(waist_y)
            feet = ankle_y if ankle_y is not None else (ly + lh)
            target_h = max(int(feet - target_y), 20)
        else:
            target_w = max(int(tw * 0.92), int(sw * 0.55))
            target_y = int(waist_y if waist_y is not None else ty + th)
            target_h = max(int(sh * 0.52), int(h_img * 0.28))
            target_x = max(0, tx + (tw - target_w) // 2)
    elif kind == "outerwear":
        target_w = int(tw * 1.08)
        target_h = int(th * 1.18)
        target_x = tx - (target_w - tw) // 2
        target_y = ty
    elif kind == "dress":
        target_w = int(tw * 1.02)
        hem = knee_y if knee_y is not None else min(ty + int(th * 2.05), sy + int(sh * 0.88))
        target_h = max(int(hem - ty), int(th * 1.4))
        target_x = tx - (target_w - tw) // 2
        target_y = ty
    else:
        # tee, shirt, other: torso only
        target_w = tw
        target_h = th
        target_x = tx
        target_y = ty

    target_x = max(0, min(target_x, w_img - 10))
    target_y = max(0, min(target_y, h_img - 10))
    target_w = max(10, min(target_w, w_img - target_x))
    target_h = max(10, min(target_h, h_img - target_y))
    return target_x, target_y, target_w, target_h


def transform_and_align_garment(
    garment_bgr: np.ndarray,
    garment_alpha: np.ndarray,
    target_shape: Tuple[int, int],
    target_box: Tuple[int, int, int, int],
    user_scale: float = 1.0,
    offset_x: int = 0,
    offset_y: int = 0,
    rotation_deg: float = 0.0,
    category: str = "Tops",
    body_info: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Places the garment onto the user canvas with uniform (aspect-preserving) scale.

    Scale matches garment body/chest width to the target clothing width.
    Sleeves are not independently stretched. Category clipping keeps shirts
    on the torso and trousers below the waist.
    """
    canvas_h, canvas_w = target_shape
    bx, by, bw, bh = target_box
    kind = normalize_category(category)
    geom = measure_garment_geometry(garment_alpha)

    xmin, ymin, crop_w, crop_h = geom["crop"]
    cropped_bgr = garment_bgr[ymin:ymin + crop_h, xmin:xmin + crop_w]
    cropped_alpha = garment_alpha[ymin:ymin + crop_h, xmin:xmin + crop_w]

    if kind == "bottoms":
        g_ref_w = geom.get("waist_width") or geom["body_width"]
        ease = 1.02
    elif kind == "outerwear":
        g_ref_w = geom["body_width"]
        ease = 1.0  # target box already includes jacket ease
    elif kind == "dress":
        g_ref_w = geom["body_width"]
        ease = 1.0
    else:
        g_ref_w = geom["body_width"]
        ease = 1.0

    scale_ratio = (float(bw) * ease) / float(max(g_ref_w, 8))
    final_w = max(int(crop_w * scale_ratio * user_scale), 10)
    final_h = max(int(crop_h * scale_ratio * user_scale), 10)

    interp = cv2.INTER_AREA if final_w < crop_w else cv2.INTER_CUBIC
    resized_bgr = cv2.resize(cropped_bgr, (final_w, final_h), interpolation=interp)
    resized_alpha = cv2.resize(cropped_alpha, (final_w, final_h), interpolation=interp)

    if abs(rotation_deg) > 0.1:
        rcx, rcy = final_w / 2.0, final_h / 2.0
        rot_mat = cv2.getRotationMatrix2D((rcx, rcy), rotation_deg, 1.0)
        resized_bgr = cv2.warpAffine(resized_bgr, rot_mat, (final_w, final_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        resized_alpha = cv2.warpAffine(resized_alpha, rot_mat, (final_w, final_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    neck = None
    if body_info and body_info.get("neck_point"):
        neck = body_info["neck_point"]
    ls = body_info.get("left_shoulder") if body_info else None
    rs = body_info.get("right_shoulder") if body_info else None

    if ls and rs:
        center_target_x = (ls[0] + rs[0]) // 2
    else:
        center_target_x = bx + bw // 2

    collar_x_final = int(geom["collar_cx"] * scale_ratio * user_scale)

    if kind == "bottoms":
        top_target_y = by
        dest_x1 = int(center_target_x - (final_w // 2) + offset_x)
    else:
        if neck is not None:
            top_target_y = int(neck[1] - max(int(bh * 0.02), 2))
        else:
            top_target_y = max(by - max(int(bh * 0.02), 1), 0)
        dest_x1 = int(center_target_x - collar_x_final + offset_x)

    dest_y1 = int(top_target_y + offset_y)
    dest_x2 = dest_x1 + final_w
    dest_y2 = dest_y1 + final_h

    canvas_bgr = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas_alpha = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

    src_x1 = max(0, -dest_x1)
    src_y1 = max(0, -dest_y1)
    src_x2 = min(final_w, canvas_w - dest_x1)
    src_y2 = min(final_h, canvas_h - dest_y1)
    clip_dx1 = max(0, dest_x1)
    clip_dy1 = max(0, dest_y1)
    clip_dx2 = min(canvas_w, dest_x2)
    clip_dy2 = min(canvas_h, dest_y2)

    if (clip_dx2 > clip_dx1) and (clip_dy2 > clip_dy1) and (src_x2 > src_x1) and (src_y2 > src_y1):
        canvas_bgr[clip_dy1:clip_dy2, clip_dx1:clip_dx2] = resized_bgr[src_y1:src_y2, src_x1:src_x2]
        canvas_alpha[clip_dy1:clip_dy2, clip_dx1:clip_dx2] = resized_alpha[src_y1:src_y2, src_x1:src_x2]

    # Category bounds — never replace the whole person mask with the garment.
    if kind == "bottoms":
        waist_cutoff = max(0, by + offset_y)
        canvas_alpha[:waist_cutoff, :] = 0
        # Keep footwear: fade the last 6% of the lower box
        if body_info and body_info.get("ankle_y"):
            fade_y = int(body_info["ankle_y"])
            if 0 < fade_y < canvas_h:
                canvas_alpha[fade_y:, :] = 0
    elif kind in ("tee", "shirt", "other"):
        max_hem_y = min(canvas_h, by + int(bh * 1.06) + offset_y)
        canvas_alpha[max_hem_y:, :] = 0
    elif kind == "outerwear":
        max_jacket_y = min(canvas_h, by + int(bh * 1.22) + offset_y)
        canvas_alpha[max_jacket_y:, :] = 0
        # Preserve an open-jacket inner shirt only when the garment itself is opaque
        # in the chest center (closed catalog photo). Transparent lapels are kept.
        v_h = int(min(final_h, bh) * 0.32)
        for dy in range(max(v_h, 0)):
            y_curr = dest_y1 + dy
            if 0 <= y_curr < canvas_h:
                v_half_w = int((bw * 0.10) * (1.0 - (dy / float(max(v_h, 1))) * 0.70))
                v_x1 = max(0, center_target_x - v_half_w)
                v_x2 = min(canvas_w, center_target_x + v_half_w)
                region = canvas_alpha[y_curr, v_x1:v_x2]
                if region.size and float(np.mean(region)) > 80:
                    canvas_alpha[y_curr, v_x1:v_x2] = (region.astype(np.float32) * 0.18).astype(np.uint8)
    elif kind == "dress":
        hem = by + int(bh * 1.02) + offset_y
        canvas_alpha[min(canvas_h, hem):, :] = 0

    # Lateral clip: keep fabric near the body; allow sleeve corridor ~1.35× half-torso
    # (enough for short sleeves, not a full-image overlay).
    if kind != "bottoms":
        arm_envelope_half = int(bw * 0.78) if kind == "outerwear" else int(bw * 0.62)
        left_bound = max(0, center_target_x - arm_envelope_half)
        right_bound = min(canvas_w, center_target_x + arm_envelope_half)
        fade_w = min(10, max(int(bw * 0.08), 3))
        if left_bound > fade_w:
            canvas_alpha[:, :left_bound - fade_w] = 0
            for i in range(fade_w):
                col = left_bound - fade_w + i
                canvas_alpha[:, col] = (canvas_alpha[:, col].astype(np.float32) * (i / float(fade_w))).astype(np.uint8)
        if right_bound + fade_w < canvas_w:
            canvas_alpha[:, right_bound + fade_w:] = 0
            for i in range(fade_w):
                col = right_bound + i
                canvas_alpha[:, col] = (canvas_alpha[:, col].astype(np.float32) * (1.0 - (i / float(fade_w)))).astype(np.uint8)

    return canvas_bgr, canvas_alpha
