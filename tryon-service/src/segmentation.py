"""
Segmentation & Contour Extraction Module
Stylinger Virtual Fashion Try-On

Classical Computer Graphics & Vision techniques implemented:
- Border-seeded background color estimation & Otsu thresholding
- Morphological Opening & Closing (contour cleanup)
- Robust GrabCut foreground segmentation with boundary protection
- Multi-ethnic skin chromaticity analysis (YCrCb + HSV)
- Silhouette + distance-transform core profiling (torso vs outstretched arms)
- Convexity-defect armpit cues when the contour is well formed
- Geometric (not skeletal) estimates of neck, shoulders, waist, hips, and legs

Limitations (explicit):
Contour width and GrabCut silhouettes are 2D proxies. They are not pose-estimation
landmarks. When arms blend into the torso, shoulder width is clamped using head
proportions so sleeves are not treated as the shoulder line.
"""

from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import cv2


def detect_skin_mask(bgr: np.ndarray) -> np.ndarray:
    """
    Classical skin detection using combined YCrCb and HSV color spaces.
    Tuned for high sensitivity across diverse Fitzpatrick skin tones (I through VI).
    """
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    mask_ycrcb = cv2.inRange(
        ycrcb,
        np.array([15, 130, 75], dtype=np.uint8),
        np.array([255, 182, 135], dtype=np.uint8)
    )
    mask_hsv = cv2.inRange(
        hsv,
        np.array([0, 15, 25], dtype=np.uint8),
        np.array([35, 220, 255], dtype=np.uint8)
    )

    skin = cv2.bitwise_and(mask_ycrcb, mask_hsv)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, kernel_close)
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, kernel_open)
    return skin


def detect_head_and_neck(bgr: np.ndarray) -> Optional[Dict[str, Any]]:
    """
    Detects the primary head/face contour in the upper half of the photograph.
    Returns head bounding box and neck/collar level. May return None.
    """
    h, w = bgr.shape[:2]
    skin = detect_skin_mask(bgr)
    skin_upper = skin[:int(h * 0.55), :]
    contours, _ = cv2.findContours(skin_upper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_heads = []
    min_area = max(int((h * w) * 0.0012), 120)

    for c in contours:
        area = cv2.contourArea(c)
        if area >= min_area:
            bx, by, bw, bh = cv2.boundingRect(c)
            aspect = bw / float(bh)
            if 0.45 <= aspect <= 1.95 and by < h * 0.40:
                valid_heads.append((c, area, bx, by, bw, bh))

    if not valid_heads:
        return None

    # Prefer the highest compact blob (face), not a large chest-skin region.
    valid_heads.sort(key=lambda item: (item[3], -item[1] / float(max(item[4] * item[5], 1))))
    _, area, hx, hy, hw, hh = valid_heads[0]
    neck_y = hy + hh
    return {
        "head_bbox": (hx, hy, hw, hh),
        "neck_point": (hx + hw // 2, neck_y),
    }


def segment_foreground_threshold(img: np.ndarray) -> np.ndarray:
    """
    Classical threshold-based segmentation using border background sampling and Otsu binarization.
    """
    bgr = img[:, :, :3] if len(img.shape) == 3 and img.shape[2] == 4 else img
    h, w = bgr.shape[:2]

    border_samples = np.vstack([
        bgr[0:max(int(h * 0.03), 3), :].reshape(-1, 3),
        bgr[:, 0:max(int(w * 0.03), 3)].reshape(-1, 3),
        bgr[:, -max(int(w * 0.03), 3):].reshape(-1, 3)
    ])
    bg_mean = np.mean(border_samples, axis=0)

    color_dist = np.linalg.norm(bgr.astype(np.float32) - bg_mean.astype(np.float32), axis=2)
    _, mask = cv2.threshold(color_dist.astype(np.uint8), 35, 255, cv2.THRESH_BINARY)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    return mask


def segment_foreground_grabcut(img: np.ndarray, iterations: int = 4) -> np.ndarray:
    """
    Robust GrabCut foreground segmentation.
    Uses safe rectangular boundary initialization.
    Falls back cleanly to thresholding if GrabCut returns degenerate masks.
    """
    bgr = img[:, :, :3] if len(img.shape) == 3 and img.shape[2] == 4 else img
    h, w = bgr.shape[:2]

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    mask = np.zeros((h, w), np.uint8)

    mx = max(int(w * 0.04), 4)
    my = max(int(h * 0.02), 4)
    rect = (mx, my, max(w - 2 * mx, 10), max(h - 2 * my, 10))

    try:
        cv2.grabCut(bgr, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)
        binary_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except Exception:
        binary_mask = segment_foreground_threshold(bgr)

    fg_pixels = np.count_nonzero(binary_mask)
    if fg_pixels < int(h * w * 0.05) or fg_pixels > int(h * w * 0.88):
        binary_mask = segment_foreground_threshold(bgr)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    return binary_mask


def _scanline_profile(mask: np.ndarray, y0: int, y1: int) -> List[Tuple[int, int, int, int]]:
    h, w = mask.shape[:2]
    y0 = int(max(0, min(y0, h - 1)))
    y1 = int(max(y0 + 1, min(y1, h)))
    scanlines = []
    for y in range(y0, y1):
        row = np.where(mask[y, :] > 0)[0]
        if len(row) > 0:
            scanlines.append((y, int(row[0]), int(row[-1]), int(row[-1] - row[0])))
    return scanlines


def _core_width_profile(mask: np.ndarray, y0: int, y1: int) -> List[Tuple[int, int, int]]:
    """
    Distance-transform 'core' width along the medial axis.
    When arms are away from the torso, this is much narrower than silhouette extent.
    When arms hang against the body, core ≈ silhouette and head-ratio clamping is required.
    """
    h, w = mask.shape[:2]
    y0 = int(max(0, min(y0, h - 1)))
    y1 = int(max(y0 + 1, min(y1, h)))
    dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    rows = []
    for y in range(y0, y1):
        xs = np.where(mask[y, :] > 0)[0]
        if len(xs) == 0:
            continue
        x_med = int(np.argmax(dist[y, :]))
        core_w = int(2.0 * dist[y, x_med])
        if core_w >= 4:
            rows.append((y, x_med, core_w))
    return rows


def _armpits_from_convexity(
    contour: np.ndarray,
    neck_y: int,
    waist_y: int,
    cx: int
) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    if contour is None or len(contour) < 30:
        return None, None
    hull = cv2.convexHull(contour, returnPoints=False)
    if hull is None or len(hull) < 4:
        return None, None
    try:
        defects = cv2.convexityDefects(contour, hull)
    except Exception:
        return None, None
    if defects is None or len(defects) == 0:
        return None, None

    left_cands = []
    right_cands = []
    # In OpenCV 4.x defects has shape (N, 1, 4); in OpenCV 5.x defects has shape (N, 4).
    # Reshaping to (-1, 4) guarantees each defect row is [s, e, f, d].
    defects_flat = defects.reshape(-1, 4)
    for i in range(defects_flat.shape[0]):
        s, e, f, d = defects_flat[i]
        far = contour[f].flatten()
        depth = float(d) / 256.0
        fx = int(far[0])
        fy = int(far[1])
        if depth < 10:
            continue
        if fy <= neck_y + 4 or fy >= waist_y:
            continue
        if fx < cx:
            left_cands.append((depth, fx, fy))
        else:
            right_cands.append((depth, fx, fy))

    left = None
    right = None
    if left_cands:
        left_cands.sort(reverse=True)
        left = (int(left_cands[0][1]), int(left_cands[0][2]))
    if right_cands:
        right_cands.sort(reverse=True)
        right = (int(right_cands[0][1]), int(right_cands[0][2]))
    return left, right


def _estimate_leg_geometry(
    mask: np.ndarray,
    waist_y: int,
    subj_bbox: Tuple[int, int, int, int]
) -> Dict[str, Any]:
    h, w = mask.shape[:2]
    sx, sy, sw, sh = subj_bbox
    feet_y = min(h - 1, sy + sh)
    lower = np.zeros_like(mask)
    y0 = max(0, waist_y)
    lower[y0:feet_y, :] = mask[y0:feet_y, :]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    lower = cv2.morphologyEx(lower, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(lower, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        if cv2.contourArea(c) < 40:
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        boxes.append((bx, by, bw, bh))
    boxes.sort(key=lambda b: b[0])

    left_leg = None
    right_leg = None
    if len(boxes) >= 2:
        left_leg = boxes[0]
        right_leg = boxes[-1]
    elif len(boxes) == 1:
        bx, by, bw, bh = boxes[0]
        mid = bx + bw // 2
        left_leg = (bx, by, max(bw // 2 - 2, 8), bh)
        right_leg = (mid + 2, by, max(bw // 2 - 2, 8), bh)

    hip_w = 0
    hip_x = sx
    hip_scan = _scanline_profile(mask, waist_y, min(waist_y + max(int(sh * 0.08), 8), h))
    if hip_scan:
        hip_w = int(np.median([s[3] for s in hip_scan]))
        hip_x = int(np.median([(s[1] + s[2]) // 2 for s in hip_scan]) - hip_w // 2)

    crotch_y = waist_y
    if left_leg and right_leg:
        crotch_y = min(left_leg[1] + left_leg[3] // 6, right_leg[1] + right_leg[3] // 6)

    knee_y = int(waist_y + 0.45 * max(feet_y - waist_y, 1))
    ankle_y = int(waist_y + 0.92 * max(feet_y - waist_y, 1))

    return {
        "left_leg_bbox": left_leg,
        "right_leg_bbox": right_leg,
        "hip_bbox": (int(max(0, hip_x)), int(waist_y), int(max(hip_w, 10)), int(max(int(sh * 0.12), 10))),
        "crotch_y": int(crotch_y),
        "knee_y": int(knee_y),
        "ankle_y": int(min(ankle_y, h - 1)),
    }


def extract_body_regions(
    img: np.ndarray,
    method: str = "grabcut"
) -> Dict[str, Any]:
    """
    Extracts the subject silhouette, a torso envelope, and geometric body cues.

    Shoulder width prefers the torso core (distance transform) over the full
    silhouette so outstretched arms do not inflate jacket scale.
    """
    h, w = img.shape[:2]

    if method == "grabcut":
        mask = segment_foreground_grabcut(img)
    else:
        mask = segment_foreground_threshold(img)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_c = None

    if not contours:
        subj_x, subj_y, subj_w, subj_h = int(w * 0.15), int(h * 0.05), int(w * 0.70), int(h * 0.90)
        cx, cy = w // 2, h // 2
        landmarks_reliable = False
    else:
        largest_c = max(contours, key=cv2.contourArea)
        subj_x, subj_y, subj_w, subj_h = cv2.boundingRect(largest_c)
        M = cv2.moments(largest_c)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx = subj_x + subj_w // 2
            cy = subj_y + subj_h // 2
        fg_frac = np.count_nonzero(mask) / float(h * w)
        landmarks_reliable = 0.08 <= fg_frac <= 0.75 and subj_h > 40

    landmark_info = detect_head_and_neck(img)
    head_detected = landmark_info is not None

    if landmark_info:
        hx, hy, hw, hh = landmark_info["head_bbox"]
        neck_x, neck_y = landmark_info["neck_point"]
        # Reject chest-sized "heads" that span most of the subject width.
        if hw > subj_w * 0.62:
            head_detected = False
            landmark_info = None

    if landmark_info:
        hx, hy, hw, hh = landmark_info["head_bbox"]
        neck_x, neck_y = landmark_info["neck_point"]
        head_w_ref = max(hw, int(subj_w * 0.18))
        head_h_ref = max(hh, int(subj_h * 0.10))
        chin_y = hy + hh
        collar_y = max(chin_y + max(int(head_h_ref * 0.06), 4), subj_y + int(head_h_ref * 0.80))
        anchor_cx = neck_x
        head_bbox = (hx, hy, head_w_ref, head_h_ref)
        neck_point = (int(anchor_cx), int(collar_y))
    else:
        head_h_ref = max(int(subj_h * 0.14), 30)
        head_w_ref = max(int(subj_w * 0.28), 25)
        collar_y = subj_y + head_h_ref
        anchor_cx = cx
        head_bbox = (anchor_cx - head_w_ref // 2, subj_y, head_w_ref, head_h_ref)
        neck_point = (int(anchor_cx), int(collar_y))
        landmarks_reliable = False

    cranial_w = float(max(head_w_ref, 20))
    # Never let the shoulder box exceed the silhouette; head-ratio is a prior, not a floor
    # that can inflate tiny cartoons or close-up faces into a giant overlay.
    span_cap = int(min(max(subj_w * 0.78, 16), w - 8))
    target_biacromial_max = max(int(min(cranial_w * 2.55, span_cap)), 16)
    target_biacromial_min = min(int(cranial_w * 2.05), target_biacromial_max)

    torso_h = int(min(max(head_h_ref * 2.05, subj_h * 0.34, 40), subj_h * 0.46, h - collar_y - 5))
    waist_y = int(min(h - 8, collar_y + torso_h))

    scan_y_min = max(subj_y, collar_y - 6)
    scan_y_max = min(subj_y + int(subj_h * 0.50), waist_y + 4, h)
    scanlines = _scanline_profile(mask, scan_y_min, scan_y_max)
    core_rows = _core_width_profile(mask, scan_y_min, scan_y_max)

    left_armpit, right_armpit = _armpits_from_convexity(largest_c, collar_y, waist_y, anchor_cx)

    shoulder_y = collar_y + int(head_h_ref * 0.28)
    if left_armpit and right_armpit:
        armpit_y = (left_armpit[1] + right_armpit[1]) // 2
        shoulder_y = int(max(collar_y + 2, armpit_y - max(int(head_h_ref * 0.22), 8)))

    # Silhouette extent vs torso core at the shoulder row.
    extent_w = subj_w
    if scanlines:
        near = [s for s in scanlines if abs(s[0] - shoulder_y) <= max(6, int(head_h_ref * 0.15))]
        if near:
            extent_w = int(np.median([s[3] for s in near]))
        else:
            extent_w = int(np.percentile([s[3] for s in scanlines], 40))

    core_w = None
    core_cx = anchor_cx
    if core_rows:
        near_core = [r for r in core_rows if abs(r[0] - shoulder_y) <= max(8, int(head_h_ref * 0.2))]
        if not near_core:
            near_core = core_rows[: max(1, len(core_rows) // 4)]
        core_w = int(np.median([r[2] for r in near_core]))
        core_cx = int(np.median([r[1] for r in near_core]))

    arms_away = False
    if core_w is not None and extent_w > int(core_w * 1.35):
        arms_away = True
        raw_sh_w = int(core_w * 1.12)
    else:
        # Arms likely against the torso: do not use full silhouette (includes upper arms).
        raw_sh_w = int(min(extent_w * 0.86, cranial_w * 2.40))

    if left_armpit and right_armpit:
        pit_span = abs(right_armpit[0] - left_armpit[0])
        if pit_span > 10:
            raw_sh_w = int(min(raw_sh_w, pit_span * 1.08 + cranial_w * 0.15))

    if arms_away and core_w is not None:
        shoulder_w = int(max(int(core_w * 1.08), min(raw_sh_w, target_biacromial_max)))
    else:
        shoulder_w = int(max(target_biacromial_min, min(raw_sh_w, target_biacromial_max)))
    shoulder_x = int(max(0, min(core_cx - shoulder_w // 2, w - shoulder_w)))
    shoulder_y = int(max(shoulder_y, collar_y))

    left_shoulder = (int(shoulder_x), int(shoulder_y))
    right_shoulder = (int(shoulder_x + shoulder_w), int(shoulder_y))

    torso_y = int(max(0, collar_y))
    torso_bbox = (int(shoulder_x), int(torso_y), int(shoulder_w), int(min(torso_h, h - torso_y)))

    lower_h = max(0, (subj_y + subj_h) - waist_y)
    has_lower_body = (lower_h >= int(subj_h * 0.22)) and (waist_y < h - 25)

    if has_lower_body:
        lower_mask = np.zeros_like(mask)
        lower_mask[waist_y:min(subj_y + subj_h, h), :] = mask[waist_y:min(subj_y + subj_h, h), :]
        lb_contours, _ = cv2.findContours(lower_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if lb_contours:
            lb_c = max(lb_contours, key=cv2.contourArea)
            lb_x, lb_y, lb_w, lb_h = cv2.boundingRect(lb_c)
        else:
            lb_x = max(0, shoulder_x + int(shoulder_w * 0.08))
            lb_y = waist_y
            lb_w = max(int(shoulder_w * 0.84), 20)
            lb_h = lower_h
        lower_body_bbox = (int(lb_x), int(lb_y), int(lb_w), int(lb_h))
    else:
        lower_body_bbox = (int(shoulder_x), int(waist_y), int(shoulder_w), int(max(lower_h, 10)))

    legs = _estimate_leg_geometry(mask, waist_y, (subj_x, subj_y, subj_w, subj_h)) if has_lower_body else {
        "left_leg_bbox": None,
        "right_leg_bbox": None,
        "hip_bbox": (int(shoulder_x), int(waist_y), int(shoulder_w), 10),
        "crotch_y": int(waist_y),
        "knee_y": int(waist_y),
        "ankle_y": int(min(h - 1, waist_y + 10)),
    }

    user_skin_full = detect_skin_mask(img)
    user_skin = cv2.bitwise_and(user_skin_full, mask)

    hip_w = legs["hip_bbox"][2] if legs.get("hip_bbox") else shoulder_w
    ratio = float(shoulder_w) / float(max(cranial_w, 25))

    sizing_profile = {
        "recommended_size": None,
        "confidence_score": 0,
        "fit_type": "Approximate 2D overlay — not a validated size match",
        "is_estimate": True,
        "size_match_supported": False,
        "estimation_method": (
            "2D silhouette + distance-transform torso core. "
            "No garment-size algorithm is applied."
        ),
        "shoulder_width_px": int(shoulder_w),
        "torso_height_px": int(torso_h),
        "head_ratio": round(float(ratio), 2),
        "landmarks": {
            "left_shoulder": left_shoulder,
            "right_shoulder": right_shoulder,
            "collar_point": (int(anchor_cx), int(collar_y)),
            "waist_y": int(waist_y),
            "hip_width_px": int(hip_w),
        }
    }

    return {
        "subject_mask": mask,
        "subject_bbox": (int(subj_x), int(subj_y), int(subj_w), int(subj_h)),
        "torso_bbox": (int(torso_bbox[0]), int(torso_bbox[1]), int(torso_bbox[2]), int(torso_bbox[3])),
        "lower_body_bbox": lower_body_bbox,
        "has_lower_body": has_lower_body,
        "waist_y": int(waist_y),
        "skin_mask": user_skin,
        "centroid": (int(cx), int(cy)),
        "head_bbox": (int(head_bbox[0]), int(head_bbox[1]), int(head_bbox[2]), int(head_bbox[3])),
        "neck_point": (int(neck_point[0]), int(neck_point[1])),
        "left_shoulder": left_shoulder,
        "right_shoulder": right_shoulder,
        "left_armpit": left_armpit,
        "right_armpit": right_armpit,
        "arms_away": bool(arms_away),
        "head_detected": bool(head_detected),
        "landmarks_reliable": bool(landmarks_reliable and head_detected),
        "left_leg_bbox": legs.get("left_leg_bbox"),
        "right_leg_bbox": legs.get("right_leg_bbox"),
        "hip_bbox": legs.get("hip_bbox"),
        "knee_y": legs.get("knee_y"),
        "ankle_y": legs.get("ankle_y"),
        "crotch_y": legs.get("crotch_y"),
        "sizing_profile": sizing_profile,
        "image_size": (int(w), int(h)),
    }
