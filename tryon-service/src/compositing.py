"""
Alpha-Mask Compositing & Blending Module
Stylinger Virtual Fashion Try-On

Handles:
- Edge feathering via Gaussian filtering on alpha boundaries
- Ambient lighting / luminance matching
- Linear alpha compositing constrained to the person (not the full frame)
- Restoration of face, hair, neck, and foreground hands
"""

from typing import Tuple, Optional
import numpy as np
import cv2


def feather_mask(alpha_mask: np.ndarray, feather_radius: int = 3) -> np.ndarray:
    """
    Feathers the edges of a binary or semi-transparent alpha mask using Gaussian blur.
    Small radius only — large blur makes garments look pasted/transparent.
    """
    if feather_radius <= 0:
        return alpha_mask

    ksize = feather_radius * 2 + 1
    feathered = cv2.GaussianBlur(alpha_mask.astype(np.float32), (ksize, ksize), 0)
    return np.clip(feathered, 0, 255).astype(np.uint8)


def balance_illumination(
    garment_bgr: np.ndarray,
    user_bgr: np.ndarray,
    alpha_mask: np.ndarray,
    blend_strength: float = 0.22
) -> np.ndarray:
    """Harmonizes garment luminance with the user photograph (LAB)."""
    mask_bool = alpha_mask > 20
    if not np.any(mask_bool):
        return garment_bgr

    user_lab = cv2.cvtColor(user_bgr, cv2.COLOR_BGR2LAB)
    user_l_mean = np.mean(user_lab[mask_bool, 0])

    garment_lab = cv2.cvtColor(garment_bgr, cv2.COLOR_BGR2LAB)
    garment_l_mean = np.mean(garment_lab[mask_bool, 0])

    if garment_l_mean <= 0:
        return garment_bgr

    ratio = user_l_mean / float(garment_l_mean)
    damped_ratio = 1.0 + (ratio - 1.0) * blend_strength

    l_channel, a_channel, b_channel = cv2.split(garment_lab)
    l_adjusted = np.clip(l_channel.astype(np.float32) * damped_ratio, 0, 255)

    user_gray = cv2.cvtColor(user_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ambient_blur = cv2.GaussianBlur(user_gray, (31, 31), 0)
    ambient_mean = np.mean(ambient_blur[mask_bool]) + 1e-5
    shading_field = np.clip(0.94 + 0.12 * (ambient_blur / ambient_mean), 0.88, 1.12)
    l_modulated = np.clip(l_adjusted * shading_field, 0, 255).astype(np.uint8)

    adjusted_lab = cv2.merge([l_modulated, a_channel, b_channel])
    return cv2.cvtColor(adjusted_lab, cv2.COLOR_LAB2BGR)


def composite_garment(
    user_bgr: np.ndarray,
    garment_bgr: np.ndarray,
    garment_alpha: np.ndarray,
    feather_radius: int = 3,
    match_lighting: bool = True,
    opacity: float = 1.0,
    head_bbox: Optional[Tuple[int, int, int, int]] = None,
    skin_mask: Optional[np.ndarray] = None,
    person_mask: Optional[np.ndarray] = None,
    category: str = "Tops"
) -> np.ndarray:
    """
    Composites the aligned garment onto the user photograph.

    The garment is multiplied into a slightly dilated person mask so it cannot
    paint over background. Face/hair and foreground skin (hands, neck) are
    restored after placement.
    """
    h, w = user_bgr.shape[:2]
    gh, gw = garment_bgr.shape[:2]

    if (h != gh) or (w != gw):
        garment_bgr = cv2.resize(garment_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
        garment_alpha = cv2.resize(garment_alpha, (w, h), interpolation=cv2.INTER_LINEAR)

    garment_alpha_clean = garment_alpha.copy()

    # Constrain overlay to the person silhouette (plus a small drape margin).
    if person_mask is not None and person_mask.shape[:2] == (h, w):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        allowed = cv2.dilate(person_mask, kernel, iterations=1)
        garment_alpha_clean = cv2.bitwise_and(garment_alpha_clean, allowed)

    # Face + hair: zero garment in an expanded head box.
    if head_bbox is not None:
        hx, hy, hw, hh = head_bbox
        y0 = max(0, hy - int(hh * 0.40))
        y1 = max(0, min(h, hy + int(hh * 0.92)))
        x0 = max(0, hx - int(hw * 0.12))
        x1 = min(w, hx + hw + int(hw * 0.12))
        garment_alpha_clean[y0:y1, x0:x1] = 0

    if skin_mask is not None and skin_mask.shape[:2] == (h, w):
        if head_bbox is not None:
            _, hy, _, hh = head_bbox
            neck_skin = np.zeros_like(skin_mask)
            neck_skin[:min(h, hy + int(hh * 1.08)), :] = skin_mask[:min(h, hy + int(hh * 1.08)), :]
            garment_alpha_clean[neck_skin > 0] = 0
            chest_y_threshold = hy + int(hh * 1.35)
        else:
            chest_y_threshold = int(h * 0.32)

        # Hands / forearms in front of clothing (crossed arms, hands on hips).
        hands_skin = np.zeros_like(skin_mask)
        hands_skin[min(h, max(chest_y_threshold, 0)):, :] = skin_mask[min(h, max(chest_y_threshold, 0)):, :]
        if np.any(hands_skin > 0):
            hands_dilated = cv2.dilate(hands_skin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            hands_feathered = cv2.GaussianBlur(hands_dilated.astype(np.float32), (7, 7), 0) / 255.0
            garment_alpha_clean = np.clip(
                garment_alpha_clean.astype(np.float32) * (1.0 - hands_feathered * 0.96),
                0, 255
            ).astype(np.uint8)

    feathered = feather_mask(garment_alpha_clean, min(int(feather_radius), 8))

    if match_lighting:
        processed_garment = balance_illumination(garment_bgr, user_bgr, feathered)
        top_edge = cv2.morphologyEx(feathered, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        shadow_map = cv2.GaussianBlur(top_edge.astype(np.float32), (11, 11), 0) / 255.0
        shadow_factor = np.clip(1.0 - 0.10 * shadow_map, 0.90, 1.0)[:, :, np.newaxis]
        garment_shaded = processed_garment.astype(np.float32) * shadow_factor
    else:
        garment_shaded = garment_bgr.astype(np.float32)

    alpha_norm = (feathered.astype(np.float32) / 255.0) * np.clip(opacity, 0.0, 1.0)
    alpha_3d = np.repeat(alpha_norm[:, :, np.newaxis], 3, axis=2)

    user_f = user_bgr.astype(np.float32)
    out_f = alpha_3d * garment_shaded + (1.0 - alpha_3d) * user_f
    return np.clip(out_f, 0, 255).astype(np.uint8)


def encode_image(img_bgr: np.ndarray, format_ext: str = "png", quality: int = 95) -> bytes:
    """Encodes a BGR image array into binary bytes (PNG or JPEG)."""
    ext = format_ext.lower().replace(".", "")
    if ext in ["jpg", "jpeg"]:
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, buffer = cv2.imencode(".jpg", img_bgr, encode_params)
    else:
        encode_params = [int(cv2.IMWRITE_PNG_COMPRESSION), 4]
        success, buffer = cv2.imencode(".png", img_bgr, encode_params)

    if not success:
        raise RuntimeError(f"Failed to encode image to {ext}")

    return buffer.tobytes()
