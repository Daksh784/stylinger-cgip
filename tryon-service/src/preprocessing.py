"""
Preprocessing Module for Classical Computer Graphics & Image Processing
Stylinger Virtual Fashion Try-On

Handles:
- Safe in-memory decoding & validation (magic headers, dimension bounds)
- Aspect-ratio preserving dimension normalization
- Noise reduction (Gaussian / Median filtering)
- Luminance-preserving contrast enhancement (CLAHE in LAB space)
- Color space transformations (BGR to RGB, BGR to HSV)
"""

import io
from typing import Tuple, Optional
import numpy as np
import cv2
from PIL import Image


class ImageValidationError(ValueError):
    """Raised when an uploaded image fails validation (format, dimensions, corruption)."""
    pass


def decode_image_safely(file_bytes: bytes, min_dim: int = 64, max_dim: int = 4096) -> np.ndarray:
    """
    Safely verifies image headers with PIL and decodes into a NumPy array with OpenCV.
    Guarantees no arbitrary code execution or file traversal, fully in-memory.
    """
    if not file_bytes:
        raise ImageValidationError("Empty image payload provided.")

    # 1. Header & Format verification with PIL
    try:
        with Image.open(io.BytesIO(file_bytes)) as pil_img:
            format_name = (pil_img.format or "").upper()
            if format_name not in {"JPEG", "JPG", "PNG", "WEBP"}:
                raise ImageValidationError(f"Unsupported image format: {format_name}. Only JPEG/PNG/WebP are accepted.")
            width, height = pil_img.size
            if width < min_dim or height < min_dim:
                raise ImageValidationError(f"Image dimensions too small: {width}x{height} (min: {min_dim}px).")
            if width > max_dim or height > max_dim:
                raise ImageValidationError(f"Image dimensions exceed limit: {width}x{height} (max: {max_dim}px).")
            pil_img.verify()
    except Exception as e:
        if isinstance(e, ImageValidationError):
            raise
        raise ImageValidationError(f"Corrupted or invalid image data: {str(e)}") from e

    # 2. Decode with OpenCV in-memory
    np_buf = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(np_buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ImageValidationError("OpenCV could not decode image buffer.")

    return img


def normalize_dimensions(img: np.ndarray, max_dimension: int = 1024) -> Tuple[np.ndarray, float]:
    """
    Downsamples the image proportionally if either dimension exceeds max_dimension.
    Returns (resized_image, scale_ratio).
    """
    h, w = img.shape[:2]
    if max(h, w) <= max_dimension:
        return img, 1.0

    if h >= w:
        scale = max_dimension / float(h)
        new_h = max_dimension
        new_w = int(round(w * scale))
    else:
        scale = max_dimension / float(w)
        new_w = max_dimension
        new_h = int(round(h * scale))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def reduce_noise(img: np.ndarray, method: str = "gaussian", ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    Suppresses sensor noise and high-frequency artifacts while retaining edge information.
    Supports 'gaussian' and 'median'.
    """
    if ksize % 2 == 0:
        ksize += 1

    if method == "median":
        return cv2.medianBlur(img, ksize)
    
    # Gaussian filter
    return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)


def adjust_contrast_brightness(
    img: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
    brightness_offset: int = 0
) -> np.ndarray:
    """
    Performs Contrast Limited Adaptive Histogram Equalization (CLAHE) on the luminance (L)
    channel in CIELAB color space. Prevents color hue distortion while balancing exposure.
    """
    # If image has an alpha channel, separate it
    has_alpha = (img.shape[2] == 4) if len(img.shape) == 3 else False
    if has_alpha:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
    else:
        bgr = img

    # Convert BGR to LAB
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)

    if brightness_offset != 0:
        l_enhanced = np.clip(l_enhanced.astype(np.int32) + brightness_offset, 0, 255).astype(np.uint8)

    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    bgr_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    if has_alpha:
        return cv2.merge([bgr_enhanced[:, :, 0], bgr_enhanced[:, :, 1], bgr_enhanced[:, :, 2], alpha])
    return bgr_enhanced


def to_hsv(img: np.ndarray) -> np.ndarray:
    """Converts a BGR image to HSV color space."""
    bgr = img[:, :, :3] if len(img.shape) == 3 and img.shape[2] == 4 else img
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
