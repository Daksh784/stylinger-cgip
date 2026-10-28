"""
Pytest fixtures and synthetic test image generators
"""

import io
import pytest
import numpy as np
import cv2
from PIL import Image


@pytest.fixture
def synthetic_user_image_bytes() -> bytes:
    """
    Creates a synthetic front-facing portrait (300x500 PNG)
    with a neutral background and a central subject silhouette.
    """
    canvas = np.full((500, 300, 3), 220, dtype=np.uint8)  # Light gray background

    # Draw body silhouette: head, torso, legs
    # Head
    cv2.circle(canvas, (150, 80), 40, (180, 150, 130), -1)
    # Torso (chest to waist)
    cv2.rectangle(canvas, (100, 130), (200, 310), (120, 90, 80), -1)
    # Legs
    cv2.rectangle(canvas, (110, 310), (145, 470), (60, 60, 90), -1)
    cv2.rectangle(canvas, (155, 310), (190, 470), (60, 60, 90), -1)

    # Encode to PNG bytes
    _, buf = cv2.imencode(".png", canvas)
    return buf.tobytes()


@pytest.fixture
def synthetic_garment_image_bytes() -> bytes:
    """
    Creates a synthetic garment flat-lay image (200x200 PNG)
    with an emerald green t-shirt on a pure white background.
    """
    canvas = np.full((200, 200, 3), 255, dtype=np.uint8)  # White background

    # Draw t-shirt shape
    # Main body
    cv2.rectangle(canvas, (50, 50), (150, 180), (80, 180, 100), -1)
    # Left sleeve
    pts_left = np.array([[50, 50], [20, 90], [45, 110], [50, 90]], np.int32)
    cv2.fillPoly(canvas, [pts_left], (80, 180, 100))
    # Right sleeve
    pts_right = np.array([[150, 50], [180, 90], [155, 110], [150, 90]], np.int32)
    cv2.fillPoly(canvas, [pts_right], (80, 180, 100))
    # Collar cutout
    cv2.circle(canvas, (100, 50), 20, (255, 255, 255), -1)

    _, buf = cv2.imencode(".png", canvas)
    return buf.tobytes()


@pytest.fixture
def synthetic_person_arms_out_bytes() -> bytes:
    """Front-facing person with arms raised away from the torso."""
    canvas = np.full((500, 400, 3), 220, dtype=np.uint8)
    cv2.circle(canvas, (200, 70), 32, (180, 150, 130), -1)
    cv2.rectangle(canvas, (155, 110), (245, 280), (120, 90, 80), -1)
    cv2.line(canvas, (155, 125), (40, 90), (120, 90, 80), 18)
    cv2.line(canvas, (245, 125), (360, 90), (120, 90, 80), 18)
    cv2.circle(canvas, (40, 90), 12, (180, 150, 130), -1)
    cv2.circle(canvas, (360, 90), 12, (180, 150, 130), -1)
    cv2.rectangle(canvas, (165, 280), (195, 460), (60, 60, 90), -1)
    cv2.rectangle(canvas, (205, 280), (235, 460), (60, 60, 90), -1)
    _, buf = cv2.imencode(".png", canvas)
    return buf.tobytes()


@pytest.fixture
def synthetic_model_garment_bytes() -> bytes:
    """Unsuitable catalog photo: a model/face wearing a jacket."""
    canvas = np.full((300, 240, 3), 250, dtype=np.uint8)
    cv2.circle(canvas, (120, 55), 32, (120, 150, 200), -1)
    cv2.rectangle(canvas, (50, 90), (190, 250), (40, 40, 50), -1)
    _, buf = cv2.imencode(".png", canvas)
    return buf.tobytes()


@pytest.fixture
def synthetic_rgba_garment_bytes() -> bytes:
    """
    Creates a synthetic garment with a transparent background (RGBA PNG).
    """
    canvas = np.zeros((150, 150, 4), dtype=np.uint8)
    # Torso
    cv2.rectangle(canvas, (30, 30), (120, 130), (50, 120, 220, 255), -1)
    _, buf = cv2.imencode(".png", canvas)
    return buf.tobytes()
