from __future__ import annotations

import base64

import cv2
import numpy as np


def decode_base64_image(encoded: str) -> np.ndarray:
    if "," in encoded:
        encoded = encoded.split(",", 1)[1]
    data = base64.b64decode(encoded)
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode base64 image")
    return image


def resize_for_model(image: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

