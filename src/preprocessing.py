"""Preprocesamiento de frames: redimensionado y suavizado."""

from typing import Optional, Tuple

import cv2
import numpy as np

import config


def resize_to_max_width(frame: np.ndarray, max_width: int = config.MAX_WIDTH) -> np.ndarray:
    """Reduce el frame manteniendo la proporcion si supera el ancho maximo."""
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / width
    new_size = (max_width, max(1, int(round(height * scale))))
    # INTER_AREA es la interpolacion recomendada para reducir imagenes
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def smooth(frame: np.ndarray) -> np.ndarray:
    """Aplica un desenfoque gaussiano leve para reducir ruido."""
    kernel = (config.BLUR_KERNEL_SIZE, config.BLUR_KERNEL_SIZE)
    return cv2.GaussianBlur(frame, kernel, 0)


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Redimensiona y, si esta activado en config, suaviza el frame."""
    processed = resize_to_max_width(frame)
    if config.USE_BLUR:
        processed = smooth(processed)
    return processed


def crop_box(frame: np.ndarray, box: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
    """Recorta una caja (x1, y1, x2, y2) acotada al frame.

    Devuelve None si el recorte es mas chico que MIN_BOX_SIDE_PX.
    """
    height, width = frame.shape[:2]
    x1 = max(0, int(box[0]))
    y1 = max(0, int(box[1]))
    x2 = min(width, int(box[2]))
    y2 = min(height, int(box[3]))
    if (x2 - x1) < config.MIN_BOX_SIDE_PX or (y2 - y1) < config.MIN_BOX_SIDE_PX:
        return None
    return frame[y1:y2, x1:x2]
