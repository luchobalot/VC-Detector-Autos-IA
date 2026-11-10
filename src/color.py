"""Medicion del porcentaje de pixeles blancos de un recorte en HSV."""

from typing import Sequence

import cv2
import numpy as np

import config


def crop_center(image: np.ndarray, fraction: float = config.CENTER_ZONE_FRACTION) -> np.ndarray:
    """Devuelve la zona central de la imagen (fraccion del alto y del ancho).

    Sirve para evitar medir asfalto, sombras o fondo que quedan en los bordes de la caja.
    """
    height, width = image.shape[:2]
    margin = (1.0 - fraction) / 2.0
    x1 = int(width * margin)
    y1 = int(height * margin)
    x2 = max(x1 + 1, int(width * (1.0 - margin)))
    y2 = max(y1 + 1, int(height * (1.0 - margin)))
    return image[y1:y2, x1:x2]


def white_mask(
    bgr_image: np.ndarray,
    lower: Sequence[int] = config.HSV_LOWER,
    upper: Sequence[int] = config.HSV_UPPER,
) -> np.ndarray:
    """Convierte BGR a HSV y devuelve la mascara binaria del rango dado."""
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))


def white_percentage(
    crop: np.ndarray,
    fraction: float = config.CENTER_ZONE_FRACTION,
    lower: Sequence[int] = config.HSV_LOWER,
    upper: Sequence[int] = config.HSV_UPPER,
) -> float:
    """Porcentaje (0-100) de pixeles blancos en la zona central del recorte."""
    center = crop_center(crop, fraction)
    mask = white_mask(center, lower, upper)
    return 100.0 * float(np.count_nonzero(mask)) / mask.size
