"""Deteccion de autos con YOLO (Ultralytics) y tracking con ID por auto."""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

import config


class DetectorError(Exception):
    """Error al cargar o ejecutar el modelo."""


@dataclass
class Detection:
    """Un auto detectado en un frame."""

    track_id: int
    confidence: float
    box: Tuple[float, float, float, float]  # x1, y1, x2, y2 en pixeles


class CarTracker:
    """Envuelve un modelo YOLO y su tracker integrado (solo clase car)."""

    def __init__(self, model_name: str = config.MODEL_NAME,
                 confidence: float = config.CONFIDENCE_THRESHOLD):
        self.confidence = confidence
        self.model = self._load_model(model_name)

    @staticmethod
    def _load_model(model_name: str):
        """Carga el modelo (se descarga solo la primera vez si hay internet)."""
        try:
            from ultralytics import YOLO
            return YOLO(model_name)
        except Exception as error:
            raise DetectorError(
                f"No se pudo cargar el modelo '{model_name}': {error}. "
                "Verifica la instalacion de ultralytics y la conexion a internet "
                "(necesaria para la primera descarga del modelo)."
            ) from error

    def track(self, frame: np.ndarray) -> List[Detection]:
        """Detecta y trackea autos en un frame; mantiene el estado entre llamadas."""
        results = self.model.track(
            frame,
            persist=True,  # conserva los IDs entre frames
            classes=[config.CAR_CLASS_ID],
            conf=self.confidence,
            tracker=config.TRACKER_CONFIG,
            verbose=False,
        )
        return self._parse_results(results[0])

    @staticmethod
    def _parse_results(result) -> List[Detection]:
        """Convierte el resultado de Ultralytics en una lista de Detection."""
        boxes = result.boxes
        # Si el tracker aun no asigno IDs, boxes.id es None
        if boxes is None or boxes.id is None:
            return []
        ids = boxes.id.int().cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        coords = boxes.xyxy.cpu().tolist()
        return [
            Detection(track_id=i, confidence=c, box=tuple(b))
            for i, c, b in zip(ids, confidences, coords)
        ]
