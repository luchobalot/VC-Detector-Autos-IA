"""Dibujo de cajas y etiquetas, y guardado de capturas de ejemplo."""

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

import config


def draw_detection(
    frame: np.ndarray,
    box: Tuple[float, float, float, float],
    track_id: int,
    white_percent: float,
    is_white: bool,
) -> None:
    """Dibuja caja y etiqueta (ID y % de blanco) sobre el frame, in situ."""
    color = config.COLOR_WHITE_CAR if is_white else config.COLOR_OTHER_CAR
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, config.BOX_THICKNESS)

    label = f"ID {track_id} {white_percent:.0f}%"
    (text_w, text_h), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, config.FONT_THICKNESS
    )
    pad = config.LABEL_PADDING
    top = max(0, y1 - text_h - baseline - 2 * pad)
    # Fondo del color de la caja para que el texto siempre se lea
    cv2.rectangle(frame, (x1, top), (x1 + text_w + 2 * pad, top + text_h + baseline + 2 * pad),
                  color, -1)
    cv2.putText(frame, label, (x1 + pad, top + text_h + pad), cv2.FONT_HERSHEY_SIMPLEX,
                config.FONT_SCALE, (255, 255, 255), config.FONT_THICKNESS, cv2.LINE_AA)


class CaptureCollector:
    """Guarda el mejor recorte (mayor confianza) de cada track para usar de evidencia."""

    def __init__(self):
        self._best: Dict[int, Tuple[float, np.ndarray]] = {}

    def update(self, track_id: int, crop: np.ndarray, confidence: float) -> None:
        """Conserva el recorte si es el de mayor confianza visto para ese track."""
        current = self._best.get(track_id)
        if current is None or confidence > current[0]:
            self._best[track_id] = (confidence, crop.copy())

    @staticmethod
    def _clear_old_captures(output_dir: Path) -> None:
        """Borra capturas de corridas anteriores para no mezclar resultados."""
        for old in output_dir.glob(f"*{config.CAPTURE_EXTENSION}"):
            old.unlink()

    def _select(self, events: List[dict], decision: str) -> List[dict]:
        """Elige hasta MAX_CAPTURES_PER_GROUP eventos con esa decision."""
        chosen = [e for e in events if e["decision"] == decision and e["track_id"] in self._best]
        return chosen[: config.MAX_CAPTURES_PER_GROUP]

    def save(self, events: List[dict], output_dir: Path) -> int:
        """Guarda capturas de aceptados y descartados; devuelve cuantas escribio."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._clear_old_captures(output_dir)
        saved = 0
        groups = (
            ("aceptado", config.DECISION_WHITE),
            ("descartado", config.DECISION_NOT_WHITE),
        )
        for prefix, decision in groups:
            for event in self._select(events, decision):
                track_id = event["track_id"]
                percent = event["porcentaje_blanco_medio"]
                name = f"{prefix}_id{track_id}_blanco{percent:.1f}{config.CAPTURE_EXTENSION}"
                cv2.imwrite(str(output_dir / name), self._best[track_id][1])
                saved += 1
        return saved
