"""Logica para convertir detecciones por frame en eventos por track ID."""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

import config


@dataclass
class Observation:
    """Medicion de un auto en un frame."""

    frame: int
    second: float
    confidence: float
    white_percent: float


@dataclass
class TrackHistory:
    """Todas las observaciones de un mismo track ID."""

    observations: List[Observation] = field(default_factory=list)


def aggregate(values: List[float], method: str = config.AGGREGATION) -> float:
    """Resume una lista de valores con mediana, promedio o percentil bajo."""
    if method == "median":
        return float(np.median(values))
    if method == "mean":
        return float(np.mean(values))
    if method == "percentile":
        # Percentil bajo: robusto a reflejos, que solo suben el blanco en parte del recorrido
        return float(np.percentile(values, config.AGGREGATION_PERCENTILE))
    raise ValueError(f"Metodo de agregacion desconocido: {method}")


class EventBuilder:
    """Acumula observaciones por track ID y arma los eventos finales."""

    def __init__(
        self,
        white_threshold: float = config.WHITE_PERCENT_THRESHOLD,
        min_frames: int = config.MIN_FRAMES_FOR_DECISION,
        method: str = config.AGGREGATION,
    ):
        self.white_threshold = white_threshold
        self.min_frames = min_frames
        self.method = method
        self.tracks: Dict[int, TrackHistory] = {}

    def add(self, track_id: int, observation: Observation) -> None:
        """Registra una observacion de un track."""
        self.tracks.setdefault(track_id, TrackHistory()).observations.append(observation)

    def running_percent(self, track_id: int) -> float:
        """Porcentaje de blanco agregado hasta el momento para un track."""
        values = [o.white_percent for o in self.tracks[track_id].observations]
        return aggregate(values, self.method)

    def is_white_so_far(self, track_id: int) -> bool:
        """Decision provisional (para dibujar) con lo observado hasta ahora."""
        return self.running_percent(track_id) >= self.white_threshold

    def _decide(self, history: TrackHistory, percent: float) -> str:
        """Decision final de un track segun cantidad de frames y porcentaje."""
        if len(history.observations) < self.min_frames:
            return config.DECISION_INSUFFICIENT
        if percent >= self.white_threshold:
            return config.DECISION_WHITE
        return config.DECISION_NOT_WHITE

    def _build_event(self, track_id: int, history: TrackHistory) -> dict:
        """Arma la fila de evento de un track."""
        obs = history.observations
        percent = aggregate([o.white_percent for o in obs], self.method)
        return {
            "track_id": track_id,
            "frame_inicio": obs[0].frame,
            "segundo_inicio": round(obs[0].second, 3),
            "frame_fin": obs[-1].frame,
            "segundo_fin": round(obs[-1].second, 3),
            "confianza_media": round(float(np.mean([o.confidence for o in obs])), 4),
            "porcentaje_blanco_medio": round(percent, 2),
            "decision": self._decide(history, percent),
        }

    def build_events(self) -> List[dict]:
        """Devuelve un evento por track ID, ordenados por frame de inicio."""
        events = [self._build_event(tid, h) for tid, h in self.tracks.items()]
        return sorted(events, key=lambda e: (e["frame_inicio"], e["track_id"]))
