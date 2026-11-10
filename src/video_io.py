"""Lectura y escritura de video con OpenCV (recordar: OpenCV usa BGR)."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np

import config


class VideoError(Exception):
    """Error al abrir, leer o escribir un video."""


@dataclass
class VideoInfo:
    """Datos basicos del video."""

    fps: float
    width: int
    height: int
    total_frames: int


def find_default_video() -> Path:
    """Devuelve el primer video de la carpeta video/ o lanza VideoError."""
    if not config.VIDEO_DIR.is_dir():
        raise VideoError(f"No existe la carpeta de videos: {config.VIDEO_DIR}")
    candidates = sorted(
        p for p in config.VIDEO_DIR.iterdir()
        if p.suffix.lower() in config.VIDEO_EXTENSIONS
    )
    if not candidates:
        raise VideoError(
            f"No hay videos en {config.VIDEO_DIR}. Copia uno alli o usa --video."
        )
    return candidates[0]


def open_video(path: Path) -> Tuple[cv2.VideoCapture, VideoInfo]:
    """Abre un video y devuelve el capturador junto con sus datos."""
    path = Path(path)
    if not path.is_file():
        raise VideoError(f"El archivo de video no existe: {path}")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise VideoError(f"OpenCV no pudo abrir el video: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        capture.release()
        raise VideoError(f"No se pudo leer el fps del video: {path}")
    info = VideoInfo(
        fps=float(fps),
        width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        total_frames=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    return capture, info


def iter_frames(
    capture: cv2.VideoCapture, max_frames: Optional[int] = None
) -> Iterator[Tuple[int, np.ndarray]]:
    """Genera (indice_de_frame, frame) hasta el final o hasta max_frames."""
    index = 0
    while max_frames is None or index < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        yield index, frame
        index += 1


def read_frame_at(capture: cv2.VideoCapture, index: int) -> np.ndarray:
    """Lee un frame puntual por indice."""
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    if not ok:
        raise VideoError(f"No se pudo leer el frame {index} del video.")
    return frame


def create_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    """Crea el escritor del video anotado (crea la carpeta si falta)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*config.OUTPUT_VIDEO_CODEC)
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise VideoError(f"No se pudo crear el video de salida: {path}")
    return writer
