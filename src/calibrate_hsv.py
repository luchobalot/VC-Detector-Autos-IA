"""Herramienta de calibracion de los rangos HSV del blanco.

Modo grafico: ventana con trackbars y mascara en vivo.
Modo sin pantalla (--headless): guarda mascaras de varios valores de prueba.
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

import color
import config
import preprocessing
import video_io

TRACKBAR_NAMES = ("H min", "H max", "S min", "S max", "V min", "V max")


def parse_args() -> argparse.Namespace:
    """Lee los argumentos de linea de comandos."""
    parser = argparse.ArgumentParser(description="Calibracion de rangos HSV del blanco.")
    parser.add_argument("--video", type=Path, default=None,
                        help="Video del que se toma el frame (por defecto, el primero de video/).")
    parser.add_argument("--frame", type=int, default=0, help="Indice del frame a usar.")
    parser.add_argument("--headless", action="store_true",
                        help="No abrir ventana: guardar mascaras de prueba como imagenes.")
    parser.add_argument("--output-dir", type=Path, default=config.CALIBRATION_DIR,
                        help="Carpeta donde se guardan las mascaras en modo headless.")
    return parser.parse_args()


def load_frame(video: Path, index: int) -> np.ndarray:
    """Lee y preprocesa un frame del video."""
    capture, _ = video_io.open_video(video)
    try:
        return preprocessing.preprocess_frame(video_io.read_frame_at(capture, index))
    finally:
        capture.release()


def gui_available() -> bool:
    """Prueba si OpenCV puede abrir ventanas en este entorno."""
    try:
        cv2.namedWindow(config.CALIBRATION_WINDOW_NAME)
        cv2.destroyWindow(config.CALIBRATION_WINDOW_NAME)
        return True
    except cv2.error:
        return False


def create_trackbars() -> None:
    """Crea los seis trackbars inicializados con los valores de config.py."""
    initial = (config.HSV_LOWER[0], config.HSV_UPPER[0],
               config.HSV_LOWER[1], config.HSV_UPPER[1],
               config.HSV_LOWER[2], config.HSV_UPPER[2])
    maxima = (config.HSV_H_MAX, config.HSV_H_MAX,
              config.HSV_SV_MAX, config.HSV_SV_MAX,
              config.HSV_SV_MAX, config.HSV_SV_MAX)
    for name, value, maximum in zip(TRACKBAR_NAMES, initial, maxima):
        cv2.createTrackbar(name, config.CALIBRATION_WINDOW_NAME, value, maximum, lambda _: None)


def read_trackbars() -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """Devuelve (lower, upper) segun la posicion actual de los trackbars."""
    h_min, h_max, s_min, s_max, v_min, v_max = (
        cv2.getTrackbarPos(n, config.CALIBRATION_WINDOW_NAME) for n in TRACKBAR_NAMES)
    return (h_min, s_min, v_min), (h_max, s_max, v_max)


def window_is_open() -> bool:
    """False si el usuario cerro la ventana con la X."""
    return cv2.getWindowProperty(config.CALIBRATION_WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1


def print_values(lower, upper) -> None:
    """Imprime los valores listos para copiar a config.py."""
    print("\nCopia estos valores en src/config.py:")
    print(f"HSV_LOWER = {tuple(lower)}")
    print(f"HSV_UPPER = {tuple(upper)}")


def run_gui(frame: np.ndarray) -> None:
    """Muestra frame y mascara lado a lado hasta que el usuario cierra la ventana."""
    cv2.namedWindow(config.CALIBRATION_WINDOW_NAME, cv2.WINDOW_NORMAL)
    create_trackbars()
    print("Ajusta los trackbars. Cerra la ventana o presiona q/ESC para terminar.")
    lower, upper = read_trackbars()
    while window_is_open():
        lower, upper = read_trackbars()
        mask = color.white_mask(frame, lower, upper)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.imshow(config.CALIBRATION_WINDOW_NAME, np.hstack([frame, mask_bgr]))
        if cv2.waitKey(config.CALIBRATION_WAIT_MS) & 0xFF in config.CALIBRATION_QUIT_KEYS:
            break
    cv2.destroyAllWindows()
    print_values(lower, upper)


def run_headless(frame: np.ndarray, output_dir: Path) -> None:
    """Guarda el frame y una mascara por cada combinacion de S max y V min de prueba."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "frame_original.png"), frame)
    for s_max in config.CALIBRATION_SWEEP_S_MAX:
        for v_min in config.CALIBRATION_SWEEP_V_MIN:
            lower = (config.HSV_LOWER[0], config.HSV_LOWER[1], v_min)
            upper = (config.HSV_UPPER[0], s_max, config.HSV_UPPER[2])
            mask = color.white_mask(frame, lower, upper)
            cv2.imwrite(str(output_dir / f"mascara_Smax{s_max}_Vmin{v_min}.png"), mask)
    print(f"Mascaras guardadas en: {output_dir}")
    print("Elegi la combinacion que mejor aisle los autos blancos y actualiza config.py.")


def main() -> int:
    """Punto de entrada."""
    args = parse_args()
    try:
        video = args.video or video_io.find_default_video()
        frame = load_frame(video, args.frame)
    except video_io.VideoError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    if args.headless or not gui_available():
        if not args.headless:
            print("No hay entorno grafico disponible: se usa el modo alternativo.")
        run_headless(frame, args.output_dir)
    else:
        run_gui(frame)
    return 0


if __name__ == "__main__":
    sys.exit(main())
