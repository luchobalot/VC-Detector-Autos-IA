"""Script principal: detecta autos blancos en un video y registra los eventos."""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

import color
import config
import events
import preprocessing
import video_io
import visualization
from detector import CarTracker, DetectorError


def parse_args() -> argparse.Namespace:
    """Define y lee los argumentos de linea de comandos."""
    parser = argparse.ArgumentParser(description="Deteccion de autos blancos en video.")
    parser.add_argument("--video", type=Path, default=None,
                        help="Ruta del video (por defecto, el primero de video/).")
    parser.add_argument("--model", default=config.MODEL_NAME, help="Modelo YOLO a usar.")
    parser.add_argument("--conf", type=float, default=config.CONFIDENCE_THRESHOLD,
                        help="Umbral de confianza de la deteccion.")
    parser.add_argument("--white-threshold", type=float, default=config.WHITE_PERCENT_THRESHOLD,
                        help="Porcentaje minimo de blanco para considerar un auto blanco.")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Procesar solo los primeros N frames.")
    parser.add_argument("--max-seconds", type=float, default=None,
                        help="Procesar solo los primeros N segundos.")
    parser.add_argument("--output-dir", type=Path, default=config.RESULTS_DIR,
                        help="Carpeta de salida.")
    parser.add_argument("--sweep", action="store_true",
                        help="Barrer los umbrales de confianza y guardar una tabla comparativa.")
    return parser.parse_args()


def resolve_frame_limit(args: argparse.Namespace, fps: float) -> Optional[int]:
    """Combina --max-frames y --max-seconds en un unico limite de frames."""
    limits = []
    if args.max_frames is not None:
        limits.append(args.max_frames)
    if args.max_seconds is not None:
        limits.append(int(args.max_seconds * fps))
    return min(limits) if limits else None


def measure_detections(frame, detections, second, frame_index, builder, collector, white_threshold):
    """Mide el blanco de cada auto del frame y devuelve (filas, datos_para_dibujar)."""
    rows, to_draw = [], []
    for det in detections:
        crop = preprocessing.crop_box(frame, det.box)
        if crop is None:
            continue
        percent = color.white_percentage(crop)
        builder.add(det.track_id, events.Observation(frame_index, second, det.confidence, percent))
        collector.update(det.track_id, crop, det.confidence)
        rows.append({
            "frame": frame_index,
            "segundo": round(second, 3),
            "track_id": det.track_id,
            "confianza": round(det.confidence, 4),
            "porcentaje_blanco": round(percent, 2),
            "es_blanco": percent >= white_threshold,
        })
        to_draw.append(det)
    return rows, to_draw


def draw_frame(frame, detections, builder) -> None:
    """Dibuja los autos del frame usando la decision acumulada de cada track."""
    for det in detections:
        visualization.draw_detection(
            frame, det.box, det.track_id,
            builder.running_percent(det.track_id),
            builder.is_white_so_far(det.track_id),
        )


def save_csv(rows: List[dict], columns: List[str], path: Path) -> pd.DataFrame:
    """Guarda filas como CSV con columnas fijas (aunque no haya filas)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows, columns=columns)
    table.to_csv(path, index=False)
    return table


def run_pipeline(
    video_path: Path,
    output_dir: Path,
    model_name: str = config.MODEL_NAME,
    confidence: float = config.CONFIDENCE_THRESHOLD,
    white_threshold: float = config.WHITE_PERCENT_THRESHOLD,
    frame_limit: Optional[int] = None,
    save_outputs: bool = True,
) -> pd.DataFrame:
    """Procesa un video completo y devuelve la tabla de eventos.

    Con save_outputs=False solo escribe los CSV (sin video ni capturas),
    que es lo que se usa en el barrido de umbrales.
    """
    capture, info = video_io.open_video(video_path)
    tracker = CarTracker(model_name, confidence)
    builder = events.EventBuilder(white_threshold)
    collector = visualization.CaptureCollector()
    writer = None
    frame_rows: List[dict] = []

    try:
        for index, raw in video_io.iter_frames(capture, frame_limit):
            frame = preprocessing.preprocess_frame(raw)
            second = index / info.fps
            detections = tracker.track(frame)
            rows, valid = measure_detections(
                frame, detections, second, index, builder, collector, white_threshold)
            frame_rows.extend(rows)

            if save_outputs:
                if writer is None:
                    height, width = frame.shape[:2]
                    writer = video_io.create_writer(
                        output_dir / config.ANNOTATED_VIDEO_FILENAME, info.fps, width, height)
                draw_frame(frame, valid, builder)
                writer.write(frame)

            if index % config.PROGRESS_EVERY_FRAMES == 0:
                print(f"  frame {index} ({second:.1f} s)")
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    event_rows = builder.build_events()
    events_table = save_csv(event_rows, config.EVENT_COLUMNS, output_dir / config.EVENTS_FILENAME)
    save_csv(frame_rows, config.FRAME_COLUMNS, output_dir / config.FRAME_DETECTIONS_FILENAME)
    if save_outputs:
        saved = collector.save(event_rows, output_dir / config.CAPTURES_DIR.name)
        print(f"  capturas guardadas: {saved}")
    return events_table


def count_decisions(events_table: pd.DataFrame, confidence: float) -> dict:
    """Resume cuantos eventos hubo de cada tipo para una fila de la tabla comparativa."""
    decisions = events_table["decision"]
    return {
        "confianza": confidence,
        "tracks_totales": len(events_table),
        "eventos_blancos": int((decisions == config.DECISION_WHITE).sum()),
        "eventos_no_blancos": int((decisions == config.DECISION_NOT_WHITE).sum()),
        "eventos_insuficientes": int((decisions == config.DECISION_INSUFFICIENT).sum()),
    }


def run_sweep(args: argparse.Namespace, video_path: Path, frame_limit: Optional[int]) -> None:
    """Corre el pipeline con cada umbral de confianza y guarda la tabla comparativa."""
    summary = []
    for confidence in config.CONFIDENCE_SWEEP:
        print(f"Barrido: confianza = {confidence:.2f}")
        out_dir = args.output_dir / config.SWEEP_DIR.name / f"conf_{confidence:.2f}"
        table = run_pipeline(video_path, out_dir, args.model, confidence,
                             args.white_threshold, frame_limit, save_outputs=False)
        summary.append(count_decisions(table, confidence))
    result = save_csv(summary, config.SWEEP_COLUMNS, args.output_dir / config.SWEEP_TABLE_FILENAME)
    print("\nTabla comparativa:")
    print(result.to_string(index=False))


def main() -> int:
    """Punto de entrada. Devuelve el codigo de salida del proceso."""
    args = parse_args()
    try:
        video_path = args.video or video_io.find_default_video()
        _, info = video_io.open_video(video_path)  # valida el video antes de cargar el modelo
        frame_limit = resolve_frame_limit(args, info.fps)
        print(f"Video: {video_path} ({info.fps:.1f} fps, {info.total_frames} frames)")
        if args.sweep:
            run_sweep(args, video_path, frame_limit)
        else:
            table = run_pipeline(video_path, args.output_dir, args.model, args.conf,
                                 args.white_threshold, frame_limit)
            print(count_decisions(table, args.conf))
            print(f"Resultados en: {args.output_dir}")
    except (video_io.VideoError, DetectorError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
