"""Configuracion central del proyecto.

Todos los parametros (rutas, umbrales, rangos HSV, etc.) viven aca.
Ningun otro modulo deberia tener numeros "magicos".
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = PROJECT_ROOT / "video"
RESULTS_DIR = PROJECT_ROOT / "resultados"
CAPTURES_DIR = RESULTS_DIR / "capturas"
CALIBRATION_DIR = RESULTS_DIR / "calibracion"
SWEEP_DIR = RESULTS_DIR / "barrido"
ANNOTATIONS_DIR = PROJECT_ROOT / "anotaciones"

EVENTS_FILENAME = "eventos.csv"
FRAME_DETECTIONS_FILENAME = "detecciones_por_frame.csv"
ANNOTATED_VIDEO_FILENAME = "video_anotado.mp4"
SWEEP_TABLE_FILENAME = "barrido_confianza.csv"
GROUND_TRUTH_PATH = ANNOTATIONS_DIR / "verdad.csv"

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

# ---------------------------------------------------------------------------
# Modelo y tracking
# ---------------------------------------------------------------------------
MODEL_NAME = "yolov8n.pt"
CAR_CLASS_ID = 2  # id de la clase "car" en COCO
CONFIDENCE_THRESHOLD = 0.25
CONFIDENCE_SWEEP = (0.25, 0.50, 0.75)
TRACKER_CONFIG = "bytetrack.yaml"

# ---------------------------------------------------------------------------
# Preprocesamiento
# ---------------------------------------------------------------------------
MAX_WIDTH = 960  # ancho maximo del frame (px); si es menor no se agranda
USE_BLUR = True
BLUR_KERNEL_SIZE = 3  # debe ser impar

# ---------------------------------------------------------------------------
# Deteccion de color blanco (HSV en escala de OpenCV: H 0-179, S y V 0-255)
# ---------------------------------------------------------------------------
HSV_LOWER = (0, 0, 200)
HSV_UPPER = (179, 40, 255)
# % minimo de pixeles blancos para decir "blanco". Ajustado con el video de prueba:
# con 40 se perdian autos blancos subexpuestos (recall 0.33).
WHITE_PERCENT_THRESHOLD = 23.0
CENTER_ZONE_FRACTION = 0.6  # fraccion central del recorte que se mide
MIN_BOX_SIDE_PX = 12  # recortes mas chicos que esto se ignoran

# ---------------------------------------------------------------------------
# Eventos
# ---------------------------------------------------------------------------
MIN_FRAMES_FOR_DECISION = 4  # frames minimos de un track para decidir
# "median", "mean" o "percentile". Se usa percentil bajo porque los autos grises
# claros suben el % de blanco solo al pasar por zonas con reflejo.
AGGREGATION = "percentile"
AGGREGATION_PERCENTILE = 20
DECISION_WHITE = "blanco"
DECISION_NOT_WHITE = "no_blanco"
DECISION_INSUFFICIENT = "insuficiente"

EVENT_COLUMNS = [
    "track_id",
    "frame_inicio",
    "segundo_inicio",
    "frame_fin",
    "segundo_fin",
    "confianza_media",
    "porcentaje_blanco_medio",
    "decision",
]
FRAME_COLUMNS = [
    "frame",
    "segundo",
    "track_id",
    "confianza",
    "porcentaje_blanco",
    "es_blanco",
]
SWEEP_COLUMNS = [
    "confianza",
    "tracks_totales",
    "eventos_blancos",
    "eventos_no_blancos",
    "eventos_insuficientes",
]

# ---------------------------------------------------------------------------
# Visualizacion y capturas (colores en BGR)
# ---------------------------------------------------------------------------
COLOR_WHITE_CAR = (0, 200, 0)  # verde
COLOR_OTHER_CAR = (0, 0, 220)  # rojo
BOX_THICKNESS = 2
FONT_SCALE = 0.5
FONT_THICKNESS = 1
LABEL_PADDING = 3
OUTPUT_VIDEO_CODEC = "mp4v"
MAX_CAPTURES_PER_GROUP = 10  # aceptados y descartados, cada grupo
CAPTURE_EXTENSION = ".jpg"
PROGRESS_EVERY_FRAMES = 50

# ---------------------------------------------------------------------------
# Calibracion HSV
# ---------------------------------------------------------------------------
CALIBRATION_WINDOW_NAME = "Calibracion HSV"
CALIBRATION_WAIT_MS = 30
CALIBRATION_QUIT_KEYS = (27, ord("q"))  # ESC o q
HSV_H_MAX = 179
HSV_SV_MAX = 255
CALIBRATION_SWEEP_S_MAX = (20, 40, 60, 80)
CALIBRATION_SWEEP_V_MIN = (160, 180, 200, 220)

# ---------------------------------------------------------------------------
# Evaluacion
# ---------------------------------------------------------------------------
TOLERANCE_SECONDS = 1.5
GROUND_TRUTH_COLUMN = "segundo"
