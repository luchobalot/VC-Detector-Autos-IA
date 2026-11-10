"""Compara resultados/eventos.csv contra la anotacion manual (verdad.csv)."""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd

import config


class EvaluationError(Exception):
    """Error al leer los archivos de evaluacion."""


def parse_args() -> argparse.Namespace:
    """Lee los argumentos de linea de comandos."""
    parser = argparse.ArgumentParser(description="Evaluacion contra anotacion manual.")
    parser.add_argument("--events", type=Path,
                        default=config.RESULTS_DIR / config.EVENTS_FILENAME)
    parser.add_argument("--truth", type=Path, default=config.GROUND_TRUTH_PATH)
    parser.add_argument("--tolerance", type=float, default=config.TOLERANCE_SECONDS,
                        help="Tolerancia en segundos.")
    return parser.parse_args()


def read_csv(path: Path, required: List[str]) -> pd.DataFrame:
    """Lee un CSV validando que exista y tenga las columnas necesarias."""
    if not path.is_file():
        raise EvaluationError(f"No existe el archivo: {path}")
    table = pd.read_csv(path)
    missing = [c for c in required if c not in table.columns]
    if missing:
        raise EvaluationError(f"Faltan columnas {missing} en {path}")
    return table


def interval_distance(second: float, start: float, end: float) -> float:
    """Distancia de un segundo a un intervalo (0 si cae adentro)."""
    if second < start:
        return start - second
    if second > end:
        return second - end
    return 0.0


def match_events(truth: List[float], events: pd.DataFrame, tolerance: float
                 ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Empareja uno a uno autos reales con eventos, priorizando los mas cercanos.

    Devuelve (pares, indices_de_verdad_sin_par, indices_de_eventos_sin_par).
    """
    candidates = []
    for gi, second in enumerate(truth):
        for ei, row in enumerate(events.itertuples()):
            dist = interval_distance(second, row.segundo_inicio, row.segundo_fin)
            if dist <= tolerance:
                candidates.append((dist, gi, ei))
    candidates.sort()
    used_truth, used_events, pairs = set(), set(), []
    for _, gi, ei in candidates:
        if gi not in used_truth and ei not in used_events:
            used_truth.add(gi)
            used_events.add(ei)
            pairs.append((gi, ei))
    missed = [i for i in range(len(truth)) if i not in used_truth]
    extra = [i for i in range(len(events)) if i not in used_events]
    return pairs, missed, extra


def safe_ratio(numerator: int, denominator: int) -> float:
    """Division que devuelve 0.0 si el denominador es cero."""
    return numerator / denominator if denominator else 0.0


def main() -> int:
    """Punto de entrada."""
    args = parse_args()
    try:
        events = read_csv(args.events, ["segundo_inicio", "segundo_fin", "decision", "track_id"])
        truth_table = read_csv(args.truth, [config.GROUND_TRUTH_COLUMN])
    except EvaluationError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    white_events = events[events["decision"] == config.DECISION_WHITE].reset_index(drop=True)
    truth = sorted(truth_table[config.GROUND_TRUTH_COLUMN].astype(float).tolist())
    pairs, missed, extra = match_events(truth, white_events, args.tolerance)

    true_positives = len(pairs)
    false_negatives = len(missed)
    false_positives = len(extra)
    precision = safe_ratio(true_positives, true_positives + false_positives)
    recall = safe_ratio(true_positives, true_positives + false_negatives)

    print(f"Tolerancia: {args.tolerance} s")
    print(f"Autos blancos reales:        {len(truth)}")
    print(f"Detectados correctamente:    {true_positives}")
    print(f"Falsos positivos:            {false_positives}")
    print(f"Falsos negativos:            {false_negatives}")
    print(f"Precision:                   {precision:.3f}")
    print(f"Recall:                      {recall:.3f}")
    if missed:
        print("Segundos no detectados:", [truth[i] for i in missed])
    if extra:
        print("Track IDs falsos positivos:", [int(white_events.loc[i, "track_id"]) for i in extra])
    return 0


if __name__ == "__main__":
    sys.exit(main())
