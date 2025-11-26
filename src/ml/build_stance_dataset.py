# src/ml/build_stance_dataset.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from ..segmentation.move_capture import (
    capture_moves_with_spec,
    load_landmarks_csv,
    series_xy,
    LMK,
)
from ..features.feature_extractor import (
    extract_stance_features_from_dict,
    FEATURE_NAMES,
)

# ---------------------------------------------------------------------
# Rutas base (asumiendo la estructura que mostraste)
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"

LANDMARKS_DIR = DATA_DIR / "landmarks" / "8yang"
DATASETS_DIR = DATA_DIR / "datasets"

DEFAULT_CONFIG = CONFIG_DIR / "default.yaml"
DEFAULT_SPEC = CONFIG_DIR / "patterns" / "8yang_spec.json"
DEFAULT_POSE_SPEC = CONFIG_DIR / "patterns" / "pose_spec.json"


def _build_landmarks_dict(csv_path: Path) -> Dict[str, np.ndarray]:
    """
    Carga el CSV de landmarks y construye un dict:
      { 'L_ANK': (n_frames,2), 'R_ANK': ..., ... }

    Reutiliza load_landmarks_csv + series_xy del módulo de segmentación
    para que todo quede consistente.
    """
    df = load_landmarks_csv(csv_path)
    nframes = int(df["frame"].max()) + 1

    landmarks_dict: Dict[str, np.ndarray] = {}
    # mismos puntos que ya usas en move_capture
    needed_points = [
        "L_SH", "R_SH",
        "L_HIP", "R_HIP",
        "L_KNEE", "R_KNEE",
        "L_ANK", "R_ANK",
        "L_HEEL", "R_HEEL",
        "L_FOOT", "R_FOOT",
        "L_WRIST", "R_WRIST",
    ]
    for name in needed_points:
        idx = LMK.get(name)
        if idx is None:
            continue
        landmarks_dict[name] = series_xy(df, idx, nframes)

    return landmarks_dict


def build_stance_dataset(
    out_path: Path = DATASETS_DIR / "stance_8yang.csv",
    config_path: Path = DEFAULT_CONFIG,
    spec_path: Path = DEFAULT_SPEC,
    pose_spec_path: Path = DEFAULT_POSE_SPEC,
) -> Path:
    """
    Recorre todos los CSV en data/landmarks/8yang/, segmenta 8yang y
    construye un dataset de posturas.

    Cada fila del CSV resultante corresponde a:
      - un movimiento (segmento)
      - su frame representativo pose_f
      - las features de postura en ese frame
      - la etiqueta 'stance_label' obtenida desde expected_stance
        de 8yang_spec.json

    Este CSV es compatible con stance_classifier.train_from_csv()
    (columna label_col='stance_label').
    """
    if not LANDMARKS_DIR.exists():
        raise SystemExit(f"[DATASET] No existe carpeta de landmarks: {LANDMARKS_DIR}")

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    csv_files = sorted(LANDMARKS_DIR.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"[DATASET] No se encontraron CSV en: {LANDMARKS_DIR}")

    print("======================================================")
    print("[DATASET] Construyendo dataset de posturas (stance)")
    print(f"[DATASET] Landmarks dir : {LANDMARKS_DIR}")
    print(f"[DATASET] Config YAML   : {config_path}")
    print(f"[DATASET] 8yang_spec    : {spec_path}")
    print(f"[DATASET] pose_spec     : {pose_spec_path}")
    print("======================================================")

    for csv_path in csv_files:
        sample_id = csv_path.stem
        video_id = sample_id

        print(f"[DATASET] Procesando {csv_path.name} (video_id={video_id})")

        # 1) Segmentación + info de movimientos (pose_f, expected_stance, etc.)
        result = capture_moves_with_spec(
            csv_path=csv_path,
            video_path=None,
            config_path=config_path,
            spec_path=spec_path,
            pose_spec_path=pose_spec_path,
            use_ml_classifier=False,   # aquí NO usamos ML, solo spec/heurística
            ml_model_path=None,
        )

        if not result.moves:
            print(f"[DATASET]  -> sin movimientos detectados, salto.")
            continue

        # 2) Landmarks dict para calcular features en el frame pose_f
        landmarks_dict = _build_landmarks_dict(csv_path)

        for m in result.moves:
            stance_label = (m.expected_stance or "").strip()
            if not stance_label:
                # si el spec no define postura esperada, saltamos
                continue

            frame_idx = int(m.pose_f)

            try:
                feats = extract_stance_features_from_dict(landmarks_dict, frame_idx)
            except Exception as e:
                print(f"[DATASET]  -> error extrayendo features en frame {frame_idx}: {e}")
                continue

            # construir fila del dataset
            row: Dict[str, Any] = {
                "video": csv_path.name,
                "video_id": video_id,
                "sample_id": sample_id,
                "move_idx": int(m.idx),
                "frame_idx": frame_idx,
                "stance_label": stance_label,
            }

            # agregar features numéricas
            for fname in FEATURE_NAMES:
                val = feats.get(fname, np.nan)
                row[fname] = float(val) if val is not None else np.nan

            rows.append(row)

    if not rows:
        raise SystemExit("[DATASET] No se generaron filas, revise los specs y CSV.")

    df = pd.DataFrame(rows)
    # opcional: eliminar filas con NaN en features clave
    df = df.dropna(subset=FEATURE_NAMES)

    print(f"[DATASET] Filas totales: {len(df)}")
    print(f"[DATASET] Clases de stance: {sorted(df['stance_label'].unique())}")

    # guardar CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[DATASET] ✅ Dataset guardado en: {out_path}")

    return out_path


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Construir dataset de posturas (stance) a partir de landmarks 8yang."
    )
    ap.add_argument(
        "--out",
        type=str,
        default=str(DATASETS_DIR / "stance_8yang.csv"),
        help="Ruta de salida del CSV de dataset.",
    )
    ap.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Ruta a config/default.yaml",
    )
    ap.add_argument(
        "--spec",
        type=str,
        default=str(DEFAULT_SPEC),
        help="Ruta a 8yang_spec.json",
    )
    ap.add_argument(
        "--pose-spec",
        type=str,
        default=str(DEFAULT_POSE_SPEC),
        help="Ruta a pose_spec.json",
    )
    args = ap.parse_args()

    build_stance_dataset(
        out_path=Path(args.out),
        config_path=Path(args.config),
        spec_path=Path(args.spec),
        pose_spec_path=Path(args.pose_spec),
    )


if __name__ == "__main__":
    main()
