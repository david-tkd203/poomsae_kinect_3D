# src/tools/extract_landmarks.py
"""
Herramienta offline para extraer landmarks de MediaPipe Pose desde videos,
integrada al proyecto de Poomsae (Pal Jang / 8yang).

- Usa el wrapper interno: src/pose/MediaPipePoseEstimator
- Respeta la estructura de salida previa:
    video_id, frame, lmk_id, x, y, z, visibility
- Aplica umbral de visibility (vis_min) -> si es bajo: x,y = NaN
- Opcional: clamp de x,y a [0,1] para mayor estabilidad

Uso típico (desde la raíz del repo):

    (.venv) python -m src.tools.extract_landmarks \
        --in data/raw/8yang_videos \
        --out-root data/landmarks \
        --alias 8yang \
        --subset train

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ..pose import MediaPipePoseEstimator
from ..config import POSE_MODEL_COMPLEXITY  # default interno del proyecto

VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi", ".mpg", ".mpeg")
N_LANDMARKS = 33


def iter_videos(path: Path) -> list[Path]:
    """
    Devuelve una lista ordenada de videos a procesar.

    - Si `path` es archivo y tiene extensión de vídeo -> lista de 1.
    - Si `path` es carpeta -> busca recursivamente por extensiones conocidas.
    """
    if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
        return [path]

    out: list[Path] = []
    for ext in VIDEO_EXTS:
        out.extend(path.rglob(f"*{ext}"))

    return sorted(out)


def extract_for_video(
    vpath: Path,
    out_csv: Path,
    resize_w: int,
    overwrite: bool,
    model_complexity: int = 2,
    det_conf: float = 0.5,
    track_conf: float = 0.5,
    vis_min: float = 0.15,
    clamp01: bool = True,
) -> None:
    """
    Extrae landmarks de MediaPipe Pose para un video y los guarda en CSV.

    Parameters
    ----------
    vpath : Path
        Ruta del video de entrada.
    out_csv : Path
        Ruta del CSV de salida.
    resize_w : int
        Ancho para redimensionar el frame manteniendo aspect ratio (0 = sin cambio).
    overwrite : bool
        Si False y el CSV existe, se omite el procesamiento.
    model_complexity : int
        Complejidad del modelo MediaPipe (0,1,2) – se alinea con POSE_MODEL_COMPLEXITY.
    det_conf : float
        Umbral de min_detection_confidence.
    track_conf : float
        Umbral de min_tracking_confidence.
    vis_min : float
        Umbral mínimo de visibility. Si visibility < vis_min -> x,y = NaN.
    clamp01 : bool
        Si True, fuerza x,y a permanecer en [0,1] (por seguridad numérica).
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists() and not overwrite:
        print(f"[SKIP] {out_csv} (existe, use --overwrite para regenerar)")
        return

    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        print(f"[WARN] No se pudo abrir: {vpath}")
        return

    # Instanciamos el wrapper interno de MediaPipe del proyecto
    pose_est = MediaPipePoseEstimator(
        model_complexity=model_complexity,
        min_detection_confidence=det_conf,
        min_tracking_confidence=track_conf,
    )

    rows: list[tuple[str, int, int, float, float, float, float]] = []
    vid = vpath.stem
    idx = 0  # frame index

    while True:
        ok, img = cap.read()
        if not ok:
            break

        # Redimensionar manteniendo aspecto si corresponde
        h, w = img.shape[:2]
        if resize_w and w != resize_w:
            nh = int(round(h * (resize_w / float(w))))
            img = cv2.resize(img, (resize_w, nh), interpolation=cv2.INTER_LINEAR)

        # Procesar con el wrapper (espera BGR)
        pose2d, _world = pose_est.process(img)

        # pose2d se espera shape (N_LANDMARKS, 4) = [x,y,z,visibility] normalizados
        if pose2d is not None and pose2d.shape[0] == N_LANDMARKS:
            for j in range(N_LANDMARKS):
                x, y, z, v = pose2d[j]

                # Normalizamos NaNs / infinities
                if not np.isfinite(x) or not np.isfinite(y) or v < vis_min:
                    x, y = np.nan, np.nan
                elif clamp01:
                    # MediaPipe a veces se sale levemente de [0,1]
                    x = 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)
                    y = 0.0 if y < 0.0 else (1.0 if y > 1.0 else y)

                rows.append((vid, idx, j, float(x), float(y), float(z), float(v)))
        else:
            # No hay landmarks confiables para este frame: rellenamos NaN/0
            for j in range(N_LANDMARKS):
                rows.append((vid, idx, j, np.nan, np.nan, 0.0, 0.0))

        idx += 1

    cap.release()
    pose_est.close()

    if not rows:
        print(f"[WARN] Sin landmarks: {vpath}")
        return

    df = pd.DataFrame(
        rows,
        columns=["video_id", "frame", "lmk_id", "x", "y", "z", "visibility"],
    )
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[OK] {out_csv}  ({len(df)} filas, frames={idx})")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Extracción de landmarks con MediaPipe Pose (proyecto Poomsae). "
            "Salida normalizada 0..1, x,y = NaN si visibility baja."
        )
    )
    ap.add_argument(
        "--in",
        dest="inp",
        required=True,
        help="Video o carpeta con videos",
    )
    ap.add_argument(
        "--out-root",
        required=True,
        help="Raíz de salida p.ej. data/landmarks",
    )
    ap.add_argument(
        "--alias",
        required=True,
        help="Alias p.ej. 8yang",
    )
    ap.add_argument(
        "--subset",
        default="",
        help="train/val/test (opcional, se concatena en la ruta de salida)",
    )
    ap.add_argument(
        "--resize-w",
        type=int,
        default=960,
        help="Ancho de procesamiento (0 = sin redimensionar)",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerar CSV si existe",
    )

    # Calidad / configuración de MediaPipe
    ap.add_argument(
        "--model-complexity",
        type=int,
        default=POSE_MODEL_COMPLEXITY,
        help=f"Complejidad del modelo Pose (default desde config: {POSE_MODEL_COMPLEXITY})",
    )
    ap.add_argument(
        "--det-conf",
        type=float,
        default=0.5,
        help="min_detection_confidence (default=0.5)",
    )
    ap.add_argument(
        "--track-conf",
        type=float,
        default=0.5,
        help="min_tracking_confidence (default=0.5)",
    )

    # Robustez
    ap.add_argument(
        "--vis-min",
        type=float,
        default=0.15,
        help="Umbral de visibility para aceptar x,y (NaN si menor)",
    )
    ap.add_argument(
        "--no-clamp01",
        action="store_true",
        help="No forzar x,y a [0,1]",
    )

    args = ap.parse_args()

    base = Path(args.inp)
    vids = iter_videos(base)
    if not vids:
        sys.exit("No se encontraron videos en la ruta de entrada.")

    out_root = Path(args.out_root) / args.alias
    if args.subset:
        out_root = out_root / args.subset

    print(f"[INFO] Videos encontrados: {len(vids)}")
    print(f"[INFO] Carpeta de salida: {out_root}")

    for vp in vids:
        out_csv = out_root / f"{vp.stem}.csv"
        extract_for_video(
            vpath=vp,
            out_csv=out_csv,
            resize_w=args.resize_w,
            overwrite=args.overwrite,
            model_complexity=args.model_complexity,
            det_conf=args.det_conf,
            track_conf=args.track_conf,
            vis_min=args.vis_min,
            clamp01=(not args.no_clamp01),
        )


if __name__ == "__main__":
    main()
