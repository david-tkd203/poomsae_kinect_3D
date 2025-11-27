# src/offline/offline_pipeline.py
"""Herramientas de procesamiento offline para una sesión "capture".

Este módulo realiza un pipeline ligero que, dado un directorio de
captura con videos (.mp4), extrae landmarks con MediaPipe, calcula un
score (placeholder) y genera un reporte XLSX.

Los cambios que hagamos aquí son deliberadamente mínimos: mejorar
mensajes, añadir docstrings y comentarios para que sea más legible
para un desarrollador que revisa el código.
"""
from __future__ import annotations

from typing import Callable, Optional, Dict, List
import os
import glob

import cv2
import numpy as np
import pandas as pd

from ..pose import MediaPipePoseEstimator
from ..config import POSE_MODEL_COMPLEXITY


ProgressCallback = Callable[[str, float], None]
#    mensaje, progreso (0.0–1.0)


def _extract_landmarks_from_video(
    video_path: str,
    mp_pose: MediaPipePoseEstimator,
    progress_cb: Optional[ProgressCallback] = None,
) -> Dict[str, np.ndarray]:
    """
    Procesa un video con MediaPipe y devuelve un diccionario con:
    - "landmarks_3d": (N, L, 3)
    - "visibility":   (N, L) opcional (si tu wrapper lo expone)
    - "timestamps":   (N,) en segundos relativos
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    lms_3d: List[np.ndarray] = []
    vis_list: List[np.ndarray] = []
    ts_list: List[float] = []

    frame_idx = 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            break

        # Aquí asumimos que tu wrapper devuelve (lm_2d, lm_3d, visibility)
        # Adáptalo si tu firma es distinta.
        res = mp_pose.process(frame_bgr)
        if isinstance(res, tuple):
            if len(res) == 3:
                lm_2d, lm_3d, visibility = res
            elif len(res) == 2:
                lm_2d, lm_3d = res
                visibility = None
            else:
                lm_3d = res
                visibility = None
        else:
            lm_3d = res
            visibility = None

        if lm_3d is not None and lm_3d.size > 0:
            lm_3d = np.asarray(lm_3d, dtype=np.float32)
            lms_3d.append(lm_3d)
            if visibility is not None:
                vis_list.append(np.asarray(visibility, dtype=np.float32))
            else:
                vis_list.append(np.ones(lm_3d.shape[:2], dtype=np.float32))
            ts_list.append(frame_idx / fps)

        frame_idx += 1

        if progress_cb is not None and total_frames > 0:
            frac = min(1.0, frame_idx / total_frames)
            progress_cb(f"Landmarks: {os.path.basename(video_path)} ({frame_idx}/{total_frames})", frac)

    cap.release()

    if not lms_3d:
        return {
            "landmarks_3d": np.zeros((0, 0, 3), dtype=np.float32),
            "visibility": np.zeros((0, 0), dtype=np.float32),
            "timestamps": np.zeros((0,), dtype=np.float32),
        }

    return {
        "landmarks_3d": np.stack(lms_3d, axis=0),
        "visibility": np.stack(vis_list, axis=0),
        "timestamps": np.asarray(ts_list, dtype=np.float32),
    }


def compute_pal_yang_score(
    landmarks_3d: np.ndarray,
    timestamps: np.ndarray,
    video_name: str,
) -> Dict[str, float]:
    """
    Stub para tu score Pal Yang. Aquí solo dejo una estructura básica
    que puedes reemplazar por tu lógica de evaluación real.
    """
    n_frames = landmarks_3d.shape[0]
    duration_s = float(timestamps[-1] - timestamps[0]) if n_frames > 1 else 0.0

    # TODO: implementar tu métrica real
    score_dummy = float(n_frames)  # solo un ejemplo

    return {
        "video": video_name,
        "frames": float(n_frames),
        "duracion_s": duration_s,
        "score_pal_yang": score_dummy,
    }


def run_offline_pipeline(
    capture_dir: str,
    progress_cb: Optional[ProgressCallback] = None,
) -> str:
    """
    Pipeline completo después de grabar:
    1) Busca videos .mp4 en capture_dir.
    2) Genera landmarks con MediaPipe y los guarda en .npz.
    3) Calcula score Pal Yang y arma un DataFrame resumen.
    4) Exporta un reporte XLSX.
    5) Devuelve la ruta al .xlsx generado.
    """
    video_files = sorted(glob.glob(os.path.join(capture_dir, "*.mp4")))
    if not video_files:
        # Mensaje más amigable y orientador para usuarios no técnicos
        raise RuntimeError(
            f"No se encontraron archivos .mp4 en '{capture_dir}'. "
            "Verifique que la grabación se haya completado y que la ruta sea correcta."
        )

    if progress_cb is not None:
        progress_cb("Inicializando MediaPipe Pose...", 0.0)

    mp_pose = MediaPipePoseEstimator(model_complexity=POSE_MODEL_COMPLEXITY)

    rows: List[Dict[str, float]] = []
    total_steps = len(video_files) * 2 + 1  # landmarks + score + reporte
    step = 0

    for vf in video_files:
        base = os.path.basename(vf)
        name, _ = os.path.splitext(base)

        # --- Paso 1: landmarks ---
        if progress_cb is not None:
            progress_cb(f"Generando landmarks para {base}", step / total_steps)

        lm_data = _extract_landmarks_from_video(vf, mp_pose)
        step += 1

        # Guardar landmarks en .npz por si quieres reprocesar o depurar
        # (esto facilita reproducibilidad sin volver a ejecutar MediaPipe).
        npz_path = os.path.join(capture_dir, f"{name}_landmarks.npz")
        np.savez_compressed(
            npz_path,
            landmarks_3d=lm_data["landmarks_3d"],
            visibility=lm_data["visibility"],
            timestamps=lm_data["timestamps"],
        )

        # --- Paso 2: score Pal Yang ---
        if progress_cb is not None:
            progress_cb(f"Calculando score Pal Yang para {base}", step / total_steps)

        score_row = compute_pal_yang_score(
            lm_data["landmarks_3d"],
            lm_data["timestamps"],
            video_name=name,
        )
        rows.append(score_row)
        step += 1

    # --- Paso 3: reporte XLSX ---
    if progress_cb is not None:
        progress_cb("Generando reporte XLSX...", step / total_steps)

    df = pd.DataFrame(rows)
    report_path = os.path.join(capture_dir, "reporte_poomsae_pal_yang.xlsx")
    df.to_excel(report_path, index=False)

    step += 1
    if progress_cb is not None:
        progress_cb("Pipeline completado", 1.0)

    mp_pose.close()
    return report_path
