# src/pose/mp_pose_wrapper.py
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import cv2
import mediapipe as mp


class MediaPipePoseEstimator:
    """
    Wrapper de MediaPipe Pose:
    - Recibe frames BGR (OpenCV).
    - Devuelve landmarks 2D y 3D (world) como arrays numpy.

    El constructor ahora expone los parámetros de confianza de detección
    y trackeo para permitir ajustar la sensibilidad desde herramientas
    que invocan este wrapper.
    """

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        mp_pose = mp.solutions.pose

        self._pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=float(min_detection_confidence),
            min_tracking_confidence=float(min_tracking_confidence),
        )
        self._mp_pose = mp_pose

    # ------------------------------------------------------------------
    def process(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Procesa un frame BGR.
        Devuelve:
        - landmarks_2d: np.ndarray (N, 3) con (x, y, visibility) en [0..1]
        - landmarks_3d: np.ndarray (N, 3) con (x, y, z) en sistema world de MediaPipe
        Si no detecta nada, devuelve arrays vacíos.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._pose.process(rgb)

        # Landmarks 2D
        if not results.pose_landmarks:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

        landmarks_2d = []
        for lm in results.pose_landmarks.landmark:
            landmarks_2d.append([lm.x, lm.y, lm.visibility])
        landmarks_2d = np.asarray(landmarks_2d, dtype=np.float32)

        # Landmarks 3D (world)
        landmarks_3d = []
        if results.pose_world_landmarks:
            for lm in results.pose_world_landmarks.landmark:
                landmarks_3d.append([lm.x, lm.y, lm.z])
        else:
            # Si por alguna razón no viene pose_world_landmarks, devolvemos array vacío
            landmarks_3d = []

        if landmarks_3d:
            landmarks_3d = np.asarray(landmarks_3d, dtype=np.float32)
        else:
            landmarks_3d = np.zeros((0, 3), dtype=np.float32)

        return landmarks_2d, landmarks_3d

    # ------------------------------------------------------------------
    def close(self):
        self._pose.close()
