# src/recording/session_recorder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
import os
import time

import cv2
import numpy as np


@dataclass
class StreamInfo:
    """Metadatos básicos de un stream de vídeo."""
    name: str
    filename: str
    fps: int
    fourcc: str = "mp4v"  # mp4 visualizable en casi todo


class StreamRecorder:
    """
    Graba un solo stream de vídeo (cualquier cámara OpenCV, incluida Kinect color).
    El tamaño de frame se infiere en la primera llamada a write().
    """
    def __init__(self, info: StreamInfo, output_dir: str):
        self.info = info
        self.output_dir = output_dir
        self.path = os.path.join(output_dir, info.filename)
        os.makedirs(output_dir, exist_ok=True)

        self._writer: Optional[cv2.VideoWriter] = None
        self._frame_size: Optional[Tuple[int, int]] = None
        self._fourcc = cv2.VideoWriter_fourcc(*info.fourcc)
        self._frame_count = 0

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def write(self, frame_bgr: np.ndarray) -> None:
        if frame_bgr is None:
            return

        h, w = frame_bgr.shape[:2]
        if self._writer is None:
            self._frame_size = (w, h)
            self._writer = cv2.VideoWriter(
                self.path,
                self._fourcc,
                self.info.fps,
                self._frame_size,
                True,
            )

        if not self._writer.isOpened():
            return

        self._writer.write(frame_bgr)
        self._frame_count += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


class Kinect3DRecorder:
    """
    Guarda información 3D para repro en tu visor:
    - opcionalmente nube de puntos por frame (Kinect point cloud)
    - opcionalmente joints 3D (world o camera space)
    Se guarda todo en un .npz comprimido.
    """
    def __init__(self, output_dir: str, filename: str = "kinect_3d_data.npz"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.path = os.path.join(output_dir, filename)

        # listas de numpy arrays por frame (longitudes variables)
        self._cloud_frames: List[np.ndarray] = []
        self._joint_frames: List[Dict[int, np.ndarray]] = []
        self._timestamps: List[float] = []

    def record_frame(
        self,
        cloud_cam: Optional[np.ndarray] = None,
        joint_positions: Optional[Dict[int, np.ndarray]] = None,
    ) -> None:
        t = time.time()
        self._timestamps.append(t)

        if cloud_cam is not None:
            # Asegurar tipo
            self._cloud_frames.append(np.asarray(cloud_cam, dtype=np.float32))
        else:
            self._cloud_frames.append(np.zeros((0, 3), dtype=np.float32))

        if joint_positions is not None:
            # guardamos como dict de int -> (3,)
            clean = {
                int(jid): np.asarray(p, dtype=np.float32)
                for jid, p in joint_positions.items()
            }
            self._joint_frames.append(clean)
        else:
            self._joint_frames.append({})

    def save(self) -> None:
        # guardamos como objetos para no forzar misma longitud
        cloud_arr = np.array(self._cloud_frames, dtype=object)
        joints_arr = np.array(self._joint_frames, dtype=object)
        ts_arr = np.array(self._timestamps, dtype=np.float64)

        np.savez_compressed(
            self.path,
            cloud_frames=cloud_arr,
            joint_frames=joints_arr,
            timestamps=ts_arr,
        )

    def close(self) -> None:
        self.save()


class RecordingSession:
    """
    Sesión que agrupa todas las cámaras:
    - Varios StreamRecorder (webcams externas, Kinect color, etc.)
    - Un Kinect3DRecorder opcional para 3D.
    """
    def __init__(self, output_dir: str, fps: int = 30, save_kinect_3d: bool = True):
        self.output_dir = output_dir
        self.fps = fps
        self.streams: Dict[str, StreamRecorder] = {}
        self.kinect_3d: Optional[Kinect3DRecorder] = (
            Kinect3DRecorder(output_dir) if save_kinect_3d else None
        )
        self._active: bool = True

    @property
    def active(self) -> bool:
        return self._active

    # ---- setup de streams ----
    def add_stream(self, stream_id: str, filename: str) -> None:
        """
        stream_id: identificador lógico, ej. "kinect_color", "cam_1", etc.
        filename: nombre del archivo mp4 a crear.
        """
        info = StreamInfo(name=stream_id, filename=filename, fps=self.fps)
        self.streams[stream_id] = StreamRecorder(info, self.output_dir)

    # ---- escritura de frames ----
    def write_frame(self, stream_id: str, frame_bgr: np.ndarray) -> None:
        if not self._active:
            return
        rec = self.streams.get(stream_id)
        if rec is None:
            return
        rec.write(frame_bgr)

    def record_kinect_3d(
        self,
        cloud_cam: Optional[np.ndarray],
        joint_positions: Optional[Dict[int, np.ndarray]],
    ) -> None:
        if not self._active or self.kinect_3d is None:
            return
        self.kinect_3d.record_frame(cloud_cam=cloud_cam, joint_positions=joint_positions)

    # ---- cierre ----
    def stop(self) -> None:
        if not self._active:
            return
        self._active = False

        for rec in self.streams.values():
            rec.close()

        if self.kinect_3d is not None:
            self.kinect_3d.close()
