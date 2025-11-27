"""Herramientas para grabar sesiones de video y datos 3D.

Este módulo agrupa utilidades para grabar streams de vídeo (con
estimación automática del FPS) y para almacenar datos 3D de Kinect
en un archivo `.npz` comprimido. La intención de los comentarios y
docstrings aquí es facilitar la comprensión a un lector humano.
"""
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
    fps: int                     # FPS objetivo (ej. 60 para tus cámaras 2QHD)
    fourcc: str = "mp4v"         # códec preferido inicial (se intentan otros si falla)


class StreamRecorder:
    """
    Graba un solo stream de vídeo (cualquier cámara OpenCV, incluida Kinect color).
    El tamaño de frame se infiere en la primera llamada a write().

    Usa un pequeño búfer inicial para estimar el FPS real a partir del reloj
    del sistema, de modo que la reproducción quede a ~1x aunque el bucle
    principal no corra exactamente al FPS objetivo.
    """
    def __init__(self, info: StreamInfo, output_dir: str):
        self.info = info
        self.output_dir = output_dir
        self.path = os.path.join(output_dir, info.filename)
        os.makedirs(output_dir, exist_ok=True)

        self._writer: Optional[cv2.VideoWriter] = None
        self._frame_size: Optional[Tuple[int, int]] = None
        # fourcc numérico (se inicializa al primer intento)
        self._fourcc: Optional[int] = None
        self._frame_count = 0

        # Para estimar FPS real
        self._buffer_frames: List[np.ndarray] = []
        self._buffer_timestamps: List[float] = []
        self._t_start: Optional[float] = None
        self._fps_estimated: Optional[float] = None

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def _create_writer(self, fps: float) -> None:
        """
        Intenta crear el VideoWriter probando varios códecs en orden:

        1) El fourcc definido en StreamInfo (por ejemplo "mp4v").
        2) "avc1" (H.264 en contenedor MP4 en muchos sistemas).
        3) "H264"
        4) "XVID"
        5) "mp4v" (fallback final).

        Esto ayuda a aprovechar mejor la calidad de codificación manteniendo
        la resolución de entrada (p.ej. 2560x1440 para 2QHD).
        """
        if self._frame_size is None:
            return

        candidates: List[str] = []
        if self.info.fourcc:
            candidates.append(self.info.fourcc)

        # Códecs alternativos, evitando duplicados
        for cc in ("avc1", "H264", "XVID", "mp4v"):
            if cc not in candidates:
                candidates.append(cc)

        writer: Optional[cv2.VideoWriter] = None
        for cc in candidates:
            fourcc = cv2.VideoWriter_fourcc(*cc)
            w = cv2.VideoWriter(
                self.path,
                fourcc,
                float(fps),
                self._frame_size,
                True,
            )
            if w.isOpened():
                writer = w
                self._fourcc = fourcc
                break

        self._writer = writer  # puede quedar None si todo falla

    def _init_writer_if_needed(self) -> None:
        """
        Crea el VideoWriter usando el FPS estimado a partir del reloj.
        Si aún no se puede estimar bien (grabación muy corta), usa info.fps.
        """
        if self._writer is not None or self._frame_size is None:
            return

        fps = float(self.info.fps)

        # Intentar estimar FPS real si hay suficientes muestras
        if self._buffer_timestamps:
            dt_total = self._buffer_timestamps[-1] - self._buffer_timestamps[0]
            n_samples = len(self._buffer_timestamps)
            # al menos 5 frames y >200ms para que la estimación tenga algo de sentido
            if dt_total > 0.2 and n_samples >= 5:
                # ⚠️ CORREGIDO: usamos (n_samples - 1) / dt_total porque hay n-1 intervalos
                fps_estimated = (n_samples - 1) / max(dt_total, 1e-3)
                # Acotar la estimación a un rango razonable
                fps_estimated = float(max(10.0, min(120.0, fps_estimated)))
                self._fps_estimated = fps_estimated
                fps = fps_estimated

        if fps <= 0:
            fps = 30.0  # fallback defensivo

        self.info.fps = int(round(fps))

        # Crear writer con códecs candidatos
        self._create_writer(fps)

        # Volcar lo que haya en el búfer
        if self._writer is not None and self._writer.isOpened() and self._buffer_frames:
            # Volcar frames acumulados en el writer recién creado
            for buffered_frame in self._buffer_frames:
                self._writer.write(buffered_frame)
            # Limpiar búferes
            self._buffer_frames.clear()
            self._buffer_timestamps.clear()

    def write(self, frame_bgr: np.ndarray) -> None:
        if frame_bgr is None:
            return

        h, w = frame_bgr.shape[:2]
        if self._frame_size is None:
            # Se mantiene la resolución nativa del frame (ej. 2560x1440 para 2QHD)
            self._frame_size = (w, h)

        now = time.time()
        if self._t_start is None:
            self._t_start = now

        # Si todavía no hay writer, acumulamos frames para estimar FPS
        if self._writer is None:
            self._buffer_frames.append(frame_bgr.copy())
            self._buffer_timestamps.append(now)
            self._frame_count += 1

            # Cuando tengamos suficientes datos, inicializamos el writer
            # (15 frames o ~1 segundo como máximo para no usar mucha memoria)
            if len(self._buffer_frames) >= 15 or (now - self._t_start) >= 1.0:
                self._init_writer_if_needed()
            return

        # Si ya hay writer, escribimos normalmente
        if self._writer.isOpened():
            self._writer.write(frame_bgr)
            self._frame_count += 1

    def close(self) -> None:
        # Si nunca se creó el writer pero hay frames en búfer,
        # creamos uno usando FPS aproximado / de fallback.
        if self._writer is None and self._buffer_frames and self._frame_size is not None:
            self._init_writer_if_needed()

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
    def __init__(self, output_dir: str, fps: int = 60, save_kinect_3d: bool = True):
        """
        fps: FPS objetivo para los streams de vídeo.
             Para tus cámaras 2QHD, se recomienda 60.
        """
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
