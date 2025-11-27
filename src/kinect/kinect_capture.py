# src/kinect/kinect_capture.py
from __future__ import annotations

from typing import Optional, Dict
import numpy as np

from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime


class KinectCapture:
    """Pequeño wrapper para Kinect v2 usado por la aplicación.

    Provee métodos de conveniencia para obtener el frame de color en
    formato BGR, las posiciones 3D de las articulaciones del primer
    cuerpo rastreado y una nube de puntos 3D del cuerpo usando Depth +
    BodyIndex. Las coordenadas devueltas están en metros y en el sistema
    de cámara.
    """

    def __init__(self, use_body_index: bool = False):
        self._use_body_index = use_body_index

        frame_sources = (
            PyKinectV2.FrameSourceTypes_Color |
            PyKinectV2.FrameSourceTypes_Body
        )
        if use_body_index:
            frame_sources |= PyKinectV2.FrameSourceTypes_Depth
            frame_sources |= PyKinectV2.FrameSourceTypes_BodyIndex

        self.kinect = PyKinectRuntime.PyKinectRuntime(frame_sources)

        # Cachés de últimos frames
        self._last_color_bgra = None
        self._last_body_frame = None
        self._last_depth = None
        self._last_body_index = None

    # ------------------------------------------------------------------
    def update_frames(self) -> None:
        """Actualizar los frames nuevos (color, body, depth, bodyIndex)."""
        if self.kinect.has_new_color_frame():
            frame = self.kinect.get_last_color_frame()
            frame = frame.reshape(
                (self.kinect.color_frame_desc.Height,
                 self.kinect.color_frame_desc.Width, 4)
            ).astype("uint8")
            self._last_color_bgra = frame

        if self.kinect.has_new_body_frame():
            self._last_body_frame = self.kinect.get_last_body_frame()

        if self._use_body_index and self.kinect.has_new_depth_frame():
            depth = self.kinect.get_last_depth_frame()
            self._last_depth = depth.reshape(
                (self.kinect.depth_frame_desc.Height,
                 self.kinect.depth_frame_desc.Width)
            )

        if self._use_body_index and self.kinect.has_new_body_index_frame():
            body_index = self.kinect.get_last_body_index_frame()
            self._last_body_index = body_index.reshape(
                (self.kinect.depth_frame_desc.Height,
                 self.kinect.depth_frame_desc.Width)
            )

    # ------------------------------------------------------------------
    def get_color_bgr(self) -> Optional[np.ndarray]:
        """Devuelve el último frame de color en formato BGR (H, W, 3) o None."""
        if self._last_color_bgra is None:
            return None
        # Kinect entrega BGRA: descartamos el canal alpha
        return self._last_color_bgra[:, :, :3].copy()

    # ------------------------------------------------------------------
    def get_first_tracked_body(self):
        """Devuelve el primer cuerpo rastreado (Body), o None si no hay."""
        if self._last_body_frame is None:
            return None

        bodies = self._last_body_frame.bodies
        for i in range(self.kinect.max_body_count):
            body = bodies[i]
            if body.is_tracked:
                return body
        return None

    # ------------------------------------------------------------------
    def get_joint_positions(self) -> Dict[int, np.ndarray]:
        """Obtiene las posiciones 3D de las articulaciones del primer cuerpo.

        Retorna
        -------
        Dict[int, np.ndarray]
            Mapeo `joint_type` -> `np.array([x, y, z])` en metros. Los
            `joint_type` corresponden a las constantes de `PyKinectV2`,
            por ejemplo `PyKinectV2.JointType_Head`.

        Nota
        ----
        Si una articulación no está trackeada se omite del diccionario.
        """
        body = self.get_first_tracked_body()
        if body is None:
            return {}

        joints = body.joints
        positions: Dict[int, np.ndarray] = {}

        for j_id in range(PyKinectV2.JointType_Count):
            joint = joints[j_id]
            if joint.TrackingState == PyKinectV2.TrackingState_NotTracked:
                continue
            positions[j_id] = np.array(
                [joint.Position.x, joint.Position.y, joint.Position.z],
                dtype=np.float32,
            )

        return positions

    # ------------------------------------------------------------------
    def get_skeleton_points_3d(self) -> np.ndarray:
        """
        Devuelve solo las posiciones 3D de los joints del primer cuerpo.
        (Se usa menos ahora, porque trabajamos con el dict completo).
        """
        positions = self.get_joint_positions()
        if not positions:
            return np.zeros((0, 3), dtype=np.float32)

        pts = np.stack(list(positions.values()), axis=0)
        return pts

    # ------------------------------------------------------------------
    def get_body_index_and_depth(self):
        """
        Devuelve (body_index, depth) si se activó USE_BODY_INDEX,
        si no, (None, None).
        """
        if not self._use_body_index:
            return None, None
        return self._last_body_index, self._last_depth

    # ------------------------------------------------------------------
    def get_body_point_cloud(
        self,
        max_points: int = 25000,
        min_depth_m: float = 0.5,
        max_depth_m: float = 8.0,   # <- antes 5.0, ahora permite más distancia
    ) -> np.ndarray:
        """Construye una nube de puntos 3D del cuerpo usando Depth + BodyIndex.

        Parámetros
        ----------
        max_points : int
            Número máximo de puntos a devolver (submuestreo aleatorio si hay más).
        min_depth_m, max_depth_m : float
            Rango de profundidad en metros para filtrar la nube.

        Devuelve un array `(N, 3)` con coordenadas `[x, y, z]` en metros.
        """
        if not self._use_body_index:
            return np.zeros((0, 3), dtype=np.float32)

        if self._last_depth is None or self._last_body_index is None:
            return np.zeros((0, 3), dtype=np.float32)

        # Depth: mm -> m
        depth = self._last_depth.astype(np.float32) / 1000.0
        body_index = self._last_body_index.astype(np.uint8)

        # 255 = fondo, 0..5 = cuerpos
        mask = (body_index != 255) & (depth > 0)
        mask &= (depth >= min_depth_m) & (depth <= max_depth_m)

        ys, xs = np.where(mask)
        n = xs.size
        if n == 0:
            return np.zeros((0, 3), dtype=np.float32)

        # Submuestreo aleatorio si hay demasiados puntos
        if n > max_points:
            idx = np.random.choice(n, max_points, replace=False)
            xs = xs[idx]
            ys = ys[idx]

        d = depth[ys, xs]

        # Intrínsecos aproximados de la cámara de profundidad Kinect v2 (512x424).
        # Estos valores son aproximados y se mantienen explícitos aquí para
        # que sea fácil cambiarlos si se calibra la cámara.
        fx = 364.815
        fy = 364.815
        cx = 256.972
        cy = 205.54

        # Coordenadas de cámara:
        #  - eje X: derecha
        #  - eje Y: arriba (por eso el signo menos en la fila)
        #  - eje Z: hacia adelante desde la Kinect
        x_cam = (xs - cx) * d / fx
        y_cam = -(ys - cy) * d / fy
        z_cam = d

        pts = np.stack([x_cam, y_cam, z_cam], axis=1).astype(np.float32)
        return pts

    # ------------------------------------------------------------------
    def close(self):
        # PyKinectRuntime no tiene close explícito, pero dejamos el método
        # por simetría y para futura extensibilidad.
        pass
