# src/kinect/__init__.py
from __future__ import annotations

from typing import Dict, Optional
import numpy as np

from pykinect2 import PyKinectV2, PyKinectRuntime


class KinectCapture:
    """
    Wrapper sobre PyKinectRuntime para Kinect v2 (Xbox One).

    Provee:
      - Lectura de frames Color, Depth, Body, BodyIndex (opcional).
      - Joints 3D en coordenadas de cámara (metros).
      - Nube de puntos 3D del cuerpo usando Depth + BodyIndex,
        ignorando el entorno en lo posible.
    """

    def __init__(self, use_body_index: bool = True):
        self._use_body_index = bool(use_body_index)

        sources = (
            PyKinectV2.FrameSourceTypes_Color
            | PyKinectV2.FrameSourceTypes_Depth
            | PyKinectV2.FrameSourceTypes_Body
        )
        if self._use_body_index:
            sources |= PyKinectV2.FrameSourceTypes_BodyIndex

        self._kinect = PyKinectRuntime.PyKinectRuntime(sources)

        # Últimos frames cacheados
        self._body_frame = None

        # Depth en 2D (H, W) y versión cruda 1D (para el mapper)
        self._depth_frame: Optional[np.ndarray] = None
        self._depth_frame_raw: Optional[np.ndarray] = None

        # BodyIndex y Color
        self._body_index_frame: Optional[np.ndarray] = None
        self._color_bgra: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    # Actualizar frames desde el runtime
    # ------------------------------------------------------------------ #
    def update_frames(self) -> None:
        """Debe llamarse una vez por iteración del bucle principal."""

        # Body (joints)
        if self._kinect.has_new_body_frame():
            self._body_frame = self._kinect.get_last_body_frame()

        # Depth
        if self._kinect.has_new_depth_frame():
            df = self._kinect.get_last_depth_frame()  # 1D
            self._depth_frame_raw = df
            h = self._kinect.depth_frame_desc.Height
            w = self._kinect.depth_frame_desc.Width
            # reshape a (H, W) en uint16 (milímetros)
            self._depth_frame = df.reshape((h, w)).astype(np.uint16)

        # BodyIndex (mask de cuerpo vs entorno)
        if self._use_body_index and self._kinect.has_new_body_index_frame():
            bif = self._kinect.get_last_body_index_frame()
            h = self._kinect.body_index_frame_desc.Height
            w = self._kinect.body_index_frame_desc.Width
            self._body_index_frame = bif.reshape((h, w))

        # Color (BGRA)
        if self._kinect.has_new_color_frame():
            cf = self._kinect.get_last_color_frame()
            h = self._kinect.color_frame_desc.Height
            w = self._kinect.color_frame_desc.Width
            self._color_bgra = cf.reshape((h, w, 4)).astype(np.uint8)

    # ------------------------------------------------------------------ #
    # Joints 3D (coordenadas de cámara Kinect, en metros)
    # ------------------------------------------------------------------ #
    def get_joint_positions(self) -> Dict[int, np.ndarray]:
        """
        Devuelve un dict {joint_type: np.array([x,y,z])} en metros,
        en coordenadas de cámara Kinect.
        """
        joints: Dict[int, np.ndarray] = {}

        if self._body_frame is None:
            return joints

        # Tomamos el primer cuerpo rastreado
        for body in self._body_frame.bodies:
            if not body.is_tracked:
                continue

            for j_type, joint in enumerate(body.joints):
                if joint.TrackingState in (
                    PyKinectV2.TrackingState_Tracked,
                    PyKinectV2.TrackingState_Inferred,
                ):
                    pos = joint.Position
                    joints[j_type] = np.array(
                        [pos.x, pos.y, pos.z], dtype=np.float32
                    )
            break  # sólo el primer cuerpo rastreado

        return joints

    # ------------------------------------------------------------------ #
    # Nube de puntos 3D del cuerpo usando Depth + BodyIndex
    # ------------------------------------------------------------------ #
    """
        Devuelve una nube de puntos en coordenadas de cámara (X,Y,Z en metros)
        usando Depth + BodyIndex.

        - Si USE_BODY_INDEX=0 o no hay frames válidos, devuelve array (0,3).
        - No usa el mapper nativo de PyKinect2 para evitar errores de firma.
        - Usa intrínsecos aproximados de Kinect v2 para reconstruir el 3D,
          suficiente para visualización del cuerpo completo.
        """
    def get_body_point_cloud(self, max_points: int = 15000) -> np.ndarray:
        """
        Devuelve una nube de puntos en coordenadas de cámara (X,Y,Z en metros)
        usando Depth + BodyIndex.

        - Si USE_BODY_INDEX=0 o no hay frames válidos, devuelve array (0,3).
        - No usa el mapper nativo de PyKinect2 para evitar errores de firma.
        - Usa intrínsecos aproximados de Kinect v2 para reconstruir el 3D,
          suficiente para visualización del cuerpo completo.
        """
        # Si no activaste BodyIndex, no hay nube
        if not getattr(self, "use_body_index", False):
            return np.zeros((0, 3), dtype=np.float32)

        # Estos atributos los estamos actualizando en update_frames()
        depth_raw = getattr(self, "_last_depth_frame", None)
        body_raw = getattr(self, "_last_body_index_frame", None)

        if depth_raw is None or body_raw is None:
            return np.zeros((0, 3), dtype=np.float32)

        depth = np.asarray(depth_raw, dtype=np.float32)
        body = np.asarray(body_raw, dtype=np.uint8)

        # Kinect v2 depth estándar: 512x424.
        total = depth.size
        if depth.ndim == 1:
            if total == 512 * 424:
                w, h = 512, 424
            else:
                # Fallback genérico: asumimos ancho 512 y calculamos alto
                w = 512
                h = total // w if w > 0 else 0
                if h <= 0 or h * w != total:
                    return np.zeros((0, 3), dtype=np.float32)
            depth = depth.reshape((h, w))
        elif depth.ndim == 2:
            h, w = depth.shape
        else:
            return np.zeros((0, 3), dtype=np.float32)

        if body.ndim == 1:
            if body.size != depth.size:
                return np.zeros((0, 3), dtype=np.float32)
            body = body.reshape((h, w))
        elif body.shape != depth.shape:
            return np.zeros((0, 3), dtype=np.float32)

        # Máscara: píxeles etiquetados como cuerpo (BodyIndex != 255) y con depth > 0
        mask = (body != 255) & (depth > 0)
        if not np.any(mask):
            return np.zeros((0, 3), dtype=np.float32)

        ys, xs = np.where(mask)

        # Submuestreo opcional para no dibujar millones de puntos
        n = xs.size
        if n > max_points:
            idx = np.random.choice(n, size=max_points, replace=False)
            xs = xs[idx]
            ys = ys[idx]

        # Kinect v2 da profundidad en milímetros → pasamos a metros
        z_m = depth[ys, xs].astype(np.float32) / 1000.0

        # Intrínsecos aproximados de Kinect v2 para Depth 512x424
        # (suficiente para visualización)
        fx = 366.0
        fy = 366.0
        cx = w / 2.0
        cy = h / 2.0

        x_m = (xs - cx) * z_m / fx
        y_m = (ys - cy) * z_m / fy

        pts = np.stack([x_m, y_m, z_m], axis=1).astype(np.float32)
        return pts

    # ------------------------------------------------------------------ #
    # Color frame en BGR (para MediaPipe)
    # ------------------------------------------------------------------ #
    def get_color_bgr(self) -> Optional[np.ndarray]:
        """
        Devuelve el último frame de color en BGR (H,W,3) uint8,
        o None si aún no hay frame.
        """
        if self._color_bgra is None:
            return None
        # BGRA -> BGR
        return self._color_bgra[:, :, :3]

    # ------------------------------------------------------------------ #
    def close(self) -> None:
        self._kinect.close()
