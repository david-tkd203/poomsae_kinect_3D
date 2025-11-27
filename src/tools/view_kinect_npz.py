# src/tools/view_kinect_npz.py
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
from PyQt5 import QtWidgets, QtCore

import pyqtgraph as pg
import pyqtgraph.opengl as gl


# ------------------------------------------------------------
# Conectividad de esqueleto Kinect v2 (joint indices estándar)
# ------------------------------------------------------------
KINECT_EDGES = [
    # Torso
    (0, 1),   # SpineBase    - SpineMid
    (1, 20),  # SpineMid     - SpineShoulder
    (20, 3),  # SpineShoulder- Head
    (0, 12),  # SpineBase    - HipLeft
    (0, 16),  # SpineBase    - HipRight,

    # Brazo izquierdo
    (20, 4),  # SpineShoulder - ShoulderLeft
    (4, 5),   # ShoulderLeft  - ElbowLeft
    (5, 6),   # ElbowLeft     - WristLeft
    (6, 7),   # WristLeft     - HandLeft
    (7, 21),  # HandLeft      - HandTipLeft
    (6, 22),  # WristLeft     - ThumbLeft

    # Brazo derecho
    (20, 8),  # SpineShoulder - ShoulderRight
    (8, 9),   # ShoulderRight - ElbowRight
    (9, 10),  # ElbowRight    - WristRight
    (10, 11), # WristRight    - HandRight
    (11, 23), # HandRight     - HandTipRight
    (10, 24), # WristRight    - ThumbRight

    # Pierna izquierda
    (12, 13), # HipLeft  - KneeLeft
    (13, 14), # KneeLeft - AnkleLeft
    (14, 15), # AnkleLeft- FootLeft

    # Pierna derecha
    (16, 17), # HipRight  - KneeRight
    (17, 18), # KneeRight - AnkleRight
    (18, 19), # AnkleRight- FootRight
]


# ------------------------------------------------------------
# Conversión de coordenadas Kinect -> PyQtGraph
# ------------------------------------------------------------
def _kinect_to_pg_coords(points: np.ndarray) -> np.ndarray:
    """Convertir coordenadas Kinect (X,Y,Z) a coordenadas usadas por PyQtGraph.

    Parameters
    ----------
    points : np.ndarray
        Array con forma (N,3) en sistema de cámara Kinect (metros).

    Returns
    -------
    np.ndarray
        Array (N,3) transformado para PyQtGraph.
    """
    if points is None or points.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    pts = points.astype(np.float32)

    # Desempaquetado explícito por claridad
    X = pts[:, 0]
    Y = pts[:, 1]
    Z = pts[:, 2]

    # Mapear ejes Kinect -> PyQtGraph
    x_pg = -X      # espejo left/right
    y_pg = Z       # profundidad hacia adelante
    z_pg = Y       # altura

    out = np.stack([x_pg, y_pg, z_pg], axis=1)
    return out


class NpzKinectViewer(QtWidgets.QMainWindow):
    def __init__(self, npz_path: Path, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle(f"Kinect NPZ Viewer - {npz_path.name}")

        self.npz_path = npz_path
        # Cargar contenido del npz y normalizar estructuras a listas
        raw = np.load(str(npz_path), allow_pickle=True)

        self.cloud_frames: List[Any] = list(raw.get("cloud_frames", []))
        self.joint_frames: List[Any] = list(raw.get("joint_frames", []))
        self.timestamps: List[float] = list(raw.get("timestamps", []))

        if not self.cloud_frames or not self.joint_frames:
            raise RuntimeError("El archivo NPZ no contiene 'cloud_frames' o 'joint_frames'.")

        self.num_frames = int(len(self.cloud_frames))
        self.frame_idx = 0

        # ---- Velocidad de reproducción basada en timestamps ----
        self.playback_speed = 1.0  # 1.0 = tiempo real
        self.base_interval_ms = self._estimate_base_interval(self.timestamps)
        self.current_interval_ms = int(self.base_interval_ms / self.playback_speed)

        # ---- Widget principal OpenGL ----
        self.view = gl.GLViewWidget()
        # Vista más frontal y un poco cercana
        self.view.setCameraPosition(distance=2.5, elevation=15, azimuth=0)
        self.view.setBackgroundColor("k")

        # Grid de piso más denso (más recuadros)
        self.grid = gl.GLGridItem()
        self.grid.setSize(4, 4)        # tamaño total
        self.grid.setSpacing(0.25, 0.25)  # recuadros más pequeños
        self.view.addItem(self.grid)

        # Eje XYZ
        self.axis = gl.GLAxisItem()
        self.axis.setSize(1, 1, 1)
        self.view.addItem(self.axis)

        # Nube de puntos (silueta corporal) con más calidad
        self.cloud_item = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3)),
            size=4.0,        # puntos más grandes
            pxMode=True,
        )
        self.view.addItem(self.cloud_item)

        # Esqueleto
        self.skeleton_item = gl.GLLinePlotItem(
            pos=np.zeros((2, 3)),
            mode="lines",
            width=2.0,
            antialias=True,
        )
        self.view.addItem(self.skeleton_item)

        # Rectángulo de base (en el piso bajo los pies)
        self.base_item = gl.GLLinePlotItem(
            pos=np.zeros((5, 3)),   # 4 vértices + cierre
            mode="line_strip",
            width=2.0,
            antialias=True,
        )
        self.view.addItem(self.base_item)

        # Layout Qt
        central = QtWidgets.QWidget()
        vlayout = QtWidgets.QVBoxLayout(central)
        vlayout.addWidget(self.view)
        self.setCentralWidget(central)

        # Barra de estado
        self.status = QtWidgets.QStatusBar()
        self.setStatusBar(self.status)

        # Timer de animación
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._next_frame)
        self.timer.start(self.current_interval_ms)

        # Atajos de teclado
        QtWidgets.QShortcut(QtCore.Qt.Key_Space, self, activated=self.toggle_play)
        QtWidgets.QShortcut(QtCore.Qt.Key_Right, self, activated=self.step_forward)
        QtWidgets.QShortcut(QtCore.Qt.Key_Left, self, activated=self.step_backward)

        self.playing = True
        self._update_frame(0)

    # ------------------------------------------------------------------
    def _estimate_base_interval(self, timestamps: Optional[List[float]]) -> int:
        """Estimar intervalo base (ms) a partir de timestamps.

        Si no hay timestamps válidos, devolvemos ~33ms (30fps).
        """
        fallback = 33
        if not timestamps or len(timestamps) < 2:
            return fallback

        ts = np.asarray(timestamps, dtype=float)
        diffs = np.diff(ts)
        diffs = diffs[np.isfinite(diffs)]
        # Filtrar intervalos inusuales
        diffs = diffs[(diffs > 1e-3) & (diffs < 1.0)]
        if diffs.size == 0:
            return fallback

        mean_dt = float(np.mean(diffs))
        ms = int(round(mean_dt * 1000.0))
        return max(10, min(200, ms))

    # ------------------------------------------------------------------
    # Control de reproducción
    # ------------------------------------------------------------------
    def _update_timer_interval(self) -> None:
        self.current_interval_ms = int(self.base_interval_ms / max(self.playback_speed, 1e-3))
        self.timer.setInterval(self.current_interval_ms)

    def toggle_play(self) -> None:
        self.playing = not self.playing

    def step_forward(self) -> None:
        self.playing = False
        self._next_frame()

    def step_backward(self) -> None:
        self.playing = False
        self.frame_idx = (self.frame_idx - 1) % self.num_frames
        self._update_frame(self.frame_idx)

    def _next_frame(self) -> None:
        if not self.playing:
            return
        self.frame_idx = (self.frame_idx + 1) % self.num_frames
        self._update_frame(self.frame_idx)

    # ------------------------------------------------------------------
    # Actualización de escena
    # ------------------------------------------------------------------
    def _update_frame(self, idx: int) -> None:
        """Actualizar visualización para el frame `idx`.

        Este método delega en helpers más pequeños para mantener cada
        sección fácil de leer y depurar.
        """
        pts_pg = self._get_cloud_points_pg(idx)
        self.cloud_item.setData(pos=pts_pg, size=4.0)

        joints_dict = self._get_joints_dict(idx)
        all_lines = self._build_skeleton_lines(joints_dict)
        self.skeleton_item.setData(pos=all_lines)

        base_pos = self._compute_base_rect(joints_dict)
        self.base_item.setData(pos=base_pos)

        # Estado y timestamp
        t = 0.0
        if idx < len(self.timestamps):
            try:
                t = float(self.timestamps[idx])
            except Exception:
                t = 0.0

        self.status.showMessage(
            f"Frame {idx+1}/{self.num_frames}  |  t = {t:.3f} s  |  Δt ≈ {self.base_interval_ms} ms"
        )

    def _get_cloud_points_pg(self, idx: int) -> np.ndarray:
        """Obtener la nube de puntos transformada a coordenadas PyQtGraph."""
        raw = self.cloud_frames[idx]
        if isinstance(raw, np.ndarray):
            pts = raw
        else:
            pts = np.asarray(raw, dtype=np.float32)

        if pts is None or pts.size == 0:
            return np.zeros((1, 3), dtype=np.float32)

        return _kinect_to_pg_coords(pts.astype(np.float32))

    def _get_joints_dict(self, idx: int) -> Dict[int, np.ndarray]:
        """Normalizar la estructura de `joint_frames[idx]` a un dict de int->np.ndarray."""
        raw = self.joint_frames[idx]
        if isinstance(raw, dict):
            return raw
        if hasattr(raw, "item"):
            maybe = raw.item()
            if isinstance(maybe, dict):
                return maybe
        return dict(raw) if raw is not None else {}

    def _build_skeleton_lines(self, joints_dict: Dict[int, np.ndarray]) -> np.ndarray:
        """Construir array de vértices para dibujar las líneas del esqueleto."""
        segments: List[np.ndarray] = []
        for j0, j1 in KINECT_EDGES:
            p0 = joints_dict.get(j0)
            p1 = joints_dict.get(j1)
            if p0 is None or p1 is None:
                continue

            p0_arr = np.asarray(p0, dtype=np.float32).reshape(1, 3)
            p1_arr = np.asarray(p1, dtype=np.float32).reshape(1, 3)
            seg = np.vstack([p0_arr, p1_arr])
            seg_pg = _kinect_to_pg_coords(seg)
            segments.append(seg_pg)

        if not segments:
            return np.zeros((2, 3), dtype=np.float32)
        return np.vstack(segments)

    def _compute_base_rect(self, joints_dict: Dict[int, np.ndarray]) -> np.ndarray:
        """Calcular el rectángulo de referencia bajo los pies.

        Devuelve un array (5,3) con 4 vértices y cierre.
        """
        base_pos = np.zeros((5, 3), dtype=np.float32)

        pL = joints_dict.get(15)  # FootLeft
        pR = joints_dict.get(19)  # FootRight

        if pL is None or pR is None:
            # Fuera de vista
            base_pos[:] = np.array([[0, 0, -10]] * 5, dtype=np.float32)
            return base_pos

        feet = np.vstack([pL, pR]).astype(np.float32)
        feet_pg = _kinect_to_pg_coords(feet)
        fL_pg, fR_pg = feet_pg[0], feet_pg[1]

        center = 0.5 * (fL_pg + fR_pg)

        # Vector entre pies en el plano X-Y (piso)
        v = fR_pg - fL_pg
        v[2] = 0.0
        norm_v = np.linalg.norm(v[:2]) + 1e-6
        dir_w = v / norm_v

        # Vector perpendicular en el piso para "profundidad" de la base
        dir_d = np.array([-dir_w[1], dir_w[0], 0.0], dtype=np.float32)

        half_w = 0.35  # ~70 cm de ancho
        half_d = 0.25  # ~50 cm de profundidad

        p1 = center + (-half_w * dir_w) + (-half_d * dir_d)
        p2 = center + ( half_w * dir_w) + (-half_d * dir_d)
        p3 = center + ( half_w * dir_w) + ( half_d * dir_d)
        p4 = center + (-half_w * dir_w) + ( half_d * dir_d)

        base_pos[0, :] = p1
        base_pos[1, :] = p2
        base_pos[2, :] = p3
        base_pos[3, :] = p4
        base_pos[4, :] = p1  # cierre

        return base_pos


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        print("Uso: python -m src.tools.view_kinect_npz ruta/al/archivo.npz")
        sys.exit(1)

    npz_path = Path(argv[0])
    if not npz_path.exists():
        print(f"NPZ no encontrado: {npz_path}")
        sys.exit(1)

    app = QtWidgets.QApplication(sys.argv)
    w = NpzKinectViewer(npz_path)
    w.resize(960, 720)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
