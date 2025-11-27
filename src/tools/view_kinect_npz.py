# src/tools/view_kinect_npz.py
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any

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


def _kinect_to_pg_coords(points: np.ndarray) -> np.ndarray:
    """
    Convierte coords Kinect (X right, Y up, Z forward) a coords PyQtGraph:
    - X -> X
    - Y -> Z
    - Z -> -Y

    Esto suele dar una vista 'natural' desde delante.
    """
    if points.size == 0:
        return points

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    return np.stack([x, z, -y], axis=1)


class NpzKinectViewer(QtWidgets.QMainWindow):
    def __init__(self, npz_path: Path, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle(f"Kinect NPZ Viewer - {npz_path.name}")

        self.npz_path = npz_path
        self.data = np.load(str(npz_path), allow_pickle=True)

        self.cloud_frames = self.data.get("cloud_frames", None)
        self.joint_frames = self.data.get("joint_frames", None)
        self.timestamps = self.data.get("timestamps", None)

        if self.cloud_frames is None or self.joint_frames is None:
            raise RuntimeError("El archivo NPZ no contiene 'cloud_frames' o 'joint_frames'.")

        self.num_frames = len(self.cloud_frames)
        self.frame_idx = 0

        # ---- Widget principal OpenGL ----
        self.view = gl.GLViewWidget()
        self.view.setCameraPosition(distance=3.0, elevation=20, azimuth=45)
        self.view.setBackgroundColor("k")

        # Ejes y grid para referencia
        self.grid = gl.GLGridItem()
        self.grid.setSize(3, 3)
        self.grid.setSpacing(0.5, 0.5)
        self.view.addItem(self.grid)

        # Eje XYZ
        self.axis = gl.GLAxisItem()
        self.axis.setSize(1, 1, 1)
        self.view.addItem(self.axis)

        # Nube de puntos (silueta corporal)
        self.cloud_item = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3)),
            size=2.0,        # tamaño de punto (subir si quieres verla más "densa")
            pxMode=True,
        )
        self.view.addItem(self.cloud_item)

        # Esqueleto: un line plot para todas las aristas
        self.skeleton_item = gl.GLLinePlotItem(
            pos=np.zeros((2, 3)),
            mode="lines",
            width=2.0,
            antialias=True,
        )
        self.view.addItem(self.skeleton_item)

        # Layout Qt
        central = QtWidgets.QWidget()
        vlayout = QtWidgets.QVBoxLayout(central)
        vlayout.addWidget(self.view)

        # Barra de estado para mostrar frame / tiempo
        self.status = QtWidgets.QStatusBar()
        self.setStatusBar(self.status)

        self.setCentralWidget(central)

        # Timer de animación
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._next_frame)
        self.timer.start(33)  # ~30 FPS de reproducción

        # Atajos de teclado
        QtWidgets.QShortcut(QtCore.Qt.Key_Space, self, activated=self.toggle_play)
        QtWidgets.QShortcut(QtCore.Qt.Key_Right, self, activated=self.step_forward)
        QtWidgets.QShortcut(QtCore.Qt.Key_Left, self, activated=self.step_backward)

        self.playing = True
        self._update_frame(0)

    # ------------------------------------------------------------------
    # Control de reproducción
    # ------------------------------------------------------------------
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
        # ---- Nube de puntos ----
        cloud = self.cloud_frames[idx]
        if isinstance(cloud, np.ndarray):
            pts = cloud
        else:
            pts = np.asarray(cloud, dtype=np.float32)

        if pts is None or pts.size == 0:
            pts_pg = np.zeros((1, 3), dtype=np.float32)
        else:
            pts_pg = _kinect_to_pg_coords(pts.astype(np.float32))

        # Ajustar tamaño si quieres ver más silueta
        self.cloud_item.setData(pos=pts_pg, size=2.0)

        # ---- Esqueleto ----
        joints_obj = self.joint_frames[idx]
        if isinstance(joints_obj, dict):
            joints_dict: Dict[int, np.ndarray] = joints_obj
        else:
            # Al cargar np.array(dtype=object), cada elemento puede ser el dict directamente
            joints_dict = joints_obj.item() if hasattr(joints_obj, "item") else joints_obj

        # Creamos un array de puntos conectados según KINECT_EDGES
        lines_pts = []
        for j0, j1 in KINECT_EDGES:
            p0 = joints_dict.get(j0, None)
            p1 = joints_dict.get(j1, None)
            if p0 is None or p1 is None:
                continue
            p0 = np.asarray(p0, dtype=np.float32).reshape(1, 3)
            p1 = np.asarray(p1, dtype=np.float32).reshape(1, 3)
            seg = np.vstack([p0, p1])
            seg_pg = _kinect_to_pg_coords(seg)
            lines_pts.append(seg_pg)

        if lines_pts:
            all_lines = np.vstack(lines_pts)
        else:
            all_lines = np.zeros((2, 3), dtype=np.float32)

        self.skeleton_item.setData(pos=all_lines)

        # ---- Estado ----
        t = float(self.timestamps[idx]) if self.timestamps is not None else 0.0
        self.status.showMessage(f"Frame {idx+1}/{self.num_frames}  |  t = {t:.3f} s")


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
