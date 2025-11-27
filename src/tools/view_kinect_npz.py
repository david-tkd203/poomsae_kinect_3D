# src/tools/view_kinect_npz.py
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
from PyQt5 import QtWidgets, QtCore

try:
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
except Exception as e:
    print("ERROR: Necesitas pyqtgraph con soporte OpenGL:")
    print("  pip install pyqtgraph PyOpenGL")
    raise e


class NPZ3DViewer(QtWidgets.QWidget):
    """
    Visor sencillo para archivos .npz generados por Kinect3DRecorder.

    Espera que el .npz tenga:
      - cloud_frames: array (N,) dtype=object, cada elemento (Mi, 3)
      - joint_frames: array (N,) dtype=object, cada elemento dict[int, (3,)]
      - timestamps : array (N,)
    """
    def __init__(self, npz_path: Path, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle(f"NPZ 3D Viewer - {npz_path.name}")

        # ----------------- cargar datos -----------------
        data = np.load(npz_path, allow_pickle=True)
        self.cloud_frames = data.get("cloud_frames", None)
        self.joint_frames = data.get("joint_frames", None)
        self.timestamps = data.get("timestamps", None)

        if self.cloud_frames is None or self.joint_frames is None:
            raise RuntimeError("El NPZ no contiene 'cloud_frames' o 'joint_frames'.")

        self.n_frames = int(len(self.cloud_frames))

        # ----------------- escena 3D -----------------
        self.view = gl.GLViewWidget()
        self.view.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )
        self.view.opts["distance"] = 2.5  # alejar un poco la cámara
        self.view.setBackgroundColor("k")  # fondo negro

        # rejilla de referencia
        grid = gl.GLGridItem()
        grid.setSize(2, 2, 0)
        grid.setSpacing(0.2, 0.2, 0.2)
        self.view.addItem(grid)

        # nube de puntos (cloud)
        self.cloud_item = gl.GLScatterPlotItem()
        self.cloud_item.setGLOptions("additive")  # blending simple
        self.view.addItem(self.cloud_item)

        # articulaciones (joints) – puntos más grandes
        self.joint_item = gl.GLScatterPlotItem()
        self.joint_item.setGLOptions("additive")
        self.view.addItem(self.joint_item)

        # ----------------- controles UI -----------------
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, max(0, self.n_frames - 1))
        self.slider.valueChanged.connect(self.set_frame)

        self.label_info = QtWidgets.QLabel()
        self.label_info.setText("Frame 0")

        # botones para activar/desactivar nubes / joints
        self.chk_cloud = QtWidgets.QCheckBox("Mostrar nube")
        self.chk_cloud.setChecked(True)
        self.chk_cloud.stateChanged.connect(self._update_visibility)

        self.chk_joints = QtWidgets.QCheckBox("Mostrar joints")
        self.chk_joints.setChecked(True)
        self.chk_joints.stateChanged.connect(self._update_visibility)

        # layout
        controls_layout = QtWidgets.QHBoxLayout()
        controls_layout.addWidget(self.slider)
        controls_layout.addWidget(self.chk_cloud)
        controls_layout.addWidget(self.chk_joints)

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.addWidget(self.view, stretch=1)
        main_layout.addLayout(controls_layout)
        main_layout.addWidget(self.label_info)

        # mostrar primer frame
        if self.n_frames > 0:
            self.set_frame(0)

    # -------------------------------------------------
    # Actualización de frame
    # -------------------------------------------------
    def set_frame(self, idx: int | None) -> None:
        if idx is None:
            return
        idx = int(idx)
        if idx < 0 or idx >= self.n_frames:
            return

        # ----- nube de puntos -----
        cloud = self.cloud_frames[idx]
        if cloud is None:
            cloud = np.zeros((0, 3), dtype=np.float32)
        cloud = np.asarray(cloud, dtype=np.float32)

        if cloud.size > 0:
            # opcional: reordenar ejes o escalar si lo necesitas
            # ej: invertir Z para que "arriba" sea +Z
            # cloud = cloud[:, [0, 2, 1]]
            self.cloud_item.setData(
                pos=cloud,
                size=1.0,
                pxMode=False,
            )
        else:
            # nube vacía → poner un punto invisible
            self.cloud_item.setData(
                pos=np.zeros((1, 3), dtype=np.float32),
                size=0.0,
                pxMode=False,
            )

        # ----- joints -----
        joints_obj: Any = self.joint_frames[idx]

        # En tu recorder guardas dict[int, np.ndarray], así que
        # deberíamos recibir directamente un dict. Pero si viene
        # envuelto en array(object), lo desempaquetamos.
        if isinstance(joints_obj, np.ndarray) and joints_obj.dtype == object and joints_obj.size == 1:
            joints = joints_obj.item()
        else:
            joints = joints_obj

        pts = None
        if isinstance(joints, dict) and joints:
            pts = np.vstack([np.asarray(p, dtype=np.float32) for p in joints.values()])

        if pts is not None and pts.size > 0:
            self.joint_item.setData(
                pos=pts,
                size=8.0,
                pxMode=False,
            )
        else:
            self.joint_item.setData(
                pos=np.zeros((1, 3), dtype=np.float32),
                size=0.0,
                pxMode=False,
            )

        # ----- texto info -----
        t = 0.0
        if self.timestamps is not None and len(self.timestamps) > idx:
            t = float(self.timestamps[idx])

        self.label_info.setText(
            f"Frame {idx+1}/{self.n_frames}  |  t = {t:.3f} s"
        )

        # aplicar visibilidad según checkboxes
        self._update_visibility()

    # -------------------------------------------------
    # Visibilidad de nubes / joints
    # -------------------------------------------------
    def _update_visibility(self) -> None:
        self.cloud_item.setVisible(self.chk_cloud.isChecked())
        self.joint_item.setVisible(self.chk_joints.isChecked())


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        print("Uso: python -m src.tools.view_kinect_npz RUTA/AL/ARCHIVO.npz")
        sys.exit(1)

    npz_path = Path(argv[0])
    if not npz_path.exists():
        print(f"ERROR: no se encontró el archivo: {npz_path}")
        sys.exit(1)

    app = QtWidgets.QApplication(sys.argv)
    viewer = NPZ3DViewer(npz_path)
    viewer.resize(900, 700)
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
