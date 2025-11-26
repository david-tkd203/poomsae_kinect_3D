# src/viz/kinect_3d_window.py
from __future__ import annotations

from typing import Optional
import numpy as np

from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from pykinect2 import PyKinectV2

from ..kinect import KinectCapture
from ..pose import MediaPipePoseEstimator
from ..config import USE_MEDIAPIPE, USE_BODY_INDEX, POSE_MODEL_COMPLEXITY, TARGET_FPS

# ---------------------------------------------------------------------
# Definición de "huesos" del esqueleto Kinect
# ---------------------------------------------------------------------
BONE_PAIRS = [
    # Columna
    (PyKinectV2.JointType_Head,           PyKinectV2.JointType_Neck),
    (PyKinectV2.JointType_Neck,           PyKinectV2.JointType_SpineShoulder),
    (PyKinectV2.JointType_SpineShoulder,  PyKinectV2.JointType_SpineMid),
    (PyKinectV2.JointType_SpineMid,       PyKinectV2.JointType_SpineBase),

    # Hombros
    (PyKinectV2.JointType_SpineShoulder,  PyKinectV2.JointType_ShoulderRight),
    (PyKinectV2.JointType_SpineShoulder,  PyKinectV2.JointType_ShoulderLeft),

    # Caderas
    (PyKinectV2.JointType_SpineBase,      PyKinectV2.JointType_HipRight),
    (PyKinectV2.JointType_SpineBase,      PyKinectV2.JointType_HipLeft),

    # Brazo derecho
    (PyKinectV2.JointType_ShoulderRight,  PyKinectV2.JointType_ElbowRight),
    (PyKinectV2.JointType_ElbowRight,     PyKinectV2.JointType_WristRight),
    (PyKinectV2.JointType_WristRight,     PyKinectV2.JointType_HandRight),
    (PyKinectV2.JointType_HandRight,      PyKinectV2.JointType_HandTipRight),
    (PyKinectV2.JointType_WristRight,     PyKinectV2.JointType_ThumbRight),

    # Brazo izquierdo
    (PyKinectV2.JointType_ShoulderLeft,   PyKinectV2.JointType_ElbowLeft),
    (PyKinectV2.JointType_ElbowLeft,      PyKinectV2.JointType_WristLeft),
    (PyKinectV2.JointType_WristLeft,      PyKinectV2.JointType_HandLeft),
    (PyKinectV2.JointType_HandLeft,       PyKinectV2.JointType_HandTipLeft),
    (PyKinectV2.JointType_WristLeft,      PyKinectV2.JointType_ThumbLeft),

    # Pierna derecha
    (PyKinectV2.JointType_HipRight,       PyKinectV2.JointType_KneeRight),
    (PyKinectV2.JointType_KneeRight,      PyKinectV2.JointType_AnkleRight),
    (PyKinectV2.JointType_AnkleRight,     PyKinectV2.JointType_FootRight),

    # Pierna izquierda
    (PyKinectV2.JointType_HipLeft,        PyKinectV2.JointType_KneeLeft),
    (PyKinectV2.JointType_KneeLeft,       PyKinectV2.JointType_AnkleLeft),
    (PyKinectV2.JointType_AnkleLeft,      PyKinectV2.JointType_FootLeft),
]

# Escala base del cuerpo en el visor
SKELETON_SCALE = 1.5


class Kinect3DWindow(QtWidgets.QWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)

        self.setWindowTitle("Kinect 3D + MediaPipe Pose Viewer")

        # Backend Kinect
        self.kinect = KinectCapture(use_body_index=USE_BODY_INDEX)

        # Backend MediaPipe (opcional)
        self.mp_pose = None
        if USE_MEDIAPIPE:
            self.mp_pose = MediaPipePoseEstimator(
                model_complexity=POSE_MODEL_COMPLEXITY
            )

        # Flags de visibilidad
        self.show_cloud = USE_BODY_INDEX
        self.show_mediapipe = USE_MEDIAPIPE
        self.show_skeleton = True
        self.show_bones = True

        # Parámetros de escala / tamaño
        self.skeleton_scale = SKELETON_SCALE
        self.cloud_point_size = 1.5

        # ---------------- Vista 3D (pyqtgraph.opengl) ----------------
        self.view = gl.GLViewWidget()
        self.view.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )
        # Fondo oscuro para más contraste
        self.view.setBackgroundColor(10, 10, 20, 255)

        # Cámara inicial: un poco elevada y girada para que se perciba el 3D
        self.view.setCameraPosition(
            distance=3.0,   # qué tan lejos está la cámara
            elevation=20,   # ángulo sobre el plano X-Y
            azimuth=45      # giro alrededor del eje Z
        )

        # Ejes de referencia
        axis = gl.GLAxisItem()
        axis.setSize(0.5, 0.5, 0.5)
        self.view.addItem(axis)

        # Piso / grilla (plano X-Y en Z=0)
        grid = gl.GLGridItem()
        grid.setSize(2, 2)
        grid.setSpacing(0.2, 0.2)
        grid.translate(0, 0, 0)  # plano z=0 (altura 0)
        self.view.addItem(grid)

        # Puntos de articulaciones Kinect (blancos)
        self.skel_item = gl.GLScatterPlotItem(
            pos=np.zeros((0, 3), dtype=np.float32),
            size=9,
            pxMode=True,
            color=(1.0, 1.0, 1.0, 1.0),  # blanco
        )
        self.view.addItem(self.skel_item)

        # Líneas ("huesos") entre articulaciones (amarillo suave)
        self.bone_pairs = BONE_PAIRS
        self.bone_items = []
        for _ in self.bone_pairs:
            line = gl.GLLinePlotItem(
                pos=np.zeros((2, 3), dtype=np.float32),
                width=2,
                antialias=True,
                color=(1.0, 1.0, 0.3, 1.0),  # amarillo
            )
            self.view.addItem(line)
            self.bone_items.append(line)

        # Nube de puntos del cuerpo (BodyIndex + Depth) - celeste translúcido
        self.body_cloud_item = gl.GLScatterPlotItem(
            pos=np.zeros((0, 3), dtype=np.float32),
            size=self.cloud_point_size,
            pxMode=True,
            color=(0.0, 0.6, 1.0, 0.25),  # azul claro translúcido
        )
        if USE_BODY_INDEX:
            self.view.addItem(self.body_cloud_item)

        # Puntos 3D de MediaPipe (world landmarks) - magenta
        self.mp_item = gl.GLScatterPlotItem(
            pos=np.zeros((0, 3), dtype=np.float32),
            size=4,
            pxMode=True,
            color=(1.0, 0.0, 1.0, 1.0),  # magenta
        )
        if USE_MEDIAPIPE:
            self.view.addItem(self.mp_item)

        # Root del esqueleto (SpineBase en coords de cámara)
        self._last_root: Optional[np.ndarray] = None

        # Construir panel de controles
        self._build_ui()

        # Timer a ~TARGET_FPS
        interval_ms = max(1, int(1000 / max(1, TARGET_FPS)))
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._on_timer)
        self.timer.start(interval_ms)

    # ------------------------------------------------------------------
    # UI lateral con sliders y checkboxes
    # ------------------------------------------------------------------
    def _build_ui(self):
        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.addWidget(self.view, 4)

        panel = QtWidgets.QWidget()
        panel_layout = QtWidgets.QVBoxLayout(panel)

        # --- Grupo: Cámara 3D ---
        group_cam = QtWidgets.QGroupBox("Cámara 3D")
        cam_layout = QtWidgets.QFormLayout(group_cam)

        self.slider_cam_dist = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_cam_dist.setRange(10, 80)   # 1.0 - 8.0
        self.slider_cam_dist.setValue(30)       # 3.0
        self.slider_cam_dist.valueChanged.connect(self._on_cam_slider_changed)

        self.slider_cam_elev = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_cam_elev.setRange(-10, 60)
        self.slider_cam_elev.setValue(20)
        self.slider_cam_elev.valueChanged.connect(self._on_cam_slider_changed)

        self.slider_cam_azim = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_cam_azim.setRange(0, 360)
        self.slider_cam_azim.setValue(45)
        self.slider_cam_azim.valueChanged.connect(self._on_cam_slider_changed)

        cam_layout.addRow("Distancia", self.slider_cam_dist)
        cam_layout.addRow("Elevación", self.slider_cam_elev)
        cam_layout.addRow("Azimut", self.slider_cam_azim)

        self.btn_cam_iso = QtWidgets.QPushButton("Vista isométrica")
        self.btn_cam_iso.clicked.connect(self._set_iso_view)
        self.btn_cam_front = QtWidgets.QPushButton("Vista frontal")
        self.btn_cam_front.clicked.connect(self._set_front_view)
        cam_layout.addRow(self.btn_cam_iso)
        cam_layout.addRow(self.btn_cam_front)

        panel_layout.addWidget(group_cam)

        # --- Grupo: Cuerpo / escala ---
        group_body = QtWidgets.QGroupBox("Cuerpo / escala")
        body_layout = QtWidgets.QFormLayout(group_body)

        self.slider_scale = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_scale.setRange(50, 250)  # 0.5 - 2.5
        self.slider_scale.setValue(int(self.skeleton_scale * 100))
        self.slider_scale.valueChanged.connect(self._on_scale_slider_changed)

        self.slider_cloud_size = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_cloud_size.setRange(1, 10)
        self.slider_cloud_size.setValue(int(self.cloud_point_size))
        self.slider_cloud_size.valueChanged.connect(self._on_cloud_size_changed)

        body_layout.addRow("Escala esqueleto", self.slider_scale)
        body_layout.addRow("Tamaño nube", self.slider_cloud_size)

        panel_layout.addWidget(group_body)

        # --- Grupo: Capas visibles ---
        group_layers = QtWidgets.QGroupBox("Capas visibles")
        layers_layout = QtWidgets.QVBoxLayout(group_layers)

        self.chk_skel = QtWidgets.QCheckBox("Esqueleto Kinect")
        self.chk_skel.setChecked(True)
        self.chk_skel.toggled.connect(self._on_layers_changed)

        self.chk_bones = QtWidgets.QCheckBox("Huesos")
        self.chk_bones.setChecked(True)
        self.chk_bones.toggled.connect(self._on_layers_changed)

        self.chk_cloud = QtWidgets.QCheckBox("Nube cuerpo (BodyIndex)")
        self.chk_cloud.setChecked(self.show_cloud)
        self.chk_cloud.toggled.connect(self._on_layers_changed)

        self.chk_mp = QtWidgets.QCheckBox("MediaPipe 3D")
        self.chk_mp.setChecked(self.show_mediapipe)
        self.chk_mp.toggled.connect(self._on_layers_changed)

        layers_layout.addWidget(self.chk_skel)
        layers_layout.addWidget(self.chk_bones)
        layers_layout.addWidget(self.chk_cloud)
        layers_layout.addWidget(self.chk_mp)

        panel_layout.addWidget(group_layers)

        # Info de controles
        info_label = QtWidgets.QLabel(
            "Mouse: rotar/zoom\nR: reset cámara · C: nube · M: MediaPipe"
        )
        info_label.setStyleSheet("color: #dddddd; font-size: 11px;")
        panel_layout.addWidget(info_label)

        panel_layout.addStretch(1)

        main_layout.addWidget(panel, 1)

    # ------------------------------------------------------------------
    # Mapeos de coordenadas Kinect -> visor 3D
    # ------------------------------------------------------------------
    def _map_to_gl_coords_single(self, p: np.ndarray, root: np.ndarray) -> np.ndarray:
        """
        p, root en coords de cámara Kinect (m):
        - centramos en root
        - escalamos
        - reordenamos ejes para el visor:
            X_gl = X_cam
            Y_gl = -Z_cam   (profundidad hacia atrás)
            Z_gl = Y_cam    (altura)
        """
        rel = (p - root) * self.skeleton_scale
        x, y, z = rel
        return np.array([x, -z, y], dtype=np.float32)

    def _map_to_gl_coords_cloud(self, pts_cam: np.ndarray, root: np.ndarray) -> np.ndarray:
        """
        Versión vectorizada para nubes de puntos.
        """
        rel = (pts_cam - root) * self.skeleton_scale
        x = rel[:, 0]
        y = rel[:, 1]
        z = rel[:, 2]
        gl_pts = np.stack([x, -z, y], axis=1)
        return gl_pts.astype(np.float32)

    # ------------------------------------------------------------------
    # Lógica de los sliders / checkboxes
    # ------------------------------------------------------------------
    def _on_cam_slider_changed(self):
        dist = self.slider_cam_dist.value() / 10.0
        elev = float(self.slider_cam_elev.value())
        azim = float(self.slider_cam_azim.value())
        self.view.setCameraPosition(distance=dist, elevation=elev, azimuth=azim)

    def _set_iso_view(self):
        """Vista isométrica (3/4) – buena sensación de 3D."""
        self.slider_cam_dist.setValue(30)
        self.slider_cam_elev.setValue(20)
        self.slider_cam_azim.setValue(45)
        self._on_cam_slider_changed()

    def _set_front_view(self):
        """Vista frontal – para ‘enderezar’ la imagen frente al usuario."""
        self.slider_cam_dist.setValue(25)
        self.slider_cam_elev.setValue(10)
        self.slider_cam_azim.setValue(0)
        self._on_cam_slider_changed()

    def _on_scale_slider_changed(self):
        self.skeleton_scale = self.slider_scale.value() / 100.0

    def _on_cloud_size_changed(self):
        self.cloud_point_size = float(self.slider_cloud_size.value())

    def _on_layers_changed(self):
        self.show_skeleton = self.chk_skel.isChecked()
        self.show_bones = self.chk_bones.isChecked()
        self.show_cloud = self.chk_cloud.isChecked()
        self.show_mediapipe = self.chk_mp.isChecked()

    # ------------------------------------------------------------------
    # Actualización de esqueleto Kinect
    # ------------------------------------------------------------------
    def _update_skeleton(self):
        joint_positions = self.kinect.get_joint_positions()

        if not joint_positions:
            self._last_root = None
            self.skel_item.setData(pos=np.zeros((0, 3), dtype=np.float32))
            for line in self.bone_items:
                line.setData(pos=np.zeros((2, 3), dtype=np.float32))
            return

        # Root del cuerpo: SpineBase si existe, si no, el primer joint que haya
        root = joint_positions.get(
            PyKinectV2.JointType_SpineBase,
            next(iter(joint_positions.values()))
        )
        self._last_root = root.copy()

        # 1) Scatter de articulaciones
        pts = []
        for _, p in joint_positions.items():
            pts.append(self._map_to_gl_coords_single(p, root))
        pts = np.asarray(pts, dtype=np.float32)

        if self.show_skeleton:
            self.skel_item.setData(pos=pts)
        else:
            self.skel_item.setData(pos=np.zeros((0, 3), dtype=np.float32))

        # 2) Líneas de los huesos
        for idx, (j0, j1) in enumerate(self.bone_pairs):
            if j0 in joint_positions and j1 in joint_positions and self.show_bones:
                p0 = self._map_to_gl_coords_single(joint_positions[j0], root)
                p1 = self._map_to_gl_coords_single(joint_positions[j1], root)
                line_pts = np.vstack([p0, p1])
            else:
                line_pts = np.zeros((2, 3), dtype=np.float32)

            self.bone_items[idx].setData(pos=line_pts)

    # ------------------------------------------------------------------
    # Actualización nube BodyIndex + Depth
    # ------------------------------------------------------------------
    def _update_body_cloud(self):
        """Actualiza la nube de puntos del cuerpo usando BodyIndex+Depth."""
        if not USE_BODY_INDEX or not self.show_cloud:
            self.body_cloud_item.setData(pos=np.zeros((0, 3), dtype=np.float32))
            return

        cloud_cam = self.kinect.get_body_point_cloud(max_points=15000)
        if cloud_cam.shape[0] == 0:
            self.body_cloud_item.setData(pos=np.zeros((0, 3), dtype=np.float32))
            return

        if self._last_root is not None:
            root = self._last_root
        else:
            # Si no hay esqueleto (ej: cuerpo de espaldas fuera del tracking),
            # centramos en el centro de masa de la nube.
            root = cloud_cam.mean(axis=0)

        gl_pts = self._map_to_gl_coords_cloud(cloud_cam, root)
        self.body_cloud_item.setData(pos=gl_pts, size=self.cloud_point_size)

    # ------------------------------------------------------------------
    # Actualización puntos 3D MediaPipe
    # ------------------------------------------------------------------
    def _update_mediapipe(self):
        """Actualiza los puntos 3D de MediaPipe (world landmarks)."""
        if not USE_MEDIAPIPE or self.mp_pose is None or not self.show_mediapipe:
            self.mp_item.setData(pos=np.zeros((0, 3), dtype=np.float32))
            return

        color_bgr = self.kinect.get_color_bgr()
        if color_bgr is None:
            self.mp_item.setData(pos=np.zeros((0, 3), dtype=np.float32))
            return

        _, mp_world = self.mp_pose.process(color_bgr)
        if mp_world.shape[0] == 0:
            self.mp_item.setData(pos=np.zeros((0, 3), dtype=np.float32))
            return

        # mp_world ya viene en metros y centrado en el cuerpo.
        scale = self.skeleton_scale
        mp_pts = mp_world * scale
        self.mp_item.setData(pos=mp_pts)

    # ------------------------------------------------------------------
    # Timer: ciclo principal
    # ------------------------------------------------------------------
    def _on_timer(self):
        """Callback del QTimer: actualiza todo el pipeline frame a frame."""
        # 1) Actualizar datos Kinect
        self.kinect.update_frames()

        # 2) Esqueleto
        self._update_skeleton()

        # 3) Nube de puntos del cuerpo
        self._update_body_cloud()

        # 4) MediaPipe
        self._update_mediapipe()

    # ------------------------------------------------------------------
    # Controles de teclado (R, C, M)
    # ------------------------------------------------------------------
    def keyPressEvent(self, event):
        key = event.key()

        if key == QtCore.Qt.Key_R:
            # Reset a vista isométrica
            self._set_iso_view()
        elif key == QtCore.Qt.Key_C and USE_BODY_INDEX:
            # Toggle nube de puntos
            self.chk_cloud.toggle()
        elif key == QtCore.Qt.Key_M and USE_MEDIAPIPE:
            # Toggle MediaPipe
            self.chk_mp.toggle()

        super().keyPressEvent(event)

    # ------------------------------------------------------------------
    def closeEvent(self, event):
        if self.mp_pose is not None:
            self.mp_pose.close()
        self.kinect.close()
        super().closeEvent(event)
