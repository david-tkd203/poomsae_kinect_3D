"""Interfaz 3D para visualización y grabación con Kinect y MediaPipe.

Este módulo contiene la ventana principal utilizada para visualizar la
información 3D proveniente de Kinect (esqueleto, nube de puntos) junto
con landmarks de MediaPipe. Los cambios realizados aquí son principalmente
cosméticos: docstrings y comentarios que facilitan la lectura por parte
de otro desarrollador.
"""
from __future__ import annotations

from typing import Optional, Dict
from datetime import datetime
import os

import numpy as np
import cv2  # OpenCV para cámaras externas

from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from pykinect2 import PyKinectV2

from ..kinect.kinect_capture import KinectCapture
from ..pose import MediaPipePoseEstimator
from ..config import USE_MEDIAPIPE, USE_BODY_INDEX, POSE_MODEL_COMPLEXITY, TARGET_FPS

# Nuevos imports para grabación y pipeline offline
from ..recording.session_recorder import RecordingSession
from ..offline.offline_pipeline import run_offline_pipeline
from .report_window import ReportWindow

# ---------------------------------------------------------------------
# Parámetros de rendimiento
# ---------------------------------------------------------------------
MP_FRAME_STRIDE = 1         # Ejecutar MediaPipe cada N frames
CAM_FRAME_STRIDE = 1         # ← AHORA 1: actualizar previews / grabación en cada frame
MAX_CLOUD_POINTS = 8000      # Máx. puntos en nube de cuerpo

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

SKELETON_SCALE = 1.5


class Kinect3DWindow(QtWidgets.QWidget):
    """Ventana principal para visualización 3D y control de grabación.

    Provee controles para alternar capas (esqueleto, huesos, avatar,
    nube de puntos), ajustar la cámara y gestionar una sesión de
    grabación. Está pensada como una UI de desarrollo: los textos son
    sencillos y los comentarios buscan explicar decisiones de diseño.
    """
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)

        self.setWindowTitle("Kinect 3D + MediaPipe Pose Viewer")

        # Backend Kinect
        self.kinect = KinectCapture(use_body_index=USE_BODY_INDEX)

        # Backend MediaPipe (opcional, para visual en tiempo real)
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
        self.show_avatar = True

        # Parámetros de escala / tamaño
        self.skeleton_scale = SKELETON_SCALE
        self.cloud_point_size = 1.5

        # Cámaras externas (OpenCV)
        self.cam_caps: Dict[int, cv2.VideoCapture] = {}
        self.cam_enabled: Dict[int, bool] = {}
        self.cam_labels: Dict[int, QtWidgets.QLabel] = {}
        self.cam_checkboxes: Dict[int, QtWidgets.QCheckBox] = {}
        self.cam_detected_count: int = 0
        self._init_cameras()

        # Estado Kinect
        self.lbl_kinect_status: Optional[QtWidgets.QLabel] = None

        # Último esqueleto válido (para evitar parpadeos)
        self._last_root: Optional[np.ndarray] = None
        self._last_joints: Dict[int, np.ndarray] = {}
        self._last_tracked_bodies: int = 0

        # Estado de detección de cuerpo + label visual rojo/verde
        self.body_detected: bool = False
        self.lbl_body_state: Optional[QtWidgets.QLabel] = None

        # Grabación de sesión + controles
        self.recording_session: Optional[RecordingSession] = None
        self._last_capture_dir: Optional[str] = None
        self.btn_record_start: Optional[QtWidgets.QPushButton] = None
        self.btn_record_stop: Optional[QtWidgets.QPushButton] = None
        self.lbl_record_state: Optional[QtWidgets.QLabel] = None

        # Vista 3D
        self.view = gl.GLViewWidget()
        self.view.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )
        self.view.setBackgroundColor(10, 10, 20, 255)

        self.view.setCameraPosition(
            distance=3.0,
            elevation=20,
            azimuth=45
        )

        axis = gl.GLAxisItem()
        axis.setSize(0.5, 0.5, 0.5)
        self.view.addItem(axis)

        grid = gl.GLGridItem()
        grid.setSize(2, 2)
        grid.setSpacing(0.2, 0.2)
        grid.translate(0, 0, 0)
        self.view.addItem(grid)

        # Esqueleto de puntos
        self.skel_item = gl.GLScatterPlotItem(
            pos=np.zeros((0, 3), dtype=np.float32),
            size=9,
            pxMode=True,
            color=(1.0, 1.0, 1.0, 1.0),
        )
        self.view.addItem(self.skel_item)

        # Líneas de huesos
        self.bone_pairs = BONE_PAIRS
        self.bone_items = []
        for _ in self.bone_pairs:
            line = gl.GLLinePlotItem(
                pos=np.zeros((2, 3), dtype=np.float32),
                width=2,
                antialias=True,
                color=(1.0, 1.0, 0.3, 1.0),
            )
            self.view.addItem(line)
            self.bone_items.append(line)

        # Avatar 3D: cápsulas
        self._capsule_mesh = gl.MeshData.cylinder(
            rows=8, cols=16, radius=[1.0, 1.0], length=1.0
        )
        self.bone_mesh_items = []
        for _ in self.bone_pairs:
            mesh = gl.GLMeshItem(
                meshdata=self._capsule_mesh,
                smooth=True,
                shader="shaded",
                glOptions="opaque",
                color=(0.1, 0.7, 1.0, 0.45),
            )
            mesh.setVisible(self.show_avatar)
            self.view.addItem(mesh)
            self.bone_mesh_items.append(mesh)

        # Nube de puntos cuerpo
        self.body_cloud_item = gl.GLScatterPlotItem(
            pos=np.zeros((0, 3), dtype=np.float32),
            size=self.cloud_point_size,
            pxMode=True,
            color=(0.0, 0.6, 1.0, 0.25),
        )
        if USE_BODY_INDEX:
            self.view.addItem(self.body_cloud_item)

        # MediaPipe 3D
        self.mp_item = gl.GLScatterPlotItem(
            pos=np.zeros((0, 3), dtype=np.float32),
            size=4,
            pxMode=True,
            color=(1.0, 0.0, 1.0, 1.0),
        )
        if USE_MEDIAPIPE:
            self.view.addItem(self.mp_item)

        # Contador de frames para stride
        self._frame_counter: int = 0

        # Construir UI
        self._build_ui()

        # Timer
        interval_ms = max(1, int(1000 / max(1, TARGET_FPS)))
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._on_timer)
        self.timer.start(interval_ms)

    # ------------------------------------------------------------------
    # Detección de cámaras externas
    # ------------------------------------------------------------------
    def _init_cameras(self, max_index: int = 8, skip_indices=None):
        # Solo omitimos la webcam integrada típica del notebook (índice 0).
        # De esta forma se vuelven a detectar las cámaras externas en 1, 2, etc.
        if skip_indices is None:
            skip_indices = {}

        for idx in range(max_index):
            if idx in skip_indices:
                continue

            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if not cap or not cap.isOpened():
                if cap:
                    cap.release()
                continue

            # Intentar configurar cámaras externas como 2QHD @ 60fps.
            # Si el dispositivo no soporta esos valores, OpenCV usará
            # lo más cercano disponible.
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)   # 2QHD width
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)  # 2QHD height
            cap.set(cv2.CAP_PROP_FPS, 60)

            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                continue

            self.cam_caps[idx] = cap
            self.cam_enabled[idx] = False

        self.cam_detected_count = len(self.cam_caps)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        main_layout = QtWidgets.QHBoxLayout(self)

        # Columna izquierda: previews (SOLO cámaras externas)
        cams_panel = QtWidgets.QWidget()
        cams_layout = QtWidgets.QVBoxLayout(cams_panel)
        cams_layout.setContentsMargins(6, 6, 6, 6)
        cams_layout.setSpacing(8)

        title = QtWidgets.QLabel("Cámaras externas")
        title.setStyleSheet("color: #dddddd; font-weight: bold;")
        cams_layout.addWidget(title)

        # Info cámaras externas
        info_text = f"Cámaras externas detectadas: {self.cam_detected_count}"
        info_label = QtWidgets.QLabel(info_text)
        info_label.setStyleSheet("color: #bbbbbb; font-size: 10px;")
        cams_layout.addWidget(info_label)

        if self.cam_detected_count == 0:
            no_cam = QtWidgets.QLabel(
                "No se detectaron cámaras externas.\n"
                "(Se omite índice 0 del notebook)"
            )
            no_cam.setWordWrap(True)
            no_cam.setStyleSheet("color: #888888; font-size: 10px;")
            cams_layout.addWidget(no_cam)
        else:
            for idx in sorted(self.cam_caps.keys()):
                box = QtWidgets.QGroupBox(f"Cam {idx}")
                box_layout = QtWidgets.QVBoxLayout(box)
                lbl = QtWidgets.QLabel("Desactivada")
                lbl.setAlignment(QtCore.Qt.AlignCenter)
                lbl.setFixedSize(220, 124)
                lbl.setStyleSheet(
                    "background-color: #202020; "
                    "border: 1px solid #444444; "
                    "color: #777777; font-size: 10px;"
                )
                box_layout.addWidget(lbl)
                self.cam_labels[idx] = lbl
                cams_layout.addWidget(box)

        cams_layout.addStretch(1)

        # Centro: visor 3D
        main_layout.addWidget(cams_panel, 1)
        main_layout.addWidget(self.view, 4)

        # Panel derecho
        panel = QtWidgets.QWidget()
        panel_layout = QtWidgets.QVBoxLayout(panel)

        # Cámara 3D
        group_cam = QtWidgets.QGroupBox("Cámara 3D")
        cam_layout = QtWidgets.QFormLayout(group_cam)

        self.slider_cam_dist = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_cam_dist.setRange(10, 80)
        self.slider_cam_dist.setValue(30)
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

        # Cuerpo / escala
        group_body = QtWidgets.QGroupBox("Cuerpo / escala")
        body_layout = QtWidgets.QFormLayout(group_body)

        self.slider_scale = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_scale.setRange(50, 250)
        self.slider_scale.setValue(int(self.skeleton_scale * 100))
        self.slider_scale.valueChanged.connect(self._on_scale_slider_changed)

        self.slider_cloud_size = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_cloud_size.setRange(1, 10)
        self.slider_cloud_size.setValue(int(self.cloud_point_size))
        self.slider_cloud_size.valueChanged.connect(self._on_cloud_size_changed)

        body_layout.addRow("Escala esqueleto", self.slider_scale)
        body_layout.addRow("Tamaño nube", self.slider_cloud_size)

        panel_layout.addWidget(group_body)

        # Capas visibles
        group_layers = QtWidgets.QGroupBox("Capas visibles")
        layers_layout = QtWidgets.QVBoxLayout(group_layers)

        self.chk_skel = QtWidgets.QCheckBox("Esqueleto Kinect")
        self.chk_skel.setChecked(True)
        self.chk_skel.toggled.connect(self._on_layers_changed)

        self.chk_bones = QtWidgets.QCheckBox("Huesos (líneas)")
        self.chk_bones.setChecked(True)
        self.chk_bones.toggled.connect(self._on_layers_changed)

        self.chk_avatar = QtWidgets.QCheckBox("Avatar 3D (cápsulas)")
        self.chk_avatar.setChecked(self.show_avatar)
        self.chk_avatar.toggled.connect(self._on_layers_changed)

        self.chk_cloud = QtWidgets.QCheckBox("Nube cuerpo (BodyIndex/Depth)")
        self.chk_cloud.setChecked(self.show_cloud)
        self.chk_cloud.toggled.connect(self._on_layers_changed)

        self.chk_mp = QtWidgets.QCheckBox("MediaPipe 3D")
        self.chk_mp.setChecked(self.show_mediapipe)
        self.chk_mp.toggled.connect(self._on_layers_changed)

        layers_layout.addWidget(self.chk_skel)
        layers_layout.addWidget(self.chk_bones)
        layers_layout.addWidget(self.chk_avatar)
        layers_layout.addWidget(self.chk_cloud)
        layers_layout.addWidget(self.chk_mp)

        panel_layout.addWidget(group_layers)

        # Cámaras externas / toggles
        group_cams = QtWidgets.QGroupBox("Cámaras externas (controles)")
        cams_right_layout = QtWidgets.QVBoxLayout(group_cams)

        if self.cam_detected_count == 0:
            lbl = QtWidgets.QLabel("No se detectaron cámaras externas\n(índices 1..7).")
            lbl.setStyleSheet("color: #888888; font-size: 10px;")
            lbl.setWordWrap(True)
            cams_right_layout.addWidget(lbl)
        else:
            for idx in sorted(self.cam_caps.keys()):
                chk = QtWidgets.QCheckBox(f"Cam {idx}")
                chk.setChecked(False)
                chk.toggled.connect(
                    lambda state, cam_idx=idx: self._on_camera_toggled(cam_idx, state)
                )
                self.cam_checkboxes[idx] = chk
                cams_right_layout.addWidget(chk)

            note = QtWidgets.QLabel(
                "Se omite la cámara integrada del notebook (índice 0)."
            )
            note.setStyleSheet("color: #888888; font-size: 9px;")
            note.setWordWrap(True)
            cams_right_layout.addWidget(note)

        panel_layout.addWidget(group_cams)

        # Estado Kinect
        self.lbl_kinect_status = QtWidgets.QLabel("Kinect: sin datos")
        self.lbl_kinect_status.setStyleSheet("color: #cccccc; font-size: 10px;")
        panel_layout.addWidget(self.lbl_kinect_status)

        # Indicador visual rojo / verde de detección de cuerpo
        self.lbl_body_state = QtWidgets.QLabel("Cuerpo: NO detectado")
        self.lbl_body_state.setStyleSheet(
            "color: #ffffff; font-size: 10px; "
            "background-color: #802020; "
            "padding: 2px 8px; "
            "border-radius: 4px;"
        )
        panel_layout.addWidget(self.lbl_body_state)

        # Grupo de grabación
        group_rec = QtWidgets.QGroupBox("Grabación / Pipeline offline")
        rec_layout = QtWidgets.QVBoxLayout(group_rec)

        self.lbl_record_state = QtWidgets.QLabel("Grabación: detenida")
        self.lbl_record_state.setStyleSheet(
            "color: #ffffff; font-size: 10px; "
            "background-color: #555555; "
            "padding: 2px 8px; "
            "border-radius: 4px;"
        )
        rec_layout.addWidget(self.lbl_record_state)

        self.btn_record_start = QtWidgets.QPushButton("Iniciar grabación")
        self.btn_record_start.clicked.connect(self._on_start_recording)
        rec_layout.addWidget(self.btn_record_start)

        self.btn_record_stop = QtWidgets.QPushButton("Detener y procesar")
        self.btn_record_stop.setEnabled(False)
        self.btn_record_stop.clicked.connect(self._on_stop_recording)
        rec_layout.addWidget(self.btn_record_stop)

        panel_layout.addWidget(group_rec)

        info_label = QtWidgets.QLabel(
            "Mouse: rotar/zoom\nR: reset cámara · C: nube · M: MediaPipe"
        )
        info_label.setStyleSheet("color: #dddddd; font-size: 11px;")
        panel_layout.addWidget(info_label)

        panel_layout.addStretch(1)

        main_layout.addWidget(panel, 1)

    # ------------------------------------------------------------------
    # Helpers de estado de grabación y cuerpo
    # ------------------------------------------------------------------
    def _set_record_label(self, text: str, bg: str = "#555555"):
        if self.lbl_record_state is not None:
            self.lbl_record_state.setText(text)
            self.lbl_record_state.setStyleSheet(
                "color: #ffffff; font-size: 10px; "
                f"background-color: {bg}; "
                "padding: 2px 8px; "
                "border-radius: 4px;"
            )

    def _set_body_state(self, detected: bool, extra: str = ""):
        """Actualiza flag lógico y label rojo/verde."""
        self.body_detected = detected
        if self.lbl_body_state is None:
            return

        if detected:
            text = "Cuerpo: DETECTADO"
            if extra:
                text += f" · {extra}"
            style = (
                "color: #ffffff; font-size: 10px; "
                "background-color: #207020; "
                "padding: 2px 8px; "
                "border-radius: 4px;"
            )
        else:
            text = "Cuerpo: NO detectado"
            if extra:
                text += f" · {extra}"
            style = (
                "color: #ffffff; font-size: 10px; "
                "background-color: #802020; "
                "padding: 2px 8px; "
                "border-radius: 4px;"
            )

        self.lbl_body_state.setText(text)
        self.lbl_body_state.setStyleSheet(style)

    # ------------------------------------------------------------------
    # Mapeos Kinect -> visor
    # ------------------------------------------------------------------
    def _map_to_gl_coords_single(self, p: np.ndarray, root: np.ndarray) -> np.ndarray:
        rel = (p - root) * self.skeleton_scale
        x, y, z = rel
        return np.array([x, -z, y], dtype=np.float32)

    def _map_to_gl_coords_cloud(self, pts_cam: np.ndarray, root: np.ndarray) -> np.ndarray:
        rel = (pts_cam - root) * self.skeleton_scale
        x = rel[:, 0]
        y = rel[:, 1]
        z = rel[:, 2]
        gl_pts = np.stack([x, -z, y], axis=1)
        return gl_pts.astype(np.float32)

    # ------------------------------------------------------------------
    # Controles UI
    # ------------------------------------------------------------------
    def _on_cam_slider_changed(self):
        dist = self.slider_cam_dist.value() / 10.0
        elev = float(self.slider_cam_elev.value())
        azim = float(self.slider_cam_azim.value())
        self.view.setCameraPosition(distance=dist, elevation=elev, azimuth=azim)

    def _set_iso_view(self):
        self.slider_cam_dist.setValue(30)
        self.slider_cam_elev.setValue(20)
        self.slider_cam_azim.setValue(45)
        self._on_cam_slider_changed()

    def _set_front_view(self):
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
        self.show_avatar = self.chk_avatar.isChecked()
        self.show_cloud = self.chk_cloud.isChecked()
        self.show_mediapipe = self.chk_mp.isChecked()
        for mesh in self.bone_mesh_items:
            mesh.setVisible(self.show_avatar)

    def _on_camera_toggled(self, cam_idx: int, enabled: bool):
        self.cam_enabled[cam_idx] = enabled
        lbl = self.cam_labels.get(cam_idx)
        if lbl is None:
            return
        if not enabled:
            lbl.clear()
            lbl.setText("Desactivada")

    # ------------------------------------------------------------------
    # Avatar 3D: cápsulas
    # ------------------------------------------------------------------
    def _update_bone_capsule(self, idx: int, p0_gl: np.ndarray, p1_gl: np.ndarray):
        if not self.show_avatar:
            self.bone_mesh_items[idx].setVisible(False)
            return

        dir_vec = p1_gl - p0_gl
        length = float(np.linalg.norm(dir_vec))
        if length < 1e-5:
            self.bone_mesh_items[idx].setVisible(False)
            return

        z_axis = dir_vec / length
        base = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        axis = np.cross(base, z_axis)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-6:
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            axis /= axis_norm

        dot = float(np.clip(np.dot(base, z_axis), -1.0, 1.0))
        angle_deg = np.degrees(np.arccos(dot))

        center = 0.5 * (p0_gl + p1_gl)
        radius = 0.06 * self.skeleton_scale

        mesh = self.bone_mesh_items[idx]
        mesh.resetTransform()
        mesh.scale(radius, radius, length)
        mesh.rotate(angle_deg, axis[0], axis[1], axis[2])
        mesh.translate(center[0], center[1], center[2])
        mesh.setVisible(True)

    # ------------------------------------------------------------------
    # Helper para debug: cuántos cuerpos ve el runtime
    # ------------------------------------------------------------------
    def _get_tracked_bodies_count(self) -> int:
        try:
            rt = self.kinect.kinect  # PyKinectRuntime.PyKinectRuntime
            if not rt.has_new_body_frame() and self._last_tracked_bodies:
                return self._last_tracked_bodies

            bf = rt.get_last_body_frame()
            if bf is None:
                return 0

            tracked = 0
            for i in range(rt.max_body_count):
                if bf.bodies[i].is_tracked:
                    tracked += 1
            self._last_tracked_bodies = tracked
            return tracked
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Esqueleto Kinect
    # ------------------------------------------------------------------
    def _update_skeleton(self):
        # Leer joints actuales desde el wrapper
        joint_positions: Dict[int, np.ndarray] = self.kinect.get_joint_positions()

        # Fallback: si este frame no trae joints pero antes sí había,
        # reutilizamos el último esqueleto válido (evita parpadeo)
        used_fallback = False
        if not joint_positions and self._last_joints:
            joint_positions = self._last_joints
            used_fallback = True
        elif joint_positions:
            self._last_joints = joint_positions

        tracked_count = self._get_tracked_bodies_count()

        # ---------------------------
        # SIN esqueleto válido
        # ---------------------------
        if not joint_positions:
            if self.lbl_kinect_status is not None:
                self.lbl_kinect_status.setText(
                    f"Kinect: sin cuerpo detectado · bodies_tracked={tracked_count}"
                )

            # No marcamos definitivo NO detectado aquí si se usa BodyIndex;
            # _update_body_cloud() puede encontrar puntos y sobreescribir.
            if not USE_BODY_INDEX:  # solo si no hay BodyIndex posible
                self._set_body_state(False, "sin joints")

            self._last_root = None
            self.skel_item.setData(pos=np.zeros((0, 3), dtype=np.float32))
            for line in self.bone_items:
                line.setData(pos=np.zeros((2, 3), dtype=np.float32))
            for mesh in self.bone_mesh_items:
                mesh.setVisible(False)
            return

        # ---------------------------
        # HAY esqueleto (nuevo o reutilizado)
        # ---------------------------
        if self.lbl_kinect_status is not None:
            extra = " (esqueleto reutilizado)" if used_fallback else ""
            self.lbl_kinect_status.setText(
                f"Kinect: cuerpo detectado · joints={len(joint_positions)} "
                f"· bodies_tracked={tracked_count}{extra}"
            )

        # Esqueleto por sí solo ya indica cuerpo
        self._set_body_state(True, f"joints={len(joint_positions)}")

        # Root del esqueleto
        root = joint_positions.get(
            PyKinectV2.JointType_SpineBase,
            next(iter(joint_positions.values()))
        )
        self._last_root = root.copy()

        gl_joints: Dict[int, np.ndarray] = {
            jid: self._map_to_gl_coords_single(p, root)
            for jid, p in joint_positions.items()
        }

        # Puntos
        if self.show_skeleton:
            pts = np.stack(list(gl_joints.values()), axis=0)
            self.skel_item.setData(pos=pts)
        else:
            self.skel_item.setData(pos=np.zeros((0, 3), dtype=np.float32))

        # Huesos + cápsulas
        for idx, (j0, j1) in enumerate(self.bone_pairs):
            if j0 in gl_joints and j1 in gl_joints:
                p0_gl = gl_joints[j0]
                p1_gl = gl_joints[j1]

                if self.show_bones:
                    line_pts = np.vstack([p0_gl, p1_gl])
                else:
                    line_pts = np.zeros((2, 3), dtype=np.float32)
                self.bone_items[idx].setData(pos=line_pts)

                self._update_bone_capsule(idx, p0_gl, p1_gl)
            else:
                self.bone_items[idx].setData(
                    pos=np.zeros((2, 3), dtype=np.float32)
                )
                self.bone_mesh_items[idx].setVisible(False)

    # ------------------------------------------------------------------
    # Nube BodyIndex + Depth
    # ------------------------------------------------------------------
    def _update_body_cloud(self):
        # Si no se usa BodyIndex, solo limpiamos visual y salimos
        if not USE_BODY_INDEX:
            self.body_cloud_item.setData(pos=np.zeros((0, 3), dtype=np.float32))
            return

        # Si el usuario apaga la capa "Nube cuerpo", no tocamos el estado
        # lógico de detección (puede seguir viniendo desde el esqueleto).
        if not self.show_cloud:
            self.body_cloud_item.setData(pos=np.zeros((0, 3), dtype=np.float32))
            return

        cloud_cam = self.kinect.get_body_point_cloud(max_points=MAX_CLOUD_POINTS)
        if cloud_cam.shape[0] == 0:
            # No hay puntos BodyIndex -> si además no hay joints, será NO detectado.
            # Aquí marcamos NO detectado por BodyIndex, pero el esqueleto puede
            # haber puesto DETECTADO antes.
            if not self.body_detected:  # solo si nadie lo marcó antes
                self._set_body_state(False, "sin puntos BodyIndex")
            self.body_cloud_item.setData(pos=np.zeros((0, 3), dtype=np.float32))
            return

        # Hay puntos BodyIndex -> esto es una evidencia fuerte de cuerpo presente.
        self._set_body_state(True, f"BodyIndex pts={cloud_cam.shape[0]}")

        if self._last_root is not None:
            root = self._last_root
        else:
            root = cloud_cam.mean(axis=0)

        gl_pts = self._map_to_gl_coords_cloud(cloud_cam, root)
        self.body_cloud_item.setData(pos=gl_pts, size=self.cloud_point_size)

    # ------------------------------------------------------------------
    # MediaPipe (visual en tiempo real, menos frecuente)
    # ------------------------------------------------------------------
    def _update_mediapipe(self):
        if not USE_MEDIAPIPE or self.mp_pose is None or not self.show_mediapipe:
            self.mp_item.setData(pos=np.zeros((0, 3), dtype=np.float32))
            return

        color_bgr = self.kinect.get_color_bgr()
        if color_bgr is None:
            self.mp_item.setData(pos=np.zeros((0, 3), dtype=np.float32))
            return

        h, w, _ = color_bgr.shape
        if w > 960:
            new_w = 960
            new_h = int(h * new_w / w)
            color_bgr_proc = cv2.resize(color_bgr, (new_w, new_h))
        else:
            color_bgr_proc = color_bgr

        _, mp_world = self.mp_pose.process(color_bgr_proc)
        if mp_world.shape[0] == 0:
            self.mp_item.setData(pos=np.zeros((0, 3), dtype=np.float32))
            return

        mp_pts = mp_world * self.skeleton_scale
        self.mp_item.setData(pos=mp_pts)

    # ------------------------------------------------------------------
    # Actualizar cámaras externas (preview + grabación si aplica)
    # ------------------------------------------------------------------
    def _update_cameras(self):
        if not self.cam_caps:
            return

        for idx, cap in self.cam_caps.items():
            if not self.cam_enabled.get(idx, False):
                continue

            lbl = self.cam_labels.get(idx)
            if lbl is None:
                continue

            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            # Si hay sesión de grabación activa, grabar esta cámara
            if self.recording_session is not None and self.recording_session.active:
                stream_id = f"cam_{idx}"
                self.recording_session.write_frame(stream_id, frame)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qimg = QtGui.QImage(
                frame_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888
            )
            pix = QtGui.QPixmap.fromImage(qimg)
            pix = pix.scaled(
                lbl.width(),
                lbl.height(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
            lbl.setPixmap(pix)
            lbl.setText("")

    # ------------------------------------------------------------------
    # Grabación: iniciar / detener
    # ------------------------------------------------------------------
    def _on_start_recording(self):
        if self.recording_session is not None and self.recording_session.active:
            return

        # Carpeta de salida con timestamp
        base_dir = os.path.join(os.getcwd(), "captures")
        os.makedirs(base_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        capture_dir = os.path.join(base_dir, f"session_{ts}")
        os.makedirs(capture_dir, exist_ok=True)
        self._last_capture_dir = capture_dir

        fps = int(TARGET_FPS) if TARGET_FPS and TARGET_FPS > 0 else 60

        self.recording_session = RecordingSession(
            output_dir=capture_dir,
            fps=fps,
            save_kinect_3d=True,
        )

        # Registrar streams: Kinect color + cámaras externas (cada una en archivo individual)
        self.recording_session.add_stream("kinect_color", "kinect_color.mp4")
        for idx in sorted(self.cam_caps.keys()):
            self.recording_session.add_stream(f"cam_{idx}", f"cam_{idx}.mp4")

        self._set_record_label(f"Grabando en: {capture_dir}", bg="#207020")
        if self.btn_record_start:
            self.btn_record_start.setEnabled(False)
        if self.btn_record_stop:
            self.btn_record_stop.setEnabled(True)

    def _on_stop_recording(self):
        if self.recording_session is None:
            return

        # Detener y cerrar archivos de grabación
        self.recording_session.stop()
        capture_dir = self.recording_session.output_dir
        self.recording_session = None

        self._set_record_label("Grabación: detenida (procesando offline...)", bg="#A07010")
        if self.btn_record_start:
            self.btn_record_start.setEnabled(True)
        if self.btn_record_stop:
            self.btn_record_stop.setEnabled(False)

        # Lanzar pipeline offline (MediaPipe, landmarks, Pal Yang, XLSX, etc.)
        self._run_offline_processing(capture_dir)

    # ------------------------------------------------------------------
    # Pipeline offline + ventana de progreso + reporte
    # ------------------------------------------------------------------
    def _run_offline_processing(self, capture_dir: str):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Procesando captura offline")
        vbox = QtWidgets.QVBoxLayout(dlg)

        lbl_status = QtWidgets.QLabel("Inicializando...", dlg)
        bar = QtWidgets.QProgressBar(dlg)
        bar.setRange(0, 100)

        vbox.addWidget(lbl_status)
        vbox.addWidget(bar)

        dlg.setModal(True)
        dlg.show()
        QtWidgets.QApplication.processEvents()

        def progress_cb(msg: str, frac: float):
            lbl_status.setText(msg)
            bar.setValue(int(frac * 100))
            QtWidgets.QApplication.processEvents()

        try:
            report_path = run_offline_pipeline(capture_dir, progress_cb=progress_cb)
        except Exception as e:
            dlg.close()
            QtWidgets.QMessageBox.critical(
                self,
                "Error en procesamiento offline",
                f"Ocurrió un error durante el pipeline offline:\n{e}",
            )
            self._set_record_label("Grabación: detenida (error en pipeline)", bg="#802020")
            return

        dlg.close()
        self._set_record_label("Grabación: detenida (pipeline completado)", bg="#207020")

        # Preguntar si se desea ver el reporte
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setWindowTitle("Procesamiento completado")
        msg_box.setText(f"Reporte generado en:\n{report_path}")
        btn_ver = msg_box.addButton("Ver reporte", QtWidgets.QMessageBox.AcceptRole)
        msg_box.addButton("Cerrar", QtWidgets.QMessageBox.RejectRole)
        msg_box.exec_()

        if msg_box.clickedButton() is btn_ver:
            w = ReportWindow(report_path, parent=self)
            w.exec_()

    # ------------------------------------------------------------------
    # Timer principal
    # ------------------------------------------------------------------
    def _on_timer(self):
        self._frame_counter += 1

        # 1) Actualizar frames de Kinect
        self.kinect.update_frames()

        # 2) Esqueleto y nube
        self._update_skeleton()
        self._update_body_cloud()  # ← BodyIndex se evalúa después y puede sobreescribir el estado

        # 3) Grabación de Kinect (color + 3D) si hay sesión activa
        if self.recording_session is not None and self.recording_session.active:
            color_bgr = self.kinect.get_color_bgr()
            if color_bgr is not None:
                self.recording_session.write_frame("kinect_color", color_bgr)

            cloud_cam = self.kinect.get_body_point_cloud(max_points=MAX_CLOUD_POINTS)
            joints = self.kinect.get_joint_positions()
            self.recording_session.record_kinect_3d(
                cloud_cam=cloud_cam,
                joint_positions=joints,
            )

        # 4) MediaPipe solo cada MP_FRAME_STRIDE frames (visual real-time)
        if self._frame_counter % MP_FRAME_STRIDE == 0:
            self._update_mediapipe()

        # 5) Previews de cámaras externas solo cada CAM_FRAME_STRIDE frames
        if self._frame_counter % CAM_FRAME_STRIDE == 0:
            self._update_cameras()

    # ------------------------------------------------------------------
    # Teclado
    # ------------------------------------------------------------------
    def keyPressEvent(self, event):
        key = event.key()

        if key == QtCore.Qt.Key_R:
            self._set_iso_view()
        elif key == QtCore.Qt.Key_C and USE_BODY_INDEX:
            self.chk_cloud.toggle()
        elif key == QtCore.Qt.Key_M and USE_MEDIAPIPE:
            self.chk_mp.toggle()

        super().keyPressEvent(event)

    # ------------------------------------------------------------------
    def closeEvent(self, event):
        # Asegurar que cualquier sesión de grabación se detenga
        if self.recording_session is not None and self.recording_session.active:
            self.recording_session.stop()
            self.recording_session = None

        if self.mp_pose is not None:
            self.mp_pose.close()
        self.kinect.close()

        for cap in self.cam_caps.values():
            try:
                cap.release()
            except Exception:
                pass

        super().closeEvent(event)
