# src/features/feature_extractor.py
"""
Módulo de extracción de features para el proyecto de Poomsae (8yang / Pal Jang).

Incluye dos tipos de extracción:

1) FeatureExtractor (series temporales de ángulos)
   - Recibe una secuencia de frames con 33 landmarks (MediaPipe Pose).
   - Canonicaliza la pose 2D (traslación, rotación, escala, reflexión).
   - Calcula ángulos articulares usando `angles_from_landmarks`.
   - Aplica suavizado temporal y estadísticas (mean, std, p10, etc.).
   - Devuelve un diccionario de features para un segmento [frame_start, frame_end].

2) Features de postura (stance):
   - `extract_stance_features`: trabaja directamente sobre un DataFrame
     de landmarks (como los CSV generados por `extract_landmarks.py`).
   - Devuelve features geométricas para clasificar posturas:
       distancia entre tobillos / ancho de hombros, offsets de cadera,
       ángulos de rodilla, orientación de pies, etc.

El objetivo es que este módulo sea usable tanto en el pipeline offline
(entrenar `rf_8yang.joblib`) como online (Kinect + MediaPipe en tiempo real).

Autor: adaptado e integrado a poomsae_kinect_3d.
"""

from __future__ import annotations

from typing import Dict, List, Iterable, Sequence, Any, Tuple, Optional

import numpy as np
import pandas as pd
from math import atan2, cos, sin

# Import internos del proyecto
from .angles import angles_from_landmarks          # Debe existir en src/features/angles.py
from ..utils.smoothing import smooth              # Debe existir en src/utils/smoothing.py


# -------------------------------------------------------------------------
# Constantes / utilidades comunes
# -------------------------------------------------------------------------

# Índices MediaPipe Pose (formato oficial, 0-based)
LMK = {
    "L_SH": 11,
    "R_SH": 12,
    "L_HIP": 23,
    "R_HIP": 24,
    # los demás (rodillas, tobillos, etc.) se usan en la parte de stance
}

# Estadísticos base que usaremos para las series angulares
STAT_FUNCS = {
    "mean": np.nanmean,
    "std": np.nanstd,
    "p10": lambda x: np.nanpercentile(x, 10),
    "p50": lambda x: np.nanpercentile(x, 50),
    "p90": lambda x: np.nanpercentile(x, 90),
    "range": lambda x: np.nanmax(x) - np.nanmin(x),
    # Sobre la propia serie: velocidad y aceleración máxima
    "max_speed": lambda v: np.nanmax(np.abs(np.gradient(v))),
    "max_acc": lambda v: np.nanmax(np.abs(np.gradient(np.gradient(v)))),
}


def _rot2d(theta: float) -> np.ndarray:
    """Matriz de rotación 2D (theta en radianes)."""
    c, s = cos(theta), sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def _ffill_bfill(arr: np.ndarray) -> np.ndarray:
    """
    Rellena NaN hacia adelante y atrás por columna.
    Devuelve una copia sin NaNs internos (si hay al menos un valor válido).
    """
    out = arr.copy()
    for k in range(out.shape[1]):
        v = out[:, k]
        if np.isnan(v).any():
            # forward fill
            for i in range(1, len(v)):
                if np.isnan(v[i]) and not np.isnan(v[i - 1]):
                    v[i] = v[i - 1]
            # backward fill
            for i in range(len(v) - 2, -1, -1):
                if np.isnan(v[i]) and not np.isnan(v[i + 1]):
                    v[i] = v[i + 1]
            out[:, k] = v
    return out


def canonicalize_xy(xy33: np.ndarray) -> np.ndarray:
    """
    canonicalize_xy(xy33) -> (33,2) canónico.

    Entrada:
        xy33: array (33,2) con coordenadas normalizadas [0..1] de MediaPipe.

    Proceso:
        - Origen = centro de caderas.
        - Eje Y = vector caderas -> hombros.
        - Escala = ancho de hombros.
        - Reflejo horizontal si es necesario para dejar L_SH.x < R_SH.x.

    Si la entrada no es válida, devuelve el array original.
    """
    if not isinstance(xy33, np.ndarray) or xy33.ndim != 2 or xy33.shape[1] < 2 or xy33.shape[0] < 25:
        return xy33

    xy = xy33[:, :2].astype(np.float32)

    # Rellenar NaNs si hay (para evitar explosiones numéricas)
    if np.isnan(xy).any():
        xy = _ffill_bfill(xy)
        xy = np.nan_to_num(xy, nan=0.0)

    # Centros de caderas y hombros
    c_hip = 0.5 * (xy[LMK["L_HIP"]] + xy[LMK["R_HIP"]])
    c_sh = 0.5 * (xy[LMK["L_SH"]] + xy[LMK["R_SH"]])

    # Trasladar para que caderas queden en el origen
    xy = xy - c_hip

    # Rotar: el vector caderas->hombros debe quedar alineado a +Y
    v = c_sh - np.array([0.0, 0.0], dtype=np.float32)
    theta = atan2(v[1], v[0])            # ángulo respecto de +X
    R = _rot2d((np.pi / 2) - theta)      # llevarlo a +Y
    xy = (R @ xy.T).T

    # Escalar por ancho de hombros
    shw = float(np.linalg.norm(xy[LMK["R_SH"]] - xy[LMK["L_SH"]])) + 1e-6
    xy = xy / shw

    # Reflejar si L_SH quedó a la derecha de R_SH
    if xy[LMK["L_SH"], 0] > xy[LMK["R_SH"], 0]:
        xy[:, 0] *= -1.0

    return xy


def _apply_canon_to_lmks(lmks_one_frame: Any) -> Any:
    """
    Aplica canonicalización a la estructura de landmarks de un frame,
    conservando columnas extra (z, visibility) si existen.

    Soporta:
        - np.ndarray shape (33,2/3/4)
        - list/tuple de 33 items con [x,y,(z),(v)]

    Si no puede operar, retorna el objeto original.
    """
    try:
        arr = np.asarray(lmks_one_frame)
        if arr.ndim != 2 or arr.shape[0] < 25 or arr.shape[1] < 2:
            return lmks_one_frame

        xy_canon = canonicalize_xy(arr[:, :2])

        # Reconstruir manteniendo columnas extra (z, vis, etc.)
        out = arr.copy()
        out[:, 0:2] = xy_canon

        if isinstance(lmks_one_frame, np.ndarray):
            return out
        else:
            # Convertimos a la misma "forma" de entrada (lista de tuplas)
            out_list = []
            for row in out:
                row = row.tolist()
                out_list.append(tuple(row))
            return out_list
    except Exception:
        return lmks_one_frame


# -------------------------------------------------------------------------
# FeatureExtractor para series de ángulos (segmentos)
# -------------------------------------------------------------------------


class FeatureExtractor:
    """
    Extrae features a partir de series temporales de ángulos articulares.

    Flujo típico para un segmento:
        1) Recibir `lmks_seq`: lista de frames, cada uno (33, 2/3/4).
        2) Canonicalizar cada frame (opcional).
        3) Calcular ángulos con `angles_from_landmarks` para las tripletas definidas.
        4) Suavizar cada serie angular con `smooth`.
        5) Calcular estadísticos (mean, std, p10, ..., max_acc).
        6) Adjuntar metadatos (frames de inicio/fin, labels, etc.).

    `angle_triplets` debe ser un dict como el definido en config/default.yaml:
        {
            "elbow_left":  [11,13,15],
            "elbow_right": [12,14,16],
            ...
        }

    `stat_func_names` es una lista de nombres a usar de STAT_FUNCS:
        ["mean","std","p10","p50","p90","range","max_speed","max_acc"]
    """

    def __init__(
        self,
        angle_triplets: Dict[str, Sequence[int]],
        stat_func_names: Sequence[str],
        smoothing_window: int = 5,
        canonicalize: bool = True,
    ) -> None:
        self.triplets = dict(angle_triplets)
        self.stat_func_names = list(stat_func_names)
        self.win = int(smoothing_window)
        self.use_canon = bool(canonicalize)

        # Mapeamos nombres a funciones reales
        self._stat_funcs = {}
        for name in self.stat_func_names:
            if name not in STAT_FUNCS:
                raise ValueError(f"Estadístico desconocido: {name}")
            self._stat_funcs[name] = STAT_FUNCS[name]

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "FeatureExtractor":
        """
        Construye el extractor desde un dict de config (e.g. cargado desde default.yaml).

        Espera algo como:
            cfg["features"]["angle_triplets"]
            cfg["features"]["stat_funcs"]
            cfg["features"]["smoothing_window"]
        """
        fcfg = cfg.get("features", {})
        angle_triplets = fcfg.get("angle_triplets", {})
        stat_names = fcfg.get("stat_funcs", list(STAT_FUNCS.keys()))
        smoothing_window = int(fcfg.get("smoothing_window", 5))
        return cls(
            angle_triplets=angle_triplets,
            stat_func_names=stat_names,
            smoothing_window=smoothing_window,
            canonicalize=True,
        )

    def segment_features(
        self,
        lmks_seq: Sequence[np.ndarray],
        frames_idx: Sequence[int],
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Calcula features para un segmento.

        Args
        ----
        lmks_seq:
            Lista (o cualquier secuencia) de landmarks por frame.
            Cada elemento debe ser shape (33, 2/3/4) con x,y,(z),(visibility).
            Normalmente sale de:
                - CSV de landmarks (offline)
                - Kinect+MediaPipe en tiempo real (online)
        frames_idx:
            Índices absolutos de frame que corresponden a este segmento.
            Debe tener la misma longitud que lmks_seq.
        meta:
            Diccionario opcional con metadatos (poom, movimiento, atleta, etc.)

        Returns
        -------
        feats: dict
            Features escalares + metadatos del segmento.
        """
        if len(lmks_seq) == 0:
            raise ValueError("lmks_seq vacío para segment_features().")

        if len(lmks_seq) != len(frames_idx):
            raise ValueError("lmks_seq y frames_idx deben tener la misma longitud.")

        # 1) Canonicalización por frame para robustez a cámara
        if self.use_canon:
            lmks_seq_proc = [_apply_canon_to_lmks(l) for l in lmks_seq]
        else:
            lmks_seq_proc = lmks_seq

        # 2) Series de ángulos por articulación definida en angle_triplets
        series: Dict[str, List[float]] = {k: [] for k in self.triplets.keys()}
        for lmks in lmks_seq_proc:
            angs = angles_from_landmarks(lmks, self.triplets)
            for k, v in angs.items():
                series[k].append(v)

        # 3) Suavizado y estadísticos
        feats: Dict[str, Any] = {}
        for name, vals in series.items():
            vals = np.asarray(vals, float)
            if vals.size == 0:
                # Si no hay datos, rellenamos con NaN
                for sname in self._stat_funcs.keys():
                    feats[f"{name}_{sname}"] = np.nan
                continue

            # Suavizado polinómico (ver src/utils/smoothing.py)
            vals = smooth(vals, win=self.win, poly=2)

            for sname, func in self._stat_funcs.items():
                try:
                    feats[f"{name}_{sname}"] = float(func(vals))
                except Exception:
                    feats[f"{name}_{sname}"] = np.nan

        # 4) Metadatos
        if meta:
            feats.update({f"meta_{k}": v for k, v in meta.items()})

        feats["meta_frame_start"] = int(frames_idx[0])
        feats["meta_frame_end"] = int(frames_idx[-1])
        feats["meta_len"] = int(len(frames_idx))

        return feats


# Alias opcional por si quieres un nombre más explícito
SegmentFeatureExtractor = FeatureExtractor


# -------------------------------------------------------------------------
# Features geométricas para clasificación de posturas (stances)
# -------------------------------------------------------------------------


def _xy_mean(
    df: pd.DataFrame,
    lmk_id: int,
    a: int,
    b: int,
) -> Tuple[float, float]:
    """
    Media de coordenadas x,y para un landmark en rango de frames [a,b].
    """
    sub = df[(df["lmk_id"] == lmk_id) & (df["frame"] >= a) & (df["frame"] <= b)][
        ["x", "y"]
    ]
    if sub.empty:
        return (np.nan, np.nan)
    m = sub.mean().to_numpy(np.float32)
    return float(m[0]), float(m[1])


def _angle3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Ángulo en grados entre 3 puntos (b es el vértice).
    """
    ba = a - b
    bc = c - b
    num = float(np.dot(ba, bc))
    den = float(np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    ang = np.degrees(np.arccos(np.clip(num / den, -1, 1)))
    return float(ang)


def extract_stance_features(
    df: pd.DataFrame,
    a: int,
    b: int,
) -> Dict[str, float]:
    """
    Extrae features geométricas para clasificación de posturas en el rango [a,b].

    Args
    ----
    df:
        DataFrame con landmarks (columnas mínimas: frame, lmk_id, x, y).
        Es compatible con los CSV generados por `extract_landmarks.py`.
    a:
        Frame inicial del segmento (inclusive).
    b:
        Frame final del segmento (inclusive).

    Returns
    -------
    Dict con 9 features:
        - ankle_dist_sw: Distancia entre tobillos / ancho de hombros.
        - hip_offset_x: Offset horizontal de cadera respecto a centro de pies.
        - hip_offset_y: Offset vertical de cadera (positivo = atrás).
        - knee_angle_left: Ángulo de rodilla izquierda.
        - knee_angle_right: Ángulo de rodilla derecha.
        - foot_angle_left: Orientación pie izquierdo.
        - foot_angle_right: Orientación pie derecho.
        - hip_behind_feet: 1.0 si cadera está detrás de pies, 0.0 si no.
        - knee_angle_diff: |knee_angle_left - knee_angle_right|.
    """
    # IDs MediaPipe relevantes
    LANK_ID, RANK_ID = 27, 28
    LKNE_ID, RKNE_ID = 25, 26
    LHIP_ID, RHIP_ID = 23, 24
    LSH_ID, RSH_ID = 11, 12
    LHEEL_ID, RHEEL_ID = 29, 30
    LFOOT_ID, RFOOT_ID = 31, 32

    # Landmarks clave (promedio en el segmento [a,b])
    LANK = np.array(_xy_mean(df, LANK_ID, a, b), np.float32)
    RANK = np.array(_xy_mean(df, RANK_ID, a, b), np.float32)
    LKNE = np.array(_xy_mean(df, LKNE_ID, a, b), np.float32)
    RKNE = np.array(_xy_mean(df, RKNE_ID, a, b), np.float32)
    LHIP = np.array(_xy_mean(df, LHIP_ID, a, b), np.float32)
    RHIP = np.array(_xy_mean(df, RHIP_ID, a, b), np.float32)
    LSH = np.array(_xy_mean(df, LSH_ID, a, b), np.float32)
    RSH = np.array(_xy_mean(df, RSH_ID, a, b), np.float32)
    LHEEL = np.array(_xy_mean(df, LHEEL_ID, a, b), np.float32)
    RHEEL = np.array(_xy_mean(df, RHEEL_ID, a, b), np.float32)
    LFOOT = np.array(_xy_mean(df, LFOOT_ID, a, b), np.float32)
    RFOOT = np.array(_xy_mean(df, RFOOT_ID, a, b), np.float32)

    # Validar datos esenciales (hombros, caderas, tobillos)
    essentials = [LANK, RANK, LHIP, RHIP, LSH, RSH]
    if any(np.isnan(v).any() for v in essentials):
        return {k: np.nan for k in FEATURE_NAMES}

    # 1) Distancia entre tobillos / ancho de hombros
    sh_w = float(np.linalg.norm(RSH - LSH) + 1e-6)
    ankle_dist = float(np.linalg.norm(RANK - LANK))
    ankle_dist_sw = ankle_dist / sh_w

    # 2) Offsets de cadera respecto a centro de pies
    feet_center = 0.5 * (LANK + RANK)
    hip_center = 0.5 * (LHIP + RHIP)
    hip_offset_x = float(hip_center[0] - feet_center[0])
    hip_offset_y = float(hip_center[1] - feet_center[1])  # Y+ = "atrás" en imagen

    # 3) Ángulos de rodillas
    if not any(np.isnan(v).any() for v in [LHIP, LKNE, LANK]):
        knee_angle_left = _angle3(LHIP, LKNE, LANK)
    else:
        knee_angle_left = np.nan

    if not any(np.isnan(v).any() for v in [RHIP, RKNE, RANK]):
        knee_angle_right = _angle3(RHIP, RKNE, RANK)
    else:
        knee_angle_right = np.nan

    # 4) Orientación de pies
    def foot_angle(heel: np.ndarray, foot: np.ndarray) -> float:
        if np.isnan(heel).any() or np.isnan(foot).any():
            return np.nan
        vec = foot - heel
        if np.linalg.norm(vec) < 1e-6:
            return np.nan
        return float(np.degrees(np.arctan2(vec[1], vec[0])))

    foot_angle_left = foot_angle(LHEEL, LFOOT)
    foot_angle_right = foot_angle(RHEEL, RFOOT)

    # 5) Indicador de cadera atrás
    hip_behind_feet = 1.0 if hip_offset_y > 0 else 0.0

    # 6) Diferencia de ángulos de rodillas
    if not np.isnan(knee_angle_left) and not np.isnan(knee_angle_right):
        knee_angle_diff = abs(knee_angle_left - knee_angle_right)
    else:
        knee_angle_diff = np.nan

    return {
        "ankle_dist_sw": ankle_dist_sw,
        "hip_offset_x": hip_offset_x,
        "hip_offset_y": hip_offset_y,
        "knee_angle_left": knee_angle_left,
        "knee_angle_right": knee_angle_right,
        "foot_angle_left": foot_angle_left,
        "foot_angle_right": foot_angle_right,
        "hip_behind_feet": hip_behind_feet,
        "knee_angle_diff": knee_angle_diff,
    }


# Nombres de features de postura (orden fijo)
FEATURE_NAMES = [
    "ankle_dist_sw",
    "hip_offset_x",
    "hip_offset_y",
    "knee_angle_left",
    "knee_angle_right",
    "foot_angle_left",
    "foot_angle_right",
    "hip_behind_feet",
    "knee_angle_diff",
]


def extract_features_batch(
    df: pd.DataFrame,
    segments: List[Tuple[int, int]],
) -> np.ndarray:
    """
    Extrae features de postura para múltiples segmentos.

    Args
    ----
    df:
        DataFrame con landmarks (CSV de `extract_landmarks.py`).
    segments:
        Lista de tuplas (a, b) con rangos de frames (inclusive).

    Returns
    -------
    Array numpy shape (n_segments, n_features).
    """
    rows = []
    for a, b in segments:
        feat_dict = extract_stance_features(df, a, b)
        rows.append([feat_dict[name] for name in FEATURE_NAMES])
    return np.asarray(rows, dtype=np.float32)


def extract_stance_features_from_dict(
    landmarks_dict: Dict[str, np.ndarray],
    frame_idx: int,
) -> Dict[str, float]:
    """
    Extrae features de postura para un solo frame, a partir de un diccionario
    de landmarks ya separado por nombre.

    Args
    ----
    landmarks_dict:
        Diccionario con arrays numpy (n_frames, 2) para cada landmark clave:
            'L_ANK','R_ANK','L_KNEE','R_KNEE','L_HIP','R_HIP',
            'L_SH','R_SH','L_HEEL','R_HEEL','L_FOOT','R_FOOT'
    frame_idx:
        Índice de frame a analizar.

    Returns
    -------
    Dict con las mismas 9 features definidas en FEATURE_NAMES.
    """
    try:
        LANK = landmarks_dict["L_ANK"][frame_idx].astype(np.float32)
        RANK = landmarks_dict["R_ANK"][frame_idx].astype(np.float32)
        LKNE = landmarks_dict["L_KNEE"][frame_idx].astype(np.float32)
        RKNE = landmarks_dict["R_KNEE"][frame_idx].astype(np.float32)
        LHIP = landmarks_dict["L_HIP"][frame_idx].astype(np.float32)
        RHIP = landmarks_dict["R_HIP"][frame_idx].astype(np.float32)
        LSH = landmarks_dict["L_SH"][frame_idx].astype(np.float32)
        RSH = landmarks_dict["R_SH"][frame_idx].astype(np.float32)
        LHEEL = landmarks_dict["L_HEEL"][frame_idx].astype(np.float32)
        RHEEL = landmarks_dict["R_HEEL"][frame_idx].astype(np.float32)
        LFOOT = landmarks_dict["L_FOOT"][frame_idx].astype(np.float32)
        RFOOT = landmarks_dict["R_FOOT"][frame_idx].astype(np.float32)
    except (KeyError, IndexError):
        return {name: np.nan for name in FEATURE_NAMES}

    # 1) Distancia entre tobillos / ancho de hombros
    ankle_dist = float(np.linalg.norm(RANK - LANK))
    shoulder_width = float(np.linalg.norm(RSH - LSH)) + 1e-6
    ankle_dist_sw = ankle_dist / shoulder_width

    # 2) Centros
    hip_center = (LHIP + RHIP) / 2.0
    feet_center = (LANK + RANK) / 2.0

    hip_offset_x = float(hip_center[0] - feet_center[0])
    hip_offset_y = float(hip_center[1] - feet_center[1])

    # 3) Ángulos de rodillas
    knee_angle_left = _angle3(LHIP, LKNE, LANK)
    knee_angle_right = _angle3(RHIP, RKNE, RANK)

    # 4) Orientación de pies
    left_foot_vec = LFOOT - LHEEL
    right_foot_vec = RFOOT - RHEEL
    foot_angle_left = float(
        np.degrees(np.arctan2(left_foot_vec[1], left_foot_vec[0]))
    )
    foot_angle_right = float(
        np.degrees(np.arctan2(right_foot_vec[1], right_foot_vec[0]))
    )

    # 5) Indicador binario de cadera atrás de pies
    hip_behind_feet = 1.0 if hip_offset_y > 0 else 0.0

    # 6) Diferencia de ángulos de rodillas
    knee_angle_diff = abs(knee_angle_left - knee_angle_right)

    return {
        "ankle_dist_sw": ankle_dist_sw,
        "hip_offset_x": hip_offset_x,
        "hip_offset_y": hip_offset_y,
        "knee_angle_left": knee_angle_left,
        "knee_angle_right": knee_angle_right,
        "foot_angle_left": foot_angle_left,
        "foot_angle_right": foot_angle_right,
        "hip_behind_feet": hip_behind_feet,
        "knee_angle_diff": knee_angle_diff,
    }
