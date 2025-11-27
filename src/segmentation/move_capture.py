"""Segmentación y clasificación de movimientos (Poomsae-aware).

Este módulo implementa un pipeline de segmentación pensado para
poomsae (ej. 8yang). Contiene:
 - `PoomsaeConfig`: carga de especificaciones y parámetros.
 - `SpecAwareSegmenter`: heurísticas y energía para detectar movimientos.
 - `PoseClassifier`: heurístico (y opción ML) para clasificar posturas.

Los cambios realizados son principalmente docstrings y comentarios
para que el código resulte más claro y "hecho por una persona".
"""
from __future__ import annotations
import math, json, yaml
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import cv2
import scipy.signal as sp_signal  # evitar choque de nombre con variables

# =====================================================================
# CONFIGURACIÓN GLOBAL / ESPECIFICACIONES POOMSAE
# =====================================================================

class PoomsaeConfig:
    """
    Carga:
      - config/default.yaml
      - config/patterns/8yang_spec.json
      - config/patterns/pose_spec.json
    y entrega helpers para acceder a umbrales y movimientos esperados.
    """

    def __init__(self,
                 spec_path: Path,
                 pose_spec_path: Path,
                 config_path: Path):
        self.spec_path = Path(spec_path)
        self.pose_spec_path = Path(pose_spec_path)
        self.config_path = Path(config_path)

        self.poomsae_spec: Optional[Dict[str, Any]] = None
        self.pose_spec: Optional[Dict[str, Any]] = None
        self.app_config: Optional[Dict[str, Any]] = None

        self.load_configs()

    def load_configs(self) -> None:
        """Carga YAML + JSON y deja todo disponible en atributos."""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.app_config = yaml.safe_load(f) or {}

            with open(self.spec_path, "r", encoding="utf-8") as f:
                self.poomsae_spec = json.load(f)

            with open(self.pose_spec_path, "r", encoding="utf-8") as f:
                self.pose_spec = json.load(f)

            # Mensaje informativo claro y útil para quien ejecute el script
            print(
                f"[CONFIG] ✅ Cargadas especificaciones: "
                f"poomsae={self.poomsae_spec.get('poomsae','?')}, "
                f"pose_spec_path={self.pose_spec_path.name}, config={self.config_path.name}"
            )
        except Exception as e:
            print(f"[CONFIG] ❌ Error cargando configuraciones: {e}")
            raise

    # ---- Helpers sobre el JSON de 8yang ---------------------------------

    def get_move_spec(self, move_idx: int, sub: str | None = None) -> Optional[Dict[str, Any]]:
        """
        Devuelve el dict de un movimiento según idx (y sub, si aplica).
        idx = índice lógico de movimiento (1..segments_total).
        """
        if not self.poomsae_spec:
            return None

        for move in self.poomsae_spec.get("moves", []):
            if move.get("idx") == move_idx:
                if sub is None or move.get("sub") == sub:
                    return move
        return None

    # ---- Helpers sobre pose_spec.json -----------------------------------

    def get_stance_thresholds(self, stance_code: str) -> Dict[str, Any]:
        if not self.pose_spec:
            return {}
        return (self.pose_spec.get("stances") or {}).get(stance_code, {})

    def get_kick_thresholds(self, kick_type: str) -> Dict[str, Any]:
        if not self.pose_spec:
            return {}
        return (self.pose_spec.get("kicks") or {}).get(kick_type, {})


# =====================================================================
# ÍNDICES MEDIAPIPE POSE
# =====================================================================

LMK = dict(
    NOSE=0,
    L_EYE_IN=1, L_EYE=2, L_EYE_OUT=3, R_EYE_IN=4, R_EYE=5, R_EYE_OUT=6,
    L_EAR=7, R_EAR=8,
    L_MOUTH=9, R_MOUTH=10,
    L_SH=11, R_SH=12, L_ELB=13, R_ELB=14, L_WRIST=15, R_WRIST=16,
    L_PINKY=17, R_PINKY=18, L_INDEX=19, R_INDEX=20, L_THUMB=21, R_THUMB=22,
    L_HIP=23, R_HIP=24, L_KNEE=25, R_KNEE=26, L_ANK=27, R_ANK=28,
    L_HEEL=29, R_HEEL=30, L_FOOT=31, R_FOOT=32
)

END_EFFECTORS = ("L_WRIST", "R_WRIST", "L_ANK", "R_ANK")


# =====================================================================
# SEGMENTADOR ESPECÍFICO PARA 8YANG
# =====================================================================

class SpecAwareSegmenter:
    """
    Segmentador que utiliza:
      - parámetros de segmentación de config/default.yaml
      - energía de efectores, ángulos y rotación
    para detectar los ~36 segmentos esperados de Taegeuk 8 Jang.
    """

    def __init__(self, config: PoomsaeConfig):
        self.config = config
        self.poomsae_spec = config.poomsae_spec or {}
        self.pose_spec = config.pose_spec or {}

        seg_cfg = (config.app_config or {}).get("segmentation", {})

        # Parámetros alineados con tu default.yaml
        self.min_segment_frames: int = seg_cfg.get("min_segment_frames", 3)
        self.max_pause_frames: int = seg_cfg.get("max_pause_frames", 2)
        self.activity_thresh: float = seg_cfg.get("activity_threshold", 0.05)
        self.peak_thresh: float = seg_cfg.get("peak_threshold", 0.07)
        self.min_peak_dist: int = seg_cfg.get("min_peak_distance", 3)
        self.smooth_win: int = seg_cfg.get("smooth_window", 2)

        self.expected_segments: int = seg_cfg.get(
            "expected_movements", self.poomsae_spec.get("segments_total", 36)
        )

    # ------------ Cálculo de energía "poomsae-aware" ---------------------

    def _compute_poomsae_aware_energy(
        self,
        angles_dict: Dict[str, np.ndarray],
        landmarks_dict: Dict[str, np.ndarray],
        fps: float
    ) -> np.ndarray:
        """Combina energía de efectores, energía angular y energía de rotación."""
        # Estimar nframes desde landmarks o ángulos
        nframes = 0
        if landmarks_dict:
            nframes = len(next(iter(landmarks_dict.values())))
        elif angles_dict:
            nframes = len(next(iter(angles_dict.values())))
        if nframes == 0:
            return np.zeros(0, dtype=np.float32)

        effector_energy = self._weighted_effector_energy(landmarks_dict, fps, nframes)
        angular_energy = self._taekwondo_angular_energy(angles_dict, fps, nframes)
        rotation_energy = self._rotation_energy(landmarks_dict, fps, nframes)

        energies = [effector_energy, angular_energy, rotation_energy]
        weights = [0.5, 0.3, 0.2]

        total = np.zeros(nframes, dtype=np.float32)
        for e, w in zip(energies, weights):
            if e.size == nframes and np.max(e) > 0:
                total += w * (e / (np.max(e) + 1e-6))

        return total

    def _weighted_effector_energy(
        self,
        landmarks_dict: Dict[str, np.ndarray],
        fps: float,
        nframes: int
    ) -> np.ndarray:
        """Energía de efectores (manos/pies) con distintos pesos."""
        effectors = ("L_WRIST", "R_WRIST", "L_ANK", "R_ANK")
        energies: List[np.ndarray] = []

        for eff in effectors:
            arr = landmarks_dict.get(eff)
            if arr is None or len(arr) < 2:
                continue
            # velocidad aproximada frame a frame
            movement = np.linalg.norm(
                np.diff(arr, axis=0, prepend=arr[0:1]), axis=1
            ) * fps

            if "WRIST" in eff:
                movement *= 1.2
            elif "ANK" in eff:
                movement *= 1.0

            energies.append(movement.astype(np.float32))

        if not energies:
            return np.zeros(nframes, dtype=np.float32)

        # max por frame entre efectores
        stacked = np.stack(energies, axis=0)
        return np.nanmax(stacked, axis=0)

    def _taekwondo_angular_energy(
        self,
        angles_dict: Dict[str, np.ndarray],
        fps: float,
        nframes: int
    ) -> np.ndarray:
        """Energía angular (codos, rodillas, caderas)."""
        if not angles_dict:
            return np.zeros(nframes, dtype=np.float32)

        grads = []
        priority = ["left_elbow", "right_elbow",
                    "left_knee", "right_knee",
                    "left_hip", "right_hip"]

        for name in priority:
            series = angles_dict.get(name)
            if series is None or len(series) < 2:
                continue
            g = np.abs(np.gradient(series)) * fps
            if "elbow" in name:
                g *= 1.3
            elif "knee" in name:
                g *= 1.5
            elif "hip" in name:
                g *= 1.0
            grads.append(g.astype(np.float32))

        if not grads:
            return np.zeros(nframes, dtype=np.float32)

        stacked = np.stack(grads, axis=0)
        # recortamos/extendemos si hiciera falta
        stacked = stacked[:, :nframes]
        return np.nanmean(stacked, axis=0)

    def _rotation_energy(
        self,
        landmarks_dict: Dict[str, np.ndarray],
        fps: float,
        nframes: int
    ) -> np.ndarray:
        """Energía de rotación del tronco (giros)."""
        l_sh = landmarks_dict.get("L_SH")
        r_sh = landmarks_dict.get("R_SH")
        if l_sh is None or r_sh is None or len(l_sh) < 2:
            return np.zeros(nframes, dtype=np.float32)

        shoulder_vec = r_sh - l_sh
        ori = np.arctan2(shoulder_vec[:, 1], shoulder_vec[:, 0])
        dori = np.abs(np.gradient(ori)) * fps * 180.0 / math.pi
        dori = dori.astype(np.float32)
        if dori.size < nframes:
            dori = np.pad(dori, (0, nframes - dori.size), mode="edge")
        else:
            dori = dori[:nframes]
        return dori

    # ------------ picos y segmentos --------------------------------------

    def _smooth_signal(self, sig: np.ndarray) -> np.ndarray:
        if sig.size == 0 or self.smooth_win <= 1:
            return sig
        win = min(self.smooth_win, sig.size)
        kernel = np.ones(win, dtype=np.float32) / float(win)
        return np.convolve(sig, kernel, mode="same")

    def _find_poomsae_peaks(self, energy_signal: np.ndarray) -> List[int]:
        """Detecta picos de energía que correspondan a movimientos."""
        if energy_signal.size < 3:
            return []

        smoothed = self._smooth_signal(energy_signal)

        q75 = np.percentile(smoothed, 75)
        q25 = np.percentile(smoothed, 25)
        dyn_thr = q25 + 0.5 * (q75 - q25)

        height_thr = max(self.peak_thresh, float(dyn_thr))

        peaks, _ = sp_signal.find_peaks(
            smoothed,
            height=height_thr,
            distance=self.min_peak_dist,
            prominence=0.08,
        )
        return peaks.astype(int).tolist()

    def _expand_segment_around_peak(
        self,
        energy_signal: np.ndarray,
        peak_idx: int,
        total_frames: int
    ) -> Tuple[int, int]:
        """Expande un pico a un segmento continuo de actividad."""
        start = peak_idx
        while start > 0 and energy_signal[start] > self.activity_thresh:
            start -= 1
        start = max(0, start)

        end = peak_idx
        while end < total_frames - 1 and energy_signal[end] > self.activity_thresh:
            end += 1
        end = min(total_frames - 1, end)

        if (end - start) < self.min_segment_frames:
            need = self.min_segment_frames - (end - start)
            extra_left = need // 2
            extra_right = need - extra_left
            start = max(0, start - extra_left)
            end = min(total_frames - 1, end + extra_right)

        return start, end

    def _merge_segments(self, segments: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Une segmentos muy cercanos (pausas cortas)."""
        if not segments:
            return []
        segments = sorted(segments, key=lambda x: x[0])
        merged = [segments[0]]

        for s, e in segments[1:]:
            ps, pe = merged[-1]
            if (s - pe - 1) <= self.max_pause_frames:
                merged[-1] = (ps, max(pe, e))
            else:
                merged.append((s, e))
        return merged

    # ------------ API pública del segmentador ----------------------------

    def find_segments(
        self,
        angles_dict: Dict[str, np.ndarray],
        landmarks_dict: Dict[str, np.ndarray],
        fps: float
    ) -> List[Tuple[int, int]]:
        """Devuelve lista de (frame_inicio, frame_fin) para cada movimiento."""
        if not angles_dict and not landmarks_dict:
            return []

        energy = self._compute_poomsae_aware_energy(angles_dict, landmarks_dict, fps)
        if energy.size == 0:
            return []

        peaks = self._find_poomsae_peaks(energy)

        segments: List[Tuple[int, int]] = []
        total_frames = energy.size

        for pk in peaks:
            a, b = self._expand_segment_around_peak(energy, pk, total_frames)
            if a >= b:
                continue
            # evitar superposición fuerte
            overlap = any(not (b < s2 or a > e2) for (s2, e2) in segments)
            if not overlap and (b - a) >= self.min_segment_frames:
                segments.append((a, b))

        merged = self._merge_segments(segments)
        # Mensaje para registro: número de segmentos detectados y sugerencia
        print(
            f"[SEGMENTER] Detectados {len(merged)} segmentos "
            f"(esperados ≈ {self.expected_segments}). Revise visualmente si es necesario."
        )
        return merged


# =====================================================================
# UTILIDADES NUMÉRICAS
# =====================================================================

def movavg(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1 or x.size == 0:
        return x
    win = min(win, len(x))
    c = np.cumsum(np.insert(x, 0, 0.0, axis=0), axis=0)
    y = (c[win:] - c[:-win]) / float(win)
    pad_l = win // 2
    pad_r = len(x) - len(y) - pad_l
    return np.pad(y, ((pad_l, pad_r), (0, 0)), mode="edge")


def central_diff(x: np.ndarray, fps: float) -> np.ndarray:
    if len(x) < 3:
        return np.zeros_like(x)
    dx = np.zeros_like(x)
    dx[1:-1] = (x[2:] - x[:-2]) * (fps / 2.0)
    dx[0] = (x[1] - x[0]) * fps
    dx[-1] = (x[-1] - x[-2]) * fps
    return dx


def unwrap_deg(a: np.ndarray) -> np.ndarray:
    out = a.copy()
    for i in range(1, len(out)):
        d = out[i] - out[i - 1]
        if d > 180:
            out[i:] -= 360
        elif d < -180:
            out[i:] += 360
    return out


def angle3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    BA = a - b
    BC = c - b
    num = float(np.dot(BA, BC))
    den = float(np.linalg.norm(BA) * np.linalg.norm(BC) + 1e-9)
    cosv = max(-1.0, min(1.0, num / den))
    return math.degrees(math.acos(cosv))


def bucket_turn(delta_deg: float) -> str:
    """Cuantiza un giro a categorías discretas."""
    d = ((delta_deg + 180) % 360) - 180
    opts = [
        (-180, "TURN_180"),
        (-135, "LEFT_135"),
        (-90, "LEFT_90"),
        (-45, "LEFT_45"),
        (0, "STRAIGHT"),
        (45, "RIGHT_45"),
        (90, "RIGHT_90"),
        (135, "RIGHT_135"),
        (180, "TURN_180"),
    ]
    return min(opts, key=lambda t: abs(d - t[0]))[1]


def dir_octant(vx: float, vy: float) -> str:
    """Dirección cardinal en 8 octantes según vector (vx,vy)."""
    ang = math.degrees(math.atan2(vy, vx))
    dirs = [
        ("E", -22.5, 22.5),
        ("SE", 22.5, 67.5),
        ("S", 67.5, 112.5),
        ("SW", 112.5, 157.5),
        ("W", 157.5, 180.0),
        ("W", -180.0, -157.5),
        ("NW", -157.5, -112.5),
        ("N", -112.5, -67.5),
        ("NE", -67.5, -22.5),
    ]
    for lab, a, b in dirs:
        if a <= ang <= b:
            return lab
    return "E"


# =====================================================================
# CARGA DE LANDMARKS DESDE CSV
# =====================================================================

def load_landmarks_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    req = {"video_id", "frame", "lmk_id", "x", "y"}
    if not req.issubset(df.columns):
        raise RuntimeError(f"CSV inválido, faltan columnas: {req - set(df.columns)}")
    return df


def series_xy(df: pd.DataFrame, lmk_id: int, nframes: int) -> np.ndarray:
    """
    Devuelve serie (nframes,2) para un landmark, con ffill + bfill y
    centro de imagen (0.5,0.5) si nunca apareció.
    """
    sub = df[df["lmk_id"] == lmk_id][["frame", "x", "y"]]
    arr = np.full((nframes, 2), np.nan, dtype=np.float32)

    if len(sub) > 0:
        idx = sub["frame"].to_numpy(dtype=int)
        xy = sub[["x", "y"]].to_numpy(np.float32)
        idx = np.clip(idx, 0, nframes - 1)
        arr[idx] = xy

    for k in range(2):
        v = arr[:, k]
        if np.isnan(v).any():
            last_valid = None
            for i in range(len(v)):
                if not np.isnan(v[i]):
                    last_valid = v[i]
                elif last_valid is not None:
                    v[i] = last_valid

            last_valid = None
            for i in range(len(v) - 1, -1, -1):
                if not np.isnan(v[i]):
                    last_valid = v[i]
                elif last_valid is not None:
                    v[i] = last_valid

            arr[:, k] = v

    arr = np.nan_to_num(arr, nan=0.5)
    return arr


# =====================================================================
# CLASIFICADOR DE POSTURA / PATADA (usa spec + RF opcional)
# =====================================================================

class PoseClassifier:
    """
    Clasificación de postura y tipo de patada:

      - Heurística pura (default)
      - o RF (RandomForest) entrenado con stance_features, si use_ml=True
    """

    def __init__(
        self,
        config: PoomsaeConfig,
        use_ml: bool = False,
        ml_model_path: Optional[Path] = None,
    ):
        self.config = config
        self.pose_spec = config.pose_spec or {}
        self.use_ml = use_ml
        self.ml_classifier = None

        if use_ml and ml_model_path:
            try:
                from src.ml.stance_classifier import StanceClassifier
                self.ml_classifier = StanceClassifier.load(Path(ml_model_path))
                print(f"[CLASSIFIER] ✅ Modelo ML cargado: {ml_model_path}")
            except Exception as e:
                print(f"[CLASSIFIER] ⚠️ Error cargando modelo ML, usando heurístico: {e}")
                self.use_ml = False

    # ---------- entrada pública ------------------------------------------

    def classify_stance(
        self,
        landmarks_dict: Dict[str, np.ndarray],
        frame_idx: int,
        debug: bool = False,
    ) -> str:
        if self.use_ml and self.ml_classifier is not None:
            return self._classify_with_ml(landmarks_dict, frame_idx, debug)
        return self._classify_heuristic(landmarks_dict, frame_idx, debug)

    def classify_kick(
        self,
        ankle_xy: np.ndarray,
        hip_xy: np.ndarray,
        kick_type: str | None = None,
    ) -> str:
        """Clasifica patada en base a altura relativa (spec pose_spec si existe)."""
        if len(ankle_xy) < 8:
            return "none"

        rel_height = hip_xy[:, 1] - ankle_xy[:, 1]
        max_h = float(np.max(rel_height))

        kicks_cfg = (self.pose_spec.get("kicks") or {}) if self.pose_spec else {}

        if kicks_cfg:
            if kick_type and kick_type in kicks_cfg:
                spec = kicks_cfg[kick_type]
                peak_min = float(spec.get("peak_above_hip_min", 0.25))
                if max_h >= peak_min:
                    return kick_type
            else:
                ap_spec = kicks_cfg.get("ap_chagi", {})
                ap_min = float(ap_spec.get("peak_above_hip_min", 0.25))
                if max_h >= ap_min:
                    return "ap_chagi"
                elif max_h >= 0.15:
                    return "arae_chagi"

        # fallback simple
        if max_h > 0.15:
            return "ap_chagi"
        elif max_h > 0.08:
            return "arae_chagi"
        return "none"

    # ---------- RandomForest ---------------------------------------------

    def _classify_with_ml(
        self,
        landmarks_dict: Dict[str, np.ndarray],
        frame_idx: int,
        debug: bool = False,
    ) -> str:
        """
        Clasificación de postura usando el modelo ML entrenado (StanceClassifier).

        Usa extract_stance_features_from_dict(...) para construir un dict de
        features y delega la alineación de columnas a self.ml_classifier.predict_from_dict().
        """
        try:
            from src.features.feature_extractor import (
                extract_stance_features_from_dict,
            )

            # Verificar que los landmarks mínimos existan
            required = [
                "L_ANK", "R_ANK", "L_KNEE", "R_KNEE",
                "L_HIP", "R_HIP", "L_SH", "R_SH",
                "L_HEEL", "R_HEEL", "L_FOOT", "R_FOOT",
            ]
            for name in required:
                if name not in landmarks_dict:
                    if debug:
                        print(f"[ML] Frame {frame_idx}: falta landmark {name}, uso heurístico")
                    return self._classify_heuristic(landmarks_dict, frame_idx, debug)

            # Extraer features geométricas para ESTE frame
            feats = extract_stance_features_from_dict(landmarks_dict, frame_idx)

            # Si hay NaN, mejor no forzar el modelo
            if any(np.isnan(v) for v in feats.values()):
                if debug:
                    print(f"[ML] Frame {frame_idx}: NaN en features, uso heurístico")
                return self._classify_heuristic(landmarks_dict, frame_idx, debug)

            # Delega en el helper del StanceClassifier (se encarga del orden de columnas)
            pred = self.ml_classifier.predict_from_dict(feats)

            if debug:
                # Aquí podríamos reconstruir X para obtener proba, pero no es crítico.
                print(f"[ML] Frame {frame_idx}: {pred}")

            return pred

        except Exception as e:
            if debug:
                print(f"[ML] Frame {frame_idx}: error ML {e}, uso heurístico")
            return self._classify_heuristic(landmarks_dict, frame_idx, debug)

    # ---------- Heurístico base ------------------------------------------

    def _classify_heuristic(
        self,
        landmarks_dict: Dict[str, np.ndarray],
        frame_idx: int,
        debug: bool = False,
    ) -> str:
        try:
            lank = landmarks_dict["L_ANK"][frame_idx]
            rank = landmarks_dict["R_ANK"][frame_idx]
            lkne = landmarks_dict["L_KNEE"][frame_idx]
            rkne = landmarks_dict["R_KNEE"][frame_idx]
            lhip = landmarks_dict["L_HIP"][frame_idx]
            rhip = landmarks_dict["R_HIP"][frame_idx]
            lsh = landmarks_dict["L_SH"][frame_idx]
            rsh = landmarks_dict["R_SH"][frame_idx]

            points = [lank, rank, lkne, rkne, lhip, rhip, lsh, rsh]
            if any(np.any(np.isnan(p)) for p in points):
                if debug:
                    print(f"[CLASSIFIER] Frame {frame_idx}: NaN en landmarks → moa_seogi")
                return "moa_seogi"

            shoulder_w = float(np.linalg.norm(rsh - lsh))
            if shoulder_w < 0.02:
                if debug:
                    print(f"[CLASSIFIER] Frame {frame_idx}: hombros casi colapsados ({shoulder_w:.4f})")
                return "moa_seogi"

            foot_dist = float(np.linalg.norm(lank - rank) / (shoulder_w + 1e-6))
            l_knee = angle3(lhip, lkne, lank)
            r_knee = angle3(rhip, rkne, rank)

            if debug:
                print(
                    f"[CLASSIFIER] Frame {frame_idx}: "
                    f"foot_dist={foot_dist:.2f}, l_knee={l_knee:.1f}, r_knee={r_knee:.1f}"
                )

            if self._check_ap_kubi(foot_dist, l_knee, r_knee, debug):
                stance = "ap_kubi"
            elif self._check_dwit_kubi(foot_dist, l_knee, r_knee, debug):
                stance = "dwit_kubi"
            elif self._check_beom_seogi(foot_dist, debug):
                stance = "beom_seogi"
            else:
                stance = "moa_seogi"

            if debug:
                print(f"[CLASSIFIER] Frame {frame_idx}: → {stance}")
            return stance

        except Exception as e:
            if debug:
                print(f"[CLASSIFIER] Error en heurístico frame {frame_idx}: {e}")
            return "moa_seogi"

    # ----- reglas simplificadas en base a tu análisis --------------------

    def _check_ap_kubi(self, foot_dist: float, l_knee: float, r_knee: float, debug=False) -> bool:
        result = foot_dist >= 2.75
        if debug:
            print(f"   ap_kubi: dist >= 2.75 → {result}")
        return result

    def _check_dwit_kubi(self, foot_dist: float, l_knee: float, r_knee: float, debug=False) -> bool:
        result = 0.5 <= foot_dist < 2.75
        if debug:
            print(f"   dwit_kubi: 0.5 <= dist < 2.75 → {result}")
        return result

    def _check_beom_seogi(self, foot_dist: float, debug=False) -> bool:
        result = foot_dist < 0.5
        if debug:
            print(f"   beom_seogi: dist < 0.5 → {result}")
        return result


# =====================================================================
# DATACLASSES DE SALIDA
# =====================================================================

@dataclass
class MoveSegment:
    idx: int
    a: int
    b: int
    t_start: float
    t_end: float
    duration: float
    active_limb: str
    speed_peak: float
    rotation_deg: float
    rotation_bucket: str
    height_end: float
    path: List[Tuple[float, float]]
    stance_pred: str
    kick_pred: str
    arm_dir: str
    pose_f: int
    pose_t: float
    expected_tech: str = ""
    expected_stance: str = ""
    expected_turn: str = ""
    match_score: float = 0.0


@dataclass
class CaptureResult:
    video_id: str
    fps: float
    nframes: int
    moves: List[MoveSegment]
    poomsae_spec: Dict[str, Any] | None = None

    def to_json(self) -> str:
        def conv(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: conv(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [conv(x) for x in obj]
            return obj

        data = {
            "video_id": self.video_id,
            "fps": float(self.fps),
            "nframes": int(self.nframes),
            "moves": [conv(asdict(m)) for m in self.moves],
        }
        if self.poomsae_spec:
            data["poomsae_spec"] = conv(self.poomsae_spec)
        return json.dumps(data, ensure_ascii=False, indent=2)


# =====================================================================
# CORE: CAPTURE_MOVES CON SPEC DE 8YANG
# =====================================================================

def _heading_series_deg(lsh: np.ndarray, rsh: np.ndarray) -> np.ndarray:
    sh_vec = rsh - lsh
    heading = np.degrees(np.arctan2(sh_vec[:, 1], sh_vec[:, 0]))
    return unwrap_deg(heading.astype(np.float32))


def _choose_active_limb(
    eff_xy: Dict[str, np.ndarray],
    a: int,
    b: int
) -> str:
    def _disp(xy: np.ndarray) -> float:
        if len(xy) < 2:
            return 0.0
        d = np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1))
        return float(np.nansum(d))

    cand = {k: eff_xy[k][a:b + 1] for k in END_EFFECTORS}
    return max(cand.items(), key=lambda kv: _disp(kv[1]))[0]


def _choose_pose_frame(smax: np.ndarray, a: int, b: int) -> int:
    if b <= a:
        return a
    lo = a + int(0.6 * (b - a))
    lo = min(max(lo, a), b)
    sub = smax[lo:b + 1]
    if sub.size == 0 or not np.isfinite(sub).any():
        return b
    off = int(np.nanargmin(sub))
    return lo + off


def arm_motion_direction(wrist_xy: np.ndarray) -> str:
    if len(wrist_xy) < 2:
        return "NONE"
    v = wrist_xy[-1] - wrist_xy[0]
    return dir_octant(float(v[0]), float(v[1]))


def _calculate_match_score(
    stance_actual: str,
    kick_actual: str,
    turn_actual: str,
    stance_expected: str,
    turn_expected: str,
    move_spec: Dict[str, Any] | None,
) -> float:
    score = 0.0
    factors = 0

    if stance_expected:
        factors += 1
        if stance_actual == stance_expected:
            score += 1.0
        elif stance_actual in ("ap_kubi", "dwit_kubi", "beom_seogi"):
            score += 0.5

    if turn_expected:
        factors += 1
        if turn_actual == turn_expected:
            score += 1.0
        elif turn_expected != "NONE" and turn_actual != "STRAIGHT":
            score += 0.7

    if move_spec and move_spec.get("category") == "KICK":
        factors += 1
        expected_kick = move_spec.get("kick_type")
        if kick_actual == expected_kick:
            score += 1.0
        elif kick_actual != "none":
            score += 0.5

    return score / factors if factors > 0 else 0.0


def resample_polyline(xy: np.ndarray, n: int) -> np.ndarray:
    if len(xy) == 0:
        return np.zeros((0, 2), np.float32)
    if len(xy) == 1:
        return np.tile(xy[:1], (n, 1))
    d = np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1))
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] <= 1e-8:
        return np.tile(xy[:1], (n, 1))
    t = np.linspace(0, s[-1], n)
    xs = np.interp(t, s, xy[:, 0])
    ys = np.interp(t, s, xy[:, 1])
    return np.stack([xs, ys], axis=1).astype(np.float32)


def capture_moves_with_spec(
    csv_path: Path,
    video_path: Optional[Path] = None,
    config_path: Path = Path("config/default.yaml"),
    spec_path: Path = Path("config/patterns/8yang_spec.json"),
    pose_spec_path: Path = Path("config/patterns/pose_spec.json"),
    use_ml_classifier: bool = False,
    ml_model_path: Optional[Path] = None,
) -> CaptureResult:
    """
    Pipeline principal:
      1) Lee CSV de landmarks (MediaPipe)
      2) Carga configs (default.yaml + specs JSON)
      3) Construye series 2D por landmark
      4) Calcula ángulos rodilla/codo/cadera
      5) Segmenta usando SpecAwareSegmenter
      6) Para cada segmento, estima:
         - extremidad activa, trayectoria, velocidad, giro, patada, postura,
           score de match con la especificación 8yang.
    """
    csv_path = Path(csv_path)
    video_path = Path(video_path) if video_path else None

    config = PoomsaeConfig(spec_path, pose_spec_path, config_path)
    df = load_landmarks_csv(csv_path)
    nframes = int(df["frame"].max()) + 1

    fps = 30.0
    if video_path and video_path.exists():
        cap = cv2.VideoCapture(str(video_path))
        fps_read = cap.get(cv2.CAP_PROP_FPS)
        if fps_read and fps_read > 0:
            fps = float(fps_read)
        cap.release()

    print(f"[CAPTURE] Procesando {csv_path.stem} (frames={nframes}, fps={fps:.1f})")

    landmarks_dict: Dict[str, np.ndarray] = {}
    needed_points = [
        "L_SH", "R_SH", "L_HIP", "R_HIP",
        "L_WRIST", "R_WRIST", "L_ANK", "R_ANK",
        "L_KNEE", "R_KNEE", "L_ELB", "R_ELB",
        "L_HEEL", "R_HEEL", "L_FOOT", "R_FOOT",
    ]
    for name in needed_points:
        idx = LMK.get(name)
        if idx is not None:
            landmarks_dict[name] = series_xy(df, idx, nframes)

    # ---- ángulos para energía -------------------------------------------
    angles_dict: Dict[str, np.ndarray] = {}
    try:
        l_hip = landmarks_dict["L_HIP"]
        r_hip = landmarks_dict["R_HIP"]
        l_knee = landmarks_dict["L_KNEE"]
        r_knee = landmarks_dict["R_KNEE"]
        l_ank = landmarks_dict["L_ANK"]
        r_ank = landmarks_dict["R_ANK"]

        min_len = min(len(l_hip), len(r_hip), len(l_knee), len(r_knee), len(l_ank), len(r_ank))

        left_knee_angles = []
        right_knee_angles = []
        for i in range(min_len):
            left_knee_angles.append(angle3(l_hip[i], l_knee[i], l_ank[i]))
            right_knee_angles.append(angle3(r_hip[i], r_knee[i], r_ank[i]))
        angles_dict["left_knee"] = np.array(left_knee_angles, dtype=np.float32)
        angles_dict["right_knee"] = np.array(right_knee_angles, dtype=np.float32)

        l_sh = landmarks_dict["L_SH"]
        r_sh = landmarks_dict["R_SH"]
        l_elb = landmarks_dict["L_ELB"]
        r_elb = landmarks_dict["R_ELB"]
        l_wri = landmarks_dict["L_WRIST"]
        r_wri = landmarks_dict["R_WRIST"]

        min_len_elb = min(len(l_sh), len(r_sh), len(l_elb), len(r_elb), len(l_wri), len(r_wri))
        left_elb_angles = []
        right_elb_angles = []
        for i in range(min_len_elb):
            left_elb_angles.append(angle3(l_sh[i], l_elb[i], l_wri[i]))
            right_elb_angles.append(angle3(r_sh[i], r_elb[i], r_wri[i]))
        angles_dict["left_elbow"] = np.array(left_elb_angles, dtype=np.float32)
        angles_dict["right_elbow"] = np.array(right_elb_angles, dtype=np.float32)

        left_hip_angles = []
        right_hip_angles = []
        for i in range(min_len):
            left_hip_angles.append(angle3(l_sh[i], l_hip[i], l_knee[i]))
            right_hip_angles.append(angle3(r_sh[i], r_hip[i], r_knee[i]))
        angles_dict["left_hip"] = np.array(left_hip_angles, dtype=np.float32)
        angles_dict["right_hip"] = np.array(right_hip_angles, dtype=np.float32)

    except Exception as e:
        print(f"[CAPTURE] ⚠️ Error calculando ángulos: {e}")

    segmenter = SpecAwareSegmenter(config)
    classifier = PoseClassifier(config, use_ml=use_ml_classifier, ml_model_path=ml_model_path)

    segments = segmenter.find_segments(angles_dict, landmarks_dict, fps)

    seg_cfg = (config.app_config or {}).get("segmentation", {})
    max_duration = float(seg_cfg.get("max_duration", 2.0))
    max_frames = int(max_duration * fps)
    filtered = [(a, b) for (a, b) in segments if (b - a) <= max_frames]

    print(f"[CAPTURE] Segmentos tras filtro duración (<= {max_duration:.2f}s): {len(filtered)}")

    lsh = landmarks_dict["L_SH"]
    rsh = landmarks_dict["R_SH"]
    lwri = landmarks_dict["L_WRIST"]
    rwri = landmarks_dict["R_WRIST"]
    lank = landmarks_dict["L_ANK"]
    rank = landmarks_dict["R_ANK"]
    lhip = landmarks_dict["L_HIP"]
    rhip = landmarks_dict["R_HIP"]

    eff_xy = {"L_WRIST": lwri, "R_WRIST": rwri, "L_ANK": lank, "R_ANK": rank}
    heading = _heading_series_deg(lsh, rsh)

    # smax global para elegir frame estable de pose
    vel_all_eff = []
    for k in END_EFFECTORS:
        vel_all_eff.append(
            np.linalg.norm(central_diff(eff_xy[k], fps), axis=1)
        )
    smax_global = np.nanmax(np.stack(vel_all_eff, axis=0), axis=0)

    moves: List[MoveSegment] = []

    for i, (a, b) in enumerate(filtered, start=1):
        if a >= b or a >= nframes or b >= nframes:
            continue

        limb = _choose_active_limb(eff_xy, a, b)

        move_spec = config.get_move_spec(i)
        expected_tech = move_spec.get("tech_es", "") if move_spec else ""
        expected_stance = move_spec.get("stance_expect", "") if move_spec else ""
        expected_turn = move_spec.get("turn_expect", "") if move_spec else ""

        path_xy = eff_xy[limb][a:b + 1, :]
        path_resampled = resample_polyline(path_xy, 20)

        seg_speeds = {
            k: np.linalg.norm(
                central_diff(movavg(eff_xy[k][a:b + 1], 3), fps), axis=1
            )
            for k in END_EFFECTORS
        }
        v_peak = float(
            max(np.nanmax(v) if v.size > 0 else 0.0 for v in seg_speeds.values())
        )

        rot_deg = float(heading[b] - heading[a]) if heading.size > b else 0.0
        rot_bucket = bucket_turn(rot_deg)

        height_end = float(path_xy[-1, 1]) if path_xy.size > 0 else 0.0

        pose_f = _choose_pose_frame(smax_global, a, b)
        pose_f = int(np.clip(pose_f, 0, nframes - 1))

        stance_pred = classifier.classify_stance(landmarks_dict, pose_f)

        hip_xy = 0.5 * (lhip + rhip)
        kick_type = move_spec.get("kick_type") if move_spec else None
        kick_pred = classifier.classify_kick(
            eff_xy[limb][a:b + 1], hip_xy[a:b + 1], kick_type
        )

        arm_dir = (
            arm_motion_direction(eff_xy[limb][a:b + 1])
            if limb in ("L_WRIST", "R_WRIST")
            else "NONE"
        )

        match_score = _calculate_match_score(
            stance_pred,
            kick_pred,
            rot_bucket,
            expected_stance,
            expected_turn,
            move_spec,
        )

        moves.append(
            MoveSegment(
                idx=i,
                a=a,
                b=b,
                t_start=a / fps,
                t_end=b / fps,
                duration=(b - a) / fps,
                active_limb=limb,
                speed_peak=v_peak,
                rotation_deg=rot_deg,
                rotation_bucket=rot_bucket,
                height_end=height_end,
                path=[(float(x), float(y)) for x, y in path_resampled],
                stance_pred=stance_pred,
                kick_pred=kick_pred,
                arm_dir=arm_dir,
                pose_f=pose_f,
                pose_t=pose_f / fps,
                expected_tech=expected_tech,
                expected_stance=expected_stance,
                expected_turn=expected_turn,
                match_score=match_score,
            )
        )

    print(f"[CAPTURE] Movimientos finales con spec 8yang: {len(moves)}")

    return CaptureResult(
        video_id=csv_path.stem,
        fps=fps,
        nframes=nframes,
        moves=moves,
        poomsae_spec=config.poomsae_spec,
    )


# =====================================================================
# FUNCIONES LEGACY (BACKWARD COMPATIBLE)
# =====================================================================

def capture_moves_enhanced(
    csv_path: Path,
    video_path: Optional[Path] = None,
    expected_movements: int = 24,
    min_duration: float = 0.2,
    max_duration: float = 2.0,
    sensitivity: float = 0.9,
) -> CaptureResult:
    """
    API legacy antigua: ahora redirige a capture_moves_with_spec
    manteniendo firma para compatibilidad.
    """
    print("[LEGACY] Usando sistema con especificaciones (8yang)")
    return capture_moves_with_spec(csv_path, video_path)


def capture_moves_from_csv(
    csv_path: Path,
    *,
    video_path: Optional[Path],
    vstart: float = 0.40,
    vstop: float = 0.15,
    min_dur: float = 0.20,
    min_gap: float = 0.15,
    smooth_win: int = 3,
    poly_n: int = 20,
    min_path_norm: float = 0.015,
    expected_n: Optional[int] = None,
) -> CaptureResult:
    """
    API legacy simplificada: ahora delega en capture_moves_with_spec.
    """
    print("[LEGACY] Redirigiendo a sistema con especificaciones (8yang)")
    return capture_moves_with_spec(csv_path, video_path)
