# src/segmentation/segmenter.py
from __future__ import annotations
from typing import Dict, List, Tuple, Any
import numpy as np
import scipy.signal as signal


class EnhancedSegmenter:
    """Segmentador mejorado orientado a Poomsae.

    Diseñado para ser legible y fácil de mantener: los comentarios y
    nombres de métodos explican la intención del código. Combina
    diferentes fuentes de 'energía' (velocidad/ángulos/orientación)
    para detectar ventanas temporales que correspondan a movimientos.
    """

    def __init__(
        self,
        min_segment_frames: int = 8,      # movimientos rápidos => ventanas cortas
        max_pause_frames: int = 5,
        activity_threshold: float = 0.15, # (no se usa directo, pero lo dejamos para tuning futuro)
        peak_threshold: float = 0.3,
        min_peak_distance: int = 10,      # evitar picos demasiado juntos
        smooth_window: int = 5,
    ) -> None:
        self.minlen = int(min_segment_frames)
        self.maxpause = int(max_pause_frames)
        self.activity_thresh = float(activity_threshold)
        self.peak_thresh = float(peak_threshold)
        self.min_peak_dist = int(min_peak_distance)
        self.smooth_win = int(smooth_window)

    # -------------------------------------------------------------------------
    # Helpers internos
    # -------------------------------------------------------------------------

    def _infer_length(
        self,
        angles_dict: Dict[str, np.ndarray],
        landmarks_dict: Dict[str, np.ndarray],
    ) -> int:
        """Intenta inferir el número de frames a partir de las series disponibles."""
        cand: List[int] = []
        for v in angles_dict.values():
            if isinstance(v, np.ndarray) and v.ndim >= 1:
                cand.append(len(v))
        for v in landmarks_dict.values():
            if isinstance(v, np.ndarray) and v.ndim >= 1:
                cand.append(len(v))
        return max(cand) if cand else 0

    # -------------------------------------------------------------------------
    # Componentes de energía
    # -------------------------------------------------------------------------

    def _compute_comprehensive_energy(
        self,
        angles_dict: Dict[str, np.ndarray],
        landmarks_dict: Dict[str, np.ndarray],
        fps: float,
    ) -> np.ndarray:
        """
        Calcula energía combinando múltiples fuentes:
        1. Cambios angulares (articulaciones)
        2. Velocidad de efectores (manos/pies)
        3. Movimiento de tronco (hombros/caderas)
        4. Cambios de orientación (giros)
        """
        total_len = self._infer_length(angles_dict, landmarks_dict)
        if total_len <= 0:
            return np.zeros(0, dtype=np.float32)

        energies: List[np.ndarray] = []

        # 1) Energía angular
        angular_energy = self._angular_energy(angles_dict, fps, total_len)
        energies.append(angular_energy)

        # 2) Energía de efectores
        effector_energy = self._effector_energy(landmarks_dict, fps, total_len)
        energies.append(effector_energy)

        # 3) Energía de tronco
        trunk_energy = self._trunk_energy(landmarks_dict, fps, total_len)
        energies.append(trunk_energy)

        # 4) Energía de orientación
        orientation_energy = self._orientation_energy(landmarks_dict, fps, total_len)
        energies.append(orientation_energy)

        # Combinar con pesos
        weights = [0.3, 0.4, 0.2, 0.1]  # efectores más importantes
        total_energy = np.zeros(total_len, dtype=np.float32)

        for energy, weight in zip(energies, weights):
            if len(energy) != total_len:
                continue
            if np.max(energy) > 0:
                energy_norm = energy / np.max(energy)
                total_energy += float(weight) * energy_norm.astype(np.float32)

        return total_energy

    def _angular_energy(
        self,
        angles_dict: Dict[str, np.ndarray],
        fps: float,
        total_len: int,
    ) -> np.ndarray:
        """Energía basada en cambios angulares."""
        if not angles_dict:
            return np.zeros(total_len, dtype=np.float32)

        grads: List[np.ndarray] = []
        for angle_series in angles_dict.values():
            arr = np.asarray(angle_series, dtype=np.float32)
            if arr.size > 1:
                g = np.abs(np.gradient(arr)) * float(fps)
                grads.append(g)

        if not grads:
            return np.zeros(total_len, dtype=np.float32)

        # Alineamos longitudes y promediamos
        min_len = min(len(g) for g in grads)
        if min_len <= 0:
            return np.zeros(total_len, dtype=np.float32)
        stacked = np.stack([g[:min_len] for g in grads], axis=0)
        energy = np.mean(stacked, axis=0)

        # Ajustar a total_len
        if len(energy) < total_len:
            pad = np.zeros(total_len - len(energy), dtype=energy.dtype)
            energy = np.concatenate([energy, pad])
        else:
            energy = energy[:total_len]
        return energy

    def _effector_energy(
        self,
        landmarks_dict: Dict[str, np.ndarray],
        fps: float,
        total_len: int,
    ) -> np.ndarray:
        """Energía de movimiento de manos y pies."""
        effectors = ["L_WRIST", "R_WRIST", "L_ANKLE", "R_ANKLE"]
        energies: List[np.ndarray] = []

        for eff in effectors:
            arr = landmarks_dict.get(eff)
            if arr is None:
                continue
            pos = np.asarray(arr, dtype=np.float32)
            if pos.ndim < 2 or len(pos) <= 1:
                continue
            vel = np.linalg.norm(np.gradient(pos, axis=0), axis=1) * float(fps)
            energies.append(vel)

        if not energies:
            return np.zeros(total_len, dtype=np.float32)

        # Alineamos longitudes y tomamos el máximo (movimiento más intenso)
        min_len = min(len(e) for e in energies)
        stacked = np.stack([e[:min_len] for e in energies], axis=0)
        energy = np.max(stacked, axis=0)

        if len(energy) < total_len:
            pad = np.zeros(total_len - len(energy), dtype=energy.dtype)
            energy = np.concatenate([energy, pad])
        else:
            energy = energy[:total_len]
        return energy

    def _trunk_energy(
        self,
        landmarks_dict: Dict[str, np.ndarray],
        fps: float,
        total_len: int,
    ) -> np.ndarray:
        """Energía de movimiento del tronco (hombros y caderas)."""
        trunk_points = ["L_SHOULDER", "R_SHOULDER", "L_HIP", "R_HIP"]
        energies: List[np.ndarray] = []

        for p in trunk_points:
            arr = landmarks_dict.get(p)
            if arr is None:
                continue
            pos = np.asarray(arr, dtype=np.float32)
            if pos.ndim < 2 or len(pos) <= 1:
                continue
            vel = np.linalg.norm(np.gradient(pos, axis=0), axis=1) * float(fps)
            energies.append(vel)

        if not energies:
            return np.zeros(total_len, dtype=np.float32)

        min_len = min(len(e) for e in energies)
        stacked = np.stack([e[:min_len] for e in energies], axis=0)
        energy = np.mean(stacked, axis=0)

        if len(energy) < total_len:
            pad = np.zeros(total_len - len(energy), dtype=energy.dtype)
            energy = np.concatenate([energy, pad])
        else:
            energy = energy[:total_len]
        return energy

    def _orientation_energy(
        self,
        landmarks_dict: Dict[str, np.ndarray],
        fps: float,
        total_len: int,
    ) -> np.ndarray:
        """Energía basada en cambios de orientación (giros)."""
        l_sh = landmarks_dict.get("L_SHOULDER")
        r_sh = landmarks_dict.get("R_SHOULDER")
        if l_sh is None or r_sh is None:
            return np.zeros(total_len, dtype=np.float32)

        l = np.asarray(l_sh, dtype=np.float32)
        r = np.asarray(r_sh, dtype=np.float32)
        if l.ndim < 2 or r.ndim < 2 or len(l) <= 1 or len(r) <= 1:
            return np.zeros(total_len, dtype=np.float32)

        n = min(len(l), len(r))
        l = l[:n]
        r = r[:n]

        shoulder_vec = r - l
        orientations = np.arctan2(shoulder_vec[:, 1], shoulder_vec[:, 0])
        orientation_change = np.abs(np.gradient(orientations)) * float(fps)

        energy = orientation_change.astype(np.float32)
        if len(energy) < total_len:
            pad = np.zeros(total_len - len(energy), dtype=energy.dtype)
            energy = np.concatenate([energy, pad])
        else:
            energy = energy[:total_len]
        return energy

    # -------------------------------------------------------------------------
    # Utilidades de señal
    # -------------------------------------------------------------------------

    def _smooth_signal(self, sig: np.ndarray) -> np.ndarray:
        """Suavizar señal para reducir ruido."""
        if len(sig) < self.smooth_win or self.smooth_win <= 1:
            return sig
        # Crear kernel de suavizado (media móvil simple)
        kernel_size = min(self.smooth_win, len(sig))
        kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
        return np.convolve(sig, kernel, mode="same")

    def _find_peaks_robust(self, energy_signal: np.ndarray) -> List[int]:
        """Encontrar picos robustos en la señal de energía."""
        if len(energy_signal) < 3:
            return []

        smoothed = self._smooth_signal(energy_signal)

        # Calcular percentiles y umbral adaptativo paso a paso
        p75 = float(np.percentile(smoothed, 75))
        p25 = float(np.percentile(smoothed, 25))
        dynamic_threshold = p25 + 0.35 * (p75 - p25)

        # Combinar umbral estático y dinámico
        effective_threshold = max(self.peak_thresh, dynamic_threshold * 0.5)

        min_prominence = max(0.05, effective_threshold * 0.3)

        peaks, _ = signal.find_peaks(
            smoothed,
            height=effective_threshold,
            distance=self.min_peak_dist,
            prominence=min_prominence,
        )

        return peaks.tolist()

    def _expand_segment_around_peak(
        self,
        energy_signal: np.ndarray,
        peak_idx: int,
        total_frames: int,
    ) -> Tuple[int, int]:
        """Expandir segmento alrededor de un pico."""
        if total_frames <= 0:
            return 0, 0

        q10 = float(np.percentile(energy_signal, 10))
        q50 = float(np.percentile(energy_signal, 50))
        local_thresh = q10 + 0.3 * (q50 - q10)

        start = int(peak_idx)
        while start > 0 and energy_signal[start] > local_thresh:
            start -= 1
        start = max(0, start)

        end = int(peak_idx)
        while end < total_frames - 1 and energy_signal[end] > local_thresh:
            end += 1
        end = min(total_frames - 1, end)

        # asegurar longitud mínima
        if (end - start) < self.minlen:
            need = self.minlen - (end - start)
            left_add = need // 2
            right_add = need - left_add
            start = max(0, start - left_add)
            end = min(total_frames - 1, end + right_add)

        return start, end

    # -------------------------------------------------------------------------
    # API pública
    # -------------------------------------------------------------------------

    def find_segments(
        self,
        angles_dict: Dict[str, np.ndarray],
        landmarks_dict: Dict[str, np.ndarray],
        fps: float,
    ) -> List[Tuple[int, int]]:
        """
        Encuentra segmentos usando energía integral.
        Devuelve una lista de (start_frame, end_frame) en índices de frame.
        """
        total_len = self._infer_length(angles_dict, landmarks_dict)
        if total_len <= 0:
            return []

        energy = self._compute_comprehensive_energy(angles_dict, landmarks_dict, fps)
        if energy.size == 0:
            return []

        peaks = self._find_peaks_robust(energy)

        segments: List[Tuple[int, int]] = []
        for peak in peaks:
            start, end = self._expand_segment_around_peak(energy, peak, len(energy))
            if (end - start) < self.minlen:
                continue

            # comprobar solapamientos con los ya existentes
            overlap = False
            for s0, s1 in segments:
                if not (end < s0 or start > s1):
                    overlap = True
                    break
            if not overlap:
                segments.append((start, end))

        # Ordenar por tiempo
        segments.sort(key=lambda x: x[0])

        # Fusionar segmentos muy cercanos
        merged: List[Tuple[int, int]] = []
        for seg in segments:
            if not merged:
                merged.append(seg)
            else:
                last_s, last_e = merged[-1]
                if seg[0] - last_e <= self.maxpause:
                    merged[-1] = (last_s, max(last_e, seg[1]))
                else:
                    merged.append(seg)

        print(f"[SEGMENTER] Detectados {len(merged)} segmentos")
        return merged

    def debug_energy(
        self,
        angles_dict: Dict[str, np.ndarray],
        landmarks_dict: Dict[str, np.ndarray],
        fps: float,
    ) -> Dict[str, Any]:
        """
        Devuelve un dict con componentes de energía y picos para depuración.
        """
        total_len = self._infer_length(angles_dict, landmarks_dict)

        out: Dict[str, Any] = {}
        out["angular"] = self._angular_energy(angles_dict, fps, total_len)
        out["effector"] = self._effector_energy(landmarks_dict, fps, total_len)
        out["trunk"] = self._trunk_energy(landmarks_dict, fps, total_len)
        out["orientation"] = self._orientation_energy(landmarks_dict, fps, total_len)
        out["total"] = self._compute_comprehensive_energy(angles_dict, landmarks_dict, fps)
        out["peaks"] = self._find_peaks_robust(out["total"])
        return out
