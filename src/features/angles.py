# src/features/angles.py
from __future__ import annotations
from typing import Mapping, Sequence, Dict, Any
import numpy as np


def _to_xy_array(lmks: Any) -> np.ndarray:
    """
    Convierte la estructura de landmarks (lista o array) a np.ndarray (N,2)
    usando solo (x,y). Asume orden MediaPipe Pose (33 puntos).
    """
    arr = np.asarray(lmks, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(
            f"Se esperaban landmarks con shape (N,>=2), recibido: {arr.shape}"
        )
    return arr[:, :2]


def _angle3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Ángulo en grados en el vértice b, dado por puntos a-b-c.

    Devuelve NaN si la geometría es degenerada (vectores casi nulos).
    """
    ba = a - b
    bc = c - b
    den = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if den < 1e-8:
        return float("nan")

    cosang = float(np.dot(ba, bc) / den)
    cosang = float(np.clip(cosang, -1.0, 1.0))
    ang = float(np.degrees(np.arccos(cosang)))
    return ang


def angles_from_landmarks(
    lmks: Any,
    triplets: Mapping[str, Sequence[int]],
) -> Dict[str, float]:
    """
    Calcula ángulos para un frame de landmarks.

    Args:
        lmks:
            Landmarks de un frame. Soporta:
               - np.ndarray (33,2/3/4)
               - lista de [x,y,(z),(vis)] por punto.
        triplets:
            Diccionario {nombre: (i,j,k)} donde j es el vértice del ángulo.

    Returns:
        Dict {nombre_ángulo: valor_en_grados}
    """
    xy = _to_xy_array(lmks)
    out: Dict[str, float] = {}

    for name, idxs in triplets.items():
        try:
            i, j, k = idxs
            a = xy[int(i)]
            b = xy[int(j)]
            c = xy[int(k)]
            out[name] = _angle3(a, b, c)
        except Exception:
            out[name] = float("nan")

    return out
