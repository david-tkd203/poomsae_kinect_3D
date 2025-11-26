# src/utils/smoothing.py
from __future__ import annotations
from typing import Union, Sequence
import numpy as np


def smooth(
    signal: Union[Sequence[float], np.ndarray],
    win: int = 5,
    poly: int = 2,   # se acepta para compatibilidad, pero no se usa
) -> np.ndarray:
    """
    Suavizado 1D simple (media móvil centrada) que ignora NaN.

    Args:
        signal:
            Serie 1D (lista o np.ndarray).
        win:
            Tamaño de ventana (impar recomendado). Si win<=1 o la serie
            es más corta que la ventana, se devuelve la señal original.
        poly:
            Parámetro placeholder para compatibilidad con firmas previas
            (p.ej. smooth(x, win=5, poly=2)). No se utiliza aquí.

    Returns:
        np.ndarray suavizado, misma longitud que `signal`.
    """
    x = np.asarray(signal, dtype=float).reshape(-1)
    n = x.size

    if win <= 1 or n < win:
        return x

    # Asegurar ventana impar
    if win % 2 == 0:
        win += 1

    half = win // 2
    out = np.empty_like(x)

    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        window = x[start:end]
        if np.all(np.isnan(window)):
            out[i] = np.nan
        else:
            out[i] = np.nanmean(window)

    return out
