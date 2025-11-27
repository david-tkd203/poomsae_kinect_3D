# src/utils/performance.py
from __future__ import annotations

import os
import platform
import ctypes
import psutil


def _boost_process_priority() -> None:
    """
    Sube la prioridad del proceso a HIGH_PRIORITY_CLASS en Windows.
    No hace overclock, pero le dice al scheduler que te dé más CPU
    frente a otras apps.
    """
    if platform.system() != "Windows":
        return

    try:
        p = psutil.Process()
        # En Windows psutil expone estas constantes
        p.nice(psutil.HIGH_PRIORITY_CLASS)
        print("[perf] Prioridad del proceso: HIGH_PRIORITY_CLASS")
    except Exception as e:
        print(f"[perf] No se pudo cambiar la prioridad del proceso: {e}")


def _boost_timer_resolution() -> None:
    """
    Pide resolución de temporizador de 1 ms en Windows.
    Mejora la precisión de QTimer / sleep, reduciendo un poco el lag.
    """
    if platform.system() != "Windows":
        return

    try:
        winmm = ctypes.WinDLL("winmm")
        res = winmm.timeBeginPeriod(1)
        if res == 0:
            print("[perf] Resolución de timer del sistema ajustada a 1 ms")
        else:
            print("[perf] timeBeginPeriod(1) devolvió código:", res)
    except Exception as e:
        print(f"[perf] No se pudo ajustar la resolución de timer: {e}")


def _configure_qt_opengl() -> None:
    """
    Fuerza a Qt a usar OpenGL de escritorio.
    (Debe llamarse ANTES de crear QApplication / importar widgets en algunos casos.)
    """
    # Si ya está seteada, no la pisamos
    if "QT_OPENGL" not in os.environ:
        os.environ["QT_OPENGL"] = "desktop"
        print("[perf] QT_OPENGL=desktop (OpenGL de escritorio)")


def setup_performance() -> None:
    """
    Llamar una vez al inicio de la aplicación.
    - Configura Qt para usar OpenGL de escritorio (aprovecha la GPU).
    - Sube prioridad del proceso en Windows.
    - Ajusta la resolución del temporizador a 1 ms en Windows.
    """
    _configure_qt_opengl()
    _boost_process_priority()
    _boost_timer_resolution()
