# src/utils/performance.py
from __future__ import annotations

import os
import platform
import ctypes
import psutil

# Opcional: intentar detectar OpenCV/CUDA
try:
    import cv2
except Exception:
    cv2 = None


def _boost_process_priority() -> None:
    """
    Sube la prioridad del proceso a HIGH_PRIORITY_CLASS en Windows.
    """
    if platform.system() != "Windows":
        return

    try:
        p = psutil.Process()
        p.nice(psutil.HIGH_PRIORITY_CLASS)
        print("[perf] Prioridad del proceso: HIGH_PRIORITY_CLASS")
    except Exception as e:
        print(f"[perf] No se pudo cambiar la prioridad del proceso: {e}")


def _boost_timer_resolution() -> None:
    """
    Pide resolución de temporizador de 1 ms en Windows.
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
    Fuerza a Qt a usar OpenGL de escritorio (aprovecha GPU para la parte gráfica).
    """
    if "QT_OPENGL" not in os.environ:
        os.environ["QT_OPENGL"] = "desktop"
        print("[perf] QT_OPENGL=desktop (OpenGL de escritorio)")


def _try_enable_opencv_cuda() -> None:
    """
    Intenta activar CUDA en OpenCV si está disponible.
    Solo funcionará si tu build de OpenCV fue compilado con WITH_CUDA=ON.
    """
    enable_cuda = os.getenv("ENABLE_CUDA", "0") == "1"

    if not enable_cuda:
        print("[perf] CUDA desactivado (ENABLE_CUDA!=1). Usando OpenCV CPU.")
        return

    if cv2 is None:
        print("[perf] OpenCV no está importado, no se puede comprobar CUDA.")
        return

    # Algunos builds ni siquiera tienen el submódulo cv2.cuda
    if not hasattr(cv2, "cuda"):
        print("[perf] Tu build de OpenCV no expone cv2.cuda (probablemente compilado sin CUDA).")
        return

    try:
        n = cv2.cuda.getCudaEnabledDeviceCount()
    except Exception as e:
        print(f"[perf] Error comprobando dispositivos CUDA en OpenCV: {e}")
        return

    if n <= 0:
        print(
            "[perf] ENABLE_CUDA=1, pero OpenCV reporta 0 dispositivos CUDA.\n"
            "       → O bien no hay GPU NVIDIA visible, o este OpenCV no fue compilado con soporte CUDA."
        )
        return

    # Si llegamos aquí, en teoría hay GPUs CUDA disponibles
    try:
        cv2.cuda.setDevice(0)
        print(f"[perf] OpenCV CUDA activado en GPU 0 de {n} disponibles.")
    except Exception as e:
        print(f"[perf] No se pudo seleccionar dispositivo CUDA en OpenCV: {e}")


def setup_performance() -> None:
    """
    Llamar una vez al inicio de la aplicación.
    - Configura Qt para usar OpenGL de escritorio.
    - Sube prioridad del proceso en Windows.
    - Ajusta la resolución del temporizador.
    - Intenta activar CUDA en OpenCV si está disponible.
    """
    _configure_qt_opengl()
    _boost_process_priority()
    _boost_timer_resolution()
    _try_enable_opencv_cuda()
