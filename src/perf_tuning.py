# src/perf_tuning.py
from __future__ import annotations

import os
import logging

import cv2

try:
    import psutil
except ImportError:
    psutil = None

from .config import (
    OPENCV_USE_OPTIMIZED,
    OPENCV_NUM_THREADS,
    PROCESS_PRIORITY,
    FORCE_QT_DESKTOP_OPENGL,
    ENABLE_CUDA,
)

log = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# 1) Preparar entorno *antes* de importar Qt (OpenGL / GPU)
# -------------------------------------------------------------------------
def prepare_qt_environment() -> None:
    """
    Debe llamarse lo más arriba posible en main_kinect.py,
    ANTES de importar PyQt5, para que Qt use el backend deseado.
    """
    if FORCE_QT_DESKTOP_OPENGL:
        # Forzar OpenGL de escritorio (en vez de ANGLE)
        os.environ.setdefault("QT_OPENGL", "desktop")
        # En Windows, asegúrate además en el Panel de NVIDIA de que
        # python.exe usa la GPU de alto rendimiento.
        log.info("QT_OPENGL=desktop establecido para usar OpenGL de escritorio.")


# -------------------------------------------------------------------------
# 2) OpenCV: threads y optimizaciones internas
# -------------------------------------------------------------------------
def configure_opencv() -> None:
    if OPENCV_USE_OPTIMIZED:
        cv2.setUseOptimized(True)
        log.info("OpenCV: optimizaciones internas habilitadas (setUseOptimized=True).")

    if OPENCV_NUM_THREADS > 0:
        cv2.setNumThreads(OPENCV_NUM_THREADS)
        log.info(f"OpenCV: usando {OPENCV_NUM_THREADS} hilos.")
    else:
        log.info("OpenCV: número de hilos por defecto (OPENCV_NUM_THREADS=0).")

    # Opcional: comprobar si hay soporte CUDA
    if ENABLE_CUDA:
        try:
            n = cv2.cuda.getCudaEnabledDeviceCount()
        except Exception:
            n = 0

        if n > 0:
            log.info(f"OpenCV CUDA: {n} dispositivos CUDA disponibles.")
        else:
            log.warning(
                "ENABLE_CUDA=1 pero OpenCV no parece tener soporte CUDA "
                "(getCudaEnabledDeviceCount==0)."
            )


# -------------------------------------------------------------------------
# 3) Prioridad del proceso (solo Windows)
# -------------------------------------------------------------------------
def set_process_priority() -> None:
    """
    Intenta subir la prioridad del proceso actual.
    Requiere psutil instalado y solo tiene efecto real en Windows.
    """
    if psutil is None:
        log.warning("psutil no está instalado; no se puede cambiar prioridad.")
        return

    try:
        p = psutil.Process()

        if os.name == "nt":
            mapping = {
                "IDLE": psutil.IDLE_PRIORITY_CLASS,
                "BELOW_NORMAL": psutil.BELOW_NORMAL_PRIORITY_CLASS,
                "NORMAL": psutil.NORMAL_PRIORITY_CLASS,
                "ABOVE_NORMAL": psutil.ABOVE_NORMAL_PRIORITY_CLASS,
                "HIGH": psutil.HIGH_PRIORITY_CLASS,
                "REALTIME": psutil.REALTIME_PRIORITY_CLASS,
            }
            prio = mapping.get(PROCESS_PRIORITY, psutil.NORMAL_PRIORITY_CLASS)
            p.nice(prio)
            log.info(f"Prioridad de proceso establecida a {PROCESS_PRIORITY}.")
        else:
            # En Linux/Mac se puede usar nice() con valores -20..19
            if PROCESS_PRIORITY in {"HIGH", "REALTIME"}:
                p.nice(-10)
                log.info("Prioridad elevada (nice=-10) establecida en sistema tipo Unix.")
    except Exception as e:
        log.warning(f"No se pudo cambiar la prioridad del proceso: {e}")


# -------------------------------------------------------------------------
# 4) Función de alto nivel para llamar desde main_kinect.py
# -------------------------------------------------------------------------
def bootstrap_performance() -> None:
    """
    Llamar una vez al inicio del programa (en main_kinect.py):

    - Prepara entorno Qt/OpenGL.
    - Configura OpenCV.
    - Ajusta prioridad del proceso.
    """
    prepare_qt_environment()
    configure_opencv()
    set_process_priority()
