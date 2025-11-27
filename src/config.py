# src/config.py
import os
from dotenv import load_dotenv

# Cargar variables desde .env (si existe)
load_dotenv()


def _env_bool(name: str, default: str = "0") -> bool:
    """
    Lee una variable de entorno y la interpreta como booleana.
    Valores considerados 'True': 1, true, True, YES, yes
    """
    return os.getenv(name, default).strip() in ("1", "true", "True", "YES", "yes")


def _env_int(name: str, default: str = "0") -> int:
    """
    Lee una variable de entorno como int, con fallback robusto.
    """
    raw = os.getenv(name, default)
    try:
        return int(str(raw).strip())
    except Exception:
        return int(default)


# ----------------------------------------------------------------------
# Flags de funcionalidad principal
# ----------------------------------------------------------------------

# Activar/desactivar MediaPipe Pose
USE_MEDIAPIPE: bool = _env_bool("USE_MEDIAPIPE", "1")

# Activar/desactivar uso de BodyIndex (para point cloud del cuerpo)
USE_BODY_INDEX: bool = _env_bool("USE_BODY_INDEX", "0")

# Complejidad del modelo de MediaPipe Pose (0, 1, 2)
POSE_MODEL_COMPLEXITY: int = _env_int("POSE_MODEL_COMPLEXITY", "1")

# FPS objetivo (solo referencia para el QTimer)
TARGET_FPS: int = _env_int("TARGET_FPS", "30")


# ----------------------------------------------------------------------
# Parámetros de performance / tuning
# ----------------------------------------------------------------------

# OpenCV: usar optimizaciones internas (setUseOptimized)
OPENCV_USE_OPTIMIZED: bool = _env_bool("OPENCV_USE_OPTIMIZED", "1")

# OpenCV: número de hilos (0 = auto / lo que decida OpenCV)
OPENCV_NUM_THREADS: int = _env_int("OPENCV_NUM_THREADS", "0")

# Prioridad del proceso (solo Windows)
# Valores esperados: IDLE, BELOW_NORMAL, NORMAL, ABOVE_NORMAL, HIGH, REALTIME
PROCESS_PRIORITY: str = os.getenv("PROCESS_PRIORITY", "NORMAL").upper().strip()

# Forzar backend OpenGL de Qt (QT_OPENGL=desktop)
FORCE_QT_DESKTOP_OPENGL: bool = _env_bool("FORCE_QT_DESKTOP_OPENGL", "0")

# Intentar usar CUDA en OpenCV (si la build de OpenCV lo soporta)
ENABLE_CUDA: bool = _env_bool("ENABLE_CUDA", "0")
