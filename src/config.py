# src/config.py
import os
from dotenv import load_dotenv

# Cargar variables desde .env (si existe)
load_dotenv()

def _env_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip() in ("1", "true", "True", "YES", "yes")

USE_MEDIAPIPE: bool = _env_bool("USE_MEDIAPIPE", "1")
USE_BODY_INDEX: bool = _env_bool("USE_BODY_INDEX", "0")

POSE_MODEL_COMPLEXITY: int = int(os.getenv("POSE_MODEL_COMPLEXITY", "1"))
TARGET_FPS: int = int(os.getenv("TARGET_FPS", "30"))
