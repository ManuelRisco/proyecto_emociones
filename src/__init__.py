# ============================================================================
# PAQUETE src — API PÚBLICA DEL PROYECTO
# Reconocimiento de Emociones Faciales
# ============================================================================
# Importaciones sugeridas para uso externo:
#
#   from src.model   import EmotionResNetSE, build_model
#   from src.predict import load_model, FaceDetector, preprocess_face
#   from src.dataset import load_data
#   from src.config  import DEVICE, EMOTION_LABELS, NUM_CLASSES
# ============================================================================

from src.config import (
    DEVICE,
    EMOTION_COLORS,
    EMOTION_EMOJIS,
    EMOTION_LABELS,
    EMOTION_LABELS_EN,
    IMG_SIZE,
    NUM_CLASSES,
)
from src.model import EmotionResNetSE, build_model, get_model_summary
from src.predict import FaceDetector, load_model, preprocess_face

__all__ = [
    # Config
    "DEVICE", "EMOTION_COLORS", "EMOTION_EMOJIS",
    "EMOTION_LABELS", "EMOTION_LABELS_EN", "IMG_SIZE", "NUM_CLASSES",
    # Model
    "EmotionResNetSE", "build_model", "get_model_summary",
    # Predict
    "FaceDetector", "load_model", "preprocess_face",
]
