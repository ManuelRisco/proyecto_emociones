# ============================================================================
# CONFIGURACIÓN GENERAL DEL PROYECTO
# Reconocimiento de Emociones Faciales
# ============================================================================

import os
import torch

# Rutas del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

for directory in [DATA_DIR, TRAIN_DIR, TEST_DIR, OUTPUT_DIR, MODELS_DIR, PLOTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Mapeo de clases y etiquetas
EMOTION_LABELS = {
    0: "Enojado",
    1: "Disgusto",
    2: "Miedo",
    3: "Feliz",
    4: "Triste",
    5: "Sorprendido",
    6: "Neutral"
}

EMOTION_LABELS_EN = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

EMOTION_EMOJIS = {
    0: "😠",
    1: "🤢",
    2: "😨",
    3: "😄",
    4: "😢",
    5: "😲",
    6: "😐"
}

EMOTION_COLORS = {
    0: (40, 40, 235),
    1: (34, 139, 34),
    2: (180, 50, 180),
    3: (0, 215, 255),
    4: (235, 120, 40),
    5: (0, 140, 255),
    6: (170, 170, 170)
}

COLORS = EMOTION_COLORS

NUM_CLASSES = len(EMOTION_LABELS)

# Parámetros de imagen y entrenamiento
IMG_SIZE = 48
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0
USE_AMP = True
LOAD_TO_VRAM = True
EARLY_STOPPING_PATIENCE = 10

# Dispositivo
def get_device():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()

# Rutas de modelos
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "mejor_modelo_emociones.pth")
LAST_MODEL_PATH = os.path.join(MODELS_DIR, "ultimo_modelo_emociones.pth")
DEFAULT_ARCHITECTURE = "resnet_se"

# Detección facial
YUNET_MODEL_PATH = os.path.join(DATA_DIR, "face_detection_yunet_2023mar.onnx")
YUNET_SCORE_THRESHOLD = 0.65
YUNET_NMS_THRESHOLD = 0.3

HAAR_CASCADE_PATH = os.path.join(DATA_DIR, "haarcascade_frontalface_default.xml")

# Suavizado temporal
SMOOTHING_ALPHA = 0.30
BBOX_SMOOTHING_ALPHA = 0.35
TRACKING_IOU_THRESHOLD = 0.30
