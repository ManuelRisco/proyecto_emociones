# ============================================================================
# CONFIGURACIÓN GENERAL DEL PROYECTO
# Reconocimiento de Emociones Faciales
# ============================================================================

import os
import sys
import torch

# ── Rutas del proyecto ────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR  = os.path.join(DATA_DIR, "test")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
MODELS_DIR  = os.path.join(OUTPUT_DIR, "models")
PLOTS_DIR   = os.path.join(OUTPUT_DIR, "plots")
LOGS_DIR    = os.path.join(OUTPUT_DIR, "logs")

for _dir in [DATA_DIR, TRAIN_DIR, TEST_DIR, OUTPUT_DIR, MODELS_DIR, PLOTS_DIR, LOGS_DIR]:
    os.makedirs(_dir, exist_ok=True)

# Dataset CSV (alternativa a carpetas train/test)
FER2013_CSV_PATH = os.path.join(DATA_DIR, "fer2013.csv")

# ── Emociones ─────────────────────────────────────────────────────────────────
EMOTION_LABELS = {
    0: "Enojado",
    1: "Disgusto",
    2: "Miedo",
    3: "Feliz",
    4: "Triste",
    5: "Sorprendido",
    6: "Neutral",
}

EMOTION_LABELS_EN = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}

EMOTION_EMOJIS = {
    0: "😠",
    1: "🤢",
    2: "😨",
    3: "😄",
    4: "😢",
    5: "😲",
    6: "😐",
}

# Colores BGR para cada emoción (OpenCV)
EMOTION_COLORS = {
    0: (40, 40, 235),
    1: (34, 139, 34),
    2: (180, 50, 180),
    3: (0, 215, 255),
    4: (235, 120, 40),
    5: (0, 140, 255),
    6: (170, 170, 170),
}

NUM_CLASSES = len(EMOTION_LABELS)

# ── Imagen y entrenamiento ────────────────────────────────────────────────────
IMG_SIZE  = 48
BATCH_SIZE = 256              # RTX 5060 Ti (8 GB VRAM): batch 256 con uint8 ocupa ~57 MB → OK
LEARNING_RATE  = 0.0005       # 0.001 causaba oscilación en val_loss (épocas 6-22)
NUM_EPOCHS     = 100          # El modelo seguía mejorando en época 49 → más margen
WEIGHT_DECAY   = 1e-4
NUM_WORKERS    = 0            # 0 = sin workers CPU (toda la data ya está en VRAM)
WARMUP_EPOCHS  = 8            # Warmup más gradual con LR menor
EARLY_STOPPING_PATIENCE = 15  # Más paciencia: la mejora final es lenta
MIXUP_ALPHA    = 0.1          # Reducido de 0.2 → menos dilución para clases minoritarias
MIN_SAMPLES_PER_CLASS = 2000  # Oversampling: clases con < 2000 imgs se repiten hasta este mínimo

# ── Flags de optimización ─────────────────────────────────────────────────────
USE_AMP           = True   # Automatic Mixed Precision (FP16 Tensor Cores)
LOAD_TO_VRAM      = True   # Cargar todo el dataset en VRAM al inicio
# torch.compile() requiere Triton, que solo está disponible en Linux.
# En Windows se desactiva automáticamente para evitar TritonMissing.
USE_TORCH_COMPILE = sys.platform != "win32"
USE_MIXUP         = True   # Mixup augmentation (alpha=0.2) durante entrenamiento

# ── Dispositivo ───────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False  # más velocidad en benchmark mode
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()

# ── Rutas de modelos ──────────────────────────────────────────────────────────
BEST_MODEL_PATH      = os.path.join(MODELS_DIR, "mejor_modelo_emociones.pth")
LAST_MODEL_PATH      = os.path.join(MODELS_DIR, "ultimo_modelo_emociones.pth")
DEFAULT_ARCHITECTURE = "resnet_se"

# ── Detección facial ──────────────────────────────────────────────────────────
YUNET_MODEL_PATH      = os.path.join(DATA_DIR, "face_detection_yunet_2023mar.onnx")
YUNET_SCORE_THRESHOLD = 0.65
YUNET_NMS_THRESHOLD   = 0.3

# ── Suavizado temporal (tiempo real) ──────────────────────────────────────────
SMOOTHING_ALPHA      = 0.30
BBOX_SMOOTHING_ALPHA = 0.35
TRACKING_IOU_THRESHOLD = 0.30

# ── Filtro de Precisión por Situación y Sensibilidad Avanzada ─────────────────
# Umbrales mínimos de confianza adaptativos. Disgusto y Sorprendido tienen umbrales
# ultra-sensibles (0.18) para capturar microexpresiones sutiles en tiempo real.
EMOTION_CONFIDENCE_THRESHOLDS = {
    0: 0.30,   # Enojado     — calibración equilibrada y sensible
    1: 0.18,   # Disgusto    — ULTRA SENSIBLE (captura muecas y nariz arrugada de inmediato)
    2: 0.35,   # Miedo       — umbral equilibrado
    3: 0.28,   # Feliz       — fácil detección
    4: 0.35,   # Triste      — umbral equilibrado
    5: 0.18,   # Sorprendido — ULTRA SENSIBLE (captura apertura de ojos/boca rápida)
    6: 0.24,   # Neutral     — umbral base
}

# Multiplicadores de sensibilidad reactiva para nivelar emociones rápidas
EMOTION_SENSITIVITY_BOOST = {
    0: 1.55,   # Boost Enojado (+55% reactividad e impulso a la barra)
    1: 1.55,   # Boost Disgusto (+55% reactividad)
    2: 1.05,
    3: 1.00,
    4: 1.05,
    5: 1.55,   # Boost Sorprendido (+55% reactividad)
    6: 0.85,   # Neutral calibrado
}

# Debounce adaptativo: Sorprendido y Enojado reaccionan en 2 frames (~33ms)
EMOTION_DEBOUNCE_BY_CLASS = {
    0: 2,      # Enojado (2 frames ~33ms, respuesta inmediata)
    1: 3,      # Disgusto (3 frames)
    2: 4,      # Miedo (4 frames)
    3: 3,      # Feliz (3 frames)
    4: 5,      # Triste (5 frames)
    5: 2,      # Sorprendido ultra-reactivo (2 frames ~33ms)
    6: 5,      # Neutral (5 frames)
}
EMOTION_DEBOUNCE_FRAMES = 4

# ── Paleta Holográfica Multi-Persona (Colores Neón Distintivos) ──────────────
PERSON_PALETTE = [
    (255, 220, 0),    # Sujeto 1: Cian Holográfico (#00DCFF)
    (0, 180, 255),    # Sujeto 2: Ámbar / Naranja Neón (#FFB400)
    (255, 70, 210),   # Sujeto 3: Magenta Cyberpunk (#D246FF)
    (60, 255, 120),   # Sujeto 4: Verde Esmeralda Matrix (#78FF3C)
    (255, 140, 20),   # Sujeto 5: Azul Cobalto Eléctrico (#148CFF)
    (0, 240, 255),    # Sujeto 6: Amarillo Neón (#FFF000)
]

# ── Memoria Biométrica de Re-Identificación (Face Re-ID) ─────────────────────
# Umbral de similitud biométrica para reconocer a un sujeto que salió de la cámara y vuelve
REID_SIMILARITY_THRESHOLD = 0.62
REID_TEMPLATE_SIZE = (28, 28)
