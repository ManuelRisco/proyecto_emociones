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

# ── Filtro de Precisión por Situación (Anti-Confusión Neutral↔Enojado) ────────
# Umbral mínimo de confianza que debe superar una emoción para mostrarse como
# dominante. Si no alcanza el umbral, se usa la segunda emoción más probable.
# Esto evita que "concentración/comiendo/leyendo" se confunda con "Enojado".
EMOTION_CONFIDENCE_THRESHOLDS = {
    0: 0.55,   # Enojado    — requiere 55% de confianza (era la que más se confundía)
    1: 0.45,   # Disgusto   — umbral moderado
    2: 0.45,   # Miedo      — umbral moderado
    3: 0.35,   # Feliz      — umbral bajo (muy distintiva)
    4: 0.42,   # Triste     — umbral moderado
    5: 0.38,   # Sorprendido — umbral bajo (muy distintiva)
    6: 0.30,   # Neutral    — umbral muy bajo (catch-all para estados ambiguos)
}

# Frames consecutivos que debe mantenerse una nueva emoción antes de cambiarla
# en el HUD (evita saltos instantáneos entre emociones similares).
# Un valor de 6 ≈ 0.1 segundos a 60 FPS, suficiente para filtrar microexpresiones
# sin añadir lag perceptible.
EMOTION_DEBOUNCE_FRAMES = 6
