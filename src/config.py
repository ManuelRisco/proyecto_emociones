# ============================================================================
# CONFIGURACIÓN GENERAL DEL PROYECTO
# Reconocimiento de Emociones Faciales con CNN
# Autor: Manuel Fabrizzio Risco Gil
# Curso: Sistemas Inteligentes y Machine Learning - Ciclo 9
# ============================================================================

import os
import torch

# ============================================================================
# RUTAS DEL PROYECTO
# ============================================================================
# Directorio raíz del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directorio de datos
DATA_DIR = os.path.join(BASE_DIR, "data")
FER2013_CSV_PATH = os.path.join(DATA_DIR, "fer2013.csv")

# Directorio de salidas
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")

# Crear directorios si no existen
for dir_path in [DATA_DIR, MODELS_DIR, PLOTS_DIR, LOGS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ============================================================================
# CLASES DE EMOCIONES
# ============================================================================
EMOTION_LABELS = {
    0: "Enojado",      # Angry
    1: "Disgusto",     # Disgust
    2: "Miedo",        # Fear
    3: "Feliz",        # Happy
    4: "Triste",       # Sad
    5: "Sorprendido",  # Surprise
    6: "Neutral",      # Neutral
}

NUM_CLASSES = len(EMOTION_LABELS)

# Emojis para visualización
EMOTION_EMOJIS = {
    0: "😠",  # Enojado
    1: "🤢",  # Disgusto
    2: "😨",  # Miedo
    3: "😃",  # Feliz
    4: "😢",  # Triste
    5: "😲",  # Sorprendido
    6: "😐",  # Neutral
}

# ============================================================================
# HIPERPARÁMETROS DEL MODELO
# ============================================================================
# Tamaño de imagen de entrada
IMG_SIZE = 48

# Parámetros de entrenamiento
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2     # Hilos para carga de datos en paralelo
USE_AMP = True      # Automatic Mixed Precision (AMP) para aprovechar Tensor Cores

# Early stopping
EARLY_STOPPING_PATIENCE = 10

# ============================================================================
# CONFIGURACIÓN DE GPU / DISPOSITIVO
# ============================================================================
def get_device():
    """
    Detecta y retorna el dispositivo disponible (GPU CUDA o CPU).
    Muestra información detallada de la GPU si está disponible.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True  # Optimización automática de cuDNN
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print("=" * 60)
        print("[GPU] GPU DETECTADA - ACELERACION CUDA ACTIVADA")
        print("=" * 60)
        print(f"   GPU: {gpu_name}")
        print(f"   Memoria: {gpu_memory:.1f} GB")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   PyTorch Version: {torch.__version__}")
        print(f"   cuDNN Benchmark: Activado")
        print(f"   Precision Mixta (AMP): {'Activada' if USE_AMP else 'Desactivada'}")
        print("=" * 60)
    else:
        device = torch.device("cpu")
        print("=" * 60)
        print("[!] GPU NO DETECTADA - Usando CPU")
        print(f"   PyTorch Version: {torch.__version__}")
        print("   Nota: El entrenamiento sera mas lento sin GPU")
        print("=" * 60)

    return device

# Dispositivo global
DEVICE = get_device()

# ============================================================================
# RUTA DEL MODELO GUARDADO
# ============================================================================
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "mejor_modelo_emociones.pth")
LAST_MODEL_PATH = os.path.join(MODELS_DIR, "ultimo_modelo_emociones.pth")

# ============================================================================
# CONFIGURACIÓN DE OPENCV (Haar Cascade para detección de rostros)
# ============================================================================
HAAR_CASCADE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "data",
    "haarcascade_frontalface_default.xml"
)

# Colores para visualización en OpenCV (BGR)
COLORS = {
    "green": (0, 255, 0),
    "red": (0, 0, 255),
    "blue": (255, 0, 0),
    "white": (255, 255, 255),
    "yellow": (0, 255, 255),
    "black": (0, 0, 0),
}
