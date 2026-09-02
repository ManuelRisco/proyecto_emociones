# ============================================================================
# UTILIDADES COMPARTIDAS DEL PROYECTO
# Reconocimiento de Emociones Faciales
# ============================================================================
# Centraliza funciones auxiliares reutilizadas en múltiples módulos:
#   - setup_utf8():           Forza UTF-8 en stdout/stderr (Windows)
#   - ensure_root_in_path():  Agrega la raíz del proyecto a sys.path
#   - get_inference_transform(): Transform cacheado para inferencia (PIL → Tensor)
#   - open_camera():          Detecta la cámara web activa evitando virtuales
#   - mixup_batch():          Mixup data augmentation en GPU
# ============================================================================

import os
import sys
from functools import lru_cache

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms


# ──────────────────────────────────────────────────────────────────────────────
# 1. ENCODING
# ──────────────────────────────────────────────────────────────────────────────

def setup_utf8() -> None:
    """
    Forza UTF-8 en stdout y stderr en Windows para evitar errores con
    la página de códigos cp1252 al imprimir emojis o caracteres latinos.
    Llamar UNA SOLA VEZ al inicio de cada punto de entrada del programa.
    """
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# 2. SYS.PATH
# ──────────────────────────────────────────────────────────────────────────────

def ensure_root_in_path() -> None:
    """
    Agrega el directorio raíz del proyecto a sys.path si no está ya presente,
    permitiendo que los módulos de src/ puedan ser importados desde cualquier
    punto de entrada (main.py o ejecución directa de un archivo en src/).
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)


# ──────────────────────────────────────────────────────────────────────────────
# 3. TRANSFORM DE INFERENCIA (CACHEADO)
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_inference_transform(img_size: int = 48) -> transforms.Compose:
    """
    Retorna el pipeline de transformación para inferencia (una sola instancia).
    Usa @lru_cache para que el objeto se construya UNA SOLA VEZ durante toda
    la sesión, evitando el overhead de reconstruirlo frame a frame en tiempo real.

    Pipeline:
        PIL Image (grayscale) → Resize → Tensor → Normalize [-1, 1]
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


def configure_camera(cap: cv2.VideoCapture) -> None:
    """
    Aplica configuraciones de hardware a la cámara web para máximo FPS y mínima latencia:
      - Códec MJPG (evita saturación de ancho de banda USB 2.0 que limita a 15-20 FPS)
      - Resolución 640x480 (óptima para equilibrio entre calidad y velocidad)
      - Solicitud de 60 FPS
      - Buffer size = 1 (evita acumulación de frames retrasados)
    """
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass


def open_camera(target_idx: int | None = None) -> tuple[cv2.VideoCapture | None, int]:
    """
    Detecta y abre la cámara web con señal real y configuración de alto rendimiento.

    Estrategia:
      1. Si se especifica target_idx, intenta abrir esa cámara directamente.
      2. De lo contrario, prueba índices [0, 1, 2, 3] con backend DSHOW y ANY,
         leyendo 3 frames para verificar que la señal no sea negra
         (cámaras virtuales inactivas como NVIDIA Broadcast emiten negro puro).
      3. Aplica configure_camera() para forzar 60 FPS y códec MJPG.

    Returns:
        (VideoCapture, camera_index) o (None, 0) si no se encontró ninguna.
    """
    if target_idx is not None:
        cap = cv2.VideoCapture(target_idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(target_idx)
        if cap.isOpened():
            configure_camera(cap)
            return cap, target_idx
        return None, target_idx

    for idx in [0, 1, 2, 3]:
        for backend in [cv2.CAP_DSHOW, cv2.CAP_ANY]:
            try:
                cap = cv2.VideoCapture(idx, backend)
                if cap.isOpened():
                    configure_camera(cap)
                    for _ in range(3):
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            if np.mean(frame) > 2.0 and np.std(frame) > 2.0:
                                return cap, idx
                    cap.release()
            except Exception:
                pass

    for idx in [0, 1, 2]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            configure_camera(cap)
            return cap, idx

    return None, 0


# ──────────────────────────────────────────────────────────────────────────────
# 5. MIXUP DATA AUGMENTATION (GPU-NATIVE)
# ──────────────────────────────────────────────────────────────────────────────

def mixup_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.2,
    num_classes: int = 7,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Aplica Mixup data augmentation directamente en GPU.

    Mixup interpola linealmente dos imágenes del batch y sus etiquetas,
    forzando al modelo a aprender representaciones más suaves entre clases.
    Es especialmente útil en FER-2013, donde las etiquetas son ruidosas
    (distintos anotadores pueden etiquetar la misma cara con distintas emociones).

    Fórmula:
        λ ~ Beta(alpha, alpha)
        x̃ = λ * x[i] + (1 - λ) * x[j]
        ỹ = λ * one_hot(y[i]) + (1 - λ) * one_hot(y[j])

    Args:
        x: Tensor de imágenes (B, C, H, W) float32, normalizado [-1, 1].
        y: Tensor de etiquetas (B,) int64.
        alpha: Parámetro de concentración de la distribución Beta (0.2 recomendado).
        num_classes: Número de clases de emoción.

    Returns:
        (x_mixed, y_soft): imágenes mezcladas y etiquetas suaves one-hot.
    """
    if alpha <= 0:
        y_one_hot = F.one_hot(y, num_classes=num_classes).float()
        return x, y_one_hot

    lam = float(np.random.beta(alpha, alpha))
    batch_size = x.size(0)
    perm = torch.randperm(batch_size, device=x.device)

    x_mixed = lam * x + (1 - lam) * x[perm]

    y_a = F.one_hot(y, num_classes=num_classes).float()
    y_b = F.one_hot(y[perm], num_classes=num_classes).float()
    y_soft = lam * y_a + (1 - lam) * y_b

    return x_mixed, y_soft


# ──────────────────────────────────────────────────────────────────────────────
# 6. FORMATO DE TIEMPO
# ──────────────────────────────────────────────────────────────────────────────

def format_time(seconds: float) -> str:
    """Convierte segundos a cadena legible 'Xm Ys'."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs}s"
