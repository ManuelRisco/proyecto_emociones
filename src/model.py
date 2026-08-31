# ============================================================================
# ARQUITECTURA DE LA RED NEURONAL CONVOLUCIONAL (CNN)
# Reconocimiento de Emociones Faciales
# Autor: Manuel Fabrizzio Risco Gil
# ============================================================================
# Arquitectura: 4 bloques convolucionales + 2 capas fully connected
#
# FLUJO DE DATOS (Input → Proceso → Output):
# ─────────────────────────────────────────────
# ENTRADA: Imagen 48x48 píxeles (escala de grises, 1 canal)
#     ↓
# PROCESO: La CNN extrae patrones automáticamente:
#   Bloque 1: Detecta bordes y texturas básicas (32 filtros)
#   Bloque 2: Detecta formas simples como ojos, cejas (64 filtros)
#   Bloque 3: Detecta partes del rostro como boca, nariz (128 filtros)
#   Bloque 4: Detecta expresiones completas del rostro (256 filtros)
#   FC1: Combina todas las características detectadas (512 neuronas)
#   FC2: Refina la decisión (256 neuronas)
#     ↓
# SALIDA: Probabilidades para cada una de las 7 emociones
# ============================================================================

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Asegurar que el directorio raiz del proyecto este en sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import NUM_CLASSES, IMG_SIZE


class EmotionCNN(nn.Module):
    """
    Red Neuronal Convolucional para Reconocimiento de Emociones Faciales.

    Arquitectura:
    - 4 bloques convolucionales, cada uno con:
      * Conv2D → BatchNorm → ReLU → MaxPool → Dropout
    - 2 capas fully connected con BatchNorm y Dropout
    - Capa de salida con 7 clases (una por emoción)

    La red aprende automáticamente a:
    1. Extraer características visuales del rostro
    2. Identificar patrones asociados a cada emoción
    3. Clasificar la emoción con alta precisión
    """

    def __init__(self, num_classes=NUM_CLASSES):
        super(EmotionCNN, self).__init__()

        # ================================================================
        # BLOQUE CONVOLUCIONAL 1: Detección de bordes y texturas
        # Input: (1, 48, 48) → Output: (32, 24, 24)
        # ================================================================
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,        # 1 canal (escala de grises)
                out_channels=32,      # 32 filtros convolucionales
                kernel_size=3,        # Filtro de 3x3
                padding=1             # Mantener dimensiones
            ),
            nn.BatchNorm2d(32),       # Normalización por lote
            nn.ReLU(inplace=True),    # Activación ReLU
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Reducir a la mitad
            nn.Dropout2d(0.25),       # Regularización (25%)
        )

        # ================================================================
        # BLOQUE CONVOLUCIONAL 2: Detección de formas simples
        # Input: (32, 24, 24) → Output: (64, 12, 12)
        # ================================================================
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
        )

        # ================================================================
        # BLOQUE CONVOLUCIONAL 3: Detección de partes faciales
        # Input: (64, 12, 12) → Output: (128, 6, 6)
        # ================================================================
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
        )

        # ================================================================
        # BLOQUE CONVOLUCIONAL 4: Detección de expresiones completas
        # Input: (128, 6, 6) → Output: (256, 3, 3)
        # ================================================================
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
        )

        # ================================================================
        # CAPAS FULLY CONNECTED (Clasificación)
        # Combinan las características para tomar la decisión final
        # ================================================================
        # Calcular el tamaño después de las convoluciones
        # 48 → 24 → 12 → 6 → 3 = 3x3x256 = 2304
        self._fc_input_size = 256 * (IMG_SIZE // 16) * (IMG_SIZE // 16)

        self.fc_block = nn.Sequential(
            # Capa Fully Connected 1
            nn.Linear(self._fc_input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),       # Mayor dropout para evitar overfitting

            # Capa Fully Connected 2
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            # Capa de Salida: 7 neuronas (una por emoción)
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        """
        Propagación hacia adelante (forward pass).

        Args:
            x: tensor de entrada con forma (batch_size, 1, 48, 48)

        Returns:
            tensor de salida con forma (batch_size, 7) - logits para cada emoción
        """
        # Pasar por bloques convolucionales
        x = self.conv_block1(x)   # Extrae bordes y texturas
        x = self.conv_block2(x)   # Extrae formas simples
        x = self.conv_block3(x)   # Extrae partes faciales
        x = self.conv_block4(x)   # Extrae expresiones completas

        # Aplanar para las capas fully connected
        x = x.view(x.size(0), -1)

        # Clasificación final
        x = self.fc_block(x)

        return x

    def predict_emotion(self, x):
        """
        Predice la emoción de una imagen.

        Args:
            x: tensor de entrada (puede ser una sola imagen o un batch)

        Returns:
            tuple: (clase_predicha, probabilidades)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
        return predicted_class, probabilities


def count_parameters(model):
    """
    Cuenta el número total de parámetros del modelo.

    Args:
        model: modelo de PyTorch

    Returns:
        tuple: (parámetros entrenables, parámetros totales)
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def get_model_summary(model):
    """
    Imprime un resumen detallado del modelo.

    Args:
        model: modelo EmotionCNN
    """
    trainable, total = count_parameters(model)

    print("\n" + "=" * 60)
    print("🧠 ARQUITECTURA DEL MODELO CNN")
    print("=" * 60)
    print(f"\n{'Capa':<40} {'Parámetros':>12}")
    print("-" * 55)

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            params = sum(p.numel() for p in module.parameters())
            print(f"  {name:<38} {params:>12,}")

    print("-" * 55)
    print(f"  {'Total Parámetros:':<38} {total:>12,}")
    print(f"  {'Parámetros Entrenables:':<38} {trainable:>12,}")
    print(f"\n  Memoria estimada: {total * 4 / (1024**2):.1f} MB (float32)")
    print("=" * 60)


if __name__ == "__main__":
    # Test del modelo
    model = EmotionCNN()
    get_model_summary(model)

    # Test con datos aleatorios
    dummy_input = torch.randn(4, 1, IMG_SIZE, IMG_SIZE)
    output = model(dummy_input)
    print(f"\n  Input shape:  {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output (logits): {output[0].detach().numpy()}")
