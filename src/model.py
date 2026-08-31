# ============================================================================
# ARQUITECTURA DE REDES NEURONALES PARA EMOCIONES FACIALES
# Reconocimiento de Emociones Faciales
# Autor: Manuel Fabrizzio Risco Gil
# Curso: Sistemas Inteligentes y Machine Learning - Ciclo 9
# ============================================================================
# Este archivo contiene:
# 1. EmotionCNN: Arquitectura CNN secuencial clásica (4 bloques)
# 2. EmotionResNetSE: Arquitectura avanzada con Conexiones Residuales
#    y Mecanismos de Atención Squeeze-and-Excitation (SE-Attention)
# 3. build_model(): Fábrica de modelos con detección automática de arquitectura
# ============================================================================

import os
import sys
import io
import torch
import torch.nn as nn
import torch.nn.functional as F

# Forzar UTF-8 en Windows para evitar errores con charmap cp1252
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

# Asegurar que el directorio raiz del proyecto este en sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import NUM_CLASSES, IMG_SIZE


# ============================================================================
# 1. MÓDULOS DE ATENCIÓN Y RESIDUALES (Para EmotionResNetSE)
# ============================================================================

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) Block.
    Recalibra adaptativamente las respuestas de los canales de características
    modelando explícitamente las interdependencias entre canales.
    Permite a la red priorizar rasgos faciales discriminativos (cejas, boca, ojos).
    """

    def __init__(self, channels: int, reduction: int = 16):
        super(SEBlock, self).__init__()
        reduced_channels = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        weights = self.fc(x).view(b, c, 1, 1)
        return x * weights


class ResidualSEBlock(nn.Module):
    """
    Bloque Residual con Conexión Skip y Módulo de Atención SE.
    Fórmula: y = ReLU(SE(Conv2(ReLU(Conv1(x)))) + Shortcut(x))
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.1):
        super(ResidualSEBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.se = SEBlock(out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # Shortcut de salto cuando cambian dimensiones o canales
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        out = self.dropout(out)

        out += residual
        out = self.relu(out)
        return out


# ============================================================================
# 2. ARQUITECTURA AVANZADA: EmotionResNetSE
# ============================================================================

class EmotionResNetSE(nn.Module):
    """
    Red Neuronal Residual Profunda con Atención por Canales (SE-ResNet)
    Especialmente optimizada para FER-2013 y expresiones faciales sutiles.

    Ventajas clave:
    - Conexiones residuales que evitan el desvanecimiento de gradiente.
    - SE Attention que amplifica micro-patrones (comisuras de labios, ceño fruncido).
    - Global Average Pooling (GAP) para evitar sobreajuste en capas densas.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super(EmotionResNetSE, self).__init__()
        self.arch_name = "resnet_se"

        # Tallo inicial (Stem)
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Etapa 1: 64 canales (48x48)
        self.stage1 = nn.Sequential(
            ResidualSEBlock(64, 64, stride=1, dropout=0.1),
            ResidualSEBlock(64, 64, stride=1, dropout=0.1),
        )

        # Etapa 2: 128 canales (24x24)
        self.stage2 = nn.Sequential(
            ResidualSEBlock(64, 128, stride=2, dropout=0.15),
            ResidualSEBlock(128, 128, stride=1, dropout=0.15),
        )

        # Etapa 3: 256 canales (12x12)
        self.stage3 = nn.Sequential(
            ResidualSEBlock(128, 256, stride=2, dropout=0.2),
            ResidualSEBlock(256, 256, stride=1, dropout=0.2),
        )

        # Etapa 4: 512 canales (6x6)
        self.stage4 = nn.Sequential(
            ResidualSEBlock(256, 512, stride=2, dropout=0.25),
            ResidualSEBlock(512, 512, stride=1, dropout=0.25),
        )

        # Cabeza clasificadora con Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x

    def predict_emotion(self, x: torch.Tensor):
        """
        Predice la emoción de una imagen.
        Retorna (clase_predicha, probabilidades).
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
        return predicted_class, probabilities


# ============================================================================
# 3. ARQUITECTURA CLÁSICA: EmotionCNN (Retrocompatibilidad)
# ============================================================================

class EmotionCNN(nn.Module):
    """
    Red Neuronal Convolucional Clásica (4 bloques secuenciales).
    Mantenida para compatibilidad con modelos previamente entrenados.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super(EmotionCNN, self).__init__()
        self.arch_name = "cnn_clasica"

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
        )

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

        self._fc_input_size = 256 * (IMG_SIZE // 16) * (IMG_SIZE // 16)

        self.fc_block = nn.Sequential(
            nn.Linear(self._fc_input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = x.view(x.size(0), -1)
        x = self.fc_block(x)
        return x

    def predict_emotion(self, x: torch.Tensor):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
        return predicted_class, probabilities


# ============================================================================
# 4. FÁBRICA Y UTILIDADES DE MODELOS
# ============================================================================

def build_model(arch: str = "resnet_se", num_classes: int = NUM_CLASSES) -> nn.Module:
    """
    Construye e inicializa el modelo según la arquitectura solicitada.
    Opciones: 'resnet_se' (Recomendado) o 'cnn_clasica'.
    """
    arch = arch.lower().strip()
    if arch in ("resnet_se", "resnet", "se_resnet", "se"):
        return EmotionResNetSE(num_classes=num_classes)
    elif arch in ("cnn_clasica", "cnn", "emotion_cnn"):
        return EmotionCNN(num_classes=num_classes)
    else:
        print(f"   [!] Arquitectura desconocida '{arch}', usando EmotionResNetSE por defecto.")
        return EmotionResNetSE(num_classes=num_classes)


def count_parameters(model: nn.Module):
    """Cuenta parámetros entrenables y totales."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def get_model_summary(model: nn.Module):
    """Imprime un resumen detallado del modelo."""
    trainable, total = count_parameters(model)
    arch_name = getattr(model, "arch_name", model.__class__.__name__)

    print("\n" + "=" * 60)
    print(f"[MODEL] ARQUITECTURA DEL MODELO: {model.__class__.__name__} ({arch_name.upper()})")
    print("=" * 60)
    print(f"\n{'Capa / Módulo':<42} {'Parámetros':>12}")
    print("-" * 55)

    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"  {name:<40} {params:>12,}")

    print("-" * 55)
    print(f"  {'Total Parámetros:':<40} {total:>12,}")
    print(f"  {'Parámetros Entrenables:':<40} {trainable:>12,}")
    print(f"\n  Memoria estimada: {total * 4 / (1024**2):.2f} MB (float32)")
    print("=" * 60)


if __name__ == "__main__":
    # Test de ambas arquitecturas
    print("\n>>> PROBANDO ARQUITECTURA EmotionResNetSE:")
    model_se = EmotionResNetSE()
    get_model_summary(model_se)
    dummy_input = torch.randn(4, 1, IMG_SIZE, IMG_SIZE)
    out_se = model_se(dummy_input)
    print(f"  Input: {dummy_input.shape} -> Output: {out_se.shape}")

    print("\n>>> PROBANDO ARQUITECTURA EmotionCNN (Clásica):")
    model_cnn = EmotionCNN()
    get_model_summary(model_cnn)
    out_cnn = model_cnn(dummy_input)
    print(f"  Input: {dummy_input.shape} -> Output: {out_cnn.shape}")
