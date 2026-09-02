# ============================================================================
# ARQUITECTURA DE REDES NEURONALES PARA EMOCIONES FACIALES
# Reconocimiento de Emociones Faciales
# ============================================================================
# Este archivo contiene:
#   1. SEBlock:          Squeeze-and-Excitation para recalibración de canales
#   2. ResidualSEBlock:  Bloque residual con SE Attention
#   3. EmotionResNetSE:  Arquitectura principal (ResNet + SE Attention)
#   4. build_model():    Fábrica con detección automática y torch.compile()
#   5. Utilidades:       count_parameters(), get_model_summary()
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import ensure_root_in_path
ensure_root_in_path()

from src.config import NUM_CLASSES, IMG_SIZE, USE_TORCH_COMPILE


# ──────────────────────────────────────────────────────────────────────────────
# 1. MÓDULOS DE ATENCIÓN
# ──────────────────────────────────────────────────────────────────────────────

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) Block.

    Recalibra adaptativamente las respuestas de los canales modelando de forma
    explícita las interdependencias entre ellos, permitiendo a la red priorizar
    rasgos faciales discriminativos (cejas, boca, ojos) sobre el ruido de fondo.

    Operación:
        1. Squeeze: Global Average Pooling → vector de contexto (B, C)
        2. Excitation: FC → ReLU → FC → Sigmoid → pesos por canal (B, C, 1, 1)
        3. Scale: x * pesos
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        weights = self.fc(x).view(b, c, 1, 1)
        return x * weights


# ──────────────────────────────────────────────────────────────────────────────
# 2. BLOQUE RESIDUAL CON SE ATTENTION
# ──────────────────────────────────────────────────────────────────────────────

class ResidualSEBlock(nn.Module):
    """
    Bloque Residual con Conexión Skip y SE Attention.

    Fórmula: y = ReLU( SE( BN(Conv2(ReLU(BN(Conv1(x))))) ) + Shortcut(x) )

    La conexión skip permite que el gradiente fluya sin atenuarse en redes
    profundas (problema del gradiente desvaneciente en CNNs sin residuales).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2     = nn.BatchNorm2d(out_channels)
        self.se      = SEBlock(out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # Shortcut: ajusta dimensiones si cambian canales o resolución espacial
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.dropout(out)

        return self.relu(out + residual)


# ──────────────────────────────────────────────────────────────────────────────
# 3. ARQUITECTURA PRINCIPAL: EmotionResNetSE
# ──────────────────────────────────────────────────────────────────────────────

class EmotionResNetSE(nn.Module):
    """
    Red Neuronal Residual con Atención por Canales (SE-ResNet).

    Especialmente optimizada para FER-2013 y expresiones faciales sutiles.

    Arquitectura (entrada: 1 × 48 × 48):
        Stem     →  64 ch  (48 × 48)
        Stage 1  →  64 ch  (48 × 48)  × 2 bloques residuales
        Stage 2  → 128 ch  (24 × 24)  × 2 bloques residuales (stride 2)
        Stage 3  → 256 ch  (12 × 12)  × 2 bloques residuales (stride 2)
        Stage 4  → 512 ch  ( 6 ×  6)  × 2 bloques residuales (stride 2)
        GAP      →       (1 × 1)
        Head     → 512 → 256 → num_classes

    Total parámetros: ~6.6M
    Memoria estimada (FP32): ~25 MB | (FP16 AMP): ~12.5 MB
    """

    arch_name = "resnet_se"

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        # Stem: extrae características de bajo nivel (bordes, texturas)
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Bloques progresivos: más canales, menor resolución espacial
        self.stage1 = nn.Sequential(
            ResidualSEBlock(64,  64,  stride=1, dropout=0.10),
            ResidualSEBlock(64,  64,  stride=1, dropout=0.10),
        )
        self.stage2 = nn.Sequential(
            ResidualSEBlock(64,  128, stride=2, dropout=0.15),
            ResidualSEBlock(128, 128, stride=1, dropout=0.15),
        )
        self.stage3 = nn.Sequential(
            ResidualSEBlock(128, 256, stride=2, dropout=0.20),
            ResidualSEBlock(256, 256, stride=1, dropout=0.20),
        )
        self.stage4 = nn.Sequential(
            ResidualSEBlock(256, 512, stride=2, dropout=0.25),
            ResidualSEBlock(512, 512, stride=1, dropout=0.25),
        )

        # Global Average Pooling: reduce 6×6 → 1×1, evita sobreajuste
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Clasificador: con BN y Dropout para regularización
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x)
        return self.classifier(x)

    def predict_emotion(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Inferencia de emoción sobre un tensor de imagen preprocesado.

        Args:
            x: Tensor (1, 1, IMG_SIZE, IMG_SIZE) normalizado [-1, 1].

        Returns:
            (predicted_class, probabilities): índice de clase y distribución softmax.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
        return predicted_class, probabilities


# ──────────────────────────────────────────────────────────────────────────────
# 4. FÁBRICA Y UTILIDADES DE MODELOS
# ──────────────────────────────────────────────────────────────────────────────

def build_model(arch: str = "resnet_se", num_classes: int = NUM_CLASSES) -> nn.Module:
    """
    Construye e inicializa el modelo según la arquitectura solicitada.

    Si USE_TORCH_COMPILE=True y PyTorch >= 2.0, aplica torch.compile()
    con el backend TorchInductor, que genera código CUDA/C++ optimizado
    en la primera ejecución (~2-3 minutos de compilación JIT).

    Args:
        arch: 'resnet_se' (único soportado actualmente).
        num_classes: Número de emociones a clasificar.

    Returns:
        Modelo listo para mover a GPU con .to(DEVICE).
    """
    arch = arch.lower().strip()
    if arch not in ("resnet_se", "resnet", "se_resnet", "se"):
        print(f"   [!] Arquitectura desconocida '{arch}', usando EmotionResNetSE.")
    model = EmotionResNetSE(num_classes=num_classes)

    if USE_TORCH_COMPILE and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("   [torch.compile] Compilado con TorchInductor (reduce-overhead).")
            print("   [!] Primera época tardará ~2-3 min extra por compilación JIT — normal.")
        except Exception as e:
            print(f"   [torch.compile] No disponible ({e}). Usando modelo estándar.")
    elif not USE_TORCH_COMPILE:
        import sys
        reason = "Windows no soporta Triton" if sys.platform == "win32" else "desactivado en config"
        print(f"   [torch.compile] Omitido — {reason}. El modelo funciona igual sin él.")

    return model


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Retorna (parámetros entrenables, parámetros totales)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    return trainable, total


def get_model_summary(model: nn.Module) -> None:
    """Imprime un resumen detallado de capas y parámetros del modelo."""
    trainable, total = count_parameters(model)
    arch_name = getattr(model, "arch_name", model.__class__.__name__)

    print("\n" + "=" * 60)
    print(f"[MODEL] {model.__class__.__name__} ({arch_name.upper()})")
    print("=" * 60)
    print(f"\n{'Capa / Módulo':<42} {'Parámetros':>12}")
    print("-" * 55)

    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"  {name:<40} {params:>12,}")

    print("-" * 55)
    print(f"  {'Total Parámetros:':<40} {total:>12,}")
    print(f"  {'Parámetros Entrenables:':<40} {trainable:>12,}")
    print(f"\n  Memoria estimada: {total * 4 / (1024**2):.1f} MB (FP32) | "
          f"{total * 2 / (1024**2):.1f} MB (FP16 AMP)")
    print("=" * 60)


if __name__ == "__main__":
    model = EmotionResNetSE()
    get_model_summary(model)
    dummy = torch.randn(4, 1, IMG_SIZE, IMG_SIZE)
    out = model(dummy)
    print(f"\n  Input: {dummy.shape} → Output: {out.shape}")
