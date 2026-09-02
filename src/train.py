# ============================================================================
# ENTRENAMIENTO DE LA RED NEURONAL (EmotionResNetSE)
# Reconocimiento de Emociones Faciales
# ============================================================================
# Técnicas de entrenamiento implementadas:
#   1. Class Weights — equilibra emociones minoritarias (Disgust ~600 vs Happy ~9k)
#   2. Label Smoothing — reduce sobreconfianza en etiquetas ruidosas de FER-2013
#   3. Mixup Augmentation — interpola pares de imágenes para mejor generalización
#   4. LR Warmup + CosineAnnealing — convergencia estable sin oscilaciones iniciales
#   5. AdamW + Gradient Clipping — estabiliza el gradiente en etiquetas ambiguas
#   6. AMP / FP16 — usa Tensor Cores para máximo throughput en VRAM
# ============================================================================

import json
import os
import time

import matplotlib
matplotlib.use("Agg")  # Backend sin GUI para guardar gráficas en servidor
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.utils import ensure_root_in_path, format_time, setup_utf8
setup_utf8()
ensure_root_in_path()

from src.config import (
    BEST_MODEL_PATH, DEFAULT_ARCHITECTURE, DEVICE,
    EARLY_STOPPING_PATIENCE, EMOTION_LABELS, LAST_MODEL_PATH,
    LEARNING_RATE, LOGS_DIR, MIXUP_ALPHA, NUM_EPOCHS, PLOTS_DIR,
    USE_AMP, WARMUP_EPOCHS, WEIGHT_DECAY,
)
from src.dataset import load_data
from src.model import build_model, get_model_summary


# ──────────────────────────────────────────────────────────────────────────────
# FOCAL LOSS
# ──────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss para clasificación multi-clase con soporte de etiquetas suaves.

    Motiva al modelo a concentrar el gradiente en los ejemplos difíciles
    (emociones ambiguas o clases minoritarias como Disgust), ignorando los
    ejemplos que ya predice con alta confianza (Happy, Neutral).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha:  Tensor de pesos por clase (class_weights del dataset).
        gamma:  Parámetro de foco. gamma=2.0 es el valor estándar de Lin et al.
        smooth: Label smoothing para reducir sobreconfianza en etiquetas ruidosas.
    """

    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        smooth: float = 0.05,
        num_classes: int = 7,
    ):
        super().__init__()
        self.alpha       = alpha
        self.gamma       = gamma
        self.smooth      = smooth
        self.num_classes = num_classes

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            inputs:  Logits (B, C) — salida del modelo antes de softmax.
            targets: Etiquetas (B,) int64 o distribuciones suaves (B, C) float32.
        """
        log_probs = torch.log_softmax(inputs, dim=1)   # (B, C)

        # Soporte para etiquetas suaves (Mixup) o etiquetas duras
        if targets.dim() == 2:
            # targets ya es una distribución (B, C) — de Mixup
            y_soft = targets.float()
        else:
            # Convertir etiquetas duras a one-hot y aplicar label smoothing
            y_hard = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1.0)
            y_soft = y_hard * (1 - self.smooth) + self.smooth / self.num_classes

        # Probabilidad de la clase ground-truth (para el factor focal)
        if targets.dim() == 2:
            p_t = (torch.exp(log_probs) * y_soft).sum(dim=1)          # (B,)
        else:
            p_t = torch.exp(log_probs).gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_weight = (1 - p_t) ** self.gamma                        # (B,)

        # Pérdida base: cross-entropy con etiquetas suaves
        loss = -(y_soft * log_probs).sum(dim=1)                        # (B,)

        # Aplicar peso focal
        loss = focal_weight * loss

        # Aplicar pesos de clase (alpha)
        if self.alpha is not None:
            if targets.dim() == 2:
                alpha_w = (self.alpha.unsqueeze(0) * y_soft).sum(dim=1)
            else:
                alpha_w = self.alpha[targets]
            loss = alpha_w * loss

        return loss.mean()


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS DE ENTRENAMIENTO
# ──────────────────────────────────────────────────────────────────────────────

def _run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    num_epochs: int,
    is_train: bool,
) -> tuple[float, float]:
    """
    Ejecuta una época de entrenamiento o validación.

    Unifica la lógica de train/eval para evitar código duplicado.
    En modo entrenamiento aplica AMP + gradient clipping.
    En modo validación solo hace forward pass con torch.no_grad().

    Returns:
        (avg_loss, accuracy_percent)
    """
    model.train() if is_train else model.eval()
    use_amp = device.type == "cuda" and USE_AMP
    phase   = "Train" if is_train else "Val  "

    running_loss = correct = total = 0
    pbar = tqdm(
        loader,
        desc=f"  Epoca {epoch+1:2d}/{num_epochs} [{phase}]",
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )

    ctx = torch.no_grad() if not is_train else torch.enable_grad()
    with ctx:
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device, non_blocking=True)

            # Soporte para etiquetas suaves (Mixup) o etiquetas duras
            if isinstance(labels, torch.Tensor) and labels.dim() == 2:
                # Mixup: labels son distribuciones suaves (B, C)
                labels_for_loss = labels.to(device, non_blocking=True)
                # Para accuracy usamos el argmax de la etiqueta suave
                hard_labels = labels_for_loss.argmax(dim=1)
            else:
                labels_for_loss = labels.to(device, non_blocking=True)
                hard_labels = labels_for_loss

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(images)
                    loss = criterion(outputs, labels_for_loss)
                if is_train:
                    scaler.scale(loss).backward()
                    # Gradient clipping: evita explosión en emociones ambiguas
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels_for_loss)
                if is_train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total   += hard_labels.size(0)
            correct += (predicted == hard_labels).sum().item()

            pbar.set_postfix({
                "Loss": f"{running_loss / (batch_idx + 1):.4f}",
                "Acc":  f"{100.0 * correct / total:.1f}%",
            })

    return running_loss / len(loader), 100.0 * correct / total


def _build_scheduler(optimizer: optim.Optimizer, num_epochs: int, warmup_epochs: int):
    """
    Combina Warmup Lineal + CosineAnnealingLR en un SequentialLR.

    - Warmup: evita grandes actualizaciones del gradiente en los primeros epochs
      cuando los pesos aún están lejos del mínimo.
    - CosineAnnealing: reduce el LR suavemente hasta eta_min=1e-6.
    """
    warmup = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs - warmup_epochs, eta_min=1e-6
    )
    return optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )


def _plot_history(history: dict, save_dir: str, arch_name: str = "") -> None:
    """Genera y guarda gráficas de loss y accuracy del entrenamiento."""
    epochs_range = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Historial de Entrenamiento — {arch_name.upper()} Emociones Faciales",
        fontsize=14, fontweight="bold",
    )

    for ax, metric, title, ylabel in zip(
        axes,
        [("train_loss", "val_loss"), ("train_acc", "val_acc")],
        ["Pérdida por Época", "Precisión por Época"],
        ["Loss", "Accuracy (%)"],
    ):
        ax.plot(epochs_range, history[metric[0]], "b-o", label="Train", markersize=3)
        ax.plot(epochs_range, history[metric[1]], "r-o", label="Validación", markersize=3)
        ax.set_xlabel("Época")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "historial_entrenamiento.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n   [GRAFICA] Historial guardado en: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# FUNCIÓN PRINCIPAL DE ENTRENAMIENTO
# ──────────────────────────────────────────────────────────────────────────────

def train(
    arch: str = DEFAULT_ARCHITECTURE,
    num_epochs: int = NUM_EPOCHS,
    lr: float = LEARNING_RATE,
) -> None:
    """
    Ciclo completo de entrenamiento de EmotionResNetSE.

    Flujo:
        1. Carga datos → VRAM (FastGPUDataLoader)
        2. Construye modelo + torch.compile() si disponible
        3. Configura pérdida (CrossEntropy + ClassWeights + LabelSmoothing)
        4. Configura optimizer (AdamW) + scheduler (Warmup + Cosine)
        5. Entrena con AMP, Mixup y Gradient Clipping
        6. Early Stopping, guardado de mejor modelo, log JSON, gráficas
    """
    print("\n" + "=" * 60)
    print(f">>> INICIANDO ENTRENAMIENTO: {arch.upper()}")
    print("=" * 60)

    # ── Datos ─────────────────────────────────────────────────────────────────
    train_loader, val_loader, _, class_weights = load_data(return_weights=True)

    # ── Modelo ────────────────────────────────────────────────────────────────
    model = build_model(arch=arch, num_classes=len(EMOTION_LABELS)).to(DEVICE)
    get_model_summary(model)

    # ── Función de pérdida ────────────────────────────────────────────────────
    # FocalLoss concentra el gradiente en clases difíciles/minoritarias.
    # Es especialmente efectivo con el desbalance de FER-2013
    # (Disgust: 360 imgs vs Happy: 6,104 imgs — ratio 17:1).
    criterion = FocalLoss(
        alpha=class_weights,
        gamma=2.0,
        smooth=0.05,
        num_classes=len(EMOTION_LABELS),
    )

    # ── Optimizador ───────────────────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = _build_scheduler(optimizer, num_epochs, WARMUP_EPOCHS)
    scaler    = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda" and USE_AMP))

    # ── Estado inicial ────────────────────────────────────────────────────────
    history: dict = {
        "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "arch": arch
    }
    best_val_loss = float("inf")
    best_val_acc  = 0.0
    patience_counter = 0

    print(f"\n   Configuración de entrenamiento:")
    print(f"   |-- Arquitectura : {arch.upper()} ({model.__class__.__name__})")
    print(f"   |-- Épocas       : {num_epochs}")
    print(f"   |-- Warmup       : {WARMUP_EPOCHS} épocas")
    print(f"   |-- LR inicial   : {lr}")
    print(f"   |-- Batch Size   : {train_loader.batch_size}")
    print(f"   |-- Dispositivo  : {DEVICE}")
    print(f"   |-- Precisión    : {'AMP FP16' if DEVICE.type == 'cuda' and USE_AMP else 'FP32'}")
    print(f"   |-- Mixup        : {'Activado (alpha=' + str(MIXUP_ALPHA) + ')' if train_loader.use_mixup else 'Desactivado'}")
    print(f"   +-- Early Stop   : {EARLY_STOPPING_PATIENCE} épocas sin mejora")

    start_time = time.time()
    last_epoch = 0

    # ══════════════════════════════════════════════════════════════════════════
    # CICLO DE ENTRENAMIENTO
    # ══════════════════════════════════════════════════════════════════════════
    for epoch in range(num_epochs):
        print(f"\n{'─' * 60}")
        last_epoch = epoch

        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, optimizer, scaler,
            DEVICE, epoch, num_epochs, is_train=True,
        )
        val_loss, val_acc = _run_epoch(
            model, val_loader, criterion, optimizer, scaler,
            DEVICE, epoch, num_epochs, is_train=False,
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\n  Resumen Época {epoch+1}/{num_epochs}:")
        print(f"     Train → Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"     Val   → Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
        print(f"     LR: {current_lr:.6f}")

        # ── Guardar mejor modelo ───────────────────────────────────────────
        improved = val_loss < best_val_loss or val_acc > best_val_acc
        if improved:
            best_val_loss = min(best_val_loss, val_loss)
            best_val_acc  = max(best_val_acc, val_acc)
            patience_counter = 0

            torch.save({
                "epoch":             epoch + 1,
                "arch":              arch,
                "model_state_dict":  model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss":          val_loss,
                "val_acc":           val_acc,
                "history":           history,
            }, BEST_MODEL_PATH)
            print(f"     [OK] MEJOR MODELO → Val Acc: {val_acc:.2f}% | Val Loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"     [...] Sin mejora ({patience_counter}/{EARLY_STOPPING_PATIENCE})")
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n  [STOP] Early Stopping en época {epoch+1}")
                break

    # ── Guardar último estado ─────────────────────────────────────────────────
    torch.save({
        "epoch":             last_epoch + 1,
        "arch":              arch,
        "model_state_dict":  model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history":           history,
    }, LAST_MODEL_PATH)

    # ── Gráficas y log ────────────────────────────────────────────────────────
    _plot_history(history, PLOTS_DIR, arch_name=arch)

    log_data = {
        "arch":                   arch,
        "total_epochs":           last_epoch + 1,
        "best_val_loss":          best_val_loss,
        "best_val_acc":           best_val_acc,
        "training_time_minutes":  (time.time() - start_time) / 60,
        "device":                 str(DEVICE),
        "history":                history,
    }
    log_path = os.path.join(LOGS_DIR, "entrenamiento_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(">>> ENTRENAMIENTO COMPLETADO")
    print("=" * 60)
    print(f"   Arquitectura      : {arch.upper()}")
    print(f"   Épocas            : {last_epoch + 1}")
    print(f"   Tiempo total      : {format_time(total_time)}")
    print(f"   Mejor Val Loss    : {best_val_loss:.4f}")
    print(f"   Mejor Val Acc     : {best_val_acc:.2f}%")
    print(f"   Modelo guardado   : {BEST_MODEL_PATH}")
    print(f"   Gráficas en       : {PLOTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    train()
