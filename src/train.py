# ============================================================================
# SCRIPT DE ENTRENAMIENTO DE LA RED NEURONAL (ResNet-SE / CNN)
# Reconocimiento de Emociones Faciales
# Autor: Manuel Fabrizzio Risco Gil
# Curso: Sistemas Inteligentes y Machine Learning - Ciclo 9
# ============================================================================
# Este script entrena la arquitectura seleccionada con:
# 1. Ponderación de Clases (Class Weights) para equilibrar emociones minoritarias
# 2. Label Smoothing para evitar sobreconfianza en etiquetas ruidosas
# 3. Optimizador AdamW + Programador de Learning Rate adaptativo
# 4. Aceleración GPU con Precisión Mixta (AMP / FP16) para Tensor Cores
# ============================================================================

import os
import sys
import io
import time
import json

# Forzar UTF-8 en Windows para evitar errores de encoding
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")  # Backend sin GUI para guardar gráficas
import matplotlib.pyplot as plt

# Asegurar que el directorio raiz del proyecto este en sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    DEVICE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE, BEST_MODEL_PATH, LAST_MODEL_PATH,
    PLOTS_DIR, LOGS_DIR, EMOTION_LABELS, USE_AMP, DEFAULT_ARCHITECTURE
)
from src.dataset import load_data
from src.model import build_model, get_model_summary


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, num_epochs):
    """
    Entrena el modelo por una época completa con optimización de GPU y Mixed Precision.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    use_cuda_amp = (device.type == "cuda" and USE_AMP)

    pbar = tqdm(
        train_loader,
        desc=f"  Epoca {epoch+1:2d}/{num_epochs} [Train]",
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    )

    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_cuda_amp:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        current_loss = running_loss / (batch_idx + 1)
        current_acc = 100.0 * correct / total
        pbar.set_postfix({"Loss": f"{current_loss:.4f}", "Acc": f"{current_acc:.1f}%"})

    avg_loss = running_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device, epoch, num_epochs):
    """
    Evalúa el modelo en el conjunto de validación con aceleración de GPU.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    use_cuda_amp = (device.type == "cuda" and USE_AMP)

    pbar = tqdm(
        val_loader,
        desc=f"  Epoca {epoch+1:2d}/{num_epochs} [Val]  ",
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    )

    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if use_cuda_amp:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(val_loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def plot_training_history(history, save_dir, arch_name=""):
    """
    Genera gráficas de loss y accuracy del entrenamiento.
    """
    epochs_range = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Historial de Entrenamiento - {arch_name.upper()} Emociones Faciales",
                 fontsize=14, fontweight="bold")

    # Gráfica de Loss
    axes[0].plot(epochs_range, history["train_loss"], "b-o",
                 label="Train Loss", markersize=3)
    axes[0].plot(epochs_range, history["val_loss"], "r-o",
                 label="Validation Loss", markersize=3)
    axes[0].set_xlabel("Epoca")
    axes[0].set_ylabel("Loss (Perdida)")
    axes[0].set_title("Perdida por Epoca")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Gráfica de Accuracy
    axes[1].plot(epochs_range, history["train_acc"], "b-o",
                 label="Train Accuracy", markersize=3)
    axes[1].plot(epochs_range, history["val_acc"], "r-o",
                 label="Validation Accuracy", markersize=3)
    axes[1].set_xlabel("Epoca")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Precision por Epoca")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "historial_entrenamiento.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n   [GRAFICA] Graficas guardadas en: {plot_path}")


def train(arch=DEFAULT_ARCHITECTURE, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE):
    """
    Función principal de entrenamiento.
    Ejecuta el ciclo completo de entrenamiento.
    """
    print("\n" + "=" * 60)
    print(f">>> INICIANDO ENTRENAMIENTO: {arch.upper()}")
    print("=" * 60)

    # ── Cargar datos con pesos de clases ──
    train_loader, val_loader, _, class_weights = load_data(return_weights=True)

    # ── Crear modelo ──
    model = build_model(arch=arch, num_classes=len(EMOTION_LABELS)).to(DEVICE)
    get_model_summary(model)

    # ── Función de pérdida con Class Weights y Label Smoothing ──
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.05
    )

    # ── Optimizador AdamW + Scheduler Cosine Annealing ──
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6
    )

    # GradScaler para Automatic Mixed Precision (AMP)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda" and USE_AMP))

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "arch": arch,
    }

    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0

    print(f"\n   Configuracion:")
    print(f"   |-- Arquitectura:  {arch.upper()} ({model.__class__.__name__})")
    print(f"   |-- Epocas:        {num_epochs}")
    print(f"   |-- Learning Rate: {lr}")
    print(f"   |-- Batch Size:    {train_loader.batch_size}")
    print(f"   |-- Dispositivo:   {DEVICE}")
    print(f"   |-- Precision:     {'Mixed Precision (FP16/AMP)' if (DEVICE.type == 'cuda' and USE_AMP) else 'FP32'}")
    print(f"   |-- Ponderacion:   Ponderacion de clases activa (Class Weights)")
    print(f"   +-- Early Stop:    {EARLY_STOPPING_PATIENCE} epocas sin mejora")

    start_time = time.time()

    # ══════════════════════════════════════════════════════════════
    # CICLO DE ENTRENAMIENTO
    # ══════════════════════════════════════════════════════════════
    for epoch in range(num_epochs):
        print(f"\n{'─' * 60}")

        # Entrenamiento
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, DEVICE,
            epoch, num_epochs
        )

        # Validación
        val_loss, val_acc = validate(
            model, val_loader, criterion, DEVICE,
            epoch, num_epochs
        )

        # Guardar historial
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Actualizar learning rate con CosineAnnealing
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Resumen de la época
        print(f"\n  Resumen Epoca {epoch+1}/{num_epochs}:")
        print(f"     Train  -> Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"     Val    -> Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
        print(f"     LR: {current_lr:.6f}")

        # ── Early Stopping y guardado del mejor modelo ──
        if val_loss < best_val_loss or val_acc > best_val_acc:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            patience_counter = 0

            # Guardar mejor modelo con metadatos de arquitectura
            torch.save({
                "epoch": epoch + 1,
                "arch": arch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "history": history,
            }, BEST_MODEL_PATH)

            print(f"     [OK] MEJOR MODELO GUARDADO (Val Acc: {val_acc:.2f}% | Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"     [...] Sin mejora ({patience_counter}/{EARLY_STOPPING_PATIENCE})")

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n  [STOP] Early Stopping activado en epoca {epoch+1}")
                break

    # ── Guardar último modelo ──
    torch.save({
        "epoch": epoch + 1,
        "arch": arch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
    }, LAST_MODEL_PATH)

    total_time = time.time() - start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)

    # ── Generar gráficas ──
    plot_training_history(history, PLOTS_DIR, arch_name=arch)

    # ── Guardar log ──
    log_path = os.path.join(LOGS_DIR, "entrenamiento_log.json")
    log_data = {
        "arch": arch,
        "total_epochs": epoch + 1,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "training_time_minutes": total_time / 60,
        "device": str(DEVICE),
        "history": history,
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(">>> ENTRENAMIENTO COMPLETADO")
    print("=" * 60)
    print(f"   Arquitectura:          {arch.upper()}")
    print(f"   Epocas completadas:    {epoch + 1}")
    print(f"   Tiempo total:          {minutes}m {seconds}s")
    print(f"   Mejor Val Loss:        {best_val_loss:.4f}")
    print(f"   Mejor Val Accuracy:    {best_val_acc:.2f}%")
    print(f"   Modelo guardado en:    {BEST_MODEL_PATH}")
    print(f"   Graficas en:           {PLOTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    train()
