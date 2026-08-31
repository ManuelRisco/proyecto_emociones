# ============================================================================
# SCRIPT DE ENTRENAMIENTO DE LA CNN
# Reconocimiento de Emociones Faciales
# Autor: Manuel Fabrizzio Risco Gil
# ============================================================================
# Este script entrena la CNN usando la GPU (RTX 5060 Ti) para:
# 1. Aprender patrones faciales de las 7 emociones
# 2. Validar el rendimiento en cada época
# 3. Guardar el mejor modelo automáticamente
# 4. Generar gráficas de entrenamiento (loss y accuracy)
# ============================================================================

import os
import sys
import io
import time
import json

# Forzar UTF-8 en Windows para evitar errores de encoding
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
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
    PLOTS_DIR, LOGS_DIR, EMOTION_LABELS, USE_AMP
)
from src.dataset import load_data
from src.model import EmotionCNN, get_model_summary


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, num_epochs):
    """
    Entrena el modelo por una época completa con optimización de GPU y Mixed Precision.

    Args:
        model: modelo CNN
        train_loader: DataLoader de entrenamiento
        criterion: función de pérdida
        optimizer: optimizador
        scaler: GradScaler para Automatic Mixed Precision
        device: dispositivo (GPU/CPU)
        epoch: número de época actual
        num_epochs: total de épocas

    Returns:
        tuple: (loss promedio, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    use_cuda_amp = (device.type == "cuda" and USE_AMP)

    pbar = tqdm(
        train_loader,
        desc=f"  Epoca {epoch+1}/{num_epochs} [Train]",
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    )

    for batch_idx, (images, labels) in enumerate(pbar):
        # Mover datos a GPU de forma no bloqueante
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Forward y Backward pass con Precision Mixta
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

        # Estadísticas
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Actualizar barra de progreso
        current_loss = running_loss / (batch_idx + 1)
        current_acc = 100.0 * correct / total
        pbar.set_postfix({"Loss": f"{current_loss:.4f}", "Acc": f"{current_acc:.1f}%"})

    avg_loss = running_loss / len(train_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def validate(model, val_loader, criterion, device, epoch, num_epochs):
    """
    Evalúa el modelo en el conjunto de validación con aceleración de GPU.

    Args:
        model: modelo CNN
        val_loader: DataLoader de validación
        criterion: función de pérdida
        device: dispositivo (GPU/CPU)
        epoch: número de época actual
        num_epochs: total de épocas

    Returns:
        tuple: (loss promedio, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    use_cuda_amp = (device.type == "cuda" and USE_AMP)

    pbar = tqdm(
        val_loader,
        desc=f"  Epoca {epoch+1}/{num_epochs} [Val]  ",
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


def plot_training_history(history, save_dir):
    """
    Genera gráficas de loss y accuracy del entrenamiento.

    Args:
        history: diccionario con listas de métricas por época
        save_dir: directorio donde guardar las gráficas
    """
    epochs_range = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Historial de Entrenamiento - CNN Emociones Faciales",
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


def train():
    """
    Función principal de entrenamiento.
    Ejecuta el ciclo completo de entrenamiento de la CNN.
    """
    print("\n" + "=" * 60)
    print(">>> INICIANDO ENTRENAMIENTO DE LA CNN")
    print("=" * 60)

    # ── Cargar datos ──
    train_loader, val_loader, _ = load_data()

    # ── Crear modelo ──
    model = EmotionCNN().to(DEVICE)
    get_model_summary(model)

    # ── Configurar entrenamiento ──
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )

    # GradScaler para Automatic Mixed Precision (AMP)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda" and USE_AMP))

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0

    print(f"\n   Configuracion:")
    print(f"   |-- Epocas:        {NUM_EPOCHS}")
    print(f"   |-- Learning Rate: {LEARNING_RATE}")
    print(f"   |-- Batch Size:    {train_loader.batch_size}")
    print(f"   |-- Dispositivo:   {DEVICE}")
    print(f"   |-- Precision:     {'Mixed Precision (FP16/AMP)' if (DEVICE.type == 'cuda' and USE_AMP) else 'FP32'}")
    print(f"   +-- Early Stop:    {EARLY_STOPPING_PATIENCE} epocas sin mejora")

    start_time = time.time()

    # ══════════════════════════════════════════════════════════════
    # CICLO DE ENTRENAMIENTO
    # ══════════════════════════════════════════════════════════════
    for epoch in range(NUM_EPOCHS):
        print(f"\n{'─' * 60}")

        # Entrenamiento
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, DEVICE,
            epoch, NUM_EPOCHS
        )

        # Validación
        val_loss, val_acc = validate(
            model, val_loader, criterion, DEVICE,
            epoch, NUM_EPOCHS
        )

        # Guardar historial
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Actualizar learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Resumen de la época
        print(f"\n  Resumen Epoca {epoch+1}/{NUM_EPOCHS}:")
        print(f"     Train  -> Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"     Val    -> Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
        print(f"     LR: {current_lr:.6f}")

        # ── Early Stopping y guardado del mejor modelo ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0

            # Guardar mejor modelo
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "history": history,
            }, BEST_MODEL_PATH)

            print(f"     [OK] MEJOR MODELO GUARDADO (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"     [...] Sin mejora ({patience_counter}/{EARLY_STOPPING_PATIENCE})")

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n  [STOP] Early Stopping activado en epoca {epoch+1}")
                break

    # ── Guardar último modelo ──
    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
    }, LAST_MODEL_PATH)

    total_time = time.time() - start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)

    # ── Generar gráficas ──
    plot_training_history(history, PLOTS_DIR)

    # ── Guardar log ──
    log_path = os.path.join(LOGS_DIR, "entrenamiento_log.json")
    log_data = {
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
    print(f"   Epocas completadas:    {epoch + 1}")
    print(f"   Tiempo total:          {minutes}m {seconds}s")
    print(f"   Mejor Val Loss:        {best_val_loss:.4f}")
    print(f"   Mejor Val Accuracy:    {best_val_acc:.2f}%")
    print(f"   Modelo guardado en:    {BEST_MODEL_PATH}")
    print(f"   Graficas en:           {PLOTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    train()
