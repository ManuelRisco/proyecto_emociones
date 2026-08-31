# ============================================================================
# SCRIPT DE EVALUACION DEL MODELO
# Reconocimiento de Emociones Faciales con CNN
# Autor: Manuel Fabrizzio Risco Gil
# ============================================================================
# Este script evalua el modelo entrenado:
# 1. Calcula metricas de rendimiento (accuracy, precision, recall, F1-score)
# 2. Genera la matriz de confusion
# 3. Muestra ejemplos de predicciones correctas e incorrectas
# ============================================================================

import os
import sys
import io
import numpy as np

# Forzar UTF-8 en Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
from tqdm import tqdm

# Asegurar que el directorio raiz del proyecto este en sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    DEVICE, BEST_MODEL_PATH, PLOTS_DIR,
    EMOTION_LABELS, EMOTION_EMOJIS, NUM_CLASSES
)
from src.dataset import load_data
from src.model import build_model
from src.predict import load_model


def load_trained_model():
    """
    Carga el mejor modelo entrenado desde disco usando el loader inteligente.
    """
    return load_model(BEST_MODEL_PATH)



def get_all_predictions(model, data_loader, device):
    """
    Obtiene todas las predicciones del modelo para un DataLoader.
    """
    all_labels = []
    all_preds = []
    all_probs = []
    all_images = []

    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="   Evaluando", ncols=80):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_images.extend(images.cpu().numpy())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
        np.array(all_images),
    )


def plot_confusion_matrix(y_true, y_pred, save_dir):
    """
    Genera y guarda la matriz de confusion.
    """
    cm = confusion_matrix(y_true, y_pred)
    labels = [EMOTION_LABELS[i] for i in range(NUM_CLASSES)]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    plt.title("Matriz de Confusion - CNN Emociones Faciales",
              fontsize=14, fontweight="bold", pad=20)
    plt.xlabel("Prediccion", fontsize=12)
    plt.ylabel("Real", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    path = os.path.join(save_dir, "matriz_confusion.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   [GRAFICA] Matriz de confusion guardada en: {path}")


def plot_confusion_matrix_normalized(y_true, y_pred, save_dir):
    """
    Genera y guarda la matriz de confusion normalizada (porcentajes).
    """
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    labels = [EMOTION_LABELS[i] for i in range(NUM_CLASSES)]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2%",
        cmap="YlOrRd",
        xticklabels=labels,
        yticklabels=labels,
        square=True,
        linewidths=0.5,
        vmin=0,
        vmax=1,
        cbar_kws={"shrink": 0.8}
    )
    plt.title("Matriz de Confusion Normalizada - CNN Emociones Faciales",
              fontsize=14, fontweight="bold", pad=20)
    plt.xlabel("Prediccion", fontsize=12)
    plt.ylabel("Real", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    path = os.path.join(save_dir, "matriz_confusion_normalizada.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   [GRAFICA] Matriz normalizada guardada en: {path}")


def plot_sample_predictions(images, y_true, y_pred, probs, save_dir, n_samples=16):
    """
    Genera una grilla con ejemplos de predicciones del modelo.
    """
    n_samples = min(n_samples, len(images))
    indices = np.random.choice(len(images), n_samples, replace=False)

    cols = 4
    rows = (n_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    fig.suptitle("Ejemplos de Predicciones - CNN Emociones Faciales",
                 fontsize=14, fontweight="bold")

    for idx, ax in enumerate(axes.flat):
        if idx < n_samples:
            i = indices[idx]
            img = images[i].squeeze()
            img = img * 0.5 + 0.5
            img = np.clip(img, 0, 1)

            real = EMOTION_LABELS[y_true[i]]
            pred = EMOTION_LABELS[y_pred[i]]
            conf = probs[i][y_pred[i]] * 100
            is_correct = y_true[i] == y_pred[i]

            ax.imshow(img, cmap="gray")
            color = "green" if is_correct else "red"
            symbol = "[OK]" if is_correct else "[X]"
            ax.set_title(
                f"{symbol} {pred}\n({conf:.0f}%) | Real: {real}",
                fontsize=9,
                color=color,
                fontweight="bold"
            )
            ax.axis("off")
        else:
            ax.axis("off")

    plt.tight_layout()
    path = os.path.join(save_dir, "ejemplos_predicciones.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   [GRAFICA] Ejemplos de predicciones guardados en: {path}")


def plot_per_class_accuracy(y_true, y_pred, save_dir):
    """
    Genera grafica de accuracy por clase (emocion).
    """
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100

    labels = [EMOTION_LABELS[i] for i in range(NUM_CLASSES)]
    colors = plt.cm.RdYlGn(per_class_acc / 100)

    plt.figure(figsize=(10, 6))
    bars = plt.barh(labels, per_class_acc, color=colors, edgecolor="gray", height=0.6)
    plt.xlabel("Accuracy (%)", fontsize=12)
    plt.title("Accuracy por Emocion", fontsize=14, fontweight="bold")
    plt.xlim(0, 100)
    plt.grid(axis="x", alpha=0.3)

    for bar, acc in zip(bars, per_class_acc):
        plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                 f"{acc:.1f}%", va="center", fontweight="bold")

    plt.tight_layout()
    path = os.path.join(save_dir, "accuracy_por_emocion.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   [GRAFICA] Accuracy por emocion guardada en: {path}")


def evaluate():
    """
    Funcion principal de evaluacion.
    Evalua el modelo entrenado en el conjunto de test.
    """
    print("\n" + "=" * 60)
    print(">>> EVALUACION DEL MODELO CNN")
    print("=" * 60)

    model = load_trained_model()
    _, _, test_loader = load_data()

    y_true, y_pred, y_probs, images = get_all_predictions(
        model, test_loader, DEVICE
    )

    overall_acc = accuracy_score(y_true, y_pred) * 100
    print(f"\n{'=' * 60}")
    print(f"RESULTADOS EN EL CONJUNTO DE TEST")
    print(f"{'=' * 60}")
    print(f"\n   Accuracy General: {overall_acc:.2f}%")
    print(f"   Total de muestras: {len(y_true)}")

    target_names = [EMOTION_LABELS[i] for i in range(NUM_CLASSES)]
    report = str(classification_report(
        y_true, y_pred,
        target_names=target_names,
        digits=4
    ))
    print(f"\n   Reporte de Clasificacion:")
    print(f"   {'─' * 52}")
    for line in report.split("\n"):
        print(f"   {line}")

    print(f"\n   Generando visualizaciones...")
    plot_confusion_matrix(y_true, y_pred, PLOTS_DIR)
    plot_confusion_matrix_normalized(y_true, y_pred, PLOTS_DIR)
    plot_sample_predictions(images, y_true, y_pred, y_probs, PLOTS_DIR)
    plot_per_class_accuracy(y_true, y_pred, PLOTS_DIR)

    print(f"\n{'=' * 60}")
    print(f"[OK] EVALUACION COMPLETADA")
    print(f"   Todas las graficas guardadas en: {PLOTS_DIR}")
    print(f"{'=' * 60}")

    return overall_acc


if __name__ == "__main__":
    evaluate()
