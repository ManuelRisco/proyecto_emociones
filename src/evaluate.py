# ============================================================================
# EVALUACIÓN DEL MODELO ENTRENADO
# Reconocimiento de Emociones Faciales
# ============================================================================
# Genera:
#   1. Reporte de clasificación (Precision, Recall, F1 por emoción)
#   2. Matriz de confusión (absoluta y normalizada)
#   3. Accuracy por clase (gráfica de barras)
#   4. Grilla de ejemplos de predicciones (correctas e incorrectas)
# ============================================================================

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

from src.utils import ensure_root_in_path, setup_utf8
setup_utf8()
ensure_root_in_path()

from src.config import (
    BEST_MODEL_PATH, DEVICE, EMOTION_LABELS, LOGS_DIR, NUM_CLASSES, PLOTS_DIR,
)
from src.dataset import load_data
from src.predict import load_model


# ──────────────────────────────────────────────────────────────────────────────
# EXTRACCIÓN DE PREDICCIONES
# ──────────────────────────────────────────────────────────────────────────────

def get_all_predictions(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Realiza inferencia sobre todo el DataLoader y retorna arrays de resultados.

    Returns:
        (y_true, y_pred, y_probs, images): arrays numpy con etiquetas reales,
        predicciones, probabilidades softmax e imágenes normalizadas.
    """
    all_labels, all_preds, all_probs, all_images = [], [], [], []

    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="   Evaluando", ncols=80):
            images  = images.to(device)
            outputs = model(images)
            probs   = torch.softmax(outputs, dim=1)
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


# ──────────────────────────────────────────────────────────────────────────────
# VISUALIZACIONES
# ──────────────────────────────────────────────────────────────────────────────

def _save_fig(path: str) -> None:
    """Guarda la figura actual y la cierra."""
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   [GRAFICA] Guardada en: {path}")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_dir: str) -> None:
    """Matriz de confusión absoluta (conteos)."""
    cm     = confusion_matrix(y_true, y_pred)
    labels = [EMOTION_LABELS[i] for i in range(NUM_CLASSES)]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
    )
    plt.title("Matriz de Confusión — CNN Emociones Faciales", fontsize=14, fontweight="bold", pad=20)
    plt.xlabel("Predicción", fontsize=12)
    plt.ylabel("Real", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    _save_fig(os.path.join(save_dir, "matriz_confusion.png"))


def plot_confusion_matrix_normalized(y_true: np.ndarray, y_pred: np.ndarray, save_dir: str) -> None:
    """Matriz de confusión normalizada por clase real (porcentajes)."""
    cm     = confusion_matrix(y_true, y_pred, normalize="true")
    labels = [EMOTION_LABELS[i] for i in range(NUM_CLASSES)]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt=".2%", cmap="YlOrRd",
        xticklabels=labels, yticklabels=labels,
        square=True, linewidths=0.5, vmin=0, vmax=1, cbar_kws={"shrink": 0.8},
    )
    plt.title("Matriz de Confusión Normalizada", fontsize=14, fontweight="bold", pad=20)
    plt.xlabel("Predicción", fontsize=12)
    plt.ylabel("Real", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    _save_fig(os.path.join(save_dir, "matriz_confusion_normalizada.png"))


def plot_per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, save_dir: str) -> None:
    """Gráfica de barras horizontales con accuracy por emoción."""
    cm              = confusion_matrix(y_true, y_pred)
    per_class_acc   = cm.diagonal() / cm.sum(axis=1) * 100
    labels          = [EMOTION_LABELS[i] for i in range(NUM_CLASSES)]
    colors          = plt.cm.RdYlGn(per_class_acc / 100)

    plt.figure(figsize=(10, 6))
    bars = plt.barh(labels, per_class_acc, color=colors, edgecolor="gray", height=0.6)
    plt.xlabel("Accuracy (%)", fontsize=12)
    plt.title("Accuracy por Emoción", fontsize=14, fontweight="bold")
    plt.xlim(0, 100)
    plt.grid(axis="x", alpha=0.3)
    for bar, acc in zip(bars, per_class_acc):
        plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                 f"{acc:.1f}%", va="center", fontweight="bold")
    _save_fig(os.path.join(save_dir, "accuracy_por_emocion.png"))


def plot_sample_predictions(
    images: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    save_dir: str,
    n_samples: int = 16,
) -> None:
    """Grilla con ejemplos de predicciones (verde=correcto, rojo=incorrecto)."""
    n_samples = min(n_samples, len(images))
    indices   = np.random.choice(len(images), n_samples, replace=False)
    cols, rows = 4, (n_samples + 3) // 4

    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    fig.suptitle("Ejemplos de Predicciones", fontsize=14, fontweight="bold")

    for idx, ax in enumerate(axes.flat):
        if idx < n_samples:
            i          = indices[idx]
            img        = np.clip(images[i].squeeze() * 0.5 + 0.5, 0, 1)
            real       = EMOTION_LABELS[y_true[i]]
            pred       = EMOTION_LABELS[y_pred[i]]
            conf       = probs[i][y_pred[i]] * 100
            is_correct = y_true[i] == y_pred[i]

            ax.imshow(img, cmap="gray")
            ax.set_title(
                f"{'[OK]' if is_correct else '[X]'} {pred}\n({conf:.0f}%) | Real: {real}",
                fontsize=9, color="green" if is_correct else "red", fontweight="bold",
            )
        ax.axis("off")

    _save_fig(os.path.join(save_dir, "ejemplos_predicciones.png"))


# ──────────────────────────────────────────────────────────────────────────────
# FUNCIÓN PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────

def evaluate() -> float:
    """
    Evalúa el mejor modelo guardado en el conjunto de test.
    Genera:
      - Accuracy general
      - Reporte por emocion: Precision / Recall / F1-score
      - Matriz de confusion (absoluta y normalizada)
      - Grilla de ejemplos de predicciones
      - Tabla visual con codigo de colores por emocion
      - Archivo de texto con el reporte completo

    Returns:
        Accuracy general (%) en el conjunto de test.
    """
    print("\n" + "=" * 60)
    print(">>> EVALUACION COMPLETA DEL MODELO (CONJUNTO DE TEST)")
    print("=" * 60)

    model = load_model(BEST_MODEL_PATH)
    _, _, test_loader = load_data()

    print("\n   Ejecutando inferencia en 7,178 imagenes del conjunto TEST...")
    y_true, y_pred, y_probs, images = get_all_predictions(model, test_loader, DEVICE)

    overall_acc = accuracy_score(y_true, y_pred) * 100
    target_names = [EMOTION_LABELS[i] for i in range(NUM_CLASSES)]
    report_str   = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    report_dict  = classification_report(
        y_true, y_pred, target_names=target_names, output_dict=True
    )

    # ── Imprimir resultados en consola ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTADOS FINALES — CONJUNTO DE TEST")
    print("=" * 60)
    print(f"\n  Accuracy General : {overall_acc:.2f}%")
    print(f"  Total muestras   : {len(y_true):,} imagenes")

    # Tabla visual de precision/recall/F1 por emocion
    print("\n" + "-" * 60)
    print(f"  {'Emocion':<14} {'Soporte':>8} {'Precision':>10} {'Recall':>9} {'F1-score':>10}")
    print("-" * 60)
    for name in target_names:
        d    = report_dict[name]
        f1   = d["f1-score"]
        sup  = int(d["support"])
        prec = d["precision"]
        rec  = d["recall"]
        # Codigo de colores en consola
        if f1 >= 0.70:
            marker = "[OK]"
        elif f1 >= 0.50:
            marker = "[~~]"
        else:
            marker = "[!] "
        print(f"  {marker} {name:<10} {sup:>8,} {prec:>10.4f} {rec:>9.4f} {f1:>10.4f}")
    print("-" * 60)
    macro = report_dict["macro avg"]
    print(
        f"  {'Macro avg':<14} {int(macro['support']):>8,}"
        f" {macro['precision']:>10.4f} {macro['recall']:>9.4f} {macro['f1-score']:>10.4f}"
    )
    print("-" * 60)

    print("\n  Reporte completo sklearn:")
    for line in report_str.split("\n"):
        print(f"  {line}")

    # ── Guardar reporte en archivo de texto ──────────────────────────────
    os.makedirs(LOGS_DIR, exist_ok=True)
    report_path = os.path.join(LOGS_DIR, "evaluacion_test.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("EVALUACION DEL MODELO — CONJUNTO DE TEST\n")
        f.write("=" * 60 + "\n")
        f.write(f"Accuracy General : {overall_acc:.2f}%\n")
        f.write(f"Total muestras   : {len(y_true):,}\n\n")
        f.write("Reporte de Clasificacion:\n")
        f.write(report_str + "\n")
    print(f"\n  [REPORTE] Guardado en: {report_path}")

    # ── Graficas ────────────────────────────────────────────────────────────
    print("\n  Generando visualizaciones...")
    plot_confusion_matrix(y_true, y_pred, PLOTS_DIR)
    plot_confusion_matrix_normalized(y_true, y_pred, PLOTS_DIR)
    plot_sample_predictions(images, y_true, y_pred, y_probs, PLOTS_DIR)
    plot_per_class_accuracy(y_true, y_pred, PLOTS_DIR)
    _plot_f1_by_class(report_dict, target_names, PLOTS_DIR)

    print(f"\n{'=' * 60}")
    print(f"[OK] EVALUACION COMPLETADA")
    print(f"     Acc Test: {overall_acc:.2f}% | Graficas: {PLOTS_DIR}")
    print(f"{'=' * 60}")

    return overall_acc


def _plot_f1_by_class(
    report_dict: dict,
    target_names: list[str],
    save_dir: str,
) -> None:
    """Grafica de barras horizontales con F1-score por emocion (con codigo de color)."""
    f1_scores = [report_dict[n]["f1-score"] for n in target_names]
    prec      = [report_dict[n]["precision"] for n in target_names]
    rec       = [report_dict[n]["recall"] for n in target_names]
    support   = [int(report_dict[n]["support"]) for n in target_names]

    # Ordenar por F1 ascendente
    order      = sorted(range(len(f1_scores)), key=lambda i: f1_scores[i])
    names_ord  = [target_names[i] for i in order]
    f1_ord     = [f1_scores[i]    for i in order]
    prec_ord   = [prec[i]         for i in order]
    rec_ord    = [rec[i]          for i in order]
    sup_ord    = [support[i]      for i in order]

    colors = ["#e74c3c" if v < 0.5 else "#f39c12" if v < 0.7 else "#27ae60" for v in f1_ord]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Reporte por Emocion — Conjunto de TEST", fontsize=14, fontweight="bold")

    # F1-score
    bars = axes[0].barh(names_ord, f1_ord, color=colors, edgecolor="gray", height=0.6)
    axes[0].axvline(0.7, color="green",  linestyle="--", alpha=0.6, label="F1=0.70")
    axes[0].axvline(0.5, color="orange", linestyle="--", alpha=0.6, label="F1=0.50")
    axes[0].set_xlabel("F1-score")
    axes[0].set_title("F1-score por Emocion")
    axes[0].set_xlim(0, 1.05)
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="x", alpha=0.3)
    for bar, v in zip(bars, f1_ord):
        axes[0].text(v + 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{v:.3f}", va="center", fontweight="bold", fontsize=10)

    # Precision vs Recall
    y_pos = range(len(names_ord))
    axes[1].barh([p + 0.18 for p in y_pos], prec_ord, height=0.35,
                 color="steelblue", label="Precision", edgecolor="gray")
    axes[1].barh([p - 0.18 for p in y_pos], rec_ord,  height=0.35,
                 color="tomato",    label="Recall",    edgecolor="gray")
    axes[1].set_yticks(list(y_pos))
    axes[1].set_yticklabels([f"{n} (n={s:,})" for n, s in zip(names_ord, sup_ord)])
    axes[1].set_xlabel("Valor")
    axes[1].set_title("Precision vs Recall por Emocion")
    axes[1].set_xlim(0, 1.1)
    axes[1].legend()
    axes[1].grid(axis="x", alpha=0.3)

    _save_fig(os.path.join(save_dir, "f1_precision_recall_por_emocion.png"))


if __name__ == "__main__":
    evaluate()
