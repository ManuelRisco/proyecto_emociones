# ============================================================================
# PREDICCION DE EMOCIONES EN IMAGENES INDIVIDUALES
# Reconocimiento de Emociones Faciales con CNN
# Autor: Manuel Fabrizzio Risco Gil
# ============================================================================
# Este script permite:
# 1. Cargar una imagen individual
# 2. Detectar el rostro automaticamente
# 3. Predecir la emocion usando la CNN entrenada
# 4. Mostrar resultados con probabilidades
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

import cv2
import cv2.data
import torch
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Asegurar que el directorio raiz del proyecto este en sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    DEVICE, BEST_MODEL_PATH, PLOTS_DIR, IMG_SIZE,
    EMOTION_LABELS, EMOTION_EMOJIS, NUM_CLASSES, COLORS
)
from src.model import EmotionCNN


def load_model():
    """Carga el modelo entrenado."""
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"\n[!] No se encontro el modelo: {BEST_MODEL_PATH}\n"
            f"   Primero entrena el modelo (opcion 1 del menu)."
        )

    model = EmotionCNN().to(DEVICE)
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def get_face_detector():
    """
    Obtiene el detector de rostros Haar Cascade de OpenCV.
    """
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        raise RuntimeError(
            "[!] No se pudo cargar el detector de rostros Haar Cascade.\n"
            "   Asegurate de tener opencv-python instalado correctamente."
        )

    return face_cascade


def preprocess_face(face_img):
    """
    Preprocesa una imagen de rostro para la CNN.
    """
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    if len(face_img.shape) == 3:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

    pil_img = Image.fromarray(face_img)
    tensor = transform(pil_img).unsqueeze(0)
    return tensor.to(DEVICE)


def predict_single_image(image_path, model=None):
    """
    Predice la emocion en una imagen dada.
    """
    if not os.path.exists(image_path):
        print(f"\n[!] Imagen no encontrada: {image_path}")
        return None

    if model is None:
        model = load_model()

    img = cv2.imread(image_path)
    if img is None:
        print(f"\n[!] No se pudo leer la imagen: {image_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = get_face_detector()
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        print("\n[!] No se detectaron rostros. Procesando imagen completa...")
        face_img = gray
        faces = [(0, 0, img.shape[1], img.shape[0])]
    else:
        print(f"\n   Rostros detectados: {len(faces)}")

    results = []
    for i, (x, y, w, h) in enumerate(faces):
        face_img = gray[y:y+h, x:x+w]

        tensor = preprocess_face(face_img)
        predicted_class, probabilities = model.predict_emotion(tensor)

        pred_idx = int(predicted_class.item())
        pred_label = EMOTION_LABELS[pred_idx]
        confidence = float(probabilities[0][pred_idx].item()) * 100

        result = {
            "face_index": i,
            "bbox": (x, y, w, h),
            "predicted_class": pred_idx,
            "predicted_label": pred_label,
            "confidence": confidence,
            "probabilities": {
                EMOTION_LABELS[j]: float(probabilities[0][j].item()) * 100
                for j in range(NUM_CLASSES)
            },
        }
        results.append(result)

        print(f"\n   {'─' * 40}")
        print(f"   Rostro #{i+1}: {pred_label} ({confidence:.1f}%)")
        print(f"   {'─' * 40}")
        print(f"   Probabilidades:")
        sorted_probs = sorted(
            result["probabilities"].items(),
            key=lambda x: x[1],
            reverse=True
        )
        for label, prob in sorted_probs:
            bar = "=" * int(prob / 2)
            print(f"     {label:>12}: {prob:5.1f}% {bar}")

    _visualize_prediction(img, results, image_path)

    return results


def _visualize_prediction(img, results, image_path):
    """
    Genera y guarda una visualizacion de la prediccion.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for r in results:
        x, y, w, h = r["bbox"]
        cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)

    axes[0].imshow(img_rgb)
    axes[0].set_title("Deteccion de Rostros", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    if results:
        r = results[0]
        sorted_probs = sorted(
            r["probabilities"].items(),
            key=lambda x: x[1],
            reverse=True
        )
        labels = [l for l, _ in sorted_probs]
        values = [v for _, v in sorted_probs]
        colors_list = plt.cm.RdYlGn(np.array(values) / 100)

        axes[1].barh(labels[::-1], values[::-1], color=colors_list[::-1],
                     edgecolor="gray", height=0.6)
        axes[1].set_xlabel("Probabilidad (%)")
        axes[1].set_title(
            f"Prediccion: {r['predicted_label']} ({r['confidence']:.1f}%)",
            fontsize=12, fontweight="bold"
        )
        axes[1].set_xlim(0, 100)
        axes[1].grid(axis="x", alpha=0.3)

        for i, (v, l) in enumerate(zip(values[::-1], labels[::-1])):
            axes[1].text(v + 1, i, f"{v:.1f}%", va="center", fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, "prediccion_imagen.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n   [GRAFICA] Visualizacion guardada en: {save_path}")


def predict_interactive():
    """
    Modo interactivo de prediccion.
    """
    print("\n" + "=" * 60)
    print(">>> PREDICCION DE EMOCIONES EN IMAGENES")
    print("=" * 60)
    print("   Cargando modelo...")

    model = load_model()

    while True:
        print("\n" + "─" * 60)
        image_path = input("   Ingresa la ruta de la imagen (o 'salir'): ").strip()

        if image_path.lower() in ("salir", "exit", "q", "quit"):
            break

        image_path = image_path.strip("\"'")

        if os.path.exists(image_path):
            predict_single_image(image_path, model)
        else:
            print(f"   [!] Archivo no encontrado: {image_path}")

    print("\n   Saliendo del modo de prediccion...")


if __name__ == "__main__":
    predict_interactive()
