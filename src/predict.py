# ============================================================================
# PREDICCIÓN DE EMOCIONES EN IMÁGENES INDIVIDUALES Y DETECTOR FACIAL
# Reconocimiento de Emociones Faciales con Deep Learning
# Autor: Manuel Fabrizzio Risco Gil
# Curso: Sistemas Inteligentes y Machine Learning - Ciclo 9
# ============================================================================
# Este script permite:
# 1. Cargar el detector neuronal YuNet (robusto ante giros y perfiles)
# 2. Cargar el modelo de emociones automáticamente (ResNet-SE o CNN clásica)
# 3. Predecir y graficar las probabilidades de cada emoción
# ============================================================================

import os
import sys
import io
import urllib.request
import numpy as np

# Forzar UTF-8 en Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

import cv2
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
    EMOTION_LABELS, EMOTION_EMOJIS, NUM_CLASSES, COLORS,
    YUNET_MODEL_PATH, YUNET_SCORE_THRESHOLD, YUNET_NMS_THRESHOLD,
    HAAR_CASCADE_PATH
)
from src.model import EmotionCNN, EmotionResNetSE, build_model


# ============================================================================
# 1. CARGADOR INTELIGENTE DE MODELOS
# ============================================================================

def load_model(model_path=BEST_MODEL_PATH):
    """
    Carga el modelo entrenado detectando automáticamente si es
    EmotionResNetSE o EmotionCNN a partir del checkpoint.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"\n[!] No se encontro el modelo: {model_path}\n"
            f"   Primero entrena el modelo (opcion 1 del menu)."
        )

    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Detección inteligente de la arquitectura guardada
    arch = checkpoint.get("arch", None)
    if arch is None:
        if any(k.startswith("stage1") or k.startswith("stem") for k in state_dict.keys()):
            arch = "resnet_se"
        else:
            arch = "cnn_clasica"

    model = build_model(arch=arch, num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    val_acc = checkpoint.get("val_acc", None)
    epoch = checkpoint.get("epoch", None)
    acc_info = f" (Val Acc: {val_acc:.2f}%)" if val_acc is not None else ""
    epoch_info = f" [Epoca {epoch}]" if epoch is not None else ""
    print(f"   [OK] Modelo cargado: {model.__class__.__name__} ({arch.upper()}){epoch_info}{acc_info}")

    return model


# ============================================================================
# 2. DETECTOR DE ROSTROS ROBUSTO (YuNet DNN + Fallback Haar)
# ============================================================================

class FaceDetector:
    """
    Detector de rostros multi-ángulo de alta precisión basado en YuNet ONNX.
    Capaz de detectar perfiles (yaw ±90°), caras inclinadas (roll/pitch) y
    eliminar falsos positivos en ropa/fondos mediante umbrales neuronales.
    """

    def __init__(self, mode="yunet", score_thresh=YUNET_SCORE_THRESHOLD):
        self.mode = mode.lower()
        self.score_thresh = score_thresh
        self.yunet = None
        self.haar = None
        self._init_detector()

    def _ensure_yunet_model(self):
        """Descarga el modelo YuNet ONNX si no existe localmente."""
        if not os.path.exists(YUNET_MODEL_PATH):
            print(f"   [DOWNLOAD] Descargando modelo YuNet ONNX (~230 KB)...")
            url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
            try:
                os.makedirs(os.path.dirname(YUNET_MODEL_PATH), exist_ok=True)
                urllib.request.urlretrieve(url, YUNET_MODEL_PATH)
                print(f"   [OK] YuNet descargado en: {YUNET_MODEL_PATH}")
            except Exception as e:
                print(f"   [!] Error al descargar YuNet: {e}")

    def _init_detector(self):
        """Inicializa el detector primario y el de respaldo."""
        if self.mode == "yunet" and hasattr(cv2, "FaceDetectorYN"):
            self._ensure_yunet_model()
            if os.path.exists(YUNET_MODEL_PATH):
                try:
                    self.yunet = cv2.FaceDetectorYN.create(
                        model=YUNET_MODEL_PATH,
                        config="",
                        input_size=(320, 320),
                        score_threshold=self.score_thresh,
                        nms_threshold=YUNET_NMS_THRESHOLD,
                        top_k=5000
                    )
                    return
                except Exception as e:
                    print(f"   [!] No se pudo instanciar YuNet: {e}. Usando Haar Cascade...")

        # Fallback a Haar Cascade
        cascade_path = HAAR_CASCADE_PATH if os.path.exists(HAAR_CASCADE_PATH) else (cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.haar = cv2.CascadeClassifier(cascade_path)
        self.mode = "haar"

    def detect(self, image_bgr):
        """
        Detecta rostros en una imagen BGR.

        Returns:
            list de dicts: [{'bbox': (x, y, w, h), 'confidence': float, 'landmarks': np.ndarray or None}]
        """
        h_img, w_img = image_bgr.shape[:2]
        results = []

        if self.mode == "yunet" and self.yunet is not None:
            self.yunet.setInputSize((w_img, h_img))
            _, faces = self.yunet.detect(image_bgr)

            if faces is not None:
                for face in faces:
                    x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                    confidence = float(face[14])

                    # Clamping a los límites de la imagen
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, w_img - x)
                    h = min(h, h_img - y)

                    if w < 16 or h < 16:
                        continue

                    # 5 puntos faciales (ojos, nariz, comisuras boca)
                    landmarks = face[4:14].reshape((5, 2)).astype(int)

                    results.append({
                        "bbox": (x, y, w, h),
                        "confidence": confidence,
                        "landmarks": landmarks
                    })
            return results

        # Modo Haar Cascade
        if self.haar is not None:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            faces = self.haar.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(32, 32)
            )
            for (x, y, w, h) in faces:
                results.append({
                    "bbox": (int(x), int(y), int(w), int(h)),
                    "confidence": 1.0,
                    "landmarks": None
                })

        return results


def get_face_detector(mode="yunet"):
    """Fábrica para obtener una instancia de FaceDetector."""
    return FaceDetector(mode=mode)


# ============================================================================
# 3. PREPROCESAMIENTO E INFERENCIA
# ============================================================================

def preprocess_face(face_img):
    """Preprocesa un recorte de rostro para la red neuronal."""
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


def predict_single_image(image_path, model=None, detector=None):
    """
    Predice las emociones de todos los rostros detectados en una imagen.
    """
    if not os.path.exists(image_path):
        print(f"\n[!] Imagen no encontrada: {image_path}")
        return None

    if model is None:
        model = load_model()

    if detector is None:
        detector = FaceDetector(mode="yunet")

    img = cv2.imread(image_path)
    if img is None:
        print(f"\n[!] No se pudo leer la imagen: {image_path}")
        return None

    detections = detector.detect(img)

    if len(detections) == 0:
        print("\n[!] No se detectaron rostros con el detector neuronal. Procesando imagen completa...")
        detections = [{
            "bbox": (0, 0, img.shape[1], img.shape[0]),
            "confidence": 1.0,
            "landmarks": None
        }]
    else:
        print(f"\n   Rostros detectados ({detector.mode.upper()}): {len(detections)}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results = []

    for i, det in enumerate(detections):
        x, y, w, h = det["bbox"]
        face_roi = gray[y:y+h, x:x+w]

        if face_roi.size == 0:
            continue

        tensor = preprocess_face(face_roi)
        predicted_class, probabilities = model.predict_emotion(tensor)

        pred_idx = int(predicted_class.item())
        pred_label = EMOTION_LABELS[pred_idx]
        confidence = float(probabilities[0][pred_idx].item()) * 100

        result = {
            "face_index": i,
            "bbox": (x, y, w, h),
            "face_confidence": det["confidence"] * 100,
            "predicted_class": pred_idx,
            "predicted_label": pred_label,
            "confidence": confidence,
            "probabilities": {
                EMOTION_LABELS[j]: float(probabilities[0][j].item()) * 100
                for j in range(NUM_CLASSES)
            },
            "landmarks": det["landmarks"]
        }
        results.append(result)

        print(f"\n   {'─' * 45}")
        print(f"   Rostro #{i+1}: {pred_label} ({confidence:.1f}%) [Det: {det['confidence']*100:.0f}%]")
        print(f"   {'─' * 45}")
        sorted_probs = sorted(
            result["probabilities"].items(),
            key=lambda item: item[1],
            reverse=True
        )
        for label, prob in sorted_probs:
            bar = "=" * int(prob / 2)
            print(f"     {label:>12}: {prob:5.1f}% {bar}")

    _visualize_prediction(img, results, image_path)
    return results


def _visualize_prediction(img, results, image_path):
    """Genera y guarda una visualización detallada de la predicción."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for r in results:
        x, y, w, h = r["bbox"]
        cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
        if r.get("landmarks") is not None:
            for pt in r["landmarks"]:
                cv2.circle(img_rgb, (int(pt[0]), int(pt[1])), 3, (255, 0, 0), -1)

    axes[0].imshow(img_rgb)
    axes[0].set_title("Detección de Rostros (YuNet)", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    if results:
        r = results[0]
        sorted_probs = sorted(
            r["probabilities"].items(),
            key=lambda item: item[1],
            reverse=True
        )
        labels = [l for l, _ in sorted_probs]
        values = [v for _, v in sorted_probs]
        colors_list = plt.cm.RdYlGn(np.array(values) / 100)

        axes[1].barh(labels[::-1], values[::-1], color=colors_list[::-1],
                     edgecolor="gray", height=0.6)
        axes[1].set_xlabel("Probabilidad (%)")
        axes[1].set_title(
            f"Predicción: {r['predicted_label']} ({r['confidence']:.1f}%)",
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
    print(f"\n   [GRAFICA] Visualización guardada en: {save_path}")


def predict_interactive():
    """Modo interactivo de predicción por consola."""
    print("\n" + "=" * 60)
    print(">>> PREDICCION DE EMOCIONES EN IMAGENES (YuNet + Deep Learning)")
    print("=" * 60)
    print("   Cargando modelo y detector neuronal...")

    model = load_model()
    detector = FaceDetector(mode="yunet")

    while True:
        print("\n" + "─" * 60)
        image_path = input("   Ingresa la ruta de la imagen (o 'salir'): ").strip()

        if image_path.lower() in ("salir", "exit", "q", "quit"):
            break

        image_path = image_path.strip("\"'")

        if os.path.exists(image_path):
            predict_single_image(image_path, model, detector)
        else:
            print(f"   [!] Archivo no encontrado: {image_path}")

    print("\n   Saliendo del modo de prediccion...")


if __name__ == "__main__":
    predict_interactive()
