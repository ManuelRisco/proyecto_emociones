# ============================================================================
# DEMO EN TIEMPO REAL CON WEBCAM
# Reconocimiento de Emociones Faciales con CNN
# Autor: Manuel Fabrizzio Risco Gil
# ============================================================================
# Este script permite:
# 1. Capturar video de la webcam en tiempo real
# 2. Detectar rostros automaticamente
# 3. Clasificar la emocion de cada rostro usando la CNN
# 4. Mostrar los resultados en pantalla con overlay visual
# ============================================================================

import os
import sys
import io

# Forzar UTF-8 en Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import cv2
import cv2.data
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

# Asegurar que el directorio raiz del proyecto este en sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    DEVICE, BEST_MODEL_PATH, IMG_SIZE,
    EMOTION_LABELS, EMOTION_EMOJIS, NUM_CLASSES, COLORS
)
from src.model import EmotionCNN
from src.predict import load_model, get_face_detector, preprocess_face


def create_overlay(frame, text, position, font_scale=0.7, color=(0, 255, 0),
                   bg_color=(0, 0, 0), thickness=2):
    """
    Crea un texto con fondo semitransparente en el frame.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

    x, y = position
    pad = 5
    cv2.rectangle(
        frame,
        (x - pad, y - text_size[1] - pad),
        (x + text_size[0] + pad, y + pad),
        bg_color,
        -1
    )
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)


def draw_probability_bar(frame, x, y, width, height, probability, label,
                         color=(0, 255, 0)):
    """
    Dibuja una barra de probabilidad horizontal en el frame.
    """
    cv2.rectangle(frame, (x, y), (x + width, y + height), (50, 50, 50), -1)

    bar_width = int(width * probability)
    cv2.rectangle(frame, (x, y), (x + bar_width, y + height), color, -1)

    cv2.rectangle(frame, (x, y), (x + width, y + height), (100, 100, 100), 1)

    text = f"{label}: {probability * 100:.0f}%"
    cv2.putText(frame, text, (x + 5, y + height - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)


def run_realtime():
    """
    Ejecuta el reconocimiento de emociones en tiempo real usando la webcam.
    """
    print("\n" + "=" * 60)
    print(">>> DEMO EN TIEMPO REAL - RECONOCIMIENTO DE EMOCIONES")
    print("=" * 60)

    print("   Cargando modelo CNN...")
    model = load_model()

    print("   Inicializando detector de rostros...")
    face_cascade = get_face_detector()

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    emotion_colors = {
        0: (0, 0, 255),     # Enojado - Rojo
        1: (0, 128, 0),     # Disgusto - Verde oscuro
        2: (128, 0, 128),   # Miedo - Morado
        3: (0, 255, 255),   # Feliz - Amarillo
        4: (255, 0, 0),     # Triste - Azul
        5: (0, 165, 255),   # Sorprendido - Naranja
        6: (128, 128, 128), # Neutral - Gris
    }

    print("   Abriendo webcam...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("\n[!] No se pudo abrir la webcam.")
        print("   Verifica que tu camara este conectada y no este en uso por otra app.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n" + "=" * 60)
    print("   [OK] Webcam activada!")
    print("   Controles:")
    print("   |-- 'q' o 'ESC': Salir")
    print("   |-- 's': Capturar screenshot")
    print("   +-- 'p': Pausar/Reanudar")
    print("=" * 60)

    paused = False
    frame_count = 0
    frame = None

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("   [!] Error al capturar frame.")
                break

        if frame is None:
            continue

        frame_count += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(48, 48)
        )

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            pil_face = Image.fromarray(face_roi)
            tensor = transform(pil_face).unsqueeze(0).to(DEVICE)

            model.eval()
            with torch.no_grad():
                outputs = model(tensor)
                probs = torch.softmax(outputs, dim=1)
                pred_idx = int(torch.argmax(probs, dim=1).item())
                confidence = float(probs[0, pred_idx].item())

            color = emotion_colors.get(pred_idx, (255, 255, 255))
            label = EMOTION_LABELS[pred_idx]
            tag_text = f"{label} ({confidence * 100:.0f}%)"

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            create_overlay(frame, tag_text, (x, y - 5), font_scale=0.6, color=color)

            bar_x = x + w + 10
            bar_y = y
            bar_width = 150
            bar_height = 18

            if bar_x + bar_width < frame.shape[1]:
                for i in range(NUM_CLASSES):
                    prob = float(probs[0, i].item())
                    bar_color = emotion_colors.get(i, (128, 128, 128))
                    draw_probability_bar(
                        frame, bar_x, bar_y + i * (bar_height + 2),
                        bar_width, bar_height, prob,
                        EMOTION_LABELS[i], bar_color
                    )

        header_text = "CNN Emociones Faciales | RTX 5060 Ti"
        create_overlay(frame, header_text, (10, 25), font_scale=0.5, color=(0, 255, 255))

        faces_text = f"Rostros: {len(faces)}"
        create_overlay(frame, faces_text, (10, 50), font_scale=0.5, color=(0, 255, 0))

        device_text = f"Device: {str(DEVICE).upper()}"
        create_overlay(frame, device_text, (10, 75), font_scale=0.5, color=(0, 200, 255))

        if paused:
            create_overlay(frame, "PAUSADO", (frame.shape[1] // 2 - 50, 30),
                          font_scale=0.8, color=(0, 0, 255))

        help_text = "q:Salir | s:Screenshot | p:Pausar"
        create_overlay(
            frame, help_text,
            (10, frame.shape[0] - 10),
            font_scale=0.4, color=(200, 200, 200)
        )

        cv2.imshow("Reconocimiento de Emociones - CNN", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            print("\n   Cerrando webcam...")
            break
        elif key == ord("s"):
            screenshot_path = f"outputs/plots/screenshot_{frame_count}.png"
            cv2.imwrite(screenshot_path, frame)
            print(f"   [FOTO] Screenshot guardado: {screenshot_path}")
        elif key == ord("p"):
            paused = not paused
            print(f"   {'[PAUSA] Pausado' if paused else '[RUN] Reanudado'}")

    cap.release()
    cv2.destroyAllWindows()
    print("   [OK] Webcam cerrada correctamente.")


if __name__ == "__main__":
    run_realtime()
