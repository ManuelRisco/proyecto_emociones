# ============================================================================
# APLICACIÓN PORTÁTIL DE RECONOCIMIENTO DE EMOCIONES
# ============================================================================

import os
import sys
import time
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Búsqueda automática de modelos
YUNET_CANDIDATES = [
    os.path.join(BASE_DIR, "models", "face_detection_yunet_2023mar.onnx"),
    os.path.join(BASE_DIR, "..", "data", "face_detection_yunet_2023mar.onnx"),
    os.path.join(BASE_DIR, "..", "outputs", "models", "face_detection_yunet_2023mar.onnx"),
    os.path.join(BASE_DIR, "face_detection_yunet_2023mar.onnx"),
]
YUNET_MODEL = next((p for p in YUNET_CANDIDATES if os.path.exists(p)), YUNET_CANDIDATES[0])

EMOTION_CANDIDATES = [
    os.path.join(BASE_DIR, "models", "modelo_emociones_cv2.onnx"),
    os.path.join(BASE_DIR, "..", "outputs", "models", "modelo_emociones_cv2.onnx"),
    os.path.join(BASE_DIR, "modelo_emociones_cv2.onnx"),
]
EMOTION_MODEL = next((p for p in EMOTION_CANDIDATES if os.path.exists(p)), EMOTION_CANDIDATES[0])

EMOTIONS = {
    0: {"name": "Enojado",     "color": (40, 40, 235)},
    1: {"name": "Disgusto",    "color": (34, 139, 34)},
    2: {"name": "Miedo",       "color": (180, 50, 180)},
    3: {"name": "Feliz",       "color": (0, 215, 255)},
    4: {"name": "Triste",      "color": (235, 120, 40)},
    5: {"name": "Sorprendido", "color": (0, 140, 255)},
    6: {"name": "Neutral",     "color": (170, 170, 170)},
}


class PortableEmotionTracker:
    def __init__(self, alpha=0.30):
        self.alpha = alpha
        self.tracks = {}
        self.next_id = 1

    def update(self, detections):
        matched = set()
        for d in detections:
            x, y, w, h = d["bbox"]
            best_iou, best_tid = 0.0, None
            for tid, t in self.tracks.items():
                if tid in matched:
                    continue
                tx, ty, tw, th = t["bbox"]
                xi1, yi1 = max(x, tx), max(y, ty)
                xi2, yi2 = min(x + w, tx + tw), min(y + h, ty + th)
                inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                union = w * h + tw * th - inter
                iou = inter / union if union > 0 else 0
                if iou > best_iou:
                    best_iou, best_tid = iou, tid

            if best_iou >= 0.30 and best_tid is not None:
                matched.add(best_tid)
                t = self.tracks[best_tid]
                t["lost"] = 0
                t["bbox"] = (
                    int(0.35 * x + 0.65 * t["bbox"][0]),
                    int(0.35 * y + 0.65 * t["bbox"][1]),
                    int(0.35 * w + 0.65 * t["bbox"][2]),
                    int(0.35 * h + 0.65 * t["bbox"][3])
                )
                t["probs"] = self.alpha * d["probs"] + (1.0 - self.alpha) * t["probs"]
                t["landmarks"] = d.get("landmarks")
            else:
                self.tracks[self.next_id] = {
                    "id": self.next_id,
                    "bbox": d["bbox"],
                    "probs": d["probs"].copy(),
                    "lost": 0,
                    "landmarks": d.get("landmarks")
                }
                matched.add(self.next_id)
                self.next_id += 1

        to_del = [tid for tid, t in self.tracks.items() if tid not in matched and t["lost"] > 10]
        for tid in to_del:
            del self.tracks[tid]
        for tid, t in self.tracks.items():
            if tid not in matched:
                t["lost"] += 1

        res = []
        for tid, t in self.tracks.items():
            if t["lost"] == 0:
                p_idx = int(np.argmax(t["probs"]))
                res.append({
                    "id": tid,
                    "bbox": t["bbox"],
                    "probs": t["probs"],
                    "pred_idx": p_idx,
                    "label": EMOTIONS[p_idx]["name"],
                    "conf": float(t["probs"][p_idx]) * 100.0,
                    "landmarks": t["landmarks"]
                })
        return res


def open_camera(target_idx=None):
    """
    Detecta automáticamente la cámara física con señal real
    evitando cámaras virtuales inactivas (NVIDIA Broadcast, etc.).
    """
    if target_idx is not None:
        cap = cv2.VideoCapture(target_idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(target_idx)
        if cap.isOpened():
            return cap, target_idx

    # Probar orden de prioridad: 2 (física habitual), 0, 1, 3
    for idx in [2, 0, 1, 3]:
        for backend in [cv2.CAP_DSHOW, cv2.CAP_ANY]:
            try:
                cap = cv2.VideoCapture(idx, backend)
                if cap.isOpened():
                    for _ in range(3):
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            # Verificar que no sea pantalla negra de cámara virtual inactiva
                            if np.mean(frame) > 2.0 and np.std(frame) > 2.0:
                                return cap, idx
                    cap.release()
            except Exception:
                pass

    # Si ninguna cumplió el filtro de brillo, devolver la primera disponible
    for idx in [2, 0, 1]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            return cap, idx

    return None, 0


def run_portable_app():
    print("\nReconocimiento de Emociones Faciales (Modo Portable)")
    print("Autores: Manuel Fabrizzio Risco Gil y Alejandro Choquihuayta\n")

    if not os.path.exists(YUNET_MODEL) or not os.path.exists(EMOTION_MODEL):
        print("Error: No se encontraron los modelos.")
        input("Presiona ENTER para salir...")
        return

    print("Cargando detector facial YuNet...")
    detector = cv2.FaceDetectorYN.create(
        model=YUNET_MODEL, config="", input_size=(320, 320),
        score_threshold=0.65, nms_threshold=0.3, top_k=5000
    )

    print("Cargando red neuronal de emociones...")
    emotion_net = cv2.dnn.readNetFromONNX(EMOTION_MODEL)
    tracker = PortableEmotionTracker(alpha=0.30)

    print("Detectando camara web activa...")
    cap, cam_idx = open_camera()
    if cap is None:
        print("\n[!] Error: No se pudo conectar con ninguna camara web.")
        input("\nPresiona ENTER para salir...")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    window_name = "Reconocimiento de Emociones"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    print(f"\nCamara ID {cam_idx} activa.")
    print("Controles:")
    print("  * 'q' o ESC : Salir")
    print("  * 'c'       : Cambiar de camara")
    print("  * 's'       : Capturar foto")
    print("  * 'p'       : Pausar\n")

    fps = 30.0
    prev_time = time.time()
    paused = False
    consecutive_failures = 0

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret or frame is None:
                consecutive_failures += 1
                if consecutive_failures > 60:
                    time.sleep(0.03)
                continue
            else:
                consecutive_failures = 0

        h_f, w_f = frame.shape[:2]
        detector.setInputSize((w_f, h_f))
        _, faces = detector.detect(frame)

        processed = []
        if faces is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for f in faces:
                x, y, w, h = int(f[0]), int(f[1]), int(f[2]), int(f[3])
                x, y = max(0, x), max(0, y)
                w, h = min(w, w_f - x), min(h, h_f - y)
                if w < 16 or h < 16:
                    continue

                roi = gray[y:y+h, x:x+w]
                resized = cv2.resize(roi, (48, 48)).astype(np.float32)
                norm = (resized / 255.0 - 0.5) / 0.5
                blob = norm.reshape(1, 1, 48, 48)

                emotion_net.setInput(blob)
                logits = emotion_net.forward()[0]
                exp = np.exp(logits - np.max(logits))
                probs = exp / np.sum(exp)

                landmarks = f[4:14].reshape((5, 2)).astype(int)
                processed.append({"bbox": (x, y, w, h), "probs": probs, "landmarks": landmarks})

        tracks = tracker.update(processed)

        # Dibujar HUD
        for t in tracks:
            x, y, w, h = t["bbox"]
            c = EMOTIONS[t["pred_idx"]]["color"]

            cv2.rectangle(frame, (x, y), (x + w, y + h), c, 2)
            clen = min(15, w // 4, h // 4)
            cv2.line(frame, (x, y), (x + clen, y), c, 3)
            cv2.line(frame, (x, y), (x, y + clen), c, 3)
            cv2.line(frame, (x+w, y+h), (x+w-clen, y+h), c, 3)
            cv2.line(frame, (x+w, y+h), (x+w, y+h-clen), c, 3)

            badge_text = f"{t['label']} {t['conf']:.0f}%"
            cv2.putText(frame, badge_text, (x, max(20, y - 8)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.55, c, 1, cv2.LINE_AA)

            for i, p_val in enumerate(t["probs"]):
                by = y + i * 18
                if x + w + 140 < w_f and by + 14 < h_f:
                    cv2.rectangle(frame, (x + w + 8, by), (x + w + 130, by + 14), (30, 30, 30), -1)
                    fill = int(122 * p_val)
                    bar_c = EMOTIONS[i]["color"] if i == t["pred_idx"] else (80, 80, 80)
                    cv2.rectangle(frame, (x + w + 8, by), (x + w + 8 + fill, by + 14), bar_c, -1)
                    cv2.putText(frame, f"{EMOTIONS[i]['name'][:4]} {p_val*100:2.0f}%",
                                (x + w + 12, by + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)

        curr_time = time.time()
        elapsed = curr_time - prev_time
        if elapsed > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / elapsed)
        prev_time = curr_time

        cv2.rectangle(frame, (0, 0), (w_f, 28), (20, 20, 20), -1)
        cv2.putText(frame, f"FPS: {fps:.1f} | Camara [{cam_idx}] | 'c' cambiar camara",
                    (10, 19), cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        elif key == ord("c"):
            # Cambiar a la siguiente cámara
            cap.release()
            next_idx = (cam_idx + 1) % 4
            cap, cam_idx = open_camera(target_idx=next_idx)
            print(f"Cambiando a Camara ID {cam_idx}...")
        elif key == ord("s"):
            fname = f"captura_{int(time.time())}.png"
            cv2.imwrite(fname, frame)
            print(f"Foto guardada: {fname}")
        elif key == ord("p"):
            paused = not paused

        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        except Exception:
            pass

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_portable_app()
