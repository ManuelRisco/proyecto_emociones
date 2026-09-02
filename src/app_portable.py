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


# Umbral m\u00ednimo de confianza para mostrar una emoci\u00f3n.
# Si no se supera, se usa la segunda m\u00e1s probable.
# Evita que "cara seria/comiendo" se muestre como "Enojado".
CONF_THRESHOLDS = {
    0: 0.55,   # Enojado      — umbral alto (muy confundible con Neutral/serio)
    1: 0.45,   # Disgusto
    2: 0.45,   # Miedo
    3: 0.35,   # Feliz        — muy distintiva, umbral bajo
    4: 0.42,   # Triste
    5: 0.38,   # Sorprendido  — muy distintiva, umbral bajo
    6: 0.30,   # Neutral      — catch-all para estados ambiguos
}
DEBOUNCE_FRAMES = 6  # frames consecutivos antes de cambiar la emoci\u00f3n mostrada


class PortableEmotionTracker:
    def __init__(self, alpha=0.30, conf_thresholds=None):
        self.alpha = alpha
        self.conf_thresholds = dict(conf_thresholds or CONF_THRESHOLDS)
        self.tracks = {}
        self.next_id = 1

    def update(self, detections, use_smoothing=True):
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
                if use_smoothing:
                    t["bbox"] = (
                        int(0.35 * x + 0.65 * t["bbox"][0]),
                        int(0.35 * y + 0.65 * t["bbox"][1]),
                        int(0.35 * w + 0.65 * t["bbox"][2]),
                        int(0.35 * h + 0.65 * t["bbox"][3])
                    )
                    t["probs"] = self.alpha * d["probs"] + (1.0 - self.alpha) * t["probs"]
                else:
                    t["bbox"] = d["bbox"]
                    t["probs"] = d["probs"].copy()
                t["landmarks"] = d.get("landmarks")
            else:
                init_idx = int(np.argmax(d["probs"]))
                self.tracks[self.next_id] = {
                    "id": self.next_id,
                    "bbox": d["bbox"],
                    "probs": d["probs"].copy(),
                    "lost": 0,
                    "landmarks": d.get("landmarks"),
                    "display_idx": init_idx,
                    "candidate_idx": init_idx,
                    "candidate_count": 0,
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
            if t["lost"] != 0:
                continue

            probs = t["probs"]
            raw_best = int(np.argmax(probs))

            # 1. Umbral de confianza: si no supera el m\u00ednimo, usar la siguiente
            if float(probs[raw_best]) >= self.conf_thresholds.get(raw_best, 0.0):
                resolved = raw_best
            else:
                sorted_idxs = np.argsort(probs)[::-1]
                resolved = int(sorted_idxs[0])
                for cand in sorted_idxs[1:]:
                    if float(probs[cand]) >= self.conf_thresholds.get(int(cand), 0.0):
                        resolved = int(cand)
                        break

            # 2. Debounce: cambiar emoci\u00f3n visible solo si persiste DEBOUNCE_FRAMES
            disp = t.get("display_idx", resolved)
            if resolved != disp:
                if resolved == t.get("candidate_idx"):
                    t["candidate_count"] = t.get("candidate_count", 0) + 1
                else:
                    t["candidate_idx"] = resolved
                    t["candidate_count"] = 1
                if t["candidate_count"] >= DEBOUNCE_FRAMES:
                    t["display_idx"] = resolved
                    t["candidate_count"] = 0
            else:
                t["candidate_idx"] = resolved
                t["candidate_count"] = 0
                t["display_idx"] = resolved

            disp_idx = t.get("display_idx", resolved)
            res.append({
                "id": tid,
                "bbox": t["bbox"],
                "probs": probs,
                "pred_idx": disp_idx,
                "label": EMOTIONS[disp_idx]["name"],
                "conf": float(probs[disp_idx]) * 100.0,
                "landmarks": t["landmarks"]
            })
        return res


def configure_camera(cap):
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass


def open_camera(target_idx=None):
    """
    Detecta automáticamente la cámara física con señal real y alto rendimiento
    evitando cámaras virtuales inactivas (NVIDIA Broadcast, etc.).
    """
    if target_idx is not None:
        cap = cv2.VideoCapture(target_idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(target_idx)
        if cap.isOpened():
            configure_camera(cap)
            return cap, target_idx
        return None, target_idx

    # Probar orden de prioridad: 0, 1, 2, 3
    for idx in [0, 1, 2, 3]:
        for backend in [cv2.CAP_DSHOW, cv2.CAP_ANY]:
            try:
                cap = cv2.VideoCapture(idx, backend)
                if cap.isOpened():
                    configure_camera(cap)
                    for _ in range(3):
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            if np.mean(frame) > 2.0 and np.std(frame) > 2.0:
                                return cap, idx
                    cap.release()
            except Exception:
                pass

    for idx in [0, 1, 2]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            configure_camera(cap)
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

    window_name = "Reconocimiento de Emociones - AI Studio Portable"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 720)

    print(f"\nCamara ID {cam_idx} activa.")
    print("Controles:")
    print("  * 'q' o ESC : Salir")
    print("  * 't'       : Activar/Desactivar Suavizado EMA")
    print("  * 'l'       : Mostrar/Ocultar Puntos Faciales (Landmarks)")
    print("  * 'c'       : Cambiar de camara")
    print("  * 's'       : Capturar foto")
    print("  * 'p'       : Pausar\n")

    fps = 60.0
    prev_time = time.time()
    paused = False
    use_smoothing = True
    show_landmarks = True
    consecutive_failures = 0

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret or frame is None:
                consecutive_failures += 1
                if consecutive_failures > 60:
                    time.sleep(0.03)
                
                # Mostrar frame de error si la cámara falla
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Error leyendo camara. Presiona 'c' para cambiar.", (50, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                frame = cv2.resize(frame, (640, 480))
                consecutive_failures = 0

        if frame is None:
            continue

        h_f, w_f = frame.shape[:2]
        # Detección escalada ultrarrápida a 320px
        scale = 320.0 / float(max(h_f, w_f))
        w_s, h_s = int(w_f * scale), int(h_f * scale)
        detect_img = cv2.resize(frame, (w_s, h_s), interpolation=cv2.INTER_LINEAR)

        detector.setInputSize((w_s, h_s))
        _, faces = detector.detect(detect_img)

        processed = []
        if faces is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            inv_s = 1.0 / scale
            for f in faces:
                x = max(0, int(f[0] * inv_s))
                y = max(0, int(f[1] * inv_s))
                w = int(f[2] * inv_s)
                h = int(f[3] * inv_s)
                w = min(w, w_f - x)
                h = min(h, h_f - y)
                if w < 16 or h < 16:
                    continue

                roi = gray[y:y+h, x:x+w]
                resized = cv2.resize(roi, (48, 48), interpolation=cv2.INTER_AREA).astype(np.float32)
                norm = (resized / 255.0 - 0.5) / 0.5
                blob = norm.reshape(1, 1, 48, 48)

                emotion_net.setInput(blob)
                logits = emotion_net.forward()[0]
                exp = np.exp(logits - np.max(logits))
                probs = exp / np.sum(exp)

                landmarks = (f[4:14].reshape((5, 2)) * inv_s).astype(int)
                processed.append({"bbox": (x, y, w, h), "probs": probs, "landmarks": landmarks})

        tracks = tracker.update(processed, use_smoothing=use_smoothing)

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

            if show_landmarks and t.get("landmarks") is not None:
                for pt in t["landmarks"]:
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (0, 255, 255), -1, cv2.LINE_AA)

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
        ema_str = "EMA:ON" if use_smoothing else "EMA:OFF"
        cv2.putText(frame, f"FPS: {fps:.1f} | Cam [{cam_idx}] | {ema_str} | YuNet ONNX",
                    (10, 19), cv2.FONT_HERSHEY_DUPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        elif key == ord("t"):
            use_smoothing = not use_smoothing
            print(f"Suavizado EMA: {'ACTIVADO' if use_smoothing else 'DESACTIVADO'}")
        elif key == ord("l"):
            show_landmarks = not show_landmarks
            print(f"Landmarks: {'VISIBLES' if show_landmarks else 'OCULTOS'}")
        elif key == ord("c"):
            # Cambiar a la siguiente cámara
            cap.release()
            for offset in range(1, 4):
                next_idx = (cam_idx + offset) % 4
                new_cap, new_idx = open_camera(target_idx=next_idx)
                if new_cap is not None and new_cap.isOpened():
                    cap = new_cap
                    cam_idx = new_idx
                    print(f"Cambiando a Camara ID {cam_idx}...")
                    break
            else:
                cap, cam_idx = open_camera(target_idx=cam_idx)
                print(f"No hay otras camaras disponibles. Volviendo a ID {cam_idx}...")
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
