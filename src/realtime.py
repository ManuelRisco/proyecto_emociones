# ============================================================================
# DEMO EN TIEMPO REAL CON WEBCAM - EMOTION AI v2.0
# Reconocimiento de Emociones Faciales con Deep Learning y Suavizado Temporal
# Autor: Manuel Fabrizzio Risco Gil
# Curso: Sistemas Inteligentes y Machine Learning - Ciclo 9
# ============================================================================
# Características principales:
# 1. Detector neuronal YuNet (robusto ante caras giradas/perfiles y sin falsos positivos)
# 2. Suavizado Temporal EMA (Exponential Moving Average) anti-parpadeo de emociones
# 3. Suavizado espacial de coordenadas para bounding boxes fluidos
# 4. HUD Moderno con métricas en tiempo real (FPS, latencia, GPU, barras animadas)
# 5. Atajos interactivos ('d' cambiar detector, 't' toggle suavizado, 'l' landmarks)
# ============================================================================

import os
import sys
import io
import time
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

# Asegurar que el directorio raiz del proyecto este en sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    DEVICE, BEST_MODEL_PATH, IMG_SIZE, PLOTS_DIR,
    EMOTION_LABELS, EMOTION_EMOJIS, NUM_CLASSES, COLORS,
    SMOOTHING_ALPHA, BBOX_SMOOTHING_ALPHA, TRACKING_IOU_THRESHOLD
)
from src.model import build_model
from src.predict import load_model, FaceDetector, preprocess_face


# ============================================================================
# PALETA DE COLORES Y ESTILOS PARA CADA EMOCIÓN (BGR)
# ============================================================================
EMOTION_THEME = {
    0: {"name": "Enojado",     "color": (40, 40, 235),   "emoji": "😠"},  # Rojo intenso
    1: {"name": "Disgusto",    "color": (34, 139, 34),   "emoji": "🤢"},  # Verde bosque
    2: {"name": "Miedo",       "color": (180, 50, 180),  "emoji": "😨"},  # Púrpura
    3: {"name": "Feliz",       "color": (0, 215, 255),   "emoji": "😄"},  # Amarillo / Oro
    4: {"name": "Triste",      "color": (235, 120, 40),  "emoji": "😢"},  # Azul cielo
    5: {"name": "Sorprendido", "color": (0, 140, 255),   "emoji": "😲"},  # Naranja
    6: {"name": "Neutral",     "color": (170, 170, 170), "emoji": "😐"},  # Gris metálico
}


# ============================================================================
# 1. RASTREADOR Y SUAVIZADOR TEMPORAL DE ROSTROS (EMA Tracker)
# ============================================================================

def compute_iou(box1, box2):
    """Calcula Intersection over Union (IoU) entre dos bounding boxes (x, y, w, h)."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + h2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h

    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


class FaceTracker:
    """
    Rastrea múltiples rostros entre fotogramas y aplica Suavizado Temporal (EMA)
    tanto a las coordenadas espaciales como a la distribución de probabilidades.
    """

    def __init__(self, alpha=SMOOTHING_ALPHA, bbox_alpha=BBOX_SMOOTHING_ALPHA, max_lost=8):
        self.alpha = alpha
        self.bbox_alpha = bbox_alpha
        self.max_lost = max_lost
        self.tracks = {}  # track_id: dict
        self.next_id = 1

    def update(self, detections, use_smoothing=True):
        """
        Asocia detecciones actuales con tracks previos y actualiza valores suavizados.

        detections: lista de dicts con {'bbox': (x,y,w,h), 'raw_probs': np.ndarray, 'landmarks': ...}
        """
        matched_tracks = set()
        matched_dets = set()

        # Matriz de IoU para matching de rostros
        track_ids = list(self.tracks.keys())
        for d_idx, det in enumerate(detections):
            best_iou = 0.0
            best_tid = None
            for tid in track_ids:
                if tid in matched_tracks:
                    continue
                iou = compute_iou(det["bbox"], self.tracks[tid]["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_iou >= TRACKING_IOU_THRESHOLD and best_tid is not None:
                matched_tracks.add(best_tid)
                matched_dets.add(d_idx)

                # Actualizar track existente
                track = self.tracks[best_tid]
                track["lost_frames"] = 0
                track["landmarks"] = det.get("landmarks")
                track["confidence"] = det.get("confidence", 1.0)

                if use_smoothing:
                    # Suavizado exponencial de Bounding Box
                    cur_x, cur_y, cur_w, cur_h = det["bbox"]
                    prev_x, prev_y, prev_w, prev_h = track["bbox"]
                    track["bbox"] = (
                        int(self.bbox_alpha * cur_x + (1 - self.bbox_alpha) * prev_x),
                        int(self.bbox_alpha * cur_y + (1 - self.bbox_alpha) * prev_y),
                        int(self.bbox_alpha * cur_w + (1 - self.bbox_alpha) * prev_w),
                        int(self.bbox_alpha * cur_h + (1 - self.bbox_alpha) * prev_h),
                    )
                    # Suavizado exponencial de Probabilidades (EMA)
                    track["probs"] = self.alpha * det["raw_probs"] + (1 - self.alpha) * track["probs"]
                    # Re-normalizar a suma 1.0
                    track["probs"] /= np.sum(track["probs"])
                else:
                    track["bbox"] = det["bbox"]
                    track["probs"] = det["raw_probs"]

        # Crear nuevos tracks para detecciones no asociadas
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_dets:
                self.tracks[self.next_id] = {
                    "id": self.next_id,
                    "bbox": det["bbox"],
                    "probs": det["raw_probs"].copy(),
                    "lost_frames": 0,
                    "landmarks": det.get("landmarks"),
                    "confidence": det.get("confidence", 1.0),
                }
                matched_tracks.add(self.next_id)
                self.next_id += 1

        # Incrementar lost_frames para tracks no vistos en este frame y purgar obsoletos
        to_delete = []
        for tid, track in self.tracks.items():
            if tid not in matched_tracks:
                track["lost_frames"] += 1
                if track["lost_frames"] > self.max_lost:
                    to_delete.append(tid)

        for tid in to_delete:
            del self.tracks[tid]

        # Retornar lista de tracks activos en este frame
        active_tracks = []
        for tid, track in self.tracks.items():
            if track["lost_frames"] == 0:
                pred_idx = int(np.argmax(track["probs"]))
                active_tracks.append({
                    "id": track["id"],
                    "bbox": track["bbox"],
                    "probs": track["probs"],
                    "pred_idx": pred_idx,
                    "pred_label": EMOTION_LABELS[pred_idx],
                    "confidence": float(track["probs"][pred_idx]) * 100,
                    "landmarks": track["landmarks"],
                    "det_confidence": track["confidence"] * 100
                })

        return active_tracks


# ============================================================================
# 2. DIBUJO Y DISEÑO DEL HUD MODERNO (Heads-Up Display)
# ============================================================================

def draw_styled_bbox(frame, bbox, color, thickness=2, corner_len=20):
    """
    Dibuja un recuadro estilizado con esquinas destacadas estilo sci-fi / studio.
    """
    x, y, w, h = bbox
    # Rectángulo base tenue
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    # Esquinas estilizadas sólidas
    clen = min(corner_len, w // 4, h // 4)
    # Top-Left
    cv2.line(frame, (x, y), (x + clen, y), color, thickness)
    cv2.line(frame, (x, y), (x, y + clen), color, thickness)
    # Top-Right
    cv2.line(frame, (x + w, y), (x + w - clen, y), color, thickness)
    cv2.line(frame, (x + w, y), (x + w, y + clen), color, thickness)
    # Bottom-Left
    cv2.line(frame, (x, y + h), (x + clen, y + h), color, thickness)
    cv2.line(frame, (x, y + h), (x, y + h - clen), color, thickness)
    # Bottom-Right
    cv2.line(frame, (x + w, y + h), (x + w - clen, y + h), color, thickness)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - clen), color, thickness)


def draw_emotion_badge(frame, x, y, label, confidence, color, track_id=None):
    """Dibuja un badge flotante con la emoción dominante y porcentaje."""
    id_str = f"#{track_id} " if track_id is not None else ""
    text = f"{id_str}{label} {confidence:.0f}%"
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.55
    thickness = 1

    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    badge_h = th + 12
    badge_w = tw + 16
    badge_y = max(5, y - badge_h - 4)

    # Fondo semitransparente con borde temático
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, badge_y), (x + badge_w, badge_y + badge_h), (20, 20, 20), -1)
    cv2.rectangle(overlay, (x, badge_y), (x + badge_w, badge_y + badge_h), color, 1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Texto
    cv2.putText(frame, text, (x + 8, badge_y + th + 5), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_probability_sidebar(frame, x, y, probs, active_idx, max_w=160, bar_h=16):
    """Dibuja un panel lateral con las barras de probabilidad de todas las emociones."""
    num_emotions = len(probs)
    total_h = num_emotions * (bar_h + 4) + 8
    sidebar_x = x
    sidebar_y = y

    # Clamping si se sale de la pantalla
    if sidebar_x + max_w + 10 > frame.shape[1]:
        sidebar_x = max(10, x - max_w - 10)
    if sidebar_y + total_h > frame.shape[0]:
        sidebar_y = max(10, frame.shape[0] - total_h - 10)

    # Fondo oscuro difuminado
    overlay = frame.copy()
    cv2.rectangle(overlay, (sidebar_x - 4, sidebar_y - 4),
                  (sidebar_x + max_w + 4, sidebar_y + total_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    for i in range(num_emotions):
        prob = float(probs[i])
        by = sidebar_y + i * (bar_h + 4)
        theme = EMOTION_THEME.get(i, {"name": EMOTION_LABELS[i], "color": (150, 150, 150)})
        bar_color = theme["color"] if i == active_idx else (90, 90, 90)

        # Canal de fondo de la barra
        cv2.rectangle(frame, (sidebar_x, by), (sidebar_x + max_w, by + bar_h), (35, 35, 35), -1)

        # Barra de progreso activa
        fill_w = int(max_w * prob)
        if fill_w > 0:
            cv2.rectangle(frame, (sidebar_x, by), (sidebar_x + fill_w, by + bar_h), bar_color, -1)

        # Borde sutil
        cv2.rectangle(frame, (sidebar_x, by), (sidebar_x + max_w, by + bar_h), (60, 60, 60), 1)

        # Texto de etiqueta y porcentaje
        name_short = EMOTION_LABELS[i][:4]
        text = f"{name_short} {prob*100:3.0f}%"
        text_color = (255, 255, 255) if i == active_idx else (190, 190, 190)
        cv2.putText(frame, text, (sidebar_x + 5, by + bar_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_color, 1, cv2.LINE_AA)


def draw_hud_header(frame, fps, latency_ms, detector_mode, use_smoothing, is_cuda, paused):
    """Dibuja la barra de estado superior con métricas técnicas."""
    h, w = frame.shape[:2]
    header_h = 42

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, header_h), (12, 12, 12), -1)
    cv2.line(overlay, (0, header_h), (w, header_h), (50, 50, 50), 1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    font = cv2.FONT_HERSHEY_DUPLEX

    # Título del Proyecto
    title = "EMOTION AI v2.0 | REAL-TIME"
    cv2.putText(frame, title, (14, 18), font, 0.45, (0, 220, 255), 1, cv2.LINE_AA)

    # Estado de Hardware
    hw_info = "GPU: RTX 5060 Ti (CUDA)" if is_cuda else "CPU Mode"
    cv2.putText(frame, hw_info, (14, 34), font, 0.38, (0, 255, 150), 1, cv2.LINE_AA)

    # Métricas de FPS y Latencia
    perf_text = f"FPS: {fps:.1f} | Latencia: {latency_ms:.1f}ms"
    cv2.putText(frame, perf_text, (w // 2 - 100, 18), font, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    # Modo de Detector y Suavizado
    det_color = (0, 255, 0) if detector_mode == "yunet" else (0, 165, 255)
    det_text = f"Detector: {detector_mode.upper()}"
    cv2.putText(frame, det_text, (w // 2 - 100, 34), font, 0.38, det_color, 1, cv2.LINE_AA)

    smooth_color = (0, 255, 200) if use_smoothing else (100, 100, 100)
    smooth_text = f"Filtro EMA: {'ACTIVADO' if use_smoothing else 'DESACTIVADO'}"
    cv2.putText(frame, smooth_text, (w - 180, 24), font, 0.38, smooth_color, 1, cv2.LINE_AA)

    if paused:
        cv2.putText(frame, "[PAUSADO]", (w // 2 - 45, h // 2), font, 0.9, (0, 0, 255), 2, cv2.LINE_AA)


def draw_hud_footer(frame):
    """Dibuja los controles de atajos en la parte inferior."""
    h, w = frame.shape[:2]
    footer_h = 24

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - footer_h), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    shortcuts = "q:Salir | d:Detector(YuNet/Haar) | t:Suavizado(EMA) | l:Landmarks | s:Foto | p:Pausa"
    cv2.putText(frame, shortcuts, (14, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)


# ============================================================================
# 3. BUCLE PRINCIPAL DE TIEMPO REAL
# ============================================================================

def run_realtime():
    """
    Ejecuta el reconocimiento de emociones en tiempo real con webcam.
    """
    print("\n" + "=" * 60)
    print(">>> DEMO EN TIEMPO REAL - EMOTION AI v2.0")
    print("=" * 60)

    print("   [1/3] Cargando red neuronal de emociones...")
    model = load_model()

    print("   [2/3] Inicializando detector neuronal YuNet (Deep Learning)...")
    detector = FaceDetector(mode="yunet")
    tracker = FaceTracker(alpha=SMOOTHING_ALPHA, bbox_alpha=BBOX_SMOOTHING_ALPHA)

    print("   [3/3] Detectando camara web activa...")
    
    def open_camera_rt(target_idx=None):
        if target_idx is not None:
            cap = cv2.VideoCapture(target_idx, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(target_idx)
            if cap.isOpened():
                return cap, target_idx
            return None, target_idx
        for idx in [2, 0, 1, 3]:
            for backend in [cv2.CAP_DSHOW, cv2.CAP_ANY]:
                try:
                    cap = cv2.VideoCapture(idx, backend)
                    if cap.isOpened():
                        for _ in range(3):
                            ret, frame = cap.read()
                            if ret and frame is not None:
                                if np.mean(frame) > 2.0 and np.std(frame) > 2.0:
                                    return cap, idx
                        cap.release()
                except Exception:
                    pass
        for idx in [2, 0, 1]:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                return cap, idx
        return None, 0

    cap, current_cam_idx = open_camera_rt()

    if not cap or not cap.isOpened():
        print("\n[!] No se pudo abrir ninguna camara web.")
        print("   Verifica que tu camara este conectada y no este en uso por otra app.")
        return

    # No forzamos la resolucion con cap.set() porque rompe algunas camaras virtuales (NVIDIA Broadcast)
    # y las hace emitir estatica/ruido de colores. En su lugar, redimensionamos con cv2.resize en el bucle.
    print("\n" + "=" * 60)
    print("   [OK] Sistema de tiempo real activo!")
    print("   Atajos de teclado:")
    print("   |-- 'd': Alternar Detector (YuNet vs Haar Cascade)")
    print("   |-- 't': Alternar Suavizado Temporal EMA (Anti-jitter)")
    print("   |-- 'l': Mostrar/Ocultar Puntos Faciales (Landmarks)")
    print("   |-- 's': Guardar captura de pantalla (HD)")
    print("   |-- 'p': Pausar / Reanudar video")
    print("   +-- 'q' o ESC: Salir")
    print("=" * 60)

    # Variables de control
    use_smoothing = True
    show_landmarks = True
    paused = False
    frame_count = 0
    fps = 30.0
    latency_ms = 0.0
    is_cuda = (DEVICE.type == "cuda")

    prev_time = time.time()
    frame = None

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("   [!] Error al leer fotograma de la webcam.")
                # Para evitar que la app se congele, creamos un frame negro con un mensaje
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Error leyendo camara. Presiona 'c' para cambiar.", (50, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                # Si se leyo correctamente, estandarizamos a 640x480 para no romper el modelo ni el HUD
                frame = cv2.resize(frame, (640, 480))
                
        if frame is None:
            continue

        start_tick = time.time()
        frame_count += 1

        if not paused:
            # 1. Detección de rostros
            raw_detections = detector.detect(frame)
            processed_dets = []

            # 2. Inferencia de emociones para cada rostro detectado
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            for det in raw_detections:
                x, y, w, h = det["bbox"]
                face_roi = gray[y:y+h, x:x+w]

                if face_roi.size == 0 or w < 16 or h < 16:
                    continue

                tensor = preprocess_face(face_roi)

                model.eval()
                with torch.no_grad():
                    if is_cuda:
                        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                            outputs = model(tensor)
                    else:
                        outputs = model(tensor)

                    probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

                processed_dets.append({
                    "bbox": (x, y, w, h),
                    "raw_probs": probs,
                    "confidence": det.get("confidence", 1.0),
                    "landmarks": det.get("landmarks")
                })

            # 3. Seguimiento y Suavizado Temporal (EMA)
            active_tracks = tracker.update(processed_dets, use_smoothing=use_smoothing)

            # 4. Renderizado en el frame
            for track in active_tracks:
                x, y, w, h = track["bbox"]
                pred_idx = track["pred_idx"]
                theme = EMOTION_THEME.get(pred_idx, {"color": (0, 255, 0)})
                color = theme["color"]

                # Dibujar bounding box estilizado
                draw_styled_bbox(frame, (x, y, w, h), color=color, thickness=2)

                # Dibujar badge flotante
                draw_emotion_badge(
                    frame, x, y, track["pred_label"], track["confidence"],
                    color=color, track_id=track["id"]
                )

                # Dibujar panel lateral de probabilidades
                draw_probability_sidebar(
                    frame, x + w + 8, y, track["probs"], active_idx=pred_idx
                )

                # Dibujar puntos faciales (landmarks) si están activos
                if show_landmarks and track["landmarks"] is not None:
                    for pt in track["landmarks"]:
                        cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (0, 255, 255), -1, cv2.LINE_AA)

        # Medir latencia y FPS
        end_tick = time.time()
        frame_time = end_tick - start_tick
        latency_ms = frame_time * 1000.0

        current_time = time.time()
        elapsed = current_time - prev_time
        if elapsed > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / elapsed)
        prev_time = current_time

        # Dibujar HUD superior e inferior
        draw_hud_header(frame, fps, latency_ms, detector.mode, use_smoothing, is_cuda, paused)
        draw_hud_footer(frame)

        # Mostrar ventana
        cv2.imshow("Reconocimiento de Emociones Faciales - AI Studio", frame)

        # Gestión de teclas
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            print("\n   [INFO] Cerrando sesion de webcam...")
            break
        elif key == ord("d"):
            new_mode = "haar" if detector.mode == "yunet" else "yunet"
            detector = FaceDetector(mode=new_mode)
            print(f"   [DETECTOR] Cambiado a: {detector.mode.upper()}")
        elif key == ord("t"):
            use_smoothing = not use_smoothing
            print(f"   [SUAVIZADO] Suavizado Temporal EMA: {'ACTIVADO' if use_smoothing else 'DESACTIVADO'}")
        elif key == ord("l"):
            show_landmarks = not show_landmarks
            print(f"   [LANDMARKS] Puntos faciales: {'VISIBLES' if show_landmarks else 'OCULTOS'}")
        elif key == ord("s"):
            os.makedirs(PLOTS_DIR, exist_ok=True)
            screenshot_path = os.path.join(PLOTS_DIR, f"screenshot_{frame_count}.png")
            cv2.imwrite(screenshot_path, frame)
            print(f"   [FOTO] Screenshot guardado: {screenshot_path}")
        elif key == ord("c"):
            cap.release()
            for offset in range(1, 4):
                next_idx = (current_cam_idx + offset) % 4
                new_cap, new_idx = open_camera_rt(target_idx=next_idx)
                if new_cap is not None and new_cap.isOpened():
                    cap = new_cap
                    current_cam_idx = new_idx
                    print(f"   [CAMARA] Cambiando a Camara ID {current_cam_idx}...")
                    break
            else:
                cap, current_cam_idx = open_camera_rt(target_idx=current_cam_idx)
                print(f"   [CAMARA] No hay otras camaras, volviendo a ID {current_cam_idx}...")
        elif key == ord("p"):
            paused = not paused
            print(f"   {'[PAUSA] Stream pausado' if paused else '[PLAY] Stream reanudado'}")

    cap.release()
    cv2.destroyAllWindows()
    print("   [OK] Webcam y ventanas liberadas exitosamente.")


if __name__ == "__main__":
    run_realtime()
