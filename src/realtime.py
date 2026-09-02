# ============================================================================
# DEMO EN TIEMPO REAL CON WEBCAM — EMOTION AI v2.0
# Reconocimiento de Emociones Faciales con Deep Learning
# ============================================================================
# Características:
#   1. Detector neuronal YuNet — robusto ante perfiles y caras giradas
#   2. Suavizado Temporal EMA — anti-parpadeo de emociones y bounding boxes
#   3. HUD Moderno — FPS, latencia, barras de probabilidad animadas
#   4. Multi-rostro — seguimiento independiente con IDs por persona
#   5. AMP (FP16) en inferencia — máximo rendimiento en Tensor Cores
#   6. Atajos: q=Salir, d=Detector, t=Suavizado, l=Landmarks, s=Foto, p=Pausa
# ============================================================================

import os
import time

import cv2
import numpy as np
import torch

from src.utils import ensure_root_in_path, open_camera, setup_utf8
setup_utf8()
ensure_root_in_path()

from src.config import (
    BEST_MODEL_PATH, BBOX_SMOOTHING_ALPHA, DEVICE, EMOTION_COLORS,
    EMOTION_LABELS, IMG_SIZE, NUM_CLASSES, PLOTS_DIR,
    SMOOTHING_ALPHA, TRACKING_IOU_THRESHOLD, USE_AMP,
)
from src.predict import FaceDetector, load_model, preprocess_face


# ──────────────────────────────────────────────────────────────────────────────
# 1. RASTREADOR TEMPORAL DE ROSTROS (EMA Tracker)
# ──────────────────────────────────────────────────────────────────────────────

def _compute_iou(box1: tuple, box2: tuple) -> float:
    """Calcula IoU entre dos bounding boxes (x, y, w, h)."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


class FaceTracker:
    """
    Rastrea múltiples rostros entre fotogramas con Suavizado Temporal (EMA).

    Aplica Exponential Moving Average (EMA) tanto a las coordenadas espaciales
    del bounding box como a la distribución de probabilidades de emoción,
    eliminando el parpadeo (jitter) que ocurriría si se mostraran los valores
    brutos frame a frame.

    Asociación de detecciones: Greedy matching por IoU.
    Purga: tracks no vistos por más de max_lost frames se eliminan.
    """

    def __init__(
        self,
        alpha: float = SMOOTHING_ALPHA,
        bbox_alpha: float = BBOX_SMOOTHING_ALPHA,
        max_lost: int = 8,
    ):
        self.alpha      = alpha
        self.bbox_alpha = bbox_alpha
        self.max_lost   = max_lost
        self.tracks: dict[int, dict] = {}
        self.next_id = 1

    def update(self, detections: list[dict], use_smoothing: bool = True) -> list[dict]:
        """
        Asocia las detecciones actuales con los tracks existentes y actualiza EMA.

        Args:
            detections: Lista de dicts {'bbox', 'raw_probs', 'confidence', 'landmarks'}.
            use_smoothing: Si False, usa valores brutos sin EMA.

        Returns:
            Lista de tracks activos con sus valores suavizados.
        """
        matched_tracks: set[int] = set()
        matched_dets:   set[int] = set()
        track_ids = list(self.tracks.keys())

        # Greedy matching por IoU máximo
        for d_idx, det in enumerate(detections):
            best_iou, best_tid = 0.0, None
            for tid in track_ids:
                if tid in matched_tracks:
                    continue
                iou = _compute_iou(det["bbox"], self.tracks[tid]["bbox"])
                if iou > best_iou:
                    best_iou, best_tid = iou, tid

            if best_iou >= TRACKING_IOU_THRESHOLD and best_tid is not None:
                matched_tracks.add(best_tid)
                matched_dets.add(d_idx)
                track = self.tracks[best_tid]
                track["lost_frames"] = 0
                track["landmarks"]   = det.get("landmarks")
                track["confidence"]  = det.get("confidence", 1.0)

                if use_smoothing:
                    # EMA para bounding box
                    cx, cy, cw, ch = det["bbox"]
                    px, py, pw, ph = track["bbox"]
                    a = self.bbox_alpha
                    track["bbox"] = (
                        int(a * cx + (1 - a) * px),
                        int(a * cy + (1 - a) * py),
                        int(a * cw + (1 - a) * pw),
                        int(a * ch + (1 - a) * ph),
                    )
                    # EMA para probabilidades de emoción
                    track["probs"] = self.alpha * det["raw_probs"] + (1 - self.alpha) * track["probs"]
                    track["probs"] /= track["probs"].sum()  # re-normalizar
                else:
                    track["bbox"]  = det["bbox"]
                    track["probs"] = det["raw_probs"]

        # Nuevos tracks para detecciones no asociadas
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_dets:
                self.tracks[self.next_id] = {
                    "id":          self.next_id,
                    "bbox":        det["bbox"],
                    "probs":       det["raw_probs"].copy(),
                    "lost_frames": 0,
                    "landmarks":   det.get("landmarks"),
                    "confidence":  det.get("confidence", 1.0),
                }
                matched_tracks.add(self.next_id)
                self.next_id += 1

        # Purgar tracks obsoletos
        for tid in [t for t, tk in self.tracks.items() if t not in matched_tracks]:
            self.tracks[tid]["lost_frames"] += 1
        self.tracks = {
            tid: tk for tid, tk in self.tracks.items()
            if tk["lost_frames"] <= self.max_lost
        }

        # Retornar solo tracks activos (vistos en este frame)
        return [
            {
                "id":             tk["id"],
                "bbox":           tk["bbox"],
                "probs":          tk["probs"],
                "pred_idx":       int(np.argmax(tk["probs"])),
                "pred_label":     EMOTION_LABELS[int(np.argmax(tk["probs"]))],
                "confidence":     float(tk["probs"][int(np.argmax(tk["probs"]))]) * 100,
                "landmarks":      tk["landmarks"],
                "det_confidence": tk["confidence"] * 100,
            }
            for tid, tk in self.tracks.items()
            if tk["lost_frames"] == 0
        ]


# ──────────────────────────────────────────────────────────────────────────────
# 2. DIBUJO DEL HUD (Heads-Up Display)
# ──────────────────────────────────────────────────────────────────────────────

def _draw_styled_bbox(
    frame: np.ndarray, bbox: tuple, color: tuple, thickness: int = 2, corner_len: int = 20
) -> None:
    """Dibuja recuadro estilizado con esquinas destacadas estilo sci-fi."""
    x, y, w, h = bbox
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    clen = min(corner_len, w // 4, h // 4)
    for (px, py), (dx, dy) in [
        ((x,     y),     (clen, 0)), ((x,     y),     (0, clen)),
        ((x+w,   y),     (-clen, 0)), ((x+w,  y),     (0, clen)),
        ((x,     y+h),   (clen, 0)), ((x,     y+h),   (0, -clen)),
        ((x+w,   y+h),   (-clen, 0)), ((x+w,  y+h),   (0, -clen)),
    ]:
        cv2.line(frame, (px, py), (px + dx, py + dy), color, thickness)


def _draw_emotion_badge(
    frame: np.ndarray, x: int, y: int,
    label: str, confidence: float, color: tuple, track_id: int | None = None,
) -> None:
    """Dibuja badge flotante con emoción dominante y porcentaje de confianza."""
    text   = f"#{track_id} {label} {confidence:.0f}%" if track_id else f"{label} {confidence:.0f}%"
    font   = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 0.55, 1)
    badge_y = max(5, y - th - 16)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, badge_y), (x + tw + 16, badge_y + th + 12), (20, 20, 20), -1)
    cv2.rectangle(overlay, (x, badge_y), (x + tw + 16, badge_y + th + 12), color, 1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.putText(frame, text, (x + 8, badge_y + th + 5), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_probability_sidebar(
    frame: np.ndarray, x: int, y: int,
    probs: np.ndarray, active_idx: int, max_w: int = 160, bar_h: int = 16,
) -> None:
    """Panel lateral con barras de probabilidad animadas para cada emoción."""
    num_emotions = len(probs)
    total_h = num_emotions * (bar_h + 4) + 8
    sidebar_x = min(x, frame.shape[1] - max_w - 14)
    sidebar_y = min(y, frame.shape[0] - total_h - 14)

    overlay = frame.copy()
    cv2.rectangle(overlay, (sidebar_x - 4, sidebar_y - 4),
                  (sidebar_x + max_w + 4, sidebar_y + total_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    for i in range(num_emotions):
        prob   = float(probs[i])
        by     = sidebar_y + i * (bar_h + 4)
        color  = EMOTION_COLORS.get(i, (150, 150, 150)) if i == active_idx else (90, 90, 90)
        fill_w = int(max_w * prob)

        cv2.rectangle(frame, (sidebar_x, by), (sidebar_x + max_w, by + bar_h), (35, 35, 35), -1)
        if fill_w > 0:
            cv2.rectangle(frame, (sidebar_x, by), (sidebar_x + fill_w, by + bar_h), color, -1)
        cv2.rectangle(frame, (sidebar_x, by), (sidebar_x + max_w, by + bar_h), (60, 60, 60), 1)

        text_color = (255, 255, 255) if i == active_idx else (190, 190, 190)
        cv2.putText(
            frame, f"{EMOTION_LABELS[i][:4]} {prob*100:3.0f}%",
            (sidebar_x + 5, by + bar_h - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_color, 1, cv2.LINE_AA,
        )


def _draw_hud_header(
    frame: np.ndarray, fps: float, latency_ms: float,
    use_smoothing: bool, is_cuda: bool, paused: bool,
) -> None:
    """Barra superior con métricas de rendimiento y estado del sistema."""
    h, w  = frame.shape[:2]
    font  = cv2.FONT_HERSHEY_DUPLEX
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 42), (12, 12, 12), -1)
    cv2.line(overlay, (0, 42), (w, 42), (50, 50, 50), 1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    cv2.putText(frame, "EMOTION AI v2.0 | REAL-TIME", (14, 18), font, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "GPU: CUDA" if is_cuda else "CPU Mode", (14, 34), font, 0.38, (0, 255, 150), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f} | Lat: {latency_ms:.1f}ms", (w // 2 - 100, 18), font, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "Det: YUNET ONNX", (w // 2 - 100, 34), font, 0.38, (0, 255, 0), 1, cv2.LINE_AA)

    smooth_color = (0, 255, 200) if use_smoothing else (100, 100, 100)
    cv2.putText(frame, f"EMA: {'ON' if use_smoothing else 'OFF'}", (w - 90, 24), font, 0.38, smooth_color, 1, cv2.LINE_AA)

    if paused:
        cv2.putText(frame, "[PAUSADO]", (w // 2 - 45, h // 2), font, 0.9, (0, 0, 255), 2, cv2.LINE_AA)


def _draw_hud_footer(frame: np.ndarray) -> None:
    """Barra inferior con atajos de teclado."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 24), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.putText(
        frame, "q:Salir | t:EMA | l:Landmarks | s:Foto | p:Pausa | c:Camara",
        (14, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1, cv2.LINE_AA,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 3. BUCLE PRINCIPAL DE TIEMPO REAL
# ──────────────────────────────────────────────────────────────────────────────

def run_realtime() -> None:
    """
    Ejecuta el reconocimiento de emociones en tiempo real con webcam.

    Flujo por frame:
        1. Leer frame → resize 640×480
        2. Detectar rostros con red neuronal YuNet
        3. Inferencia de emoción por rostro (AMP FP16)
        4. Tracking + suavizado EMA
        5. Renderizar HUD
        6. Gestionar atajos de teclado
    """
    print("\n" + "=" * 60)
    print(">>> DEMO EN TIEMPO REAL — EMOTION AI v2.0")
    print("=" * 60)

    # ── Carga única (fuera del bucle) ─────────────────────────────────────────
    print("   [1/3] Cargando red neuronal de emociones...")
    model = load_model()
    model.eval()

    print("   [2/3] Inicializando detector neuronal YuNet...")
    detector = FaceDetector()
    tracker  = FaceTracker(alpha=SMOOTHING_ALPHA, bbox_alpha=BBOX_SMOOTHING_ALPHA)

    print("   [3/3] Buscando cámara web activa...")
    cap, current_cam_idx = open_camera()

    if not cap or not cap.isOpened():
        print("\n[!] No se pudo abrir ninguna cámara. Verifica que esté conectada.")
        return

    WINDOW_NAME = "Reconocimiento de Emociones Faciales - AI Studio"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 960, 720)

    print("\n" + "=" * 60)
    print("   [OK] Sistema activo. Atajos:")
    print("   |-- 't': Suavizado Temporal EMA")
    print("   |-- 'l': Puntos Faciales (Landmarks)")
    print("   |-- 's': Capturar foto")
    print("   |-- 'p': Pausar / Reanudar")
    print("   |-- 'c': Cambiar cámara")
    print("   +-- 'q' / ESC: Salir")
    print("=" * 60)

    use_smoothing  = True
    show_landmarks = True
    paused         = False
    frame_count    = 0
    fps            = 60.0
    latency_ms     = 0.0
    is_cuda        = DEVICE.type == "cuda"
    prev_time      = time.time()
    frame          = None

    while True:
        # ── Captura ───────────────────────────────────────────────────────────
        if not paused:
            ret, frame = cap.read()
            if not ret or frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Error leyendo camara. Presiona 'c'.", (50, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                frame = cv2.resize(frame, (640, 480))

        if frame is None:
            continue

        start_tick = time.time()
        frame_count += 1

        # ── Inferencia ────────────────────────────────────────────────────────
        if not paused:
            # Detección ultrarrápida escalada a 320px
            raw_detections = detector.detect(frame, max_dim=320)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            processed_dets = []

            for det in raw_detections:
                x, y, w, h = det["bbox"]
                face_roi = gray[y:y+h, x:x+w]
                if face_roi.size == 0 or w < 16 or h < 16:
                    continue

                tensor = preprocess_face(face_roi)
                with torch.no_grad():
                    if is_cuda:
                        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                            outputs = model(tensor)
                    else:
                        outputs = model(tensor)
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

                processed_dets.append({
                    "bbox":       (x, y, w, h),
                    "raw_probs":  probs,
                    "confidence": det.get("confidence", 1.0),
                    "landmarks":  det.get("landmarks"),
                })

            active_tracks = tracker.update(processed_dets, use_smoothing=use_smoothing)

            # ── Renderizado ───────────────────────────────────────────────────
            for track in active_tracks:
                x, y, w, h  = track["bbox"]
                pred_idx     = track["pred_idx"]
                color        = EMOTION_COLORS.get(pred_idx, (0, 255, 0))

                _draw_styled_bbox(frame, (x, y, w, h), color=color)
                _draw_emotion_badge(frame, x, y, track["pred_label"], track["confidence"],
                                    color=color, track_id=track["id"])
                _draw_probability_sidebar(frame, x + w + 8, y, track["probs"], active_idx=pred_idx)

                if show_landmarks and track["landmarks"] is not None:
                    for pt in track["landmarks"]:
                        cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (0, 255, 255), -1, cv2.LINE_AA)

        # ── FPS y latencia ────────────────────────────────────────────────────
        latency_ms   = (time.time() - start_tick) * 1000.0
        curr_time    = time.time()
        elapsed      = curr_time - prev_time
        if elapsed > 0:
            fps = 0.9 * fps + 0.1 / elapsed
        prev_time = curr_time

        _draw_hud_header(frame, fps, latency_ms, use_smoothing, is_cuda, paused)
        _draw_hud_footer(frame)

        cv2.imshow(WINDOW_NAME, frame)

        # ── Teclado ───────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key == ord("t"):
            use_smoothing = not use_smoothing
            print(f"   [EMA] {'ACTIVADO' if use_smoothing else 'DESACTIVADO'}")
        elif key == ord("l"):
            show_landmarks = not show_landmarks
            print(f"   [LANDMARKS] {'VISIBLES' if show_landmarks else 'OCULTOS'}")
        elif key == ord("s"):
            os.makedirs(PLOTS_DIR, exist_ok=True)
            path = os.path.join(PLOTS_DIR, f"screenshot_{frame_count}.png")
            cv2.imwrite(path, frame)
            print(f"   [FOTO] Guardada: {path}")
        elif key == ord("c"):
            cap.release()
            for offset in range(1, 4):
                new_cap, new_idx = open_camera(target_idx=(current_cam_idx + offset) % 4)
                if new_cap and new_cap.isOpened():
                    cap, current_cam_idx = new_cap, new_idx
                    print(f"   [CAMARA] → ID {current_cam_idx}")
                    break
        elif key == ord("p"):
            paused = not paused
            print(f"   {'[PAUSA]' if paused else '[PLAY]'}")

        # Salir si el usuario cierra la ventana con la 'X'
        try:
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
        except Exception:
            pass

    cap.release()
    cv2.destroyAllWindows()
    print("   [OK] Webcam liberada.")


if __name__ == "__main__":
    run_realtime()
