# ============================================================================
# RECONOCIMIENTO DE EMOCIONES EN TIEMPO REAL CON WEBCAM (BIOMETRIC AI STUDIO)
# ============================================================================
# Características avanzadas:
#   1. Memoria Biométrica Facial (Re-ID): Recuerda al sujeto si sale y regresa.
#   2. Alta Sensibilidad: Captura inmediata de "Sorprendido" y "Disgusto".
#   3. Gráficas Independientes Multi-Persona: Dock lateral derecho por sujeto.
#   4. Silueta Irregular Escalonada Holográfica: Estilo Sci-Fi con láser de escaneo.
#   5. Suavizado Temporal EMA: Cero parpadeo a 60 FPS con ventana ajustable.
# ============================================================================

import math
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
    EMOTION_CONFIDENCE_THRESHOLDS, EMOTION_DEBOUNCE_FRAMES,
    EMOTION_SENSITIVITY_BOOST, EMOTION_DEBOUNCE_BY_CLASS, PERSON_PALETTE,
    REID_SIMILARITY_THRESHOLD, REID_TEMPLATE_SIZE,
)
from src.predict import FaceDetector, load_model, preprocess_face


# ──────────────────────────────────────────────────────────────────────────────
# 1. FIRMA BIOMÉTRICA Y RE-IDENTIFICACIÓN FACIAL (Long-Term Face Memory)
# ──────────────────────────────────────────────────────────────────────────────

def _extract_biometric_signature(
    gray_face: np.ndarray, landmarks: np.ndarray | None, bbox: tuple
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Extrae una firma biométrica multimodal invariante a escala e iluminación:
    1. Template normalizado del rostro (28x28).
    2. Histograma de distribución de luminosidad (16 bins).
    3. Vector de proporciones anatómicas invariantes a partir de los 5 landmarks.
    """
    # 1. Template facial normalizado
    tpl = cv2.resize(gray_face, REID_TEMPLATE_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)
    t_mean, t_std = np.mean(tpl), np.std(tpl)
    tpl_norm = (tpl - t_mean) / (t_std + 1e-5)

    # 2. Histograma normalizado
    hist = cv2.calcHist([gray_face], [0], None, [16], [0, 256]).flatten()
    h_norm = hist / (np.linalg.norm(hist) + 1e-5)

    # 3. Proporciones geométricas invariantes
    geom_vec = None
    if landmarks is not None and len(landmarks) >= 5:
        p_re = landmarks[0].astype(float)  # ojo derecho
        p_le = landmarks[1].astype(float)  # ojo izquierdo
        p_n  = landmarks[2].astype(float)  # nariz
        p_rm = landmarks[3].astype(float)  # boca der
        p_lm = landmarks[4].astype(float)  # boca izq

        eye_dist = np.linalg.norm(p_re - p_le)
        eye_mid = (p_re + p_le) / 2.0
        nose_dist = np.linalg.norm(eye_mid - p_n)
        mouth_mid = (p_rm + p_lm) / 2.0
        mouth_dist = np.linalg.norm(eye_mid - mouth_mid)
        mouth_w = np.linalg.norm(p_rm - p_lm)

        base = max(5.0, eye_dist)
        geom_vec = np.array([
            eye_dist / max(10.0, float(bbox[2])),
            nose_dist / base,
            mouth_dist / base,
            mouth_w / base,
        ], dtype=np.float32)

    return (tpl_norm, h_norm, geom_vec)


def _compute_biometric_similarity(sig1: tuple, sig2: tuple) -> float:
    """Calcula similitud biométrica (0.0 a 1.0) entre dos firmas faciales."""
    t1, h1, g1 = sig1
    t2, h2, g2 = sig2

    # Similitud de template facial (coseno)
    t1_f, t2_f = t1.flatten(), t2.flatten()
    sim_tpl = float(np.dot(t1_f, t2_f) / (np.linalg.norm(t1_f) * np.linalg.norm(t2_f) + 1e-5))
    sim_tpl = max(0.0, min(1.0, (sim_tpl + 1.0) / 2.0))

    # Similitud de histograma
    sim_hist = max(0.0, float(np.dot(h1, h2)))

    # Similitud de proporciones geométricas
    sim_geom = 0.5
    if g1 is not None and g2 is not None:
        diff = np.linalg.norm(g1 - g2)
        sim_geom = max(0.0, min(1.0, 1.0 / (1.0 + diff * 2.0)))

    # Fusión ponderada de biometría
    return 0.50 * sim_tpl + 0.30 * sim_geom + 0.20 * sim_hist


def _compute_iou(box1: tuple, box2: tuple) -> float:
    """Calcula IoU entre dos bounding boxes (x, y, w, h)."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# 2. RASTREADOR BIOMÉTRICO CON RE-IDENTIFICACIÓN Y SENSIBILIDAD AVANZADA
# ──────────────────────────────────────────────────────────────────────────────

class FaceTracker:
    """
    Rastrea múltiples rostros en tiempo real con:
    - Suavizado temporal EMA espacial y de probabilidades.
    - Memoria Biométrica Re-ID: Recuerda a un sujeto cuando sale y regresa.
    - Alta sensibilidad en 'Sorprendido' y 'Disgusto' con asistencia geométrica.
    - Debounce adaptativo para transiciones orgánicas sin parpadeos.
    """

    def __init__(
        self,
        alpha: float = SMOOTHING_ALPHA,
        bbox_alpha: float = BBOX_SMOOTHING_ALPHA,
        max_lost: int = 10,
        debounce_frames: int = EMOTION_DEBOUNCE_FRAMES,
        conf_thresholds: dict | None = None,
        sensitivity_boost: dict | None = None,
        debounce_by_class: dict | None = None,
    ):
        self.alpha              = alpha
        self.bbox_alpha         = bbox_alpha
        self.max_lost           = max_lost
        self.debounce_frames    = debounce_frames
        self.conf_thresholds    = dict(conf_thresholds or EMOTION_CONFIDENCE_THRESHOLDS)
        self.sensitivity_boost  = dict(sensitivity_boost or EMOTION_SENSITIVITY_BOOST)
        self.debounce_by_class  = dict(debounce_by_class or EMOTION_DEBOUNCE_BY_CLASS)

        self.tracks: dict[int, dict] = {}
        self.gallery: dict[int, dict] = {}  # Memoria biométrica a largo plazo
        self.next_id = 1

    def update(
        self,
        detections: list[dict],
        frame_gray: np.ndarray | None = None,
        use_smoothing: bool = True,
    ) -> list[dict]:
        matched_tracks: set[int] = set()
        matched_dets:   set[int] = set()
        track_ids = list(self.tracks.keys())

        # ── 1. Asociación espacial rápida por IoU (personas que siguen en cámara)
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

                # EMA Bbox
                if use_smoothing:
                    cx, cy, cw, ch = det["bbox"]
                    px, py, pw, ph = track["bbox"]
                    a = self.bbox_alpha
                    track["bbox"] = (
                        int(a * cx + (1 - a) * px),
                        int(a * cy + (1 - a) * py),
                        int(a * cw + (1 - a) * pw),
                        int(a * ch + (1 - a) * ph),
                    )
                    # EMA Probabilidades
                    track["probs"] = self.alpha * det["raw_probs"] + (1 - self.alpha) * track["probs"]
                    track["probs"] /= (track["probs"].sum() + 1e-7)
                else:
                    track["bbox"]  = det["bbox"]
                    track["probs"] = det["raw_probs"]

                # Actualizar firma biométrica acumulada en galería
                if frame_gray is not None:
                    bx, by, bw, bh = track["bbox"]
                    face_roi = frame_gray[by:by+bh, bx:bx+bw]
                    if face_roi.size > 0:
                        sig = _extract_biometric_signature(face_roi, track["landmarks"], track["bbox"])
                        if best_tid in self.gallery:
                            self._update_gallery_signature(best_tid, sig)

        # ── 2. Re-Identificación Biométrica de Sujetos que Salieron y Vuelven ──
        active_ids_in_screen = {tid for tid in matched_tracks}
        for d_idx, det in enumerate(detections):
            if d_idx in matched_dets:
                continue

            bx, by, bw, bh = det["bbox"]
            sig = None
            if frame_gray is not None and bw >= 16 and bh >= 16:
                face_roi = frame_gray[by:by+bh, bx:bx+bw]
                if face_roi.size > 0:
                    sig = _extract_biometric_signature(face_roi, det.get("landmarks"), det["bbox"])

            assigned_id = None
            reidentified_flag = False

            # Comparar con sujetos registrados que NO están en pantalla actualmente
            if sig is not None and self.gallery:
                best_sim, candidate_id = 0.0, None
                for gal_id, gal_data in self.gallery.items():
                    if gal_id in active_ids_in_screen:
                        continue  # Ya está presente otra persona con este ID
                    sim = _compute_biometric_similarity(sig, gal_data["signature"])
                    if sim > best_sim:
                        best_sim, candidate_id = sim, gal_id

                if best_sim >= REID_SIMILARITY_THRESHOLD and candidate_id is not None:
                    assigned_id = candidate_id
                    reidentified_flag = True
                    self._update_gallery_signature(assigned_id, sig)

            # Si no coincide con nadie registrado, es un nuevo sujeto
            if assigned_id is None:
                assigned_id = self.next_id
                self.next_id += 1
                if sig is not None:
                    self.gallery[assigned_id] = {
                        "signature": sig,
                        "samples": 1,
                        "person_color": PERSON_PALETTE[(assigned_id - 1) % len(PERSON_PALETTE)],
                        "person_label": f"SUJETO #{assigned_id:02d}",
                    }

            init_boosted = det["raw_probs"].copy()
            for emo_i, boost_val in self.sensitivity_boost.items():
                init_boosted[emo_i] *= boost_val
            init_idx = int(np.argmax(init_boosted))

            self.tracks[assigned_id] = {
                "id":                 assigned_id,
                "bbox":               det["bbox"],
                "probs":              det["raw_probs"].copy(),
                "lost_frames":        0,
                "landmarks":          det.get("landmarks"),
                "confidence":         det.get("confidence", 1.0),
                "display_idx":        init_idx,
                "candidate_idx":      init_idx,
                "candidate_count":    0,
                "reidentified_banner": 45 if reidentified_flag else 0,
            }
            active_ids_in_screen.add(assigned_id)
            matched_dets.add(d_idx)

        # ── 3. Purgar tracks temporales (la galería NUNCA se borra) ─────────────
        for tid in [t for t, tk in self.tracks.items() if t not in matched_tracks and t not in active_ids_in_screen]:
            self.tracks[tid]["lost_frames"] += 1

        self.tracks = {
            tid: tk for tid, tk in self.tracks.items()
            if tk["lost_frames"] <= self.max_lost
        }

        # ── 4. Sensibilidad reactiva y debounce adaptativo ───────────────────
        result = []
        for tid, tk in self.tracks.items():
            if tk["lost_frames"] != 0:
                continue

            probs = tk["probs"]
            boosted = probs.copy()

            # Boost multiplicador de sensibilidad
            for emo_i, boost_val in self.sensitivity_boost.items():
                boosted[emo_i] *= boost_val

            # Asistencia biométrica por Landmarks faciales (apertura de boca / compresión nasal)
            lm = tk.get("landmarks")
            if lm is not None and len(lm) >= 5:
                p_re, p_le = lm[0].astype(float), lm[1].astype(float)
                p_n        = lm[2].astype(float)
                p_rm, p_lm = lm[3].astype(float), lm[4].astype(float)
                eye_dist   = max(5.0, float(np.linalg.norm(p_re - p_le)))
                mouth_mid  = (p_rm + p_lm) / 2.0
                mouth_open = float(np.linalg.norm(p_n - mouth_mid)) / eye_dist

                # Boca abierta o cejas levantadas -> Sorpresa
                if mouth_open > 0.82:
                    boosted[5] *= 1.45  # Sorprendido

                # Nariz y labio superior fruncidos -> Disgusto
                if mouth_open < 0.52:
                    boosted[1] *= 1.40  # Disgusto

            sum_b = boosted.sum()
            if sum_b > 0:
                boosted /= sum_b

            raw_best_idx = int(np.argmax(boosted))

            # Umbral de confianza adaptativo
            threshold = self.conf_thresholds.get(raw_best_idx, 0.0)
            if float(boosted[raw_best_idx]) >= threshold:
                resolved_idx = raw_best_idx
            else:
                sorted_idxs = np.argsort(boosted)[::-1]
                resolved_idx = int(sorted_idxs[0])
                for candidate in sorted_idxs[1:]:
                    if float(boosted[candidate]) >= self.conf_thresholds.get(int(candidate), 0.0):
                        resolved_idx = int(candidate)
                        break

            # Debounce ultra-rápido por emoción
            req_debounce = self.debounce_by_class.get(resolved_idx, self.debounce_frames)
            if resolved_idx != tk["display_idx"]:
                if resolved_idx == tk.get("candidate_idx"):
                    tk["candidate_count"] = tk.get("candidate_count", 0) + 1
                else:
                    tk["candidate_idx"]   = resolved_idx
                    tk["candidate_count"] = 1

                if tk["candidate_count"] >= req_debounce:
                    tk["display_idx"]     = resolved_idx
                    tk["candidate_count"] = 0
            else:
                tk["candidate_idx"]   = resolved_idx
                tk["candidate_count"] = 0

            display_idx = tk["display_idx"]
            person_color = PERSON_PALETTE[(tid - 1) % len(PERSON_PALETTE)]
            person_label = f"SUJETO #{tid:02d}"

            if tk.get("reidentified_banner", 0) > 0:
                tk["reidentified_banner"] -= 1

            result.append({
                "id":                  tid,
                "person_label":        person_label,
                "person_color":        person_color,
                "bbox":                tk["bbox"],
                "probs":               probs,
                "boosted_probs":       boosted,
                "pred_idx":            display_idx,
                "raw_pred_idx":        raw_best_idx,
                "pred_label":          EMOTION_LABELS[display_idx],
                "confidence":          float(boosted[display_idx]) * 100,
                "landmarks":           tk["landmarks"],
                "det_confidence":      tk["confidence"] * 100,
                "reidentified_banner": tk.get("reidentified_banner", 0),
            })
        return result

    def _update_gallery_signature(self, subj_id: int, new_sig: tuple) -> None:
        """Actualiza la firma promedio del sujeto con suavizado exponencial."""
        if subj_id not in self.gallery:
            return
        g = self.gallery[subj_id]
        t_old, h_old, g_old = g["signature"]
        t_new, h_new, g_new = new_sig

        t_up = 0.85 * t_old + 0.15 * t_new
        h_up = 0.85 * h_old + 0.15 * h_new
        g_up = g_old
        if g_old is not None and g_new is not None:
            g_up = 0.85 * g_old + 0.15 * g_new

        g["signature"] = (t_up, h_up, g_up)
        g["samples"] += 1


# ──────────────────────────────────────────────────────────────────────────────
# 3. DIBUJO DEL HUD HOLOGRÁFICO Y REPORTES MULTI-PERSONA
# ──────────────────────────────────────────────────────────────────────────────

def _draw_styled_bbox(
    frame: np.ndarray, bbox: tuple, color: tuple, thickness: int = 2, corner_len: int = 20
) -> None:
    """Dibuja recuadro estilizado con esquinas destacadas estilo sci-fi."""
    x, y, w, h = bbox
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    clen = min(corner_len, w // 4, h // 4)
    for (px, py), (dx, dy) in [
        ((x,     y),     (clen, 0)), ((x,     y),     (0, clen)),
        ((x+w,   y),     (-clen, 0)), ((x+w,  y),     (0, clen)),
        ((x,     y+h),   (clen, 0)), ((x,     y+h),   (0, -clen)),
        ((x+w,   y+h),   (-clen, 0)), ((x+w,  y+h),   (0, -clen)),
    ]:
        cv2.line(frame, (px, py), (px + dx, py + dy), color, thickness)


def _draw_stepped_hologram_silhouette(
    frame: np.ndarray,
    bbox: tuple,
    person_color: tuple,
    person_label: str,
    frame_count: int,
    block_size: int = 14,
) -> tuple[int, int]:
    """
    Dibuja una silueta biométrica escalonada e irregular en cuadrícula ortogonal (90°),
    idéntica a las capturas holográficas de referencia (pixelated stepped sci-fi contour),
    con línea de escaneo láser horizontal pulsante y métrica de temperatura simulada.
    """
    h_f, w_f = frame.shape[:2]
    x, y, w, h = bbox
    cx = x + w // 2

    top_y = max(45, y - int(0.35 * h))
    bot_y = min(h_f - 30, y + int(2.10 * h))

    grid_span_w = max(int(w * 2.35), 70)
    left_x  = max(6, cx - grid_span_w // 2)
    right_x = min(w_f - 6, cx + grid_span_w // 2)

    cell_w = max(2, (right_x - left_x) // block_size)
    cell_h = max(3, (bot_y - top_y) // block_size)

    mask_grid = np.zeros((cell_h, cell_w), dtype=np.uint8)
    center_col = cell_w // 2

    for row in range(cell_h):
        prog = row / max(1, cell_h - 1)
        if prog < 0.28:
            rad = max(1, int((0.30 + 0.70 * (prog / 0.28)) * (w * 0.58 / block_size)))
        elif prog < 0.40:
            rad = max(1, int(w * 0.46 / block_size))
        elif prog < 0.58:
            shoulder_expand = (prog - 0.40) / 0.18
            rad = max(1, int((w * 0.55 + shoulder_expand * w * 0.60) / block_size))
        else:
            rad = max(1, int((w * 1.08 - 0.12 * prog * w) / block_size))

        c_min = max(0, center_col - rad)
        c_max = min(cell_w, center_col + rad + 1)
        mask_grid[row, c_min:c_max] = 255

    pixel_mask = cv2.resize(
        mask_grid,
        (cell_w * block_size, cell_h * block_size),
        interpolation=cv2.INTER_NEAREST,
    )

    full_mask = np.zeros((h_f, w_f), dtype=np.uint8)
    paste_y2 = min(h_f, top_y + pixel_mask.shape[0])
    paste_x2 = min(w_f, left_x + pixel_mask.shape[1])
    h_p = paste_y2 - top_y
    w_p = paste_x2 - left_x

    if h_p > 0 and w_p > 0:
        full_mask[top_y:paste_y2, left_x:paste_x2] = pixel_mask[:h_p, :w_p]

    contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Capa 1: Resplandor (Glow)
        overlay = frame.copy()
        cv2.drawContours(overlay, contours, -1, person_color, 4)
        cv2.addWeighted(overlay, 0.40, frame, 0.60, 0, frame)

        # Capa 2: Trazo blanco
        cv2.drawContours(frame, contours, -1, (255, 255, 255), 1)

        # Capa 3: Línea de escaneo láser pulsante horizontal
        scan_t = (math.sin(time.time() * 3.2 + float(bbox[0]) * 0.02) + 1.0) / 2.0
        laser_y = int(top_y + scan_t * (bot_y - top_y))

        cv2.line(frame, (left_x, laser_y), (paste_x2, laser_y), person_color, 2)
        cv2.line(frame, (left_x, laser_y), (paste_x2, laser_y), (255, 255, 255), 1)

        cv2.circle(frame, (left_x, laser_y), 3, (255, 255, 255), -1)
        cv2.circle(frame, (paste_x2, laser_y), 4, person_color, -1)
        cv2.circle(frame, (paste_x2, laser_y), 2, (255, 255, 255), -1)

        # Capa 4: Tag biométrico de telemetría flotante (como en la referencia)
        tag_x = paste_x2 + 8
        if tag_x + 58 < w_f:
            cv2.line(frame, (paste_x2, laser_y), (tag_x, laser_y), person_color, 1)
            t_overlay = frame.copy()
            cv2.rectangle(t_overlay, (tag_x, laser_y - 10), (tag_x + 56, laser_y + 10), (20, 20, 30), -1)
            cv2.rectangle(t_overlay, (tag_x, laser_y - 10), (tag_x + 56, laser_y + 10), person_color, 1)
            cv2.addWeighted(t_overlay, 0.85, frame, 0.15, 0, frame)
            temp_sim = 36.2 + ((bbox[0] * 19) % 7) * 0.1
            cv2.putText(frame, f"{temp_sim:.1f} C", (tag_x + 5, laser_y + 4),
                        cv2.FONT_HERSHEY_DUPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    return (paste_x2 if 'paste_x2' in locals() else (x + w), y + h // 2)


def _draw_emotion_badge(
    frame: np.ndarray, x: int, y: int,
    label: str, confidence: float, person_color: tuple, person_label: str,
    reidentified_banner: int = 0,
) -> tuple[int, int]:
    """Dibuja badge superior identificando al sujeto, su emoción y alerta de Re-ID."""
    text = f"{person_label}: {label.upper()} {confidence:.0f}%"
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 0.45, 1)
    badge_y = max(46, y - th - 14)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, badge_y), (x + tw + 18, badge_y + th + 10), (15, 15, 20), -1)
    cv2.rectangle(overlay, (x, badge_y), (x + tw + 18, badge_y + th + 10), person_color, 1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    cv2.circle(frame, (x + 8, badge_y + th // 2 + 5), 4, person_color, -1)
    cv2.putText(frame, text, (x + 18, badge_y + th + 3), font, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    # Banner visual cuando el sujeto fue reconocido al regresar a la cámara
    if reidentified_banner > 0:
        re_txt = "MEMORIA: RE-IDENTIFICADO"
        (rw, rh), _ = cv2.getTextSize(re_txt, font, 0.34, 1)
        rx = x
        ry = badge_y - rh - 6
        cv2.rectangle(frame, (rx, ry), (rx + rw + 8, ry + rh + 4), (0, 160, 80), -1)
        cv2.putText(frame, re_txt, (rx + 4, ry + rh + 1), font, 0.34, (255, 255, 255), 1, cv2.LINE_AA)

    return (x + tw + 18, badge_y + th // 2 + 5)


def _draw_probability_card(
    frame: np.ndarray,
    card_x: int, card_y: int, card_w: int, card_h: int,
    track: dict,
) -> None:
    """Renderiza una tarjeta gráfica independiente de probabilidades para una persona."""
    person_color = track["person_color"]
    person_label = track["person_label"]
    pred_idx     = track["pred_idx"]
    probs        = track["boosted_probs"]

    overlay = frame.copy()
    cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), (12, 14, 20), -1)
    cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), person_color, 1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    header_text = f"{person_label} | {track['pred_label'].upper()}"
    cv2.putText(frame, header_text, (card_x + 8, card_y + 15),
                cv2.FONT_HERSHEY_DUPLEX, 0.38, person_color, 1, cv2.LINE_AA)
    cv2.line(frame, (card_x + 6, card_y + 20), (card_x + card_w - 6, card_y + 20), (50, 55, 70), 1)

    num_emos = len(probs)
    avail_bar_h = card_h - 26
    bar_h = max(9, avail_bar_h // num_emos - 2)
    max_bar_w = card_w - 65

    for i in range(num_emos):
        prob = float(probs[i])
        by = card_y + 24 + i * (bar_h + 2)
        is_active = (i == pred_idx)
        bar_color = person_color if is_active else (70, 75, 85)

        fill_w = int(max_bar_w * prob)
        cv2.rectangle(frame, (card_x + 36, by), (card_x + 36 + max_bar_w, by + bar_h), (25, 28, 35), -1)
        if fill_w > 0:
            cv2.rectangle(frame, (card_x + 36, by), (card_x + 36 + fill_w, by + bar_h), bar_color, -1)

        text_c = (255, 255, 255) if is_active else (160, 160, 170)
        lbl_short = EMOTION_LABELS[i][:3]
        cv2.putText(frame, lbl_short, (card_x + 8, by + bar_h - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, text_c, 1, cv2.LINE_AA)
        cv2.putText(frame, f"{prob*100:2.0f}%", (card_x + 36 + max_bar_w + 4, by + bar_h - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, text_c, 1, cv2.LINE_AA)


def _draw_biometric_reports(
    frame: np.ndarray,
    tracks: list[dict],
    show_graphs: bool = True,
) -> None:
    """
    Gestiona el reporte multi-persona:
    - 1 persona: Gráfica flotante adyacente conectada al rostro.
    - 2 o más personas: Dock lateral derecho estructurado en tarjetas independientes
      para cada sujeto, con líneas directrices conectando cada rostro con su gráfica.
    """
    if not show_graphs or not tracks:
        return

    h_f, w_f = frame.shape[:2]
    num_tracks = len(tracks)

    if num_tracks == 1:
        track = tracks[0]
        x, y, w, h = track["bbox"]
        card_w, card_h = 175, 125
        card_x = x + w + 14
        card_y = max(48, min(y, h_f - card_h - 30))
        if card_x + card_w >= w_f:
            card_x = max(10, x - card_w - 14)

        _draw_probability_card(frame, card_x, card_y, card_w, card_h, track)
        cv2.line(frame, (x + w, y + h // 2), (card_x, y + h // 2), track["person_color"], 1, cv2.LINE_AA)

    else:
        dock_w = 175
        dock_x = w_f - dock_w - 8
        avail_h = h_f - 46 - 28 - 10
        card_h = min(120, max(85, (avail_h - (num_tracks - 1) * 8) // num_tracks))

        for idx, track in enumerate(tracks):
            card_y = 48 + idx * (card_h + 8)
            if card_y + card_h > h_f - 26:
                break

            _draw_probability_card(frame, dock_x, card_y, dock_w, card_h, track)

            fx, fy, fw, fh = track["bbox"]
            face_anchor = (fx + fw, fy + fh // 2)
            dock_anchor = (dock_x, card_y + card_h // 2)

            overlay = frame.copy()
            cv2.line(overlay, face_anchor, dock_anchor, track["person_color"], 1, cv2.LINE_AA)
            cv2.circle(overlay, face_anchor, 3, track["person_color"], -1)
            cv2.circle(overlay, dock_anchor, 3, track["person_color"], -1)
            cv2.addWeighted(overlay, 0.40, frame, 0.60, 0, frame)


def _draw_hud_header(
    frame: np.ndarray, fps: float, latency_ms: float,
    use_smoothing: bool, is_cuda: bool, paused: bool,
    show_silhouette: bool = True, num_persons: int = 0, gallery_count: int = 0,
) -> None:
    """Barra superior con métricas de rendimiento, estado multi-persona y memoria Re-ID."""
    h, w  = frame.shape[:2]
    font  = cv2.FONT_HERSHEY_DUPLEX
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 42), (12, 12, 16), -1)
    cv2.line(overlay, (0, 42), (w, 42), (0, 200, 240), 1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    cv2.putText(frame, "EMOTION AI v2.0 | RE-ID BIOMETRICS", (14, 18), font, 0.44, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"ACTIVOS: {num_persons} | MEMORIA RE-ID: {gallery_count} SUJETOS",
                (14, 34), font, 0.36, (0, 255, 160) if num_persons > 0 else (140, 140, 140), 1, cv2.LINE_AA)

    cv2.putText(frame, f"FPS: {fps:.1f} | Lat: {latency_ms:.1f}ms", (w // 2 - 110, 18), font, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "DET: YUNET ONNX", (w // 2 - 110, 34), font, 0.38, (0, 230, 255), 1, cv2.LINE_AA)

    holo_c = (0, 240, 255) if show_silhouette else (90, 90, 90)
    cv2.putText(frame, f"HOLO: {'ON' if show_silhouette else 'OFF'}", (w - 170, 18), font, 0.38, holo_c, 1, cv2.LINE_AA)

    smooth_c = (0, 255, 180) if use_smoothing else (90, 90, 90)
    cv2.putText(frame, f"EMA: {'ON' if use_smoothing else 'OFF'}", (w - 170, 34), font, 0.38, smooth_c, 1, cv2.LINE_AA)

    if paused:
        cv2.putText(frame, "[PAUSADO]", (w // 2 - 45, h // 2), font, 0.9, (0, 0, 255), 2, cv2.LINE_AA)


def _draw_hud_footer(frame: np.ndarray, angry_threshold: float = 0.50) -> None:
    """Barra inferior con atajos de teclado y controles interactivos."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 24), (w, h), (12, 12, 16), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.putText(
        frame,
        f"q:Salir | h:Silueta Holo | g:Graficas | t:EMA | l:Landmarks | +/-:Umbral Enojado({angry_threshold:.0%}) | c:Cam | s:Foto",
        (12, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (210, 220, 230), 1, cv2.LINE_AA,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 4. BUCLE PRINCIPAL DE TIEMPO REAL
# ──────────────────────────────────────────────────────────────────────────────

def run_realtime() -> None:
    """
    Ejecuta el reconocimiento de emociones en tiempo real con webcam.

    Innovaciones integradas:
        - Memoria Biométrica Facial (Re-ID): Identifica al sujeto al salir y volver.
        - Silueta biométrica escalonada e irregular en cuadrícula con escáner láser.
        - Detección y reportes multi-persona con gráficas independientes por sujeto.
        - Sensibilidad aumentada en 'Sorprendido' y 'Disgusto' (umbral 0.18).
        - Seguimiento y suavizado temporal EMA en 60 FPS con ventana ajustable.
    """
    print("\n" + "=" * 60)
    print(">>> DEMO EN TIEMPO REAL — EMOTION AI v2.0 (RE-ID BIOMETRICS)")
    print("=" * 60)

    print("   [1/3] Cargando red neuronal de emociones...")
    model = load_model()
    model.eval()

    print("   [2/3] Inicializando detector neuronal YuNet...")
    detector = FaceDetector()
    tracker  = FaceTracker(
        alpha=SMOOTHING_ALPHA,
        bbox_alpha=BBOX_SMOOTHING_ALPHA,
        debounce_frames=EMOTION_DEBOUNCE_FRAMES,
    )

    print("   [3/3] Buscando cámara web activa...")
    cap, current_cam_idx = open_camera()

    if not cap or not cap.isOpened():
        print("\n[!] No se pudo abrir ninguna cámara. Verifica que esté conectada.")
        return

    WINDOW_NAME = "Reconocimiento de Emociones Faciales - AI Studio"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 960, 720)

    print("\n" + "=" * 60)
    print("   [OK] Sistema biométrico activo. Atajos:")
    print("   |-- 'h': Silueta Holográfica Escalonada (Sci-Fi)")
    print("   |-- 'g': Mostrar/Ocultar Gráficas de Emoción")
    print("   |-- 't': Suavizado Temporal EMA")
    print("   |-- 'l': Puntos Faciales (Landmarks)")
    print("   |-- '+/-': Ajustar umbral de Enojado en vivo")
    print("   |-- 's': Capturar foto")
    print("   |-- 'p': Pausar / Reanudar")
    print("   |-- 'c': Cambiar cámara")
    print("   +-- 'q' / ESC: Salir")
    print("=" * 60)

    use_smoothing   = True
    show_landmarks  = True
    show_silhouette = True
    show_graphs     = True
    paused          = False
    frame_count     = 0
    fps             = 30.0
    latency_ms      = 0.0
    is_cuda         = DEVICE.type == "cuda"
    prev_time       = time.time()
    frame           = None

    while True:
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

        active_tracks = []
        if not paused:
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

            active_tracks = tracker.update(processed_dets, frame_gray=gray, use_smoothing=use_smoothing)

            # Renderizado de siluetas, recuadros y badges
            for track in active_tracks:
                x, y, w, h   = track["bbox"]
                p_color      = track["person_color"]
                p_label      = track["person_label"]

                # 1. Silueta irregular escalonada holográfica
                if show_silhouette:
                    _draw_stepped_hologram_silhouette(
                        frame, (x, y, w, h), person_color=p_color,
                        person_label=p_label, frame_count=frame_count,
                    )

                # 2. Bbox estilizado y badge de emoción
                _draw_styled_bbox(frame, (x, y, w, h), color=p_color)
                _draw_emotion_badge(
                    frame, x, y, track["pred_label"], track["confidence"],
                    person_color=p_color, person_label=p_label,
                    reidentified_banner=track.get("reidentified_banner", 0),
                )

                # 3. Puntos faciales (Landmarks)
                if show_landmarks and track["landmarks"] is not None:
                    for pt in track["landmarks"]:
                        cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (0, 255, 255), -1, cv2.LINE_AA)

            # 4. Reportes y gráficas independientes multi-persona
            _draw_biometric_reports(frame, active_tracks, show_graphs=show_graphs)

        latency_ms = (time.time() - start_tick) * 1000.0
        curr_time  = time.time()
        elapsed    = curr_time - prev_time
        if elapsed > 0:
            fps = 0.9 * fps + 0.1 / elapsed
        prev_time = curr_time

        _draw_hud_header(
            frame, fps, latency_ms, use_smoothing, is_cuda, paused,
            show_silhouette=show_silhouette, num_persons=len(active_tracks),
            gallery_count=len(tracker.gallery),
        )
        _draw_hud_footer(frame, angry_threshold=tracker.conf_thresholds[0])

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key == ord("h"):
            show_silhouette = not show_silhouette
            print(f"   [SILUETA HOLOGRAMA] {'ACTIVADA' if show_silhouette else 'DESACTIVADA'}")
        elif key == ord("g"):
            show_graphs = not show_graphs
            print(f"   [GRAFICAS BIOMETRICAS] {'VISIBLES' if show_graphs else 'OCULTAS'}")
        elif key == ord("t"):
            use_smoothing = not use_smoothing
            print(f"   [EMA] {'ACTIVADO' if use_smoothing else 'DESACTIVADO'}")
        elif key == ord("l"):
            show_landmarks = not show_landmarks
            print(f"   [LANDMARKS] {'VISIBLES' if show_landmarks else 'OCULTOS'}")
        elif key in (ord("+"), ord("=")):
            tracker.conf_thresholds[0] = min(0.95, tracker.conf_thresholds[0] + 0.05)
            print(f"   [UMBRAL Enojado] → {tracker.conf_thresholds[0]:.0%}")
        elif key == ord("-"):
            tracker.conf_thresholds[0] = max(0.10, tracker.conf_thresholds[0] - 0.05)
            print(f"   [UMBRAL Enojado] → {tracker.conf_thresholds[0]:.0%}")
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
