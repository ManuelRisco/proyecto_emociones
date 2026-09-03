# ============================================================================
# APLICACIÓN PORTÁTIL DE RECONOCIMIENTO DE EMOCIONES (PORTABLE v2.0 - RE-ID)
# ============================================================================
# Autores: Manuel Fabrizzio Risco Gil y Alejandro Choquihuayta
# Dependencias: Únicamente opencv-python y numpy (sin PyTorch, sin GPU obligatoria)
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

# Colores característicos de cada emoción (tradicionales)
EMOTIONS = {
    0: {"name": "Enojado",     "color": (40, 40, 235)},   # Rojo
    1: {"name": "Disgusto",    "color": (34, 139, 34)},   # Verde bosque
    2: {"name": "Miedo",       "color": (180, 50, 180)},  # Morado
    3: {"name": "Feliz",       "color": (0, 215, 255)},   # Amarillo/Dorado
    4: {"name": "Triste",      "color": (235, 120, 40)},  # Azul
    5: {"name": "Sorprendido", "color": (0, 140, 255)},   # Naranja
    6: {"name": "Neutral",     "color": (170, 170, 170)}, # Gris
}

# Umbrales mínimos de confianza adaptativos
CONF_THRESHOLDS = {
    0: 0.25,   # Enojado     — ULTRA SENSIBLE (baja exigencia para responder al ceño fruncido)
    1: 0.18,   # Disgusto    — ULTRA SENSIBLE
    2: 0.35,   # Miedo
    3: 0.28,   # Feliz
    4: 0.35,   # Triste
    5: 0.18,   # Sorprendido — ULTRA SENSIBLE
    6: 0.24,   # Neutral
}

SENSITIVITY_BOOST = {
    0: 1.65,   # Boost Enojado (+65% reactividad e impulso a la barra)
    1: 1.55,   # Boost Disgusto (+55%)
    2: 1.05,
    3: 1.00,
    4: 1.05,
    5: 1.55,   # Boost Sorprendido (+55%)
    6: 0.85,
}

DEBOUNCE_BY_CLASS = {
    0: 2,      # Enojado (2 frames ~33ms, respuesta inmediata)
    1: 3,      # Disgusto (3 frames)
    2: 4,      # Miedo
    3: 3,      # Feliz
    4: 5,      # Triste
    5: 2,      # Sorprendido ultra-rápido (2 frames)
    6: 5,      # Neutral
}

REID_SIMILARITY_THRESHOLD = 0.62


# ──────────────────────────────────────────────────────────────────────────────
# 1. BIOMETRÍA Y RE-IDENTIFICACIÓN (PORTABLE)
# ──────────────────────────────────────────────────────────────────────────────

def _extract_biometric_signature_portable(gray_face, landmarks, bbox):
    tpl = cv2.resize(gray_face, (28, 28), interpolation=cv2.INTER_AREA).astype(np.float32)
    tpl_norm = (tpl - np.mean(tpl)) / (np.std(tpl) + 1e-5)
    hist = cv2.calcHist([gray_face], [0], None, [16], [0, 256]).flatten()
    h_norm = hist / (np.linalg.norm(hist) + 1e-5)

    geom_vec = None
    if landmarks is not None and len(landmarks) >= 5:
        p_re, p_le = landmarks[0].astype(float), landmarks[1].astype(float)
        p_n        = landmarks[2].astype(float)
        p_rm, p_lm = landmarks[3].astype(float), landmarks[4].astype(float)

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


def _compute_biometric_similarity_portable(sig1, sig2):
    t1, h1, g1 = sig1
    t2, h2, g2 = sig2

    t1_f, t2_f = t1.flatten(), t2.flatten()
    sim_tpl = float(np.dot(t1_f, t2_f) / (np.linalg.norm(t1_f) * np.linalg.norm(t2_f) + 1e-5))
    sim_tpl = max(0.0, min(1.0, (sim_tpl + 1.0) / 2.0))

    sim_hist = max(0.0, float(np.dot(h1, h2)))

    sim_geom = 0.5
    if g1 is not None and g2 is not None:
        diff = np.linalg.norm(g1 - g2)
        sim_geom = max(0.0, min(1.0, 1.0 / (1.0 + diff * 2.0)))

    return 0.50 * sim_tpl + 0.30 * sim_geom + 0.20 * sim_hist


class PortableEmotionTracker:
    def __init__(self, alpha=0.30, conf_thresholds=None):
        self.alpha = alpha
        self.conf_thresholds = dict(conf_thresholds or CONF_THRESHOLDS)
        self.tracks = {}
        self.gallery = {}  # Memoria Re-ID
        self.next_id = 1

    def update(self, detections, frame_gray=None, use_smoothing=True):
        matched = set()
        track_ids = list(self.tracks.keys())

        # 1. Matching espacial IoU
        for d_idx, d in enumerate(detections):
            x, y, w, h = d["bbox"]
            best_iou, best_tid = 0.0, None
            for tid in track_ids:
                if tid in matched:
                    continue
                tx, ty, tw, th = self.tracks[tid]["bbox"]
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
                    t["probs"] /= t["probs"].sum()
                else:
                    t["bbox"] = d["bbox"]
                    t["probs"] = d["probs"].copy()
                t["landmarks"] = d.get("landmarks")

                if frame_gray is not None:
                    bx, by, bw, bh = t["bbox"]
                    f_roi = frame_gray[by:by+bh, bx:bx+bw]
                    if f_roi.size > 0 and best_tid in self.gallery:
                        sig = _extract_biometric_signature_portable(f_roi, t["landmarks"], t["bbox"])
                        g_old = self.gallery[best_tid]["signature"]
                        t_up = 0.85 * g_old[0] + 0.15 * sig[0]
                        h_up = 0.85 * g_old[1] + 0.15 * sig[1]
                        g_up = 0.85 * g_old[2] + 0.15 * sig[2] if g_old[2] is not None and sig[2] is not None else g_old[2]
                        self.gallery[best_tid]["signature"] = (t_up, h_up, g_up)

        # 2. Re-identificación para no asociados
        active_screen_ids = {tid for tid in matched}
        for d_idx, d in enumerate(detections):
            x, y, w, h = d["bbox"]
            is_matched = any(self.tracks[tid]["bbox"] == d["bbox"] for tid in matched if tid in self.tracks)
            if is_matched:
                continue

            sig = None
            if frame_gray is not None and w >= 16 and h >= 16:
                f_roi = frame_gray[y:y+h, x:x+w]
                if f_roi.size > 0:
                    sig = _extract_biometric_signature_portable(f_roi, d.get("landmarks"), d["bbox"])

            assigned_id = None
            reid_banner = False
            if sig is not None and self.gallery:
                best_sim, cand_id = 0.0, None
                for gal_id, gal_data in self.gallery.items():
                    if gal_id in active_screen_ids:
                        continue
                    sim = _compute_biometric_similarity_portable(sig, gal_data["signature"])
                    if sim > best_sim:
                        best_sim, cand_id = sim, gal_id

                if best_sim >= REID_SIMILARITY_THRESHOLD and cand_id is not None:
                    assigned_id = cand_id
                    reid_banner = True

            if assigned_id is None:
                assigned_id = self.next_id
                self.next_id += 1
                if sig is not None:
                    self.gallery[assigned_id] = {
                        "signature": sig,
                        "person_label": f"SUJETO #{assigned_id:02d}",
                    }

            init_boosted = d["probs"].copy()
            for emo_i, boost_val in SENSITIVITY_BOOST.items():
                init_boosted[emo_i] *= boost_val
            init_idx = int(np.argmax(init_boosted))

            self.tracks[assigned_id] = {
                "id": assigned_id,
                "bbox": d["bbox"],
                "probs": d["probs"].copy(),
                "lost": 0,
                "landmarks": d.get("landmarks"),
                "display_idx": init_idx,
                "candidate_idx": init_idx,
                "candidate_count": 0,
                "reid_banner": 45 if reid_banner else 0,
            }
            active_screen_ids.add(assigned_id)

        # 3. Purgar
        to_del = [tid for tid, t in self.tracks.items() if tid not in matched and tid not in active_screen_ids and t["lost"] > 10]
        for tid in to_del:
            del self.tracks[tid]
        for tid, t in self.tracks.items():
            if tid not in matched and tid not in active_screen_ids:
                t["lost"] += 1

        # 4. Decisión y sensibilidad
        res = []
        for tid, t in self.tracks.items():
            if t["lost"] != 0:
                continue

            probs = t["probs"]
            boosted = probs.copy()
            for emo_i, boost in SENSITIVITY_BOOST.items():
                boosted[emo_i] *= boost

            lm = t.get("landmarks")
            if lm is not None and len(lm) >= 5:
                p_re, p_le = lm[0].astype(float), lm[1].astype(float)
                p_n        = lm[2].astype(float)
                p_rm, p_lm = lm[3].astype(float), lm[4].astype(float)
                eye_dist   = max(5.0, float(np.linalg.norm(p_re - p_le)))
                mouth_open = float(np.linalg.norm(p_n - (p_rm + p_lm) / 2.0)) / eye_dist
                if mouth_open > 0.82:
                    boosted[5] *= 1.45  # Sorprendido
                elif mouth_open < 0.55:
                    boosted[1] *= 1.35  # Disgusto
                    boosted[0] *= 1.40  # Enojado (labios tensos / ceño apretado)

            if boosted.sum() > 0:
                boosted /= boosted.sum()

            raw_best = int(np.argmax(boosted))

            if float(boosted[raw_best]) >= self.conf_thresholds.get(raw_best, 0.0):
                resolved = raw_best
            else:
                sorted_idxs = np.argsort(boosted)[::-1]
                resolved = int(sorted_idxs[0])
                for cand in sorted_idxs[1:]:
                    if float(boosted[cand]) >= self.conf_thresholds.get(int(cand), 0.0):
                        resolved = int(cand)
                        break

            req_db = DEBOUNCE_BY_CLASS.get(resolved, 5)
            disp = t.get("display_idx", resolved)
            if resolved != disp:
                if resolved == t.get("candidate_idx"):
                    t["candidate_count"] = t.get("candidate_count", 0) + 1
                else:
                    t["candidate_idx"] = resolved
                    t["candidate_count"] = 1
                if t["candidate_count"] >= req_db:
                    t["display_idx"] = resolved
                    t["candidate_count"] = 0
            else:
                t["candidate_idx"] = resolved
                t["candidate_count"] = 0
                t["display_idx"] = resolved

            disp_idx = t.get("display_idx", resolved)
            emotion_color = EMOTIONS[disp_idx]["color"]
            p_label = f"SUJETO #{tid:02d}"

            if t.get("reid_banner", 0) > 0:
                t["reid_banner"] -= 1

            res.append({
                "id": tid,
                "emotion_color": emotion_color,
                "person_label": p_label,
                "bbox": t["bbox"],
                "probs": probs,
                "boosted_probs": boosted,
                "pred_idx": disp_idx,
                "label": EMOTIONS[disp_idx]["name"],
                "conf": float(boosted[disp_idx]) * 100.0,
                "landmarks": t["landmarks"],
                "reid_banner": t.get("reid_banner", 0),
            })
        return res


# ──────────────────────────────────────────────────────────────────────────────
# 2. DIBUJO DEL HUD Y REPORTES (PORTABLE)
# ──────────────────────────────────────────────────────────────────────────────

def _draw_probability_card_portable(frame, card_x, card_y, card_w, card_h, track):
    emotion_color = track["emotion_color"]
    person_label  = track["person_label"]
    pred_idx      = track["pred_idx"]
    probs         = track["boosted_probs"]

    overlay = frame.copy()
    cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), (12, 14, 20), -1)
    cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), emotion_color, 1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    header_text = f"{person_label} | {track['label'].upper()}"
    cv2.putText(frame, header_text, (card_x + 8, card_y + 15),
                cv2.FONT_HERSHEY_DUPLEX, 0.38, emotion_color, 1, cv2.LINE_AA)
    cv2.line(frame, (card_x + 6, card_y + 20), (card_x + card_w - 6, card_y + 20), (50, 55, 70), 1)

    num_emos = len(probs)
    avail_bar_h = card_h - 26
    bar_h = max(9, avail_bar_h // num_emos - 2)
    max_bar_w = card_w - 65

    for i in range(num_emos):
        prob = float(probs[i])
        by = card_y + 24 + i * (bar_h + 2)
        is_active = (i == pred_idx)
        bar_color = EMOTIONS[i]["color"]

        fill_w = int(max_bar_w * prob)
        cv2.rectangle(frame, (card_x + 36, by), (card_x + 36 + max_bar_w, by + bar_h), (25, 28, 35), -1)
        if fill_w > 0:
            if is_active:
                cv2.rectangle(frame, (card_x + 36, by), (card_x + 36 + fill_w, by + bar_h), bar_color, -1)
            else:
                dimmed = tuple(int(c * 0.45) for c in bar_color)
                cv2.rectangle(frame, (card_x + 36, by), (card_x + 36 + fill_w, by + bar_h), dimmed, -1)

        text_c = (255, 255, 255) if is_active else (150, 150, 150)
        lbl_short = EMOTIONS[i]["name"][:3]
        cv2.putText(frame, lbl_short, (card_x + 8, by + bar_h - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, text_c, 1, cv2.LINE_AA)
        cv2.putText(frame, f"{prob*100:2.0f}%", (card_x + 36 + max_bar_w + 4, by + bar_h - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, text_c, 1, cv2.LINE_AA)


def _draw_biometric_reports_portable(frame, tracks, show_graphs=True):
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

        _draw_probability_card_portable(frame, card_x, card_y, card_w, card_h, track)
        cv2.line(frame, (x + w, y + h // 2), (card_x, y + h // 2), track["emotion_color"], 1, cv2.LINE_AA)
    else:
        dock_w = 175
        dock_x = w_f - dock_w - 8
        avail_h = h_f - 46 - 28 - 10
        card_h = min(120, max(85, (avail_h - (num_tracks - 1) * 8) // num_tracks))

        for idx, track in enumerate(tracks):
            card_y = 48 + idx * (card_h + 8)
            if card_y + card_h > h_f - 26:
                break

            _draw_probability_card_portable(frame, dock_x, card_y, dock_w, card_h, track)

            fx, fy, fw, fh = track["bbox"]
            face_anchor = (fx + fw, fy + fh // 2)
            dock_anchor = (dock_x, card_y + card_h // 2)

            overlay = frame.copy()
            cv2.line(overlay, face_anchor, dock_anchor, track["emotion_color"], 1, cv2.LINE_AA)
            cv2.circle(overlay, face_anchor, 3, track["emotion_color"], -1)
            cv2.circle(overlay, dock_anchor, 3, track["emotion_color"], -1)
            cv2.addWeighted(overlay, 0.40, frame, 0.60, 0, frame)


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
    if target_idx is not None:
        cap = cv2.VideoCapture(target_idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(target_idx)
        if cap.isOpened():
            configure_camera(cap)
            return cap, target_idx
        return None, target_idx

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
    print("\nReconocimiento de Emociones Faciales (Modo Portable v2.0 - RE-ID)")
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
    print("  * 'g'       : Mostrar/Ocultar Gráficas Biométricas")
    print("  * 'l'       : Mostrar/Ocultar Puntos Faciales (Landmarks)")
    print("  * 'c'       : Cambiar de camara")
    print("  * 's'       : Capturar foto (Limpia: solo cámara, rostros y gráficas)")
    print("  * 'p'       : Pausar\n")

    fps = 60.0
    prev_time = time.time()
    paused = False
    show_landmarks  = True
    show_graphs     = True
    consecutive_failures = 0
    frame_count = 0
    clean_capture = None

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret or frame is None:
                consecutive_failures += 1
                if consecutive_failures > 60:
                    time.sleep(0.03)

                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Error leyendo camara. Presiona 'c' para cambiar.", (50, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                frame = cv2.resize(frame, (640, 480))
                consecutive_failures = 0

        if frame is None:
            continue

        frame_count += 1
        h_f, w_f = frame.shape[:2]
        scale = 320.0 / float(max(h_f, w_f))
        w_s, h_s = int(w_f * scale), int(h_f * scale)
        detect_img = cv2.resize(frame, (w_s, h_s), interpolation=cv2.INTER_LINEAR)

        detector.setInputSize((w_s, h_s))
        _, faces = detector.detect(detect_img)

        processed = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if faces is not None:
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

        tracks = tracker.update(processed, frame_gray=gray, use_smoothing=True)

        # Dibujar HUD
        for t in tracks:
            x, y, w, h = t["bbox"]
            emo_color  = t["emotion_color"]
            p_label    = t["person_label"]

            # Bbox con esquinas y color de la emoción detectada
            cv2.rectangle(frame, (x, y), (x + w, y + h), emo_color, 2)
            clen = min(15, w // 4, h // 4)
            cv2.line(frame, (x, y), (x + clen, y), emo_color, 3)
            cv2.line(frame, (x, y), (x, y + clen), emo_color, 3)
            cv2.line(frame, (x+w, y+h), (x+w-clen, y+h), emo_color, 3)
            cv2.line(frame, (x+w, y+h), (x+w, y+h-clen), emo_color, 3)

            badge_text = f"{p_label}: {t['label']} {t['conf']:.0f}%"
            cv2.putText(frame, badge_text, (x, max(20, y - 8)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.48, emo_color, 1, cv2.LINE_AA)

            if t.get("reid_banner", 0) > 0:
                cv2.rectangle(frame, (x, max(3, y - 26)), (x + 160, max(21, y - 8)), (0, 160, 80), -1)
                cv2.putText(frame, "RE-IDENTIFICADO", (x + 4, max(17, y - 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

            if show_landmarks and t.get("landmarks") is not None:
                for pt in t["landmarks"]:
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (0, 255, 255), -1, cv2.LINE_AA)

        # Gráficas independientes multi-persona
        _draw_biometric_reports_portable(frame, tracks, show_graphs=show_graphs)

        # Captura limpia para fotos (solo cámara, recuadros, gráficas y puntos, sin barras de menú)
        clean_capture = frame.copy()

        curr_time = time.time()
        elapsed = curr_time - prev_time
        if elapsed > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / elapsed)
        prev_time = curr_time

        cv2.rectangle(frame, (0, 0), (w_f, 28), (15, 15, 20), -1)
        cv2.putText(frame, f"FPS: {fps:.1f} | Cam [{cam_idx}] | EMA: AUTO | {len(tracks)} ACTIVOS | {len(tracker.gallery)} RE-ID",
                    (10, 19), cv2.FONT_HERSHEY_DUPLEX, 0.38, (0, 220, 255), 1, cv2.LINE_AA)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        elif key == ord("g"):
            show_graphs = not show_graphs
            print(f"Gráficas de Emociones: {'VISIBLES' if show_graphs else 'OCULTAS'}")
        elif key == ord("l"):
            show_landmarks = not show_landmarks
            print(f"Landmarks: {'VISIBLES' if show_landmarks else 'OCULTOS'}")
        elif key == ord("c"):
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
            save_img = clean_capture if clean_capture is not None else frame
            cv2.imwrite(fname, save_img)
            print(f"Foto limpia guardada: {fname}")
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
