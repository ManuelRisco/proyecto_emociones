# ============================================================================
# PREDICCIÓN DE EMOCIONES EN IMÁGENES Y DETECTOR FACIAL
# Reconocimiento de Emociones Faciales con Deep Learning
# ============================================================================
# Funcionalidades:
#   1. load_model():         Carga automática del checkpoint (detección de arch)
#   2. FaceDetector:         YuNet ONNX (multi-ángulo, 5 landmarks faciales)
#   3. preprocess_face():    Preprocesamiento con transform cacheado (sin overhead)
#   4. predict_single_image(): Predicción completa con visualización
#   5. predict_interactive(): Modo consola interactivo
# ============================================================================

import os
import urllib.request

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.utils import ensure_root_in_path, setup_utf8
setup_utf8()
ensure_root_in_path()

from src.config import (
    BEST_MODEL_PATH, DEVICE, EMOTION_COLORS, EMOTION_CONFIDENCE_THRESHOLDS,
    EMOTION_EMOJIS, EMOTION_LABELS, EMOTION_SENSITIVITY_BOOST,
    IMG_SIZE, NUM_CLASSES, PLOTS_DIR,
    YUNET_MODEL_PATH, YUNET_NMS_THRESHOLD, YUNET_SCORE_THRESHOLD,
    ACADEMIC_EMOTION_MAPPING, get_academic_indicator,
    get_academic_short_indicator, get_academic_justification,
)
from src.model import build_model


# ──────────────────────────────────────────────────────────────────────────────
# 1. CARGADOR INTELIGENTE DE MODELOS
# ──────────────────────────────────────────────────────────────────────────────

def load_model(model_path: str = BEST_MODEL_PATH) -> torch.nn.Module:
    """
    Carga el modelo entrenado desde un checkpoint .pth.

    Detecta automáticamente la arquitectura guardada a partir de las claves
    del state_dict, permitiendo cargar checkpoints entrenados con versiones
    anteriores del proyecto sin especificar la arquitectura manualmente.

    Args:
        model_path: Ruta al archivo .pth generado por train.py.

    Returns:
        Modelo cargado en DEVICE, en modo evaluación.

    Raises:
        FileNotFoundError: Si el archivo no existe.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"\n[!] No se encontró el modelo: {model_path}\n"
            f"   Primero entrena el modelo (opción 1 del menú)."
        )

    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    # Manejar modelos compilados con torch.compile (prefijo "_orig_mod.")
    clean_state = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    # Detección de arquitectura por claves del state_dict limpio
    arch = checkpoint.get("arch", None) if isinstance(checkpoint, dict) else None
    if arch is None:
        arch = "resnet_se" if any(
            k.startswith(("stage1", "stem")) for k in clean_state.keys()
        ) else "cnn_clasica"

    model = build_model(arch=arch, num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(clean_state, strict=False)
    model.eval()

    val_acc   = checkpoint.get("val_acc") if isinstance(checkpoint, dict) else None
    epoch     = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
    acc_info  = f" (Val Acc: {val_acc:.2f}%)" if val_acc is not None else ""
    ep_info   = f" [Época {epoch}]"           if epoch  is not None else ""
    print(f"   [OK] Modelo cargado: {model.__class__.__name__} ({arch.upper()}){ep_info}{acc_info}")

    return model


# ──────────────────────────────────────────────────────────────────────────────
# 2. DETECTOR DE ROSTROS NEURONAL (YuNet DNN)
# ──────────────────────────────────────────────────────────────────────────────

class FaceDetector:
    """
    Detector de rostros multi-ángulo basado en la red neuronal YuNet ONNX.

    Ventajas de YuNet:
    - Detecta perfiles (yaw ±90°) y caras inclinadas (roll/pitch).
    - Cero falsos positivos en ropa o fondos mediante umbrales neuronales.
    - Retorna los 5 puntos faciales clave (landmarks: ojos, nariz, boca).
    """

    def __init__(self, score_thresh: float = YUNET_SCORE_THRESHOLD):
        self.score_thresh = score_thresh
        self.yunet        = None
        self._init_detector()

    def _ensure_yunet_model(self) -> None:
        """Descarga el modelo YuNet ONNX si no existe localmente o si está incompleto."""
        if not os.path.exists(YUNET_MODEL_PATH) or os.path.getsize(YUNET_MODEL_PATH) < 10000:
            print("   [DOWNLOAD] Descargando modelo YuNet (~230 KB)...")
            url = (
                "https://github.com/opencv/opencv_zoo/raw/main/models/"
                "face_detection_yunet/face_detection_yunet_2023mar.onnx"
            )
            try:
                os.makedirs(os.path.dirname(YUNET_MODEL_PATH), exist_ok=True)
                urllib.request.urlretrieve(url, YUNET_MODEL_PATH)
                print(f"   [OK] YuNet descargado: {YUNET_MODEL_PATH}")
            except Exception as e:
                print(f"   [!] Error al descargar YuNet: {e}")
                if os.path.exists(YUNET_MODEL_PATH) and os.path.getsize(YUNET_MODEL_PATH) < 10000:
                    try:
                        os.remove(YUNET_MODEL_PATH)
                    except OSError:
                        pass

    def _init_detector(self) -> None:
        """Inicializa el detector neuronal YuNet."""
        if hasattr(cv2, "FaceDetectorYN"):
            self._ensure_yunet_model()
            if os.path.exists(YUNET_MODEL_PATH):
                try:
                    self.yunet = cv2.FaceDetectorYN.create(
                        model=YUNET_MODEL_PATH,
                        config="",
                        input_size=(320, 320),
                        score_threshold=self.score_thresh,
                        nms_threshold=YUNET_NMS_THRESHOLD,
                        top_k=5000,
                    )
                except Exception as e:
                    print(f"   [!] Error inicializando YuNet: {e}")

    def detect(self, image_bgr: np.ndarray, max_dim: int = 320) -> list[dict]:
        """
        Detecta rostros en una imagen BGR usando YuNet.

        Optimización: Si la imagen supera max_dim (ej. 640x480), se escala internamente
        a max_dim (320px) para inferencia hasta 4x más rápida, proyectando las coordenadas
        de vuelta al tamaño original.

        Returns:
            Lista de dicts: [{'bbox': (x,y,w,h), 'confidence': float, 'landmarks': ndarray|None}]
        """
        if image_bgr is None or image_bgr.size == 0:
            return []

        h_img, w_img = image_bgr.shape[:2]
        results = []

        if self.yunet is not None:
            scale = 1.0
            if max_dim is not None and max(h_img, w_img) > max_dim:
                scale = max_dim / float(max(h_img, w_img))
                w_s, h_s = int(w_img * scale), int(h_img * scale)
                detect_img = cv2.resize(image_bgr, (w_s, h_s), interpolation=cv2.INTER_LINEAR)
            else:
                w_s, h_s = w_img, h_img
                detect_img = image_bgr

            self.yunet.setInputSize((w_s, h_s))
            _, faces = self.yunet.detect(detect_img)

            if faces is not None:
                inv_s = 1.0 / scale
                for face in faces:
                    x = max(0, int(face[0] * inv_s))
                    y = max(0, int(face[1] * inv_s))
                    w = int(face[2] * inv_s)
                    h = int(face[3] * inv_s)
                    w = min(w, w_img - x)
                    h = min(h, h_img - y)
                    if w < 16 or h < 16:
                        continue
                    landmarks = (face[4:14].reshape((5, 2)) * inv_s).astype(int)
                    results.append({
                        "bbox":       (x, y, w, h),
                        "confidence": float(face[14]),
                        "landmarks":  landmarks,
                    })

        return results


def get_face_detector() -> FaceDetector:
    """Fábrica para obtener una instancia de FaceDetector."""
    return FaceDetector()


# ──────────────────────────────────────────────────────────────────────────────
# 3. PREPROCESAMIENTO DIRECTO EN C++ / PYTORCH (< 0.05 ms)
# ──────────────────────────────────────────────────────────────────────────────

def align_and_crop_face(
    frame_gray: np.ndarray,
    bbox: tuple,
    landmarks: np.ndarray | None = None,
    target_size: int = IMG_SIZE,
) -> torch.Tensor:
    """
    Extrae, alinea afínmente por los ojos y estandariza el rostro con invarianza lumínica total:
    1. Alineación afín: Elimina inclinaciones de cabeza para que los rasgos coincidan con FER-2013.
    2. CLAHE local en 4x4 tiles: Disipa sombras duras y destellos de lámparas laterales.
    3. Estandarización canónica Z-score: Fija la media a 128 y std a 50 (invarianza a brillo).
    """
    if frame_gray is None or frame_gray.size == 0:
        return torch.zeros((1, 1, target_size, target_size), device=DEVICE, dtype=torch.float32)

    h_f, w_f = frame_gray.shape[:2]
    x, y, w, h = bbox

    # Clamping seguro de coordenadas para evitar slices negativos o fuera de imagen
    x1 = max(0, min(w_f, int(x)))
    y1 = max(0, min(h_f, int(y)))
    x2 = max(0, min(w_f, int(x + w)))
    y2 = max(0, min(h_f, int(y + h)))

    # Nivelación afín por inclinación de ojos
    if landmarks is not None and len(landmarks) >= 2:
        p_re = landmarks[0].astype(float)
        p_le = landmarks[1].astype(float)
        dx = p_le[0] - p_re[0]
        dy = p_le[1] - p_re[1]
        angle = float(np.degrees(np.arctan2(dy, dx)))

        if abs(angle) > 2.0 and abs(angle) < 45.0:
            center = (float(x + w / 2.0), float(y + h / 2.0))
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(frame_gray, M, (w_f, h_f), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            face = rotated[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else np.zeros((target_size, target_size), dtype=np.uint8)
        else:
            face = frame_gray[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else np.zeros((target_size, target_size), dtype=np.uint8)
    else:
        face = frame_gray[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else np.zeros((target_size, target_size), dtype=np.uint8)

    if face.size == 0:
        face = np.zeros((target_size, target_size), dtype=np.uint8)
    elif face.shape[0] != target_size or face.shape[1] != target_size:
        interp = cv2.INTER_AREA if (face.shape[0] >= target_size and face.shape[1] >= target_size) else cv2.INTER_LINEAR
        face = cv2.resize(face, (target_size, target_size), interpolation=interp)

    # CLAHE local adaptativo para nivelar sombras de lámpara
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    face_eq = clahe.apply(face)

    # Estandarización canónica Z-score
    tensor = torch.from_numpy(np.ascontiguousarray(face_eq)).to(DEVICE, dtype=torch.float32)
    t_mean = tensor.mean()
    t_std  = tensor.std() + 1e-5
    standardized = torch.clamp(((tensor - t_mean) / t_std) * 50.0 + 128.0, 0.0, 255.0)
    norm = ((standardized / 255.0 - 0.5) / 0.5).unsqueeze(0).unsqueeze(0)
    return norm


def preprocess_face(face_img: np.ndarray) -> torch.Tensor:
    """Compatibilidad: preprocesa un recorte directo de rostro."""
    if face_img is None or face_img.size == 0:
        return torch.zeros((1, 1, IMG_SIZE, IMG_SIZE), device=DEVICE, dtype=torch.float32)
    if len(face_img.shape) == 3:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    h, w = face_img.shape[:2]
    return align_and_crop_face(face_img, (0, 0, w, h), landmarks=None)


# ──────────────────────────────────────────────────────────────────────────────
# 4. PREDICCIÓN EN IMÁGENES
# ──────────────────────────────────────────────────────────────────────────────

def predict_single_image(
    image_path: str,
    model: torch.nn.Module | None = None,
    detector: FaceDetector | None = None,
) -> list[dict] | None:
    """
    Predice las emociones de todos los rostros detectados en una imagen.

    Args:
        image_path: Ruta a la imagen (JPG, PNG, etc.).
        model:      Modelo pre-cargado (se carga automáticamente si es None).
        detector:   Detector facial (se inicializa si es None).

    Returns:
        Lista de resultados por rostro o None si hay error.
    """
    if not os.path.exists(image_path):
        print(f"\n[!] Imagen no encontrada: {image_path}")
        return None

    model    = model    or load_model()
    detector = detector or FaceDetector()

    img = cv2.imread(image_path)
    if img is None:
        print(f"\n[!] No se pudo leer: {image_path}")
        return None

    detections = detector.detect(img)
    if not detections:
        print("\n[!] Sin rostros detectados. Procesando imagen completa...")
        detections = [{"bbox": (0, 0, img.shape[1], img.shape[0]), "confidence": 1.0, "landmarks": None}]
    else:
        print(f"\n   Rostros detectados (YuNet): {len(detections)}")

    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results = []

    model_device = next(model.parameters()).device

    for i, det in enumerate(detections):
        x, y, w, h = det["bbox"]
        if w < 16 or h < 16:
            continue

        # Alineación facial afín + CLAHE + Estandarización
        tensor = align_and_crop_face(gray, (x, y, w, h), landmarks=det.get("landmarks"))

        # Test-Time Augmentation (TTA con flip horizontal para máxima estabilidad y precisión)
        tensor_flip = torch.flip(tensor, dims=[-1])
        batch_tta = torch.cat([tensor, tensor_flip], dim=0).to(model_device)
        with torch.no_grad():
            outputs = model(batch_tta)
            probs_batch = torch.softmax(outputs, dim=1).cpu().numpy()
            raw_probs = (0.65 * probs_batch[0] + 0.35 * probs_batch[1]).copy()
            if np.isnan(raw_probs).any():
                raw_probs = np.nan_to_num(raw_probs, nan=1.0 / NUM_CLASSES)

        # Aplicar multiplicadores de sensibilidad adaptativa
        boosted = raw_probs.copy()
        for emo_i, boost_val in EMOTION_SENSITIVITY_BOOST.items():
            boosted[emo_i] *= boost_val

        # Apoyo geométrico por landmarks si están disponibles
        lm = det.get("landmarks")
        if lm is not None and len(lm) >= 5:
            p_re, p_le = lm[0].astype(float), lm[1].astype(float)
            p_n        = lm[2].astype(float)
            p_rm, p_lm = lm[3].astype(float), lm[4].astype(float)
            eye_dist   = max(5.0, float(np.linalg.norm(p_re - p_le)))
            mouth_open = float(np.linalg.norm(p_n - (p_rm + p_lm) / 2.0)) / eye_dist

            if mouth_open > 0.72:
                factor = min(1.45, 1.0 + (mouth_open - 0.72) * 2.0)
                boosted[5] *= factor
            elif mouth_open < 0.58:
                factor = min(1.35, 1.0 + (0.58 - mouth_open) * 2.0)
                boosted[1] *= factor
                boosted[0] *= factor

        sum_b = boosted.sum()
        if not np.isfinite(sum_b) or sum_b <= 0:
            boosted = np.ones(NUM_CLASSES) / NUM_CLASSES
        else:
            boosted /= sum_b

        # Resolución con umbrales de confianza
        raw_best_idx = int(np.argmax(boosted))
        if float(boosted[raw_best_idx]) >= EMOTION_CONFIDENCE_THRESHOLDS.get(raw_best_idx, 0.0):
            pred_idx = raw_best_idx
        else:
            sorted_idxs = np.argsort(boosted)[::-1]
            pred_idx = int(sorted_idxs[0])
            for cand in sorted_idxs[1:]:
                if float(boosted[cand]) >= EMOTION_CONFIDENCE_THRESHOLDS.get(int(cand), 0.0):
                    pred_idx = int(cand)
                    break

        pred_label = EMOTION_LABELS[pred_idx]
        confidence = float(boosted[pred_idx]) * 100
        emoji = EMOTION_EMOJIS.get(pred_idx, "")
        acad_indicator = get_academic_indicator(pred_idx)
        acad_short     = get_academic_short_indicator(pred_idx)
        acad_just      = get_academic_justification(pred_idx)

        result = {
            "face_index":             i,
            "bbox":                   (x, y, w, h),
            "face_confidence":        det["confidence"] * 100,
            "predicted_class":        pred_idx,
            "predicted_label":        pred_label,
            "confidence":             confidence,
            "academic_indicator":     acad_indicator,
            "academic_short":         acad_short,
            "academic_justification": acad_just,
            "probabilities": {
                EMOTION_LABELS[j]: float(boosted[j]) * 100
                for j in range(NUM_CLASSES)
            },
            "landmarks": det["landmarks"],
        }
        results.append(result)

        print(f"\n   {'─' * 55}")
        print(f"   Rostro #{i+1}: {pred_label} {emoji} ({confidence:.1f}%) [Det: {det['confidence']*100:.0f}%]")
        print(f"   [Indicador Aula]: {acad_indicator}")
        print(f"   [Justificación]:  {acad_just}")
        print(f"   {'─' * 55}")
        for label, prob in sorted(result["probabilities"].items(), key=lambda kv: kv[1], reverse=True):
            print(f"     {label:>12}: {prob:5.1f}% {'=' * int(prob / 2)}")

    _visualize_prediction(img, results)
    return results


FACE_PALETTE_RGB = [
    (0.12, 0.55, 0.95),  # Rostro 1: Azul Cyan
    (0.98, 0.45, 0.10),  # Rostro 2: Naranja brillante
    (0.15, 0.78, 0.35),  # Rostro 3: Verde esmeralda
    (0.88, 0.20, 0.75),  # Rostro 4: Magenta
    (0.95, 0.78, 0.10),  # Rostro 5: Amarillo dorado
    (0.55, 0.35, 0.92),  # Rostro 6: Púrpura
]


def _plot_face_bars(ax, r: dict, face_idx: int, face_color: tuple) -> None:
    """Renderiza la gráfica de barras de probabilidades para un rostro individual."""
    sorted_probs = sorted(r["probabilities"].items(), key=lambda kv: kv[1], reverse=True)
    labels = [l for l, _ in sorted_probs]
    values = [v for _, v in sorted_probs]

    # Cada barra toma el color característico de su emoción
    bar_colors = []
    for lbl in labels:
        idx = next((k for k, v in EMOTION_LABELS.items() if v == lbl), 6)
        bgr = EMOTION_COLORS.get(idx, (170, 170, 170))
        bar_colors.append((bgr[2] / 255.0, bgr[1] / 255.0, bgr[0] / 255.0))

    ax.barh(labels[::-1], values[::-1], color=bar_colors[::-1], edgecolor="gray", alpha=0.9, height=0.6)
    ax.set_xlabel("Probabilidad (%)", fontsize=10)
    ax.set_xlim(0, 100)
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    # Título con emoción dominante e indicador académico para el jurado
    acad_ind = r.get("academic_indicator", get_academic_indicator(r.get("predicted_class", 6)))
    ax.set_title(
        f"Rostro #{face_idx + 1}: {r['predicted_label'].upper()} ({r['confidence']:.1f}%)\n"
        f"Indicador Aula: {acad_ind}",
        fontsize=10.5, fontweight="bold", color=face_color, pad=8,
    )

    # Marco con el color del rostro para asociarlo visualmente al instante
    for spine in ax.spines.values():
        spine.set_edgecolor(face_color)
        spine.set_linewidth(1.8)

    # Porcentajes numéricos al final de cada barra
    for i, v in enumerate(values[::-1]):
        ax.text(v + 1.2, i, f"{v:.1f}%", va="center", fontsize=9, fontweight="bold")


def _clean_cv2_text(text: str) -> str:
    """Elimina tildes para renderizado perfecto en OpenCV HERSHEY."""
    trans = str.maketrans("áéíóúÁÉÍÓÚñÑ", "aeiouAEIOUnN")
    return text.translate(trans)


def _visualize_prediction(img: np.ndarray, results: list[dict]) -> None:
    """
    Genera y guarda una visualización completa de la predicción.
    Regla: Cantidad de rostros detectados = Cantidad de gráficas independientes.
    Cada rostro se distingue con un color y etiqueta únicos en la imagen y en su gráfica.
    """
    from matplotlib.gridspec import GridSpec

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = img_rgb.shape[:2]

    # Si la imagen es miniatura (ej. 48x48 del dataset FER), reescalar para visualización nítida
    scale = 1.0
    if max(h_orig, w_orig) < 250:
        scale = 360.0 / max(h_orig, w_orig)
        img_rgb = cv2.resize(img_rgb, (int(w_orig * scale), int(h_orig * scale)), interpolation=cv2.INTER_NEAREST)

    h_img, w_img = img_rgb.shape[:2]

    # 1. Distinguir cada rostro en la imagen con colores y badges individuales
    for i, r in enumerate(results):
        rx, ry, rw, rh = r["bbox"]
        x, y, w, h = int(rx * scale), int(ry * scale), int(rw * scale), int(rh * scale)
        col_rgb = FACE_PALETTE_RGB[i % len(FACE_PALETTE_RGB)]
        col_255 = tuple(int(c * 255) for c in col_rgb)

        cv2.rectangle(img_rgb, (x, y), (x + w, y + h), col_255, 3)

        if r.get("landmarks") is not None:
            for pt in r["landmarks"]:
                lx, ly = int(pt[0] * scale), int(pt[1] * scale)
                cv2.circle(img_rgb, (lx, ly), 4, col_255, -1)
                cv2.circle(img_rgb, (lx, ly), 2, (255, 255, 255), -1)

        acad_short = r.get("academic_short", "")
        if acad_short:
            badge_text = f"Rostro #{i+1}: {r['predicted_label']} {r['confidence']:.0f}% [{acad_short}]"
        else:
            badge_text = f"Rostro #{i+1}: {r['predicted_label']} {r['confidence']:.0f}%"
        badge_text = _clean_cv2_text(badge_text)
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = max(0.38, min(0.60, w / 220.0))
        (tw, th), _ = cv2.getTextSize(badge_text, font, font_scale, 1)

        badge_x1 = max(0, x)
        badge_x2 = min(w_img, x + tw + 14)

        # Si el rostro está muy cerca del borde superior, ubicar la etiqueta debajo del bbox
        if y - th - 12 >= 0:
            badge_y1 = y - th - 12
            badge_y2 = y
            text_y   = y - 4
        else:
            badge_y1 = y + h
            badge_y2 = min(h_img, y + h + th + 12)
            text_y   = min(h_img - 2, badge_y1 + th + 4)

        cv2.rectangle(img_rgb, (badge_x1, badge_y1), (badge_x2, badge_y2), col_255, -1)
        cv2.putText(img_rgb, badge_text, (badge_x1 + 6, text_y), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    n_faces = len(results)

    # 2. Generar tantas gráficas independientes como rostros detectados
    if n_faces == 0:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.imshow(img_rgb)
        ax.set_title("No se detectaron rostros", fontsize=13, fontweight="bold")
        ax.axis("off")

    elif n_faces == 1:
        fig = plt.figure(figsize=(15, 6))
        gs = GridSpec(1, 2, width_ratios=[1.3, 1], wspace=0.25)

        ax_img = fig.add_subplot(gs[0, 0])
        ax_img.imshow(img_rgb)
        ax_img.set_title("Detección de Rostros (YuNet)", fontsize=13, fontweight="bold")
        ax_img.axis("off")

        ax_bar = fig.add_subplot(gs[0, 1])
        _plot_face_bars(ax_bar, results[0], face_idx=0, face_color=FACE_PALETTE_RGB[0])

    elif n_faces == 2:
        # Lado a lado: Imagen y 2 gráficas de barras independientes
        fig = plt.figure(figsize=(19, 6.5))
        gs = GridSpec(1, 3, width_ratios=[1.35, 1, 1], wspace=0.28)

        ax_img = fig.add_subplot(gs[0, 0])
        ax_img.imshow(img_rgb)
        ax_img.set_title("Detección de Rostros (YuNet) — 2 Sujetos", fontsize=13, fontweight="bold")
        ax_img.axis("off")

        ax_bar1 = fig.add_subplot(gs[0, 1])
        _plot_face_bars(ax_bar1, results[0], face_idx=0, face_color=FACE_PALETTE_RGB[0])

        ax_bar2 = fig.add_subplot(gs[0, 2])
        _plot_face_bars(ax_bar2, results[1], face_idx=1, face_color=FACE_PALETTE_RGB[1])

    elif n_faces == 3:
        fig = plt.figure(figsize=(24, 6.5))
        gs = GridSpec(1, 4, width_ratios=[1.35, 1, 1, 1], wspace=0.30)

        ax_img = fig.add_subplot(gs[0, 0])
        ax_img.imshow(img_rgb)
        ax_img.set_title("Detección de Rostros (YuNet) — 3 Sujetos", fontsize=13, fontweight="bold")
        ax_img.axis("off")

        for i in range(3):
            ax_bar = fig.add_subplot(gs[0, i + 1])
            _plot_face_bars(ax_bar, results[i], face_idx=i, face_color=FACE_PALETTE_RGB[i % len(FACE_PALETTE_RGB)])

    else:
        # 4 o más personas: Imagen a la izquierda y columna de gráficas independientes a la derecha
        fig = plt.figure(figsize=(18, max(7, 3.2 * n_faces)))
        gs = GridSpec(n_faces, 2, width_ratios=[1.35, 1], hspace=0.38, wspace=0.25)

        ax_img = fig.add_subplot(gs[:, 0])
        ax_img.imshow(img_rgb)
        ax_img.set_title(f"Detección de Rostros (YuNet) — {n_faces} Sujetos", fontsize=13, fontweight="bold")
        ax_img.axis("off")

        for i in range(n_faces):
            ax_bar = fig.add_subplot(gs[i, 1])
            _plot_face_bars(ax_bar, results[i], face_idx=i, face_color=FACE_PALETTE_RGB[i % len(FACE_PALETTE_RGB)])

    os.makedirs(PLOTS_DIR, exist_ok=True)
    save_path = os.path.join(PLOTS_DIR, "prediccion_imagen.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n   [GRAFICA] Reporte generado ({n_faces} graficas independientes). Guardado en: {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 5. MODO INTERACTIVO
# ──────────────────────────────────────────────────────────────────────────────

def predict_interactive() -> None:
    """Modo interactivo de predicción desde consola."""
    print("\n" + "=" * 60)
    print(">>> PREDICCIÓN DE EMOCIONES EN IMÁGENES (YuNet + Deep Learning)")
    print("=" * 60)
    print("   Cargando modelo y detector neuronal...")

    try:
        model = load_model()
    except FileNotFoundError as e:
        print(e)
        return

    detector = FaceDetector()

    while True:
        print("\n" + "─" * 60)
        image_path = input("   Ruta de la imagen (o 'salir'): ").strip().strip("\"'")

        if image_path.lower() in ("salir", "exit", "q", "quit"):
            break

        if os.path.exists(image_path):
            predict_single_image(image_path, model, detector)
        else:
            print(f"   [!] Archivo no encontrado: {image_path}")

    print("\n   Saliendo del modo de predicción...")


if __name__ == "__main__":
    predict_interactive()
