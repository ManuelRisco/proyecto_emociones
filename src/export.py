# ============================================================================
# EXPORTACIÓN DEL MODELO Y PAQUETE PORTÁTIL (.ZIP)
# Reconocimiento de Emociones Faciales
# ============================================================================
# Genera un archivo ZIP listo para distribuir que incluye:
#   - modelo_emociones_cv2.onnx  (inferencia con OpenCV DNN, sin PyTorch)
#   - face_detection_yunet_2023mar.onnx
#   - app_portable.py            (demo webcam standalone con solo opencv+numpy)
#   - EJECUTAR_WEBCAM.bat        (lanzador Windows)
#   - LEEME.txt                  (instrucciones)
# ============================================================================

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import shutil
import tempfile
import urllib.request
import zipfile

import torch

from src.utils import ensure_root_in_path, setup_utf8
setup_utf8()
ensure_root_in_path()

from src.config import BEST_MODEL_PATH, MODELS_DIR, OUTPUT_DIR, YUNET_MODEL_PATH
from src.predict import load_model


# ──────────────────────────────────────────────────────────────────────────────
# EXPORTACIÓN ONNX
# ──────────────────────────────────────────────────────────────────────────────

def export_onnx(model_path: str = BEST_MODEL_PATH, output_dir: str = MODELS_DIR) -> str:
    """
    Exporta el modelo PyTorch entrenado a formato ONNX.

    ONNX permite ejecutar el modelo con OpenCV DNN sin necesidad de
    PyTorch instalado en el equipo destino — ideal para distribución.

    Args:
        model_path: Checkpoint .pth de PyTorch.
        output_dir: Carpeta donde se guardará el .onnx.

    Returns:
        Ruta al archivo ONNX generado.
    """
    model = load_model(model_path)
    model.eval().cpu()

    os.makedirs(output_dir, exist_ok=True)
    onnx_path   = os.path.join(output_dir, "modelo_emociones_cv2.onnx")
    dummy_input = torch.randn(1, 1, 48, 48)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        dynamo=False,
        input_names=["input"],
        output_names=["output"],
    )
    print(f"   [OK] Modelo ONNX exportado: {onnx_path}")
    return onnx_path


# ──────────────────────────────────────────────────────────────────────────────
# PAQUETE PORTABLE ZIP
# ──────────────────────────────────────────────────────────────────────────────

_BAT_CONTENT = '''\
@echo off
setlocal
cd /d "%~dp0"
title Reconocimiento de Emociones Faciales

echo ============================================================
echo   RECONOCIMIENTO DE EMOCIONES FACIALES (PORTABLE)
echo   Autores: Manuel Fabrizzio Risco Gil y Alejandro Choquihuayta
echo ============================================================
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo [!] Python no encontrado. Descargalo en: https://www.python.org/
    echo     Marca "Add python.exe to PATH" al instalar.
    pause & exit /b 1
)

python -c "import cv2, numpy" >nul 2>&1
if errorlevel 1 (
    echo [*] Instalando dependencias OpenCV y NumPy...
    python -m pip install -r requirements_portable.txt
    if errorlevel 1 ( echo [!] Error instalando. Verifica tu internet. & pause & exit /b 1 )
)

echo [*] Iniciando camara web...
python app_portable.py
pause
'''

_README_CONTENT = '''\
RECONOCIMIENTO DE EMOCIONES FACIALES (PORTABLE v2.0)
Autores: Manuel Fabrizzio Risco Gil y Alejandro Choquihuayta

INSTRUCCIONES:
  1. Doble clic en EJECUTAR_WEBCAM.bat para abrir la camara.
  2. La ventana es ajustable (puedes arrastrar los bordes o maximizarla).
  3. Controles en pantalla:
       q / ESC : Salir
       t       : Activar / Desactivar Suavizado Temporal EMA
       l       : Mostrar / Ocultar Puntos Faciales (Landmarks)
       c       : Cambiar camara
       s       : Guardar captura de pantalla
       p       : Pausar / Reanudar
'''


def create_portable_bundle() -> str | None:
    """
    Genera el archivo .ZIP portable listo para distribuir en cualquier PC.

    El bundle incluye únicamente lo necesario para ejecutar la demo webcam:
    OpenCV + NumPy son las únicas dependencias (sin PyTorch).

    Returns:
        Ruta al archivo ZIP generado, o None si falló.
    """
    print("\n--- Generando paquete portable (.ZIP) ---")

    # 1. Exportar modelo a ONNX
    print("1. Exportando modelo PyTorch → ONNX...")
    onnx_path = export_onnx()
    if not onnx_path or not os.path.exists(onnx_path):
        print("   [!] Error: No se pudo generar el ONNX.")
        return None

    # 2. Preparar directorio temporal
    temp_dir   = tempfile.mkdtemp()
    bundle_dir = os.path.join(temp_dir, "bundle")
    models_dir = os.path.join(bundle_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Copiar modelos
    shutil.copy2(onnx_path, os.path.join(models_dir, "modelo_emociones_cv2.onnx"))
    if os.path.exists(YUNET_MODEL_PATH):
        shutil.copy2(YUNET_MODEL_PATH, os.path.join(models_dir, "face_detection_yunet_2023mar.onnx"))
    else:
        url = (
            "https://github.com/opencv/opencv_zoo/raw/main/models/"
            "face_detection_yunet/face_detection_yunet_2023mar.onnx"
        )
        urllib.request.urlretrieve(url, os.path.join(models_dir, "face_detection_yunet_2023mar.onnx"))

    # Copiar script portable
    src_app = os.path.join(BASE_DIR, "src", "app_portable.py")
    shutil.copy2(src_app, os.path.join(bundle_dir, "app_portable.py"))

    # Archivos de texto
    with open(os.path.join(bundle_dir, "requirements_portable.txt"), "w", encoding="utf-8") as f:
        f.write("opencv-python>=4.8.0\nnumpy>=1.20.0\n")
    with open(os.path.join(bundle_dir, "EJECUTAR_WEBCAM.bat"), "w", encoding="utf-8") as f:
        f.write(_BAT_CONTENT)
    with open(os.path.join(bundle_dir, "LEEME.txt"), "w", encoding="utf-8") as f:
        f.write(_README_CONTENT)

    # 3. Comprimir a ZIP
    print("2. Creando archivo .ZIP...")
    zip_path = os.path.join(OUTPUT_DIR, "Emotion_AI_Portable_v2.0.zip")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(bundle_dir):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, bundle_dir))

    # Copia adicional en el directorio padre para fácil acceso
    semana1_zip = os.path.join(os.path.dirname(BASE_DIR), "Emotion_AI_Portable_v2.0.zip")
    try:
        shutil.copy2(zip_path, semana1_zip)
    except Exception:
        pass

    shutil.rmtree(temp_dir, ignore_errors=True)

    zip_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"[OK] ZIP generado: {zip_path} ({zip_mb:.1f} MB)")
    return zip_path


if __name__ == "__main__":
    create_portable_bundle()
