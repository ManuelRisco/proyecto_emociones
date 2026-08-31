# ============================================================================
# EXPORTACIÓN DEL MODELO Y PAQUETE PORTÁTIL (.ZIP)
# ============================================================================

import os
import sys
import shutil
import zipfile
import tempfile
import torch
import cv2
import numpy as np

# Forzar UTF-8 en Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

# Asegurar que el directorio raíz esté en sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import BASE_DIR, OUTPUT_DIR, MODELS_DIR, BEST_MODEL_PATH, YUNET_MODEL_PATH
from src.predict import load_model


def export_onnx(model_path=BEST_MODEL_PATH, output_dir=MODELS_DIR):
    """Exporta el modelo a formato ONNX compatible con OpenCV DNN."""
    model = load_model(model_path)
    model.eval().cpu()

    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, "modelo_emociones_cv2.onnx")
    dummy_input = torch.randn(1, 1, 48, 48, requires_grad=False)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        dynamo=False,
        input_names=["input"],
        output_names=["output"]
    )
    return onnx_path


def create_portable_bundle():
    """Genera únicamente el archivo comprimido .ZIP portable con el modelo y ejecutor."""
    print("\n--- Generando paquete portable (.ZIP) ---")

    # 1. Exportar modelo ONNX
    print("1. Exportando modelo a ONNX...")
    onnx_path = export_onnx()
    if not onnx_path or not os.path.exists(onnx_path):
        print("Error: No se pudo generar el modelo ONNX.")
        return None

    # 2. Crear estructura en directorio temporal
    temp_dir = tempfile.mkdtemp()
    bundle_path = os.path.join(temp_dir, "bundle")
    models_dir = os.path.join(bundle_path, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Copiar modelos
    shutil.copy2(onnx_path, os.path.join(models_dir, "modelo_emociones_cv2.onnx"))

    if os.path.exists(YUNET_MODEL_PATH):
        shutil.copy2(YUNET_MODEL_PATH, os.path.join(models_dir, "face_detection_yunet_2023mar.onnx"))
    else:
        url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        import urllib.request
        urllib.request.urlretrieve(url, os.path.join(models_dir, "face_detection_yunet_2023mar.onnx"))

    # Copiar script portable y requerimientos
    src_app = os.path.join(BASE_DIR, "src", "app_portable.py")
    shutil.copy2(src_app, os.path.join(bundle_path, "app_portable.py"))

    with open(os.path.join(bundle_path, "requirements_portable.txt"), "w", encoding="utf-8") as f:
        f.write("opencv-python>=4.8.0\nnumpy>=1.20.0\n")

    # Lanzador .BAT para Windows
    bat_content = '''@echo off
setlocal
cd /d "%~dp0"
title Reconocimiento de Emociones Faciales

echo ============================================================
echo   RECONOCIMIENTO DE EMOCIONES FACIALES
echo   Autores: Manuel Fabrizzio Risco Gil y Alejandro Choquihuayta
echo ============================================================
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo [!] Error: No se encontro Python instalado en este equipo.
    echo     Por favor descarga e instala Python desde: https://www.python.org/
    echo     Recuerda marcar la casilla 'Add python.exe to PATH' al instalar.
    echo.
    pause
    exit /b 1
)

python -c "import cv2, numpy" >nul 2>&1
if errorlevel 1 (
    echo [*] Instalando librerias necesarias OpenCV y NumPy ...
    echo.
    python -m pip install -r requirements_portable.txt
    if errorlevel 1 (
        echo [!] Error instalando librerias. Revisa tu conexion a Internet.
        pause
        exit /b 1
    )
)

echo [*] Iniciando camara web...
echo.
python app_portable.py

if errorlevel 1 (
    echo.
    echo [!] La aplicacion se cerro.
    pause
) else (
    echo.
    echo [OK] Aplicacion finalizada.
    pause
)
'''
    with open(os.path.join(bundle_path, "EJECUTAR_WEBCAM.bat"), "w", encoding="utf-8") as f:
        f.write(bat_content)

    # Instrucciones breves
    readme_content = '''RECONOCIMIENTO DE EMOCIONES FACIALES (PORTABLE)
Autores: Manuel Fabrizzio Risco Gil y Alejandro Choquihuayta

INSTRUCCIONES:
1. Haz doble clic en EJECUTAR_WEBCAM.bat para abrir la camara.
2. Controles:
   - 'q' o ESC: Salir
   - 'c': Cambiar camara
   - 's': Guardar foto
   - 'p': Pausar/Reanudar
'''
    with open(os.path.join(bundle_path, "LEEME.txt"), "w", encoding="utf-8") as f:
        f.write(readme_content)

    # 3. Comprimir en el archivo ZIP final
    print("2. Creando archivo .ZIP...")
    zip_path = os.path.join(OUTPUT_DIR, "Emotion_AI_Portable_v2.0.zip")
    semana1_zip = os.path.join(os.path.dirname(BASE_DIR), "Emotion_AI_Portable_v2.0.zip")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(bundle_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, bundle_path)
                zipf.write(file_path, arcname)

    try:
        shutil.copy2(zip_path, semana1_zip)
    except Exception:
        pass

    # Limpiar carpeta temporal
    shutil.rmtree(temp_dir, ignore_errors=True)

    zip_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"[OK] Archivo ZIP generado: {zip_path} ({zip_mb:.1f} MB)")
    print(f"     Copia guardada en: {semana1_zip}\n")

    return zip_path


if __name__ == "__main__":
    create_portable_bundle()
