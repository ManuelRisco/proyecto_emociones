# ============================================================================
# SCRIPT PRINCIPAL - MENU INTERACTIVO
# Reconocimiento de Emociones Faciales con CNN
# Autor: Manuel Fabrizzio Risco Gil
# Curso: Sistemas Inteligentes y Machine Learning - Ciclo 9
# ============================================================================
# Este es el punto de entrada principal del proyecto.
# Proporciona un menu interactivo con todas las funcionalidades:
# 1. Entrenar modelo CNN
# 2. Evaluar modelo entrenado
# 3. Predecir emocion en imagen
# 4. Demo en tiempo real con webcam
# 5. Generar documento Word
# 6. Salir
# ============================================================================

import os
import sys
import io
import subprocess

# Forzar UTF-8 en stdout/stderr para evitar errores de encoding en Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Auto-reinicio transparente con .venv si se ejecuta con el Python global sin librerías
try:
    import torch
except ImportError:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    venv_py = os.path.join(base_dir, ".venv", "Scripts", "python.exe")
    if os.path.exists(venv_py) and sys.executable.lower() != venv_py.lower():
        result = subprocess.run([venv_py] + sys.argv)
        sys.exit(result.returncode)
    else:
        print("[!] Error: No se encontro el paquete 'torch' ni el entorno .venv.")
        print(f"    Por favor ejecuta el script usando: {venv_py} main.py")
        sys.exit(1)

# Asegurar que el directorio raiz del proyecto este en el path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_banner():
    """Muestra el banner del proyecto."""
    banner = """
+==============================================================+
|                                                              |
|     RECONOCIMIENTO DE EMOCIONES FACIALES CON CNN             |
|                                                              |
|     Red Neuronal Convolucional con Aceleracion GPU           |
|                                                              |
|     Autor: Manuel Fabrizzio Risco Gil                        |
|     Curso: Sistemas Inteligentes y Machine Learning          |
|     Ciclo: 9no                                               |
|                                                              |
+==============================================================+

    [INPUT] ENTRADA       [CNN] PROCESO          [OUTPUT] SALIDA
    ---------------      ----------------       ----------------
    Imagen del     ->    La IA aprende y   ->   Emocion
    rostro               procesa patrones       clasificada
    (48x48 px)           automaticamente        + probabilidad
    """
    print(banner)


def print_menu():
    """Muestra el menu principal."""
    print("\n" + "=" * 60)
    print("  MENU PRINCIPAL")
    print("=" * 60)
    print("  1. [TRAIN]    Entrenar modelo CNN")
    print("  2. [EVAL]     Evaluar modelo entrenado")
    print("  3. [PREDICT]  Predecir emocion en imagen")
    print("  4. [WEBCAM]   Demo en tiempo real (webcam)")
    print("  5. [WORD]     Generar documento Word")
    print("  6. [ARCH]     Ver arquitectura del modelo")
    print("  7. [EXIT]     Salir")
    print("=" * 60)


def option_train():
    """Ejecuta el entrenamiento del modelo."""
    from src.train import train
    train()


def option_evaluate():
    """Ejecuta la evaluacion del modelo."""
    from src.evaluate import evaluate
    evaluate()


def option_predict():
    """Ejecuta la prediccion en imagenes."""
    from src.predict import predict_interactive
    predict_interactive()


def option_realtime():
    """Ejecuta el demo en tiempo real."""
    from src.realtime import run_realtime
    run_realtime()


def option_generate_word():
    """Genera los documentos Word (Informe académico y Guía explicativa)."""
    from src.generate_document import generate_word_document
    from src.generate_complete_guide import generate_comprehensive_guide
    print("\n  [1/2] Generando Informe Académico Formal...")
    generate_word_document()
    print("\n  [2/2] Generando Guía Explicativa Completa...")
    generate_comprehensive_guide()


def option_show_architecture():
    """Muestra la arquitectura del modelo."""
    from src.model import EmotionCNN, get_model_summary
    from src.config import DEVICE
    model = EmotionCNN().to(DEVICE)
    get_model_summary(model)


def main():
    """Funcion principal del programa."""
    print_banner()

    while True:
        print_menu()

        try:
            choice = input("\n  Selecciona una opcion (1-7): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  Hasta luego!")
            break

        if choice == "1":
            option_train()
        elif choice == "2":
            option_evaluate()
        elif choice == "3":
            option_predict()
        elif choice == "4":
            option_realtime()
        elif choice == "5":
            option_generate_word()
        elif choice == "6":
            option_show_architecture()
        elif choice == "7":
            print("\n  Hasta luego!")
            break
        else:
            print("\n  [!] Opcion no valida. Intenta de nuevo.")

        input("\n  Presiona ENTER para continuar...")


if __name__ == "__main__":
    main()
