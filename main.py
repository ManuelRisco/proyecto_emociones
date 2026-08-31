# ============================================================================
# MENU PRINCIPAL DEL PROYECTO
# Reconocimiento de Emociones Faciales
# Autores: Manuel Fabrizzio Risco Gil y Alejandro Choquihuayta, Mario Julian
# Curso: Sistemas Inteligentes y Machine Learning - Ciclo 9
# ============================================================================

import os
import sys
import subprocess

# Forzar UTF-8 en stdout/stderr en Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

# Auto-reinicio con .venv si se ejecuta desde Python global sin dependencias
try:
    import torch
except ImportError:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    venv_py = os.path.join(base_dir, ".venv", "Scripts", "python.exe")
    if os.path.exists(venv_py) and sys.executable.lower() != venv_py.lower():
        result = subprocess.run([venv_py] + sys.argv)
        sys.exit(result.returncode)
    else:
        print("Error: No se encontro el paquete 'torch' ni el entorno .venv.")
        sys.exit(1)

# Añadir raíz al sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_banner():
    print("""
=============================================================================
  RECONOCIMIENTO DE EMOCIONES FACIALES CON REDES NEURONALES
  Autores: Manuel Fabrizzio Risco Gil y Alejandro Choquihuayta
  Curso: Sistemas Inteligentes y Machine Learning (Ciclo 9)
=============================================================================
""")


def print_menu():
    print("Opciones disponibles:")
    print("  1. Entrenar modelo (ResNet-SE o CNN)")
    print("  2. Evaluar modelo en conjunto de Test")
    print("  3. Predecir emocion en una imagen")
    print("  4. Iniciar reconocimiento en tiempo real (Webcam)")
    print("  5. Generar documentos Word (Informe y Guia)")
    print("  6. Exportar programa portable (.ZIP para cualquier PC)")
    print("  7. Ver resumen de arquitectura del modelo")
    print("  8. Salir")
    print("-" * 65)


def option_train():
    from src.train import train
    print("\nSelecciona la arquitectura:")
    print("  1. EmotionResNetSE (ResNet + SE Attention) [Recomendado]")
    print("  2. EmotionCNN (CNN Clasica)")
    sub_choice = input("Opcion (1-2) [default: 1]: ").strip()
    arch = "resnet_se" if sub_choice in ("", "1") else "cnn_clasica"
    train(arch=arch)


def option_evaluate():
    from src.evaluate import evaluate
    evaluate()


def option_predict():
    from src.predict import predict_interactive
    predict_interactive()


def option_realtime():
    from src.realtime import run_realtime
    run_realtime()


def option_generate_word():
    from src.generate_document import generate_word_document
    from src.generate_complete_guide import generate_comprehensive_guide
    print("\nGenerando Informe Academico...")
    generate_word_document()
    print("Generando Guia de Estudio...")
    generate_comprehensive_guide()


def option_export():
    from src.export import create_portable_bundle
    create_portable_bundle()


def option_show_architecture():
    from src.model import EmotionResNetSE, EmotionCNN, get_model_summary
    from src.config import DEVICE
    print("\n--- EmotionResNetSE ---")
    model_se = EmotionResNetSE().to(DEVICE)
    get_model_summary(model_se)
    print("\n--- EmotionCNN ---")
    model_cnn = EmotionCNN().to(DEVICE)
    get_model_summary(model_cnn)


def main():
    print_banner()

    while True:
        print_menu()

        try:
            choice = input("Selecciona una opcion (1-8): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nPrograma finalizado.")
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
            option_export()
        elif choice == "7":
            option_show_architecture()
        elif choice == "8":
            print("Programa finalizado.")
            break
        else:
            print("Opcion invalida. Intenta nuevamente.")

        try:
            input("\nPresiona ENTER para continuar...")
            print()
        except (KeyboardInterrupt, EOFError):
            print("\nPrograma finalizado.")
            break


if __name__ == "__main__":
    main()
