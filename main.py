# ============================================================================
# MENÚ PRINCIPAL DEL PROYECTO
# Reconocimiento de Emociones Faciales
# Autores: Manuel Fabrizzio Risco Gil y Alejandro Choquihuayta
# Curso: Sistemas Inteligentes y Machine Learning — Ciclo 9
# ============================================================================

import os
import sys
import subprocess

# Centralizar UTF-8 usando utils si está disponible, con fallback inline
try:
    from src.utils import setup_utf8
    setup_utf8()
except ImportError:
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore
        except Exception:
            pass

# Auto-reinicio con .venv si se ejecuta con Python global sin dependencias
try:
    import torch  # noqa: F401
except ImportError:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    venv_py  = os.path.join(base_dir, ".venv", "Scripts", "python.exe")
    if os.path.exists(venv_py) and sys.executable.lower() != venv_py.lower():
        sys.exit(subprocess.run([venv_py] + sys.argv).returncode)
    print("Error: PyTorch no instalado y no se encontró el entorno .venv.")
    sys.exit(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ──────────────────────────────────────────────────────────────────────────────
# PANTALLA DE BIENVENIDA Y MENÚ
# ──────────────────────────────────────────────────────────────────────────────

def print_banner() -> None:
    print("""
=============================================================================
  RECONOCIMIENTO DE EMOCIONES FACIALES CON REDES NEURONALES
  Autores: Manuel Fabrizzio Risco Gil y Alejandro Choquihuayta
  Curso: Sistemas Inteligentes y Machine Learning (Ciclo 9)
=============================================================================
""")


def print_menu() -> None:
    print("Opciones disponibles:")
    print("  1. Entrenar modelo (EmotionResNetSE con GPU VRAM + AMP)")
    print("  2. Evaluar modelo en conjunto de Test")
    print("  3. Predecir emoción en una imagen")
    print("  4. Iniciar reconocimiento en tiempo real (Webcam - Modo Aula)")
    print("  5. Exportar programa portable (.ZIP para cualquier PC)")
    print("  6. Ver resumen de arquitectura del modelo")
    print("  7. Ver Mapeo Pedagógico de Emociones en Aula (Justificación ante el Jurado)")
    print("  8. Compilar Informe Final de Tesis UPN en Word (.docx)")
    print("  9. Salir")
    print("-" * 75)


# ──────────────────────────────────────────────────────────────────────────────
# OPCIONES DEL MENÚ
# ──────────────────────────────────────────────────────────────────────────────

def option_train() -> None:
    from src.train import train
    train()


def option_evaluate() -> None:
    from src.evaluate import evaluate
    evaluate()


def option_predict() -> None:
    from src.predict import predict_interactive
    predict_interactive()


def option_realtime() -> None:
    from src.realtime import run_realtime
    run_realtime()


def option_export() -> None:
    from src.export import create_portable_bundle
    create_portable_bundle()


def option_show_architecture() -> None:
    from src.model import EmotionResNetSE, get_model_summary
    from src.config import DEVICE
    model = EmotionResNetSE().to(DEVICE)
    get_model_summary(model)


def option_show_academic_mapping() -> None:
    """Muestra el mapeo pedagógico de las 7 emociones y su justificación ante el jurado."""
    print("\n" + "=" * 90)
    print(" 3. MAPEO DE LAS 7 EMOCIONES BÁSICAS CON LA PROBLEMÁTICA ACADÉMICA")
    print("=" * 90)
    print("""Para que el jurado no cuestione por qué usas un clasificador de 7 emociones
(alegría, tristeza, enojo, sorpresa, neutral, temor, disgusto) en un contexto académico,
debes justificar el significado de cada etiqueta en el aula:
""")

    table_data = [
        ("Neutral",          "Atención / Estado de concentración",          "Representa el estado base de escucha o lectura fluida."),
        ("Tristeza",         "Desmotivación / Agotamiento",                 "La acumulación continua de esta etiqueta sugiere fatiga o desánimo frente al material."),
        ("Enojo",            "Frustración / Incomprensión",                 "Surge cuando el estudiante no comprende un ejercicio o enfrenta dificultades técnicas/conceptuales."),
        ("Sorpresa",         "Comprensión repentina (Insight) o Asombro",   "Indica momentos de alto impacto en la explicación o dinamismo en la clase."),
        ("Temor / Disgusto", "Ansiedad / Rechazo a la tarea",               "Asociado al estrés previo a una evaluación, sobrecarga académica o disgusto por la dinámica."),
        ("Alegría",          "Compromiso (Engagement) / Satisfacción",      "Refleja un ambiente de aprendizaje receptivo y comprensión clara del tema."),
    ]

    col_w = [18, 42, 60]
    sep = "+" + "+".join("-" * (w + 2) for w in col_w) + "+"
    header = "| " + " | ".join(f"{h:<{col_w[i]}}" for i, h in enumerate(["Emoción del Dataset", "Indicador Académico Asociado", "Justificación dentro de la Problemática"])) + " |"

    print(sep)
    print(header)
    print(sep)

    for emo, ind, jus in table_data:
        # Wrap de texto para que quepa perfectamente
        print(f"| {emo:<{col_w[0]}} | {ind:<{col_w[1]}} | {jus:<{col_w[2]}} |")
    print(sep)
    print("""
   [*] Sustento Pedagógico:
       El sistema no realiza diagnósticos clínicos ni invasión médica; actúa como un sensor
       óptico no intrusivo de dinámicas afectivas de grupo que orienta la toma de decisiones
       docente (ej. detectar momentos de frustración colectiva o evaluar la atención base).
""")


def option_word() -> None:
    """Compila el informe final de tesis UPN en formato Word (.docx) bajo APA 7."""
    from scripts.build_full_thesis_report import build_complete_document
    build_complete_document()


# ──────────────────────────────────────────────────────────────────────────────
# BUCLE PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────

_MENU_OPTIONS = {
    "1": option_train,
    "2": option_evaluate,
    "3": option_predict,
    "4": option_realtime,
    "5": option_export,
    "6": option_show_architecture,
    "7": option_show_academic_mapping,
    "8": option_word,
}


def main() -> None:
    print_banner()

    while True:
        print_menu()

        try:
            choice = input("Selecciona una opción (1-9): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nPrograma finalizado.")
            break

        if choice == "9":
            print("Programa finalizado.")
            break

        handler = _MENU_OPTIONS.get(choice)
        if handler:
            handler()
        else:
            print("Opción inválida. Intenta nuevamente.")

        try:
            input("\nPresiona ENTER para continuar...")
            print()
        except (KeyboardInterrupt, EOFError):
            print("\nPrograma finalizado.")
            break


if __name__ == "__main__":
    main()
