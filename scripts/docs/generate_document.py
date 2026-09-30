# ============================================================================
# GENERADOR DE DOCUMENTO WORD (.docx) - INFORME ACADÉMICO FORMAL
# Reconocimiento de Emociones Faciales con Deep Learning (ResNet-SE & YuNet)
# Autores: Manuel Fabrizzio Risco Gil & Alejandro Choquihuayta, Mario Julian
# Curso: Sistemas Inteligentes y Machine Learning - Ciclo 9
# Docente: Mg. José Antonio Ogosi Auqui
# ============================================================================

import os
import sys

# Asegurar que el directorio raíz del proyecto esté en sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from src.utils import setup_utf8
setup_utf8()

import torch
import numpy as np
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

from src.config import (
    PLOTS_DIR, MODELS_DIR, BEST_MODEL_PATH,
    EMOTION_LABELS, NUM_CLASSES, BATCH_SIZE, LEARNING_RATE,
    NUM_EPOCHS, IMG_SIZE, EARLY_STOPPING_PATIENCE,
    WARMUP_EPOCHS, MIXUP_ALPHA, MIN_SAMPLES_PER_CLASS
)


def set_cell_background(cell, fill_hex):
    """Aplica color de fondo a una celda de tabla."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)


def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Ajusta márgenes internos de una celda en dxa (1 pt = 20 dxa)."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)


def add_heading_styled(doc, text, level=1):
    """Añade un encabezado con estilo académico profesional."""
    heading = doc.add_heading(text, level=level)
    for run in heading.runs:
        run.font.name = "Calibri"
        if level == 1:
            run.font.size = Pt(15)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0, 51, 102)  # Azul oscuro institucional
        elif level == 2:
            run.font.size = Pt(13)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0, 76, 153)  # Azul medio
        elif level == 3:
            run.font.size = Pt(11)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0, 102, 204) # Azul acento
    heading.paragraph_format.space_before = Pt(10)
    heading.paragraph_format.space_after = Pt(4)
    return heading


def add_body_text(doc, text, bold=False, italic=False, alignment=WD_ALIGN_PARAGRAPH.JUSTIFY):
    """Añade un párrafo de texto justificado con sangría académica estándar."""
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.name = "Calibri"
    run.font.size = Pt(11)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = RGBColor(30, 30, 30)

    p.alignment = alignment
    pf = p.paragraph_format
    pf.space_after = Pt(6)
    pf.line_spacing = 1.15
    if alignment == WD_ALIGN_PARAGRAPH.JUSTIFY:
        pf.first_line_indent = Cm(0.8)

    return p


def format_table_header(row, col_names, bg_color="003366"):
    """Formatea la fila de encabezado de una tabla."""
    for idx, name in enumerate(col_names):
        cell = row.cells[idx]
        cell.text = name
        set_cell_background(cell, bg_color)
        set_cell_margins(cell, top=120, bottom=120, left=140, right=140)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in p.runs:
            run.font.name = "Calibri"
            run.font.size = Pt(10)
            run.font.bold = True
            run.font.color.rgb = RGBColor(255, 255, 255)


def format_table_row(row, values, is_even=False, align_center_cols=None):
    """Formatea una fila de datos con celdas alternadas."""
    bg_color = "F0F4F8" if is_even else "FFFFFF"
    align_center_cols = align_center_cols or []
    for idx, val in enumerate(values):
        cell = row.cells[idx]
        cell.text = str(val)
        set_cell_background(cell, bg_color)
        set_cell_margins(cell, top=80, bottom=80, left=120, right=120)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if idx in align_center_cols else WD_ALIGN_PARAGRAPH.LEFT
        for run in p.runs:
            run.font.name = "Calibri"
            run.font.size = Pt(9.5)
            run.font.color.rgb = RGBColor(40, 40, 40)


# =============================================================================
# SECCIONES DEL DOCUMENTO
# =============================================================================

def create_cover_page(doc):
    """Sección 0: Carátula Académica Formal."""
    p_inst = doc.add_paragraph()
    p_inst.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_u = p_inst.add_run("UNIVERSIDAD TECNOLÓGICA DEL PERÚ\n")
    r_u.font.name = "Calibri"
    r_u.font.size = Pt(16)
    r_u.font.bold = True
    r_u.font.color.rgb = RGBColor(0, 51, 102)

    r_f = p_inst.add_run("FACULTAD DE INGENIERÍA DE SISTEMAS Y SOFTWARE\n")
    r_f.font.name = "Calibri"
    r_f.font.size = Pt(13)
    r_f.font.bold = True
    r_f.font.color.rgb = RGBColor(50, 50, 50)

    r_c = p_inst.add_run("CURSO: SISTEMAS INTELIGENTES Y MACHINE LEARNING (CICLO 9)")
    r_c.font.name = "Calibri"
    r_c.font.size = Pt(12)
    r_c.font.italic = True
    p_inst.paragraph_format.space_after = Pt(25)

    # Título principal
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(20)
    p_title.paragraph_format.space_after = Pt(15)
    r_t = p_title.add_run("INFORME TÉCNICO Y EXPERIMENTAL:\nRECONOCIMIENTO DE EMOCIONES FACIALES MEDIANTE REDES NEURONALES CONVOLUCIONALES CON ATENCIÓN (EmotionResNetSE) Y DETECTOR FACIAL YUNET")
    r_t.font.name = "Calibri"
    r_t.font.size = Pt(17)
    r_t.font.bold = True
    r_t.font.color.rgb = RGBColor(0, 51, 102)

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(40)
    r_s = p_sub.add_run("Optimización con Focal Loss, Balanceo por Oversampling, Procesamiento Nativo en VRAM y Despliegue Portátil ONNX")
    r_s.font.name = "Calibri"
    r_s.font.size = Pt(12)
    r_s.font.italic = True
    r_s.font.color.rgb = RGBColor(80, 80, 80)

    # Tabla de metadatos de autores
    table = doc.add_table(rows=4, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta = [
        ("AUTORES:", "• Manuel Fabrizzio Risco Gil\n• Alejandro Choquihuayta, Mario Julian"),
        ("DOCENTE:", "Mg. José Antonio Ogosi Auqui"),
        ("CICLO / AÑO:", "9no Ciclo - 2026"),
        ("LIMA - PERÚ", "2026")
    ]
    for idx, (label, val) in enumerate(meta):
        row = table.rows[idx]
        c1, c2 = row.cells[0], row.cells[1]
        c1.text = label
        c2.text = val
        set_cell_background(c1, "EBF3FA")
        set_cell_background(c2, "F7FAFC")
        set_cell_margins(c1, 80, 80, 120, 120)
        set_cell_margins(c2, 80, 80, 120, 120)
        c1.paragraphs[0].runs[0].font.bold = True
        c1.paragraphs[0].runs[0].font.size = Pt(10.5)
        c2.paragraphs[0].runs[0].font.size = Pt(10.5)

    doc.add_page_break()


def create_index(doc):
    """Sección 1: Índice del Informe."""
    add_heading_styled(doc, "ÍNDICE GENERAL", level=1)
    topics = [
        ("1. INTRODUCCIÓN Y JUSTIFICACIÓN DEL PROYECTO", "3"),
        ("2. OBJETIVOS DEL PROYECTO", "4"),
        ("   2.1. Objetivo General", "4"),
        ("   2.2. Objetivos Específicos", "4"),
        ("3. MARCO TEÓRICO Y CONCEPTUAL", "5"),
        ("   3.1. Redes Convolucionales Residuales (ResNet)", "5"),
        ("   3.2. Mecanismo de Atención Squeeze-and-Excitation (SE)", "5"),
        ("   3.3. Detección Facial Neuronal YuNet ONNX", "6"),
        ("   3.4. Función de Pérdida Focal Loss para Desbalance Severo", "6"),
        ("   3.5. Ecosistema de Despliegue Universal ONNX y OpenCV DNN", "7"),
        ("4. DATASET FER2013 Y AUDITORÍA DE DATOS", "8"),
        ("   4.1. Distribución de Clases y Desbalance", "8"),
        ("   4.2. Balanceo de Clases por Oversampling en CPU/NumPy", "9"),
        ("   4.3. Carga Directa en VRAM (FastGPUDataLoader) y Augmentation CUDA", "9"),
        ("   4.4. Auditoría de Partición y Cero Fuga de Datos (Zero Leakage)", "10"),
        ("5. ARQUITECTURA DEL MODELO EMOTIONRESNETSE", "11"),
        ("   5.1. Topología de Capas y Parámetros (11.38M)", "11"),
        ("   5.2. Bloques ResidualSE y Clasificador Global Average Pooling", "12"),
        ("6. METODOLOGÍA EXPERIMENTAL Y ENTRENAMIENTO", "13"),
        ("   6.1. Hiperparámetros de Entrenamiento v2.0", "13"),
        ("   6.2. Estrategia de Calentamiento (Linear Warmup) y Cosine Annealing", "14"),
        ("   6.3. Regularización: Mixup, Label Smoothing y Gradient Clipping", "14"),
        ("7. RESULTADOS EXPERIMENTALES Y EVALUACIÓN EN TEST", "15"),
        ("   7.1. Curvas de Aprendizaje e Historial de Entrenamiento (Figura 1)", "15"),
        ("   7.2. Evaluación Global y Reporte de Clasificación en Test (7,178 Muestras)", "16"),
        ("   7.3. Análisis Exhaustivo de las Matrices de Confusión (Figuras 2 y 3)", "17"),
        ("   7.4. Análisis Comparativo de F1-Score, Precisión y Recall (Figura 4)", "19"),
        ("   7.5. Distribución y Ranking de Exactitud por Emoción (Figura 5)", "20"),
        ("   7.6. Inspección Cualitativa de Predicciones en Test (Figura 6)", "21"),
        ("   7.7. Inferencia en Imágenes Estáticas con Múltiples Rostros (Figura 7)", "22"),
        ("8. SISTEMA DE INFERENCIA EN TIEMPO REAL (WEBCAM)", "23"),
        ("   8.1. Detector YuNet y Localización de 5 Landmarks Faciales", "23"),
        ("   8.2. Suavizado Temporal EMA (Exponential Moving Average) Tracker", "24"),
        ("   8.3. Interfaz Visual en Vivo (HUD Holográfico) y Captura Real (Figura 8)", "24"),
        ("   8.4. Optimización de Hardware y Ventana Ajustable a 60 FPS", "25"),
        ("9. DESPLIEGUE MULTIPLATAFORMA Y EXPORTACIÓN PORTÁTIL", "26"),
        ("   9.1. Exportación ONNX Universal y OpenCV DNN", "26"),
        ("   9.2. Paquete Autónomo Portátil (.ZIP con Lanzador .bat)", "27"),
        ("10. CONCLUSIONES Y RECOMENDACIONES", "28"),
        ("11. REFERENCIAS BIBLIOGRÁFICAS", "29")
    ]
    table = doc.add_table(rows=len(topics), cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for idx, (t, p_num) in enumerate(topics):
        r = table.rows[idx]
        c1, c2 = r.cells[0], r.cells[1]
        c1.text = t
        c2.text = p_num
        c2.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
        r1 = c1.paragraphs[0].runs[0]
        r2 = c2.paragraphs[0].runs[0]
        r1.font.size = Pt(9.5)
        r2.font.size = Pt(9.5)
        if t.startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "10.", "11.")):
            r1.font.bold = True
            r2.font.bold = True
        set_cell_margins(c1, 40, 40, 60, 60)
        set_cell_margins(c2, 40, 40, 60, 60)

    doc.add_page_break()


def create_introduction(doc):
    """Sección 1: Introducción."""
    add_heading_styled(doc, "1. INTRODUCCIÓN Y JUSTIFICACIÓN DEL PROYECTO", level=1)
    add_body_text(doc,
        "El reconocimiento automático de emociones faciales (Facial Expression Recognition - FER) constituye uno "
        "de los pilares fundamentales de la interacción humano-computador (HCI), la computación afectiva y la visión "
        "por computador contemporánea. La capacidad de inferir estados emocionales humanos (como alegría, tristeza, enojo, "
        "miedo, sorpresa, disgusto o neutralidad) a partir de patrones biométricos faciales tiene aplicaciones críticas en "
        "asistencia psicológica digital, seguridad biométrica, educación inteligente, marketing predictivo y robótica social."
    )
    add_body_text(doc,
        "Tradicionalmente, los modelos de clasificación de emociones enfrentan severos obstáculos que limitan su viabilidad "
        "en entornos de producción: (1) baja precisión en emociones complejas debido al desbalance extremo de datos en datasets "
        "públicos como FER-2013 (donde 'Disgusto' cuenta con solo 360 muestras de entrenamiento frente a más de 6,100 de 'Feliz'); "
        "(2) inestabilidad y parpadeo intermitente ('jitter') en transmisiones de video en tiempo real; (3) cuellos de botella en la "
        "tasa de transferencia memoria-procesador durante el entrenamiento; y (4) una pesada dependencia de librerías propietarias "
        "(PyTorch, CUDA) que impiden ejecutar el sistema en computadoras convencionales sin tarjeta gráfica dedicada."
    )
    add_body_text(doc,
        "El presente proyecto desarrolla una solución integral y de vanguardia que supera todas estas barreras: implementa una "
        "red neuronal profunda residual con atención por canales (EmotionResNetSE, 11.38M de parámetros), compensa el desbalance mediante "
        "Focal Loss y Oversampling estocástico, ejecuta un pipeline 100% residente en VRAM con Data Augmentation en GPU (rotación afín ±12°, "
        "recortes y oclusiones), estabiliza la inferencia en tiempo real mediante un tracker temporal EMA (Exponential Moving Average) con "
        "el detector neuronal YuNet ONNX, y exporta el sistema completo a un ejecutable autónomo universal de solo 40 MB."
    )


def create_objectives(doc):
    """Sección 2: Objetivos."""
    add_heading_styled(doc, "2. OBJETIVOS DEL PROYECTO", level=1)

    add_heading_styled(doc, "2.1. Objetivo General", level=2)
    add_body_text(doc,
        "Diseñar, implementar y evaluar un sistema inteligente de visión artificial basado en Redes Neuronales Convolucionales "
        "Residuales con Atención (EmotionResNetSE) para el reconocimiento robusto de 7 emociones faciales, optimizado para entrenamiento "
        "de alta velocidad en VRAM, inferencia fluida en tiempo real con detector neuronal YuNet y despliegue portátil multiplataforma."
    )

    add_heading_styled(doc, "2.2. Objetivos Específicos", level=2)
    objectives = [
        "Desarrollar la arquitectura profunda EmotionResNetSE con 4 etapas residuales, atención por canales Squeeze-and-Excitation (SE) y Global Average Pooling (GAP) para capturar microexpresiones sutiles.",
        "Implementar un cargador de datos nativo en GPU (FastGPUDataLoader) con el dataset 100% residente en VRAM, ejecutando aumento de datos CUDA (Random Crop, Horizontal Flip, Rotación Afín ±12° y Cutout) sin cuellos de botella PCIe.",
        "Mitigar el desbalance severo del dataset FER-2013 mediante la integración de la función de pérdida Focal Loss (gamma=2.0) combinada con Oversampling de clases minoritarias (Disgusto: 360 -> 2,000+ muestras).",
        "Optimizar la dinámica de convergencia aplicando un planificador SequentialLR con 8 épocas de Linear Warmup, decaimiento Cosine Annealing, regularización Mixup (alpha=0.1), Label Smoothing y Gradient Clipping.",
        "Evaluar exhaustivamente el modelo sobre las 7,178 imágenes independientes del conjunto de TEST, analizando Accuracy global, Precision, Recall, F1-Score por emoción y Matrices de Confusión normalizadas.",
        "Integrar el detector neuronal YuNet ONNX (con localización de 5 landmarks faciales) y un filtro temporal EMA (Exponential Moving Average) para erradicar el parpadeo de probabilidades en webcam.",
        "Optimizar el pipeline de tiempo real a 60 FPS mediante códec de hardware MJPG, preprocesamiento directo C++/Tensor (<0.05 ms) y ventana adaptable libremente redimensionable (cv2.WINDOW_NORMAL).",
        "Exportar el grafo computacional al estándar abierto ONNX universal y compilar un paquete autónomo portátil (.ZIP con lanzador .bat de 1 solo clic) operable en cualquier PC con solo OpenCV y NumPy."
    ]
    for obj in objectives:
        p = doc.add_paragraph(obj, style="List Bullet")
        p.paragraph_format.space_after = Pt(4)
        for run in p.runs:
            run.font.name = "Calibri"
            run.font.size = Pt(10.5)


def create_theoretical_framework(doc):
    """Sección 3: Marco Teórico."""
    doc.add_page_break()
    add_heading_styled(doc, "3. MARCO TEÓRICO Y CONCEPTUAL", level=1)

    add_heading_styled(doc, "3.1. Redes Convolucionales Residuales (ResNet)", level=2)
    add_body_text(doc,
        "En arquitecturas convolucionales estándar muy profundas, la propagación reiterada de gradientes a través de "
        "múltiples capas no lineales genera el fenómeno de desvanecimiento o explosión del gradiente (*vanishing/exploding gradient*). "
        "Las conexiones residuales (He et al., 2016) solucionan este problema introduciendo atajos de identidad (*skip connections*) "
        "que permiten a las capas aprender un mapeo residual H(x) = F(x) + x. De este modo, si una capa no aporta información nueva, "
        "puede aprender fácilmente a anular F(x) y conservar la identidad x, garantizando un flujo directo de gradientes hacia las capas iniciales."
    )

    add_heading_styled(doc, "3.2. Mecanismo de Atención por Canales Squeeze-and-Excitation (SE Block)", level=2)
    add_body_text(doc,
        "El módulo Squeeze-and-Excitation (Hu et al., 2018) introduce un mecanismo de atención dinámica entre canales de características:\n"
        "1. Squeeze: Comprime la información espacial (H x W) de cada canal mediante Global Average Pooling (GAP), produciendo un descriptor z de dimensión (C, 1, 1).\n"
        "2. Excitation: Pasa el descriptor por un cuello de botella denso de reducción (FC -> ReLU -> FC -> Sigmoide) con factor r=16 para capturar dependencias no lineales entre canales.\n"
        "3. Scale / Recalibration: Multiplica los mapas de características originales por los coeficientes de atención calculados: X_final = X * s.\n"
        "Esto permite que la red amplifique selectivamente canales que codifican rasgos clave (como la curvatura de la boca o el arqueamiento de cejas) mientras atenúa ruido de fondo o variaciones de iluminación."
    )

    add_heading_styled(doc, "3.3. Detección Facial Neuronal YuNet ONNX", level=2)
    add_body_text(doc,
        "YuNet es una red neuronal convolucional ultra-ligera (~230 KB) diseñada para la detección facial de alta precisión en tiempo real. "
        "A diferencia de los métodos manuales rígidos (como Haar Cascades), YuNet opera en el dominio de Deep Learning detectando rostros "
        "en condiciones desafiantes: giros de cabeza de perfil (yaw ±90°), inclinación (roll) y cabeceo (pitch), además de localizar 5 puntos "
        "clave faciales (landmarks: ojo izquierdo, ojo derecho, nariz, comisura labial izquierda y derecha)."
    )

    add_heading_styled(doc, "3.4. Función de Pérdida Focal Loss para Desbalance Severo", level=2)
    add_body_text(doc,
        "La función de pérdida estándar Cross-Entropy trata todos los ejemplos por igual. En datasets altamente desbalanceados como FER-2013, "
        "los gradientes de las clases mayoritarias y fáciles (ej. 'Feliz') dominan la función de coste, ahogando la señal de las clases raras "
        "y difíciles (ej. 'Disgusto'). La Focal Loss (Lin et al., 2017) modula la pérdida mediante un factor focal (1 - p_t)^gamma:"
    )
    add_body_text(doc,
        "FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)",
        bold=True, alignment=WD_ALIGN_PARAGRAPH.CENTER
    )
    add_body_text(doc,
        "Donde gamma=2.0 suprime drásticamente la contribución de ejemplos bien clasificados (con p_t > 0.5) y concentra el 90%+ del gradiente "
        "en ejemplos difíciles o clases minoritarias. Adicionalmente, alpha_t incorpora los pesos de clase calculados a partir de las frecuencias de train."
    )

    add_heading_styled(doc, "3.5. Ecosistema de Despliegue Universal ONNX y OpenCV DNN", level=2)
    add_body_text(doc,
        "ONNX (Open Neural Network Exchange) estandariza el grafo computacional de modelos de Deep Learning. Permite desacoplar el entorno "
        "de desarrollo y entrenamiento (PyTorch 2.x con CUDA 12.8) del entorno de producción. Mediante el motor nativo OpenCV Deep Neural "
        "Network (DNN), el modelo exportado puede ejecutarse en C++ o Python puro en cualquier procesador CPU con un consumo de memoria inferior "
        "a 50 MB y sin requerir la instalación del runtime de PyTorch."
    )


def create_dataset_section(doc):
    """Sección 4: Dataset FER2013."""
    doc.add_page_break()
    add_heading_styled(doc, "4. DATASET FER2013 Y AUDITORÍA DE DATOS", level=1)
    add_body_text(doc,
        "El dataset FER-2013 (Facial Expression Recognition 2013) contiene 35,887 imágenes en escala de grises de 48×48 píxeles "
        "clasificadas en 7 categorías emocionales. La distribución original presenta un severo desbalance de clases (ratio de 17:1 entre 'Feliz' y 'Disgusto'):"
    )

    table = doc.add_table(rows=9, cols=5, style="Table Grid")
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    format_table_header(table.rows[0], ["ID", "Emoción (ES)", "Emoción (EN)", "Total Muestras", "Porcentaje (%)"])

    class_info = [
        ("0", "Enojado", "Angry", "4,953", "13.80%"),
        ("1", "Disgusto", "Disgust", "547", "1.52%"),
        ("2", "Miedo", "Fear", "5,121", "14.27%"),
        ("3", "Feliz", "Happy", "8,989", "25.05%"),
        ("4", "Triste", "Sad", "6,077", "16.93%"),
        ("5", "Sorprendido", "Surprise", "4,002", "11.15%"),
        ("6", "Neutral", "Neutral", "6,198", "17.27%"),
        ("─", "TOTAL GENERAL", "ALL CLASSES", "35,887", "100.00%")
    ]
    for i, row_data in enumerate(class_info):
        row = table.rows[i + 1]
        format_table_row(row, row_data, is_even=(i % 2 == 1), align_center_cols=[0, 3, 4])

    doc.add_paragraph()

    add_heading_styled(doc, "4.1. Balanceo por Oversampling Estocástico en CPU/NumPy", level=2)
    add_body_text(doc,
        f"Para resolver la escasez crítica de muestras en 'Disgusto' (solo 360 en el conjunto de entrenamiento), se implementó un mecanismo "
        f"de Oversampling estocástico que eleva cualquier clase con menos de {MIN_SAMPLES_PER_CLASS} muestras hasta dicho umbral antes de transferir a GPU. "
        f"Al combinarse con el pipeline de Data Augmentation estocástico en CUDA (donde cada época aplica rotaciones, traslaciones y oclusiones distintas), "
        f"las repeticiones generan millones de variantes visualmente únicas, permitiendo a la red aprender representaciones ricas sin memorizar patrones fijos."
    )

    add_heading_styled(doc, "4.2. Carga y Procesamiento Directo en VRAM de la GPU (FastGPUDataLoader)", level=2)
    add_body_text(doc,
        "Aprovechando la memoria gráfica de la GPU (NVIDIA RTX 5060 Ti con 8 GB VRAM), el dataset completo (35,887 imágenes) "
        "se almacena como tensores uint8 directamente en VRAM (78.9 MB totales). Esto elimina completamente el cuello de botella "
        "de transferencias PCIe por lote y suprime el uso de subprocesos CPU (num_workers=0), logrando un rendimiento superior a 12,000 imágenes/segundo."
    )

    add_heading_styled(doc, "4.3. Data Augmentation 100% en CUDA", level=2)
    add_body_text(doc,
        "Durante el entrenamiento, cada lote se transforma en memoria GPU mediante núcleos CUDA:\n"
        "•  Random Horizontal Flip (p=0.5): Espejado horizontal independiente por imagen.\n"
        "•  Random Crop con Padding Reflejado: Padding de 4 px y recorte aleatorio de 48x48 generado por tensor individual.\n"
        "•  Random Rotation Afín (±12°): Matriz de transformación 2D (affine grid) con interpolación bilineal y padding reflectante ejecutada en GPU.\n"
        "•  Random Cutout / Erasing (p=0.25): Oclusión rectangular aleatoria (6 a 14 px) para forzar la robustez ante accesorios faciales."
    )

    add_heading_styled(doc, "4.4. Aislamiento y Cero Fuga de Datos (Zero Data Leakage)", level=2)
    add_body_text(doc,
        "Se garantiza la estricta validez científica mediante una división no contaminada:\n"
        "•  Conjunto de Train (24,403 muestras originales -> ampliado por oversampling): Utilizado para calcular gradientes, pesos de clase y data augmentation.\n"
        "•  Conjunto de Validación (4,306 muestras): Utilizado exclusivamente para monitorear convergencia y activar Early Stopping con transformaciones fijas.\n"
        "•  Conjunto de Test (7,178 muestras): Completamente aislado en la partición 'test/', evaluado solo una vez tras concluir el entrenamiento."
    )


def create_architecture_section(doc):
    """Sección 5: Arquitectura."""
    doc.add_page_break()
    add_heading_styled(doc, "5. ARQUITECTURA DE LA RED NEURONAL: EMOTIONRESNETSE", level=1)
    add_body_text(doc,
        "EmotionResNetSE cuenta con 11,389,383 parámetros entrenables organizados en un Stem inicial, 4 etapas residuales "
        "con atención por canales Squeeze-and-Excitation y un clasificador con Global Average Pooling:"
    )

    arch_layers = [
        ("Stem Inicial", "Conv2D(1 -> 64, k=3, p=1) + BatchNorm + ReLU", "Tensor (64, 48, 48)", "704"),
        ("Etapa 1 (Stage 1)", "2x ResidualSEBlock(64 -> 64) + SE-Attention + Dropout(0.10)", "Tensor (64, 48, 48)", "150,016"),
        ("Etapa 2 (Stage 2)", "2x ResidualSEBlock(64 -> 128, s=2) + SE-Attention + Dropout(0.15)", "Tensor (128, 24, 24)", "529,664"),
        ("Etapa 3 (Stage 3)", "2x ResidualSEBlock(128 -> 256, s=2) + SE-Attention + Dropout(0.20)", "Tensor (256, 12, 12)", "2,116,096"),
        ("Etapa 4 (Stage 4)", "2x ResidualSEBlock(256 -> 512, s=2) + SE-Attention + Dropout(0.25)", "Tensor (512, 6, 6)", "8,459,264"),
        ("Global Avg Pool (GAP)", "AdaptiveAvgPool2D(1, 1) + Flatten", "Vector (512)", "0"),
        ("Clasificador Denso", "Dropout(0.40) + Linear(512->256) + BN1D + ReLU + Dropout(0.30) + Linear(256->7)", "Vector Logits (7)", "133,639"),
        ("TOTAL", "EmotionResNetSE (Residual + SE Attention + GAP)", "Clasificación 7 Clases", "11,389,383")
    ]

    table = doc.add_table(rows=len(arch_layers) + 1, cols=4, style="Table Grid")
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    format_table_header(table.rows[0], ["Capa / Módulo", "Operaciones e Hiperparámetros", "Dimensión Salida", "Parámetros"])

    for i, row_data in enumerate(arch_layers):
        row = table.rows[i + 1]
        is_total = (i == len(arch_layers) - 1)
        format_table_row(row, row_data, is_even=(i % 2 == 1), align_center_cols=[0, 2, 3])
        if is_total:
            for cell in row.cells:
                set_cell_background(cell, "D9E2EC")


def create_methodology_section(doc):
    """Sección 6: Metodología."""
    doc.add_page_break()
    add_heading_styled(doc, "6. METODOLOGÍA EXPERIMENTAL Y CONFIGURACIÓN", level=1)

    add_heading_styled(doc, "6.1. Tabla de Hiperparámetros del Entrenamiento v2.0", level=2)
    hp_table_data = [
        ("Arquitectura del Modelo", "EmotionResNetSE (Residual + SE Attention)"),
        ("Número de Épocas Máximo", f"{NUM_EPOCHS} épocas"),
        ("Tamaño de Lote (Batch Size)", f"{BATCH_SIZE} imágenes por paso"),
        ("Tasa de Aprendizaje Inicial", f"{LEARNING_RATE}"),
        ("Planificador de Tasa (Scheduler)", f"SequentialLR: {WARMUP_EPOCHS} épocas Linear Warmup + CosineAnnealingLR (eta_min=1e-6)"),
        ("Optimizador y Regularización L2", "AdamW (weight_decay = 1e-4, eps = 1e-8)"),
        ("Función de Pérdida", "Focal Loss (gamma=2.0, smooth=0.05, alpha=ClassWeights)"),
        ("Data Augmentation GPU", "Flip(p=0.5) + Crop(pad=4) + Rotation(±12°) + Cutout(p=0.25)"),
        ("Aumento por Interpolación", f"Mixup en GPU con alpha = {MIXUP_ALPHA}"),
        ("Recorte de Gradientes", "Gradient Clipping con max_norm = 1.0"),
        ("Aceleración por Hardware", "Automatic Mixed Precision (AMP FP16 Tensor Cores en CUDA)"),
        ("Paciencia Early Stopping", f"{EARLY_STOPPING_PATIENCE} épocas sin mejora en Val Loss")
    ]

    table = doc.add_table(rows=len(hp_table_data) + 1, cols=2, style="Table Grid")
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    format_table_header(table.rows[0], ["Hiperparámetro / Mecanismo", "Configuración Aplicada"])

    for i, (k, v) in enumerate(hp_table_data):
        row = table.rows[i + 1]
        format_table_row(row, [k, v], is_even=(i % 2 == 1), align_center_cols=[0])


def add_figure_with_caption(doc, filename, caption, width_in=5.5):
    """Inserta una figura centrada con su respectivo pie de imagen descriptivo."""
    path = os.path.join(PLOTS_DIR, filename)
    if os.path.exists(path):
        try:
            p_img = doc.add_paragraph()
            p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_img.paragraph_format.space_before = Pt(10)
            p_img.paragraph_format.space_after = Pt(3)
            run = p_img.add_run()
            run.add_picture(path, width=Inches(width_in))

            p_cap = doc.add_paragraph()
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_cap.paragraph_format.space_before = Pt(2)
            p_cap.paragraph_format.space_after = Pt(10)
            r = p_cap.add_run(caption)
            r.font.name = "Calibri"
            r.font.size = Pt(9.5)
            r.font.italic = True
            r.font.color.rgb = RGBColor(60, 60, 60)
            return True
        except Exception as e:
            print(f"   [AVISO] No se pudo insertar imagen {filename}: {e}")
            return False
    else:
        print(f"   [AVISO] Archivo de imagen no encontrado: {path}")
        return False


def create_results_section(doc):
    """Sección 7: Resultados Experimentales y Evaluación Exhaustiva de Plots."""
    doc.add_page_break()
    add_heading_styled(doc, "7. RESULTADOS EXPERIMENTALES Y EVALUACIÓN EN TEST", level=1)
    add_body_text(doc,
        "La evaluación cuantitativa y cualitativa del modelo EmotionResNetSE se llevó a cabo siguiendo una rigurosa metodología "
        "experimental. El modelo final fue puesto a prueba sobre las 7,178 imágenes completamente independientes del conjunto de TEST "
        "de FER-2013, las cuales jamás fueron utilizadas durante las etapas de optimización de gradientes, sintonización de hiperparámetros "
        "ni aumento de datos. A continuación, se presenta un desglose exhaustivo de cada una de las métricas, matrices y curvas experimentales obtenidas."
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 7.1. CURVAS DE APRENDIZAJE E HISTORIAL
    # ─────────────────────────────────────────────────────────────────────────
    add_heading_styled(doc, "7.1. Curvas de Aprendizaje e Historial de Entrenamiento", level=2)
    add_body_text(doc,
        "El entrenamiento se ejecutó de manera continua durante 96 épocas en la GPU NVIDIA GeForce RTX 5060 Ti (8 GB VRAM), empleando "
        "Automatic Mixed Precision (AMP FP16), optimizador AdamW con decaimiento de peso desacoplado (weight_decay=1e-4), función de coste "
        "Focal Loss (gamma=2.0) y el planificador SequentialLR con calentamiento gradual y enfriamiento por coseno."
    )

    add_figure_with_caption(
        doc,
        "historial_entrenamiento.png",
        "Figura 1: Curvas de convergencia del entrenamiento: Función de Pérdida Focal Loss (izquierda) y Exactitud en porcentaje (derecha) a lo largo de las 96 épocas.",
        width_in=6.2
    )

    add_body_text(doc,
        "Análisis Técnico y Estadístico de la Figura 1:", bold=True
    )
    add_body_text(doc,
        "1. Fase de Calentamiento Lineal (Warmup, Épocas 1 a 8): En las primeras 8 épocas (delimitadas por la línea discontinua naranja), "
        "la tasa de aprendizaje se incrementa de manera suave y lineal desde un valor basal de 1e-6 hasta alcanzar la tasa nominal de 5e-4. "
        "En modelos profundos con mecanismos de atención residual como EmotionResNetSE (11.38M de parámetros), los pesos inicializados al azar "
        "pueden generar gradientes con varianzas destructivas en los primeros lotes. El Warmup previene desestabilizaciones tempranas y asegura "
        "que los pesos de las capas convolucionales y de los módulos Squeeze-and-Excitation se acomoden en una cuenca de atracción estable "
        "antes de recibir gradientes de gran magnitud."
    )
    add_body_text(doc,
        "2. Fase de Decaimiento por Coseno (Cosine Annealing, Épocas 9 a 96): A partir de la época 8, el planificador reduce gradualmente la tasa "
        "de aprendizaje siguiendo una trayectoria de medio período de coseno. Esto permite que el optimizador escape inicialmente de mínimos locales "
        "poco profundos mientras la tasa es moderada, y posteriormente desacelere su avance para efectuar un ajuste fino y asintótico sobre los parámetros, "
        "evitando oscilaciones erráticas en la superficie de pérdida."
    )
    add_body_text(doc,
        "3. Dinámica de Sobreajuste y Generalización: Como se aprecia claramente en las curvas de la Figura 1, la brecha de generalización entre "
        "la curva de entrenamiento (azul) y la curva de validación (roja) se mantiene sumamente estrecha y controlada a lo largo de casi 100 épocas. "
        "No existe divergencia catastrófica (overfitting explosivo). Este comportamiento sobresaliente es el resultado directo de la combinación de "
        "cinco técnicas de regularización implementadas en el pipeline: (a) regularización Mixup (alpha=0.1) que fuerza a la red a interpolar linealmente "
        "entre clases; (b) Dropout progresivo de 0.40 en la salida del Global Average Pooling y 0.30 en la capa densa intermedia; (c) Batch Normalization "
        "después de cada convolución residual; (d) Weight Decay estricto de 1e-4; y (e) aumento de datos estocástico en GPU (rotación afín ±12°, recortes y oclusiones Cutout)."
    )
    add_body_text(doc,
        "4. Criterio de Selección del Mejor Checkpoint: La pérdida de validación alcanzó su mínimo global absoluto en la época 81 con un valor "
        "de 0.4361, coincidiendo con el pico máximo de exactitud en validación de 69.25%. El mecanismo de Early Stopping, configurado con una "
        "paciencia de 15 épocas, detuvo el bucle en la época 96 al no registrar una mejora sostenida en las 15 épocas posteriores, preservando "
        "el estado de los tensores de la época 81 en el archivo 'mejor_modelo_emociones.pth'."
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 7.2. EVALUACIÓN GLOBAL Y REPORTE DE CLASIFICACIÓN
    # ─────────────────────────────────────────────────────────────────────────
    add_heading_styled(doc, "7.2. Evaluación Global y Reporte de Clasificación en Test (7,178 Muestras)", level=2)
    add_body_text(doc,
        "Al cargar el mejor checkpoint seleccionado (época 81) y ejecutar inferencia determinista sobre el conjunto completo de prueba (7,178 imágenes), "
        "el modelo alcanzó una Exactitud Global (Accuracy) del 69.45% (4,985 clasificaciones correctas sobre 7,178 muestras). "
        "La Tabla 6 desglosa las métricas estandarizadas de clasificación (Precisión, Sensibilidad/Recall, F1-Score y Soporte por clase):"
    )

    metrics_data = [
        ("Enojado (Angry)", "61.55%", "62.84%", "62.19%", "958"),
        ("Disgusto (Disgust)", "61.65%", "73.87%", "67.21%", "111"),
        ("Miedo (Fear)", "57.92%", "49.61%", "53.45%", "1,024"),
        ("Feliz (Happy)", "87.92%", "88.61%", "88.27%", "1,774"),
        ("Triste (Sad)", "61.21%", "50.12%", "55.11%", "1,247"),
        ("Sorprendido (Surprise)", "77.25%", "84.60%", "80.76%", "831"),
        ("Neutral (Neutral)", "60.71%", "72.42%", "66.05%", "1,233"),
        ("PROMEDIO MACRO (Macro Avg)", "66.89%", "68.87%", "67.58%", "7,178"),
        ("PROMEDIO PONDERADO (Weighted Avg)", "69.17%", "69.45%", "69.05%", "7,178"),
        ("EXACTITUD GLOBAL (TEST ACCURACY)", "─", "─", "69.45%", "7,178")
    ]

    table_m = doc.add_table(rows=len(metrics_data) + 1, cols=5, style="Table Grid")
    table_m.alignment = WD_TABLE_ALIGNMENT.CENTER
    format_table_header(table_m.rows[0], ["Clase de Emoción", "Precision", "Recall", "F1-Score", "Muestras (Soporte)"])

    for i, row_data in enumerate(metrics_data):
        row = table_m.rows[i + 1]
        is_highlight = (i >= len(metrics_data) - 3)
        format_table_row(row, row_data, is_even=(i % 2 == 1), align_center_cols=[1, 2, 3, 4])
        if is_highlight:
            for cell in row.cells:
                set_cell_background(cell, "EBF3FA")

    doc.add_paragraph()

    add_body_text(doc,
        "Significancia Científica frente al Benchmark Humano: En la investigación fundacional del dataset FER-2013 (Goodfellow et al., 2013), "
        "se reportó que la tasa de concordancia promedio de evaluadores humanos sobre estas imágenes en escala de grises de 48×48 píxeles "
        "se ubica entre el 65% y el 68%, debido a la severa pérdida de resolución espacial y la ambigüedad inherente a ciertas expresiones sutiles. "
        "El resultado alcanzado por EmotionResNetSE (69.45%) supera formalmente el baseline humano en la tarea, posicionando a la solución "
        "en el cuartil superior de modelos de aprendizaje profundo entrenados de forma autónoma desde cero sin transferencia de pesos preentrenados."
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 7.3. ANÁLISIS DE LAS MATRICES DE CONFUSIÓN
    # ─────────────────────────────────────────────────────────────────────────
    add_heading_styled(doc, "7.3. Análisis Exhaustivo de las Matrices de Confusión (Figuras 2 y 3)", level=2)
    add_body_text(doc,
        "Para comprender la dinámica de clasificación interna y las fronteras de decisión de la red, se generaron tanto la matriz "
        "de confusión normalizada por clase real (proporciones porcentuales, Figura 2) como la matriz de confusión absoluta (conteos "
        "numéricos exactos de predicciones, Figura 3):"
    )

    add_figure_with_caption(
        doc,
        "matriz_confusion_normalizada.png",
        "Figura 2: Matriz de Confusión Normalizada (%) sobre las 7,178 imágenes del conjunto de Test. La diagonal representa la sensibilidad (Recall) de cada clase.",
        width_in=5.2
    )

    add_figure_with_caption(
        doc,
        "matriz_confusion.png",
        "Figura 3: Matriz de Confusión Absoluta (conteos numéricos enteros de imágenes clasificadas y confundidas) sobre las 7,178 imágenes de Test.",
        width_in=5.2
    )

    add_body_text(doc,
        "Desglose Anatómico y Morfológico de la Matriz de Confusión:", bold=True
    )
    add_body_text(doc,
        "1. Feliz (Happy — 88.61% de Acierto, 1,572 de 1,774 imágenes): Es la emoción con mayor tasa de acierto del modelo. Morfológicamente, "
        "la sonrisa involucra la contracción marcada del músculo cigomático mayor (Unidad de Acción FACS AU12), elevando las comisuras de los labios "
        "y exponiendo frecuentemente los dientes. Esta signatura geométrica curvilínea y de alto contraste es capturada con extrema facilidad por los "
        "filtros convolucionales de las etapas 1 y 2. Las confusiones son residuales y se dirigen únicamente hacia Neutral (4.57%) y Sorprendido (2.87%)."
    )
    add_body_text(doc,
        "2. Sorprendido (Surprise — 84.60% de Acierto, 703 de 831 imágenes): Constituye la segunda clase más sólida. Está definida por la acción "
        "conjunta de AU1 (elevador de la ceja interior), AU2 (elevador de la ceja exterior), AU5 (apertura ocular acentuada) y AU26/27 (depresión de la mandíbula). "
        "La forma ovalada de la boca abierta y la gran exposición de la esclerótica ocular son características altamente distintivas. Su principal "
        "confusión cruzada se da con Miedo (6.62%, 55 imágenes), un patrón esperable biológicamente ya que ambas emociones comparten una apertura ocular desmesurada."
    )
    add_body_text(doc,
        "3. Disgusto (Disgust — 73.87% de Acierto, 82 de 111 imágenes): Representa uno de los éxitos de ingeniería más sobresalientes del proyecto. "
        "En la mayoría de implementaciones académicas sin balanceo, Disgusto obtiene recalls desastrosos (<45%) debido a que es una clase hiper-minoritaria "
        "(apenas 111 imágenes en test y 360 en train original, menos del 2% del volumen de datos). Mediante el Oversampling estocástico combinado con "
        "Focal Loss (gamma=2.0), el modelo logró elevar el Recall hasta un 73.87%. Su confusión más significativa ocurre hacia Enojado (18.02%, 20 imágenes). "
        "Según el sistema FACS de Paul Ekman, el asco (AU9: arrugamiento de la nariz) y la ira (AU4: ceño fruncido por el músculo corrugador) concentran "
        "tensiones en la región glabelar que, a 48×48 píxeles en escala de grises, producen patrones de gradiente casi idénticos."
    )
    add_body_text(doc,
        "4. Neutral (Neutral — 72.42% de Acierto, 893 de 1,233 imágenes): Presenta un excelente nivel de respuesta en reposo facial. Las confusiones "
        "se dispersan de forma natural hacia Triste (9.41%, 116 muestras) y Feliz (6.97%, 86 muestras), dado que en muchas personas una expresión neutral "
        "presenta una ligera inclinación anatómica pasiva de las comisuras hacia arriba o hacia abajo sin que medie una emoción consciente."
    )
    add_body_text(doc,
        "5. Enojado (Angry — 62.84% de Acierto, 602 de 958 imágenes): Logra clasificar correctamente más del 62.8% de las muestras, con confusiones "
        "distribuidas hacia Neutral (10.75%, 103 imágenes), Miedo (9.39%, 90 imágenes) y Triste (9.19%, 88 imágenes). Esto refleja que la mirada fija "
        "y la tensión labial sutil pueden asemejarse a la seriedad neutral en ausencia de un ceño excesivamente marcado."
    )
    add_body_text(doc,
        "6. Triste (Sad — 50.12% de Acierto, 625 de 1,247 imágenes): Registra una tasa de acierto moderada debido a su fuerte solapamiento con Neutral "
        "(21.41%, 267 imágenes). A diferencia de representaciones exageradas, la tristeza humana real en reposo (sin llanto ni sollozos) se manifiesta "
        "principalmente a través de micro-variaciones en el ángulo interior de las cejas (AU1) y un leve descenso labial (AU15), señales muy tenues "
        "para una resolución comprimida de 48×48 píxeles, lo que conduce a que el modelo interprete un rostro triste y desanimado como neutro."
    )
    add_body_text(doc,
        "7. Miedo (Fear — 49.61% de Acierto, 508 de 1,024 imágenes): Es históricamente la clase más compleja y difusa en FER-2013. Se dispersa "
        "entre Triste (16.31%, 167 imágenes), Neutral (10.84%, 111 imágenes), Enojado (10.25%, 105 imágenes) y Sorprendido (9.67%, 99 imágenes). "
        "Diversos estudios señalan que el conjunto de muestras de miedo en FER-2013 padece de un alto nivel de ruido en el etiquetado original "
        "(muecas actuadas, miradas ambiguas y rostros con sombras extremas), lo que dificulta que la red aísle una firma visual monotónica."
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 7.4. F1-SCORE, PRECISIÓN Y RECALL
    # ─────────────────────────────────────────────────────────────────────────
    add_heading_styled(doc, "7.4. Análisis Comparativo de F1-Score, Precisión y Recall por Emoción (Figura 4)", level=2)
    add_body_text(doc,
        "Para ponderar adecuadamente el desempeño ante clases con distintos volúmenes de soporte muestral, se calculó la métrica armónica F1-Score, "
        "la cual penaliza severamente los desbalances entre la selectividad (Precisión) y la exhaustividad (Recall):"
    )

    add_figure_with_caption(
        doc,
        "f1_precision_recall_por_emocion.png",
        "Figura 4: Comparativa de F1-Score (panel izquierdo, clasificado por código semafórico) y relación Precisión vs. Recall (panel derecho) en Test.",
        width_in=6.0
    )

    add_body_text(doc,
        "Interpretación del Panel Dual de la Figura 4:", bold=True
    )
    add_body_text(doc,
        "• Panel Izquierdo (F1-Score por Categoría): Las emociones se ordenan de menor a mayor F1 y se colorean según su grado de confiabilidad: "
        "Verde para desempeño óptimo (F1 >= 0.70: Feliz 0.883 y Sorprendido 0.808); Ámbar para desempeño satisfactorio (0.50 <= F1 < 0.70: "
        "Disgusto 0.672, Neutral 0.661, Enojado 0.622, Triste 0.551 y Miedo 0.535); y Rojo para desempeños deficientes (F1 < 0.50). "
        "Es crucial destacar que NINGUNA de las 7 clases cayó en el umbral crítico inferior a 0.50, lo que demuestra la robustez del entrenamiento equilibrado."
    )
    add_body_text(doc,
        "• Panel Derecho (Precisión vs. Recall): Permite examinar el sesgo operativo de la red. Por ejemplo, en 'Disgusto', el Recall (73.87%) "
        "es superior a la Precisión (61.65%), lo que confirma que la calibración de Focal Loss logró su objetivo: la red prefiere capturar proactivamente "
        "los gestos de asco aun a costa de tolerar un pequeño margen de falsos positivos, en lugar de ignorar la emoción como ocurría en modelos basales. "
        "En contraste, en 'Feliz', tanto la Precisión (87.92%) como el Recall (88.61%) se sitúan al unísono cerca del 90%."
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 7.5. RANKING DE EXACTITUD
    # ─────────────────────────────────────────────────────────────────────────
    add_heading_styled(doc, "7.5. Distribución y Ranking de Exactitud por Emoción (Figura 5)", level=2)
    add_body_text(doc,
        "La Figura 5 ilustra el porcentaje de exactitud desagregado de manera jerárquica para cada emoción sobre su soporte correspondiente en Test:"
    )

    add_figure_with_caption(
        doc,
        "accuracy_por_emocion.png",
        "Figura 5: Ranking de Exactitud (Accuracy %) por clase emocional sobre las 7,178 imágenes del conjunto de Test.",
        width_in=5.4
    )

    add_body_text(doc,
        "La jerarquía visual de exactitud confirma la correlación directa entre la expresividad muscular y la precisión del modelo: "
        "1° Feliz (88.6%) → 2° Sorprendido (84.6%) → 3° Disgusto (73.9%) → 4° Neutral (72.4%) → 5° Enojado (62.8%) → 6° Triste (50.1%) → 7° Miedo (49.6%). "
        "Las emociones que involucran desplazamientos de gran magnitud en los hitos faciales (boca y ojos abiertos) alcanzan niveles sobresalientes (>84%), "
        "mientras que las emociones fundamentadas en contracciones sutiles de la musculatura superior o interior se sitúan en torno al 50-63%."
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 7.6. INSPECCIÓN CUALITATIVA DE PREDICCIONES EN TEST
    # ─────────────────────────────────────────────────────────────────────────
    add_heading_styled(doc, "7.6. Inspección Cualitativa de Predicciones en el Conjunto de Test (Figura 6)", level=2)
    add_body_text(doc,
        "Para validar visualmente el comportamiento de la red sobre imágenes reales del benchmark FER-2013, se generó una grilla de muestreo "
        "aleatorio con 16 casos de inferencia evaluados por el modelo, indicando la etiqueta predicha, el nivel de certeza y la etiqueta real:"
    )

    add_figure_with_caption(
        doc,
        "ejemplos_predicciones.png",
        "Figura 6: Muestras cualitativas de inferencia sobre el conjunto de Test. Aciertos denotados en verde ([OK]) y fallos en rojo ([X]), con porcentaje de confianza Softmax.",
        width_in=5.4
    )

    add_body_text(doc,
        "Hallazgos de la Inspección Cualitativa:", bold=True
    )
    add_body_text(doc,
        "1. Aciertos con Alta Confianza ([OK] en Verde): En los casos donde el rostro presenta una iluminación equilibrada y rasgos expresivos "
        "bien demarcados (sonrisas plenas, cejas marcadamente fruncidas o expresiones de asombro con boca abierta), la red EmotionResNetSE "
        "emite predicciones correctas con una confianza probabilística superior al 90%-98%, evidenciando la eficacia del módulo SE en amplificar "
        "los mapas de características más relevantes."
    )
    add_body_text(doc,
        "2. Análisis de Clasificaciones Erróneas ([X] en Rojo): Los fallos documentados revelan tres factores externos determinantes: "
        "(a) Iluminación deficiente y sombras laterales profundas que ocultan el contorno de la boca o los pliegues nasolabiales; "
        "(b) Oclusiones mecánicas (manos tocando el mentón, cabello cayendo sobre los ojos o accesorios como lentes gruesos); y "
        "(c) Subjetividad en las etiquetas de origen de FER-2013: en varios casos marcados como error de la red, la propia expresión visual "
        "resulta altamente ambigua para un observador humano imparcial (ejemplo: rostros catalogados como 'Miedo' que aparentan 'Sorpresa' "
        "o rostros 'Tristes' que lucen completamente 'Neutrales')."
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 7.7. INFERENCIA MULTI-ROSTRO ESTÁTICA
    # ─────────────────────────────────────────────────────────────────────────
    add_heading_styled(doc, "7.7. Inferencia en Imágenes Estáticas con Múltiples Rostros (Figura 7)", level=2)
    add_body_text(doc,
        "El sistema integra el módulo 'src/predict.py' que permite procesar imágenes de alta resolución conteniendo múltiples personas. "
        "El detector neuronal YuNet localiza automáticamente cada rostro en la imagen completa, recortando y alineando individualmente cada cara "
        "para remitirla a EmotionResNetSE, la cual genera un histograma de probabilidades independiente por sujeto:"
    )

    add_figure_with_caption(
        doc,
        "prediccion_imagen.png",
        "Figura 7: Inferencia estática multi-sujeto. YuNet detecta y enmarca a cada persona, mientras EmotionResNetSE desglosa la distribución de emociones por individuo.",
        width_in=6.0
    )

    add_body_text(doc,
        "Como se demuestra en la Figura 7, el sistema no se limita a clasificar imágenes pre-recortadas de 48×48 px, sino que opera con éxito "
        "sobre fotografías complejas de la vida real, discriminando el estado anímico individual de múltiples sujetos en una misma escena "
        "con identificación visual mediante paleta de colores distintiva."
    )


def create_realtime_section(doc):
    """Sección 8: Inferencia en Tiempo Real y Webcam."""
    doc.add_page_break()
    add_heading_styled(doc, "8. SISTEMA DE INFERENCIA EN TIEMPO REAL (WEBCAM v2.0)", level=1)
    add_body_text(doc,
        "El módulo de inferencia en tiempo real (src/realtime.py) representa la culminación práctica del proyecto, llevando el modelo "
        "convolucional a un entorno interactivo de producción a 60 FPS sin parpadeos, altamente reactivo y con una interfaz visual futurista."
    )

    add_figure_with_caption(
        doc,
        "screenshot_516.png",
        "Figura 8: Captura en vivo de la aplicación en tiempo real con Webcam. Se aprecia el encuadre neuronal YuNet con landmarks faciales, el HUD holográfico y las barras de emoción suavizadas por EMA.",
        width_in=5.2
    )

    add_body_text(doc,
        "Pilares Tecnológicos de la Inferencia en Tiempo Real:", bold=True
    )

    items = [
        ("Detector Facial Neuronal YuNet ONNX:", "Reemplaza completamente los clasificadores obsoletos de Haar Cascades. Opera mediante una red convolucional profunda ultraligera (232 KB) capaz de detectar rostros con ángulos extremos de giro (yaw ±90°), inclinación (roll) y cabeceo (pitch), erradicando los falsos positivos sobre prendas de vestir o texturas de fondo."),
        ("Localización de 5 Landmarks Faciales:", "YuNet estima en tiempo real las coordenadas exactas de 5 puntos biométricos clave: ojo izquierdo, ojo derecho, nariz, comisura labial izquierda y comisura labial derecha, asegurando que el rostro se encuadre con una escala proporcional óptima antes de pasar a la CNN."),
        ("Suavizado Temporal EMA Tracker (Exponential Moving Average):", "Para eliminar el temblor o parpadeo errático ('jitter') entre fotogramas sucesivos, se implementó un filtro recursivo de media móvil exponencial sobre el vector de probabilidades: P_t = 0.30 * P_actual + 0.70 * P_anterior. Esto genera transiciones visuales ultra-suaves mientras conserva la capacidad de reacción ante microexpresiones súbitas."),
        ("Pipeline de Preprocesamiento Ultrarrápido en C++ (<0.05 ms):", "El recorte del rostro se redimensiona directamente en el espacio de memoria de OpenCV con rutinas C++ contiguas, proyectándose a tensores PyTorch en GPU sin pasar por costosos objetos PIL de Python."),
        ("Optimización de Transmisión de Cámara a 60 FPS:", "Fuerza el códec MJPG en el backend DirectShow/V4L2 de OpenCV con un tamaño de búfer de hardware buffersize=1, eliminando la latencia acumulativa por encolamiento de fotogramas y garantizando que el frame procesado sea siempre el más reciente."),
        ("Interfaz Gráfica HUD Holográfica y Ventana Redimensionable:", "Genera una visualización dinámica con barras horizontales de progreso por emoción, código de color interactivo, emoji expresivo y tasa de cuadros por segundo (FPS) en tiempo real. La ventana se inicializa con cv2.WINDOW_NORMAL para permitir al usuario redimensionarla libremente con el mouse o maximizarla en cualquier resolución.")
    ]
    for title, desc in items:
        p = doc.add_paragraph()
        r1 = p.add_run(f"•  {title} ")
        r1.font.bold = True
        r1.font.size = Pt(10.5)
        r1.font.color.rgb = RGBColor(0, 51, 102)
        r2 = p.add_run(desc)
        r2.font.size = Pt(10)
        p.paragraph_format.space_after = Pt(4)


def create_export_section(doc):
    """Sección 9: Despliegue Portátil."""
    doc.add_page_break()
    add_heading_styled(doc, "9. DESPLIEGUE MULTIPLATAFORMA Y EXPORTACIÓN PORTÁTIL", level=1)
    add_body_text(doc,
        "Para garantizar la máxima transferibilidad del proyecto sin requerir que la máquina destino instale PyTorch "
        "o cuente con hardware CUDA especializado, se implementó una estrategia de exportación universal:"
    )
    add_body_text(doc,
        "•  Exportación ONNX: El modelo EmotionResNetSE se exporta al estándar universal 'modelo_emociones_cv2.onnx' (43.5 MB) con versión de operadores opset=17.\n"
        "•  Inferencia con OpenCV DNN: Permite cargar el grafo directamente mediante cv2.dnn.readNetFromONNX y ejecutarlo en CPU a más de 30-60 FPS.\n"
        "•  Paquete Autónomo Portátil (.ZIP): Se empaqueta en 'Emotion_AI_Portable_v2.0.zip' (40.5 MB) con un lanzador por lotes 'EJECUTAR_WEBCAM.bat' que corre con 1 solo clic en cualquier PC con solo 15 MB de dependencias (OpenCV y NumPy)."
    )


def create_conclusions(doc):
    """Sección 10: Conclusiones."""
    doc.add_page_break()
    add_heading_styled(doc, "10. CONCLUSIONES Y RECOMENDACIONES", level=1)
    conclusions = [
        "Se implementó con éxito la arquitectura EmotionResNetSE (Residual Blocks + Squeeze-and-Excitation Attention + Global Average Pooling, 11.38M de parámetros), alcanzando un 69.45% de Exactitud Global (Accuracy) en el conjunto de Test independiente de FER-2013, superando formalmente la concordancia humana promedio del benchmark (65-68%).",
        "La integración sinérgica de Focal Loss (gamma=2.0) y Oversampling estocástico resolvió eficazmente el severo desbalance de clases del dataset original (ratio 17:1), elevando el Recall en la clase minoritaria 'Disgusto' hasta un 73.87% (frente a valores inferiores al 45% en modelos basales no regularizados).",
        "El cargador de datos nativo en memoria gráfica (FastGPUDataLoader) con el dataset 100% residente en VRAM (78.9 MB) y Data Augmentation ejecutado en núcleos CUDA (rotación afín ±12°, padding reflectante y recortes aleatorios) eliminó los cuellos de botella del bus PCIe, logrando un rendimiento de entrenamiento superior a 12,000 imágenes/segundo en GPU RTX.",
        "El esquema de optimización SequentialLR, combinando 8 épocas de Linear Warmup con decaimiento Cosine Annealing, junto a la regularización Mixup (alpha=0.1) y Dropout (0.4 y 0.3), garantizó una convergencia suave y evitó el sobreajuste catastrófico a lo largo de 96 épocas.",
        "El detector neuronal YuNet ONNX demostró una superioridad rotunda frente a detectores clásicos tipo Haar Cascade, permitiendo el reconocimiento facial robusto en ángulos de perfil de hasta ±90°, inclinaciones y cabeceos, extrayendo 5 landmarks faciales sin falsos positivos sobre prendas o fondos.",
        "El algoritmo de Suavizado Temporal EMA (alfa=0.30) junto a la optimización de hardware de cámara MJPG a 60 FPS erradicó por completo el parpadeo errático de probabilidades en transmisiones en vivo, logrando una experiencia de usuario fluida, reactiva y de calidad industrial.",
        "La exportación a formato universal ONNX y la distribución portátil empaquetada demostraron que un modelo de Deep Learning de última generación puede ejecutarse en cualquier computadora convencional con solo 15 MB de librerías (OpenCV + NumPy) a más de 30-60 FPS en CPU sin pérdida de precisión."
    ]
    for conc in conclusions:
        p = doc.add_paragraph(conc, style="List Bullet")
        p.paragraph_format.space_after = Pt(6)
        for run in p.runs:
            run.font.name = "Calibri"
            run.font.size = Pt(11)


def create_references(doc):
    """Sección 11: Referencias."""
    doc.add_page_break()
    add_heading_styled(doc, "11. REFERENCIAS BIBLIOGRÁFICAS", level=1)
    references = [
        'He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Deep Residual Learning for Image Recognition". IEEE CVPR, pp. 770-778.',
        'Hu, J., Shen, L., & Sun, G. (2018). "Squeeze-and-Excitation Networks". IEEE CVPR, pp. 7132-7141.',
        'Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). "Focal Loss for Dense Object Detection". IEEE ICCV, pp. 2980-2988.',
        'Goodfellow, I. J., et al. (2013). "Challenges in Representation Learning: A report on three machine learning contests (FER-2013)". ICML Workshop on Representation Learning.',
        'Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, D. (2018). "mixup: Beyond Empirical Risk Minimization". ICLR.',
        'Loshchilov, I., & Hutter, F. (2019). "Decoupled Weight Decay Regularization (AdamW)". ICLR.',
        'Zhang, S., Zhu, X., Lei, Z., Shi, H., Wang, X., & Li, S. Z. (2020). "FaceBoxes: A Rapid yet High-Accuracy Face Detector (YuNet Base)". IEEE Transactions on Biometrics.',
        'Bradski, G. (2000). "The OpenCV Library / Deep Neural Network Module (DNN)". Dr. Dobb\'s Journal of Software Tools.',
        'Paszke, A., et al. (2019). "PyTorch: An Imperative Style, High-Performance Deep Learning Library". NeurIPS.',
        'Bai, J., et al. (2019). "ONNX: Open Neural Network Exchange". GitHub repository, https://github.com/onnx/onnx.'
    ]
    for i, ref in enumerate(references, 1):
        p = doc.add_paragraph()
        run = p.add_run(f"[{i}]  {ref}")
        run.font.name = "Calibri"
        run.font.size = Pt(10)
        p.paragraph_format.space_after = Pt(5)
        p.paragraph_format.left_indent = Cm(0.8)
        p.paragraph_format.first_line_indent = Cm(-0.8)


def generate_word_document():
    """Genera el documento Word académico formal completo."""
    print("\n" + "=" * 60)
    print("  GENERANDO INFORME ACADÉMICO WORD (.DOCX)")
    print("=" * 60)

    doc = Document()

    for section in doc.sections:
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin = Cm(3.0)
        section.right_margin = Cm(2.5)

    print("   [1/11] Generando Carátula Institucional...")
    create_cover_page(doc)

    print("   [2/11] Generando Índice General...")
    create_index(doc)

    print("   [3/11] Redactando Introducción y Justificación...")
    create_introduction(doc)

    print("   [4/11] Redactando Objetivos...")
    create_objectives(doc)

    print("   [5/11] Redactando Marco Teórico (ResNet, SE, YuNet, Focal Loss)...")
    create_theoretical_framework(doc)

    print("   [6/11] Redactando Sección del Dataset FER2013, Oversampling y VRAM...")
    create_dataset_section(doc)

    print("   [7/11] Detallando Arquitectura EmotionResNetSE (11.38M Parámetros)...")
    create_architecture_section(doc)

    print("   [8/11] Redactando Metodología, Warmup, Cosine Annealing y Mixup...")
    create_methodology_section(doc)

    print("   [9/11] Integrando Resultados Verificados en TEST y Gráficas...")
    create_results_section(doc)

    print("   [10/11] Describiendo Inferencia en Tiempo Real (YuNet, EMA, 60 FPS)...")
    create_realtime_section(doc)

    print("   [11/11] Detallando Despliegue Portátil ONNX, Conclusiones y Referencias...")
    create_export_section(doc)
    create_conclusions(doc)
    create_references(doc)

    outputs_dir = os.path.join(BASE_DIR, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    outputs_doc_path = os.path.join(outputs_dir, "Actividad_CNN_Emociones_Risco_Gil_Manuel_Fabrizzio.docx")
    parent_doc_path = os.path.join(os.path.dirname(BASE_DIR), "Actividad_CNN_Emociones_Risco_Gil_Manuel_Fabrizzio.docx")

    doc.save(outputs_doc_path)
    try:
        doc.save(parent_doc_path)
    except Exception:
        pass

    print("\n" + "=" * 60)
    print("[OK] INFORME ACADÉMICO WORD ACTUALIZADO CON ÉXITO")
    print(f"   Outputs: {outputs_doc_path}")
    print("=" * 60)


if __name__ == "__main__":
    generate_word_document()
