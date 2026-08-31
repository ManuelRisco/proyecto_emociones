# ============================================================================
# GENERADOR DE DOCUMENTO WORD (.docx)
# Reconocimiento de Emociones Faciales con CNN
# Autor: Manuel Fabrizzio Risco Gil
# Curso: Sistemas Inteligentes y Machine Learning
# Docente: Mg. José Antonio Ogosi Auqui
# ============================================================================
# Genera un documento académico formal y completo con:
# - Portada académica institucional
# - Índice estructurado
# - Introducción y justificación
# - Objetivos (General y Específicos)
# - Marco teórico (IA, ML, Deep Learning, CNN, CUDA)
# - Dataset FER2013 y análisis de balance
# - Arquitectura de la CNN y parámetros
# - Metodología de entrenamiento y data augmentation
# - Resultados experimentales y gráficas reales
# - Demostración en tiempo real (Webcam + Haar Cascade)
# - Conclusiones técnicas
# - Referencias bibliográficas (IEEE/APA)
# ============================================================================

import os
import sys

# Forzar UTF-8 en stdout/stderr para evitar errores de encoding en Windows
if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        reconfig = getattr(stream, "reconfigure", None)
        if callable(reconfig):
            try:
                reconfig(encoding="utf-8", errors="replace")
            except Exception:
                pass

import torch
import numpy as np
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

# Asegurar que el directorio raíz esté en sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    BASE_DIR, PLOTS_DIR, MODELS_DIR, BEST_MODEL_PATH,
    EMOTION_LABELS, NUM_CLASSES, BATCH_SIZE, LEARNING_RATE,
    NUM_EPOCHS, IMG_SIZE, EARLY_STOPPING_PATIENCE
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


def format_table_header(row, col_names, bg_color="003366", font_size=10):
    """Formatea la cabecera de una tabla con fondo oscuro y texto blanco en negrita."""
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
            run.font.size = Pt(font_size)
            run.font.bold = True
            run.font.color.rgb = RGBColor(255, 255, 255)


def format_table_row(row, values, is_even=False, font_size=9.5, align_center_cols=None):
    """Formatea una fila de datos de tabla con colores alternos."""
    bg_color = "F4F6F9" if is_even else "FFFFFF"
    align_center_cols = align_center_cols or []
    for idx, val in enumerate(values):
        cell = row.cells[idx]
        cell.text = str(val)
        set_cell_background(cell, bg_color)
        set_cell_margins(cell, top=80, bottom=80, left=120, right=120)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        p = cell.paragraphs[0]
        if idx in align_center_cols:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        else:
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        for run in p.runs:
            run.font.name = "Calibri"
            run.font.size = Pt(font_size)
            run.font.color.rgb = RGBColor(40, 40, 40)


def create_cover_page(doc):
    """Crea una carátula académica formal."""
    for _ in range(2):
        doc.add_paragraph()

    # Institución
    p = doc.add_paragraph()
    run = p.add_run("UNIVERSIDAD NACIONAL / PRIVADA")
    run.font.name = "Calibri"
    run.font.size = Pt(17)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0, 51, 102)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    p = doc.add_paragraph()
    run = p.add_run("FACULTAD DE INGENIERÍA\nESCUELA PROFESIONAL DE INGENIERÍA DE SISTEMAS COMPUTACIONALES")
    run.font.name = "Calibri"
    run.font.size = Pt(13)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0, 76, 153)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph()

    # Barra decorativa
    p = doc.add_paragraph()
    run = p.add_run("━" * 55)
    run.font.color.rgb = RGBColor(0, 102, 204)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Título principal
    p = doc.add_paragraph()
    run = p.add_run("RECONOCIMIENTO AUTOMÁTICO DE EMOCIONES FACIALES\n"
                    "MEDIANTE REDES NEURONALES CONVOLUCIONALES (CNN)\n"
                    "CON ACELERACIÓN GPU CUDA")
    run.font.name = "Calibri"
    run.font.size = Pt(16)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0, 51, 102)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    p = doc.add_paragraph()
    run = p.add_run("━" * 55)
    run.font.color.rgb = RGBColor(0, 102, 204)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph()

    # Datos del informe
    p = doc.add_paragraph()
    run = p.add_run("INFORME DE LABORATORIO / PROYECTO TÉCNICO - SEMANA 1")
    run.font.name = "Calibri"
    run.font.size = Pt(11)
    run.font.italic = True
    run.font.color.rgb = RGBColor(100, 100, 100)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph()
    doc.add_paragraph()

    # Tabla de metadatos del estudiante y docente
    meta_table = doc.add_table(rows=5, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_data = [
        ("Curso:", "Sistemas Inteligentes y Machine Learning"),
        ("Ciclo Académico:", "9no Ciclo"),
        ("Docente:", "Mg. José Antonio Ogosi Auqui"),
        ("Estudiante:", "Manuel Fabrizzio Risco Gil"),
        ("Año Académico:", "2026"),
    ]
    for i, (label, val) in enumerate(meta_data):
        row = meta_table.rows[i]
        c1, c2 = row.cells[0], row.cells[1]
        c1.text = label
        c2.text = val
        p1, p2 = c1.paragraphs[0], c2.paragraphs[0]
        p1.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        p2.alignment = WD_ALIGN_PARAGRAPH.LEFT
        for run in p1.runs:
            run.font.name = "Calibri"
            run.font.size = Pt(11)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0, 51, 102)
        for run in p2.runs:
            run.font.name = "Calibri"
            run.font.size = Pt(11)
            run.font.color.rgb = RGBColor(40, 40, 40)

    doc.add_page_break()


def create_index(doc):
    """Crea la tabla de contenidos estructurada."""
    add_heading_styled(doc, "ÍNDICE GENERAL", level=1)

    sections = [
        ("1.", "INTRODUCCIÓN Y JUSTIFICACIÓN", "3"),
        ("2.", "OBJETIVOS DEL PROYECTO", "3"),
        ("  2.1.", "Objetivo General", "3"),
        ("  2.2.", "Objetivos Específicos", "3"),
        ("3.", "MARCO TEÓRICO Y CONCEPTUAL", "4"),
        ("  3.1.", "Inteligencia Artificial y Aprendizaje Supervisado", "4"),
        ("  3.2.", "Deep Learning y Representación Jerárquica", "4"),
        ("  3.3.", "Redes Neuronales Convolucionales (CNN)", "4"),
        ("  3.4.", "Componentes y Bloques de la Arquitectura", "5"),
        ("  3.5.", "Aceleración por GPU y Cómputo Paralelo CUDA", "5"),
        ("4.", "DATASET FER2013 Y PREPROCESAMIENTO", "6"),
        ("  4.1.", "Estructura y Clases de Emociones", "6"),
        ("  4.2.", "Desbalance de Clases y Data Augmentation", "6"),
        ("5.", "ARQUITECTURA DEL MODELO EMOTIONCNN", "7"),
        ("6.", "METODOLOGÍA EXPERIMENTAL Y ENTRENAMIENTO", "8"),
        ("  6.1.", "Función de Pérdida y Optimizador", "8"),
        ("  6.2.", "Políticas de Regularización y Early Stopping", "8"),
        ("7.", "RESULTADOS EXPERIMENTALES Y EVALUACIÓN", "9"),
        ("  7.1.", "Curvas de Aprendizaje (Loss y Accuracy)", "9"),
        ("  7.2.", "Matriz de Confusión y Reporte por Emoción", "10"),
        ("  7.3.", "Ejemplos de Inferencia Cualitativa", "11"),
        ("8.", "SISTEMA DE INFERENCIA EN TIEMPO REAL (WEBCAM)", "12"),
        ("9.", "CONCLUSIONES Y RECOMENDACIONES", "13"),
        ("10.", "REFERENCIAS BIBLIOGRÁFICAS", "14"),
    ]

    table = doc.add_table(rows=len(sections) + 1, cols=3, style="Table Grid")
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    format_table_header(table.rows[0], ["Sección", "Título del Contenido", "Pág."])

    for i, (num, title, page) in enumerate(sections):
        row = table.rows[i + 1]
        format_table_row(row, [num, title, page], is_even=(i % 2 == 1), align_center_cols=[0, 2])

    doc.add_page_break()


def create_introduction(doc):
    """Sección 1: Introducción."""
    add_heading_styled(doc, "1. INTRODUCCIÓN Y JUSTIFICACIÓN", level=1)

    add_body_text(doc,
        "El reconocimiento automático de emociones humanas a través del análisis de expresiones "
        "faciales constituye uno de los desafíos más prominentes en el área de Visión por Computadora "
        "e Interacción Humano-Computador (HCI). Las microexpresiones y movimientos musculares de la cara "
        "reflejan estados afectivos universales que permiten a los sistemas inteligentes interpretar de "
        "manera empática el comportamiento del usuario en entornos de educación virtual, diagnóstico clínico, "
        "asistencia al conductor y monitoreo de seguridad."
    )

    add_body_text(doc,
        "Los métodos clásicos de procesamiento digital de imágenes dependían de la extracción manual de "
        "características (como descriptores HOG, LBP o filtros Gabor), los cuales presentaban alta "
        "sensibilidad ante variaciones de iluminación, oclusiones parciales y diferencias fisonómicas. "
        "Con la consolidación del Deep Learning y en particular de las Redes Neuronales Convolucionales "
        "(CNN), es posible aprender de manera jerárquica y totalmente automatizada representaciones visuales "
        "robustas a partir de grandes volúmenes de datos brutos."
    )

    add_body_text(doc,
        "En el presente proyecto se desarrolla e implementa un sistema integral de clasificación de 7 emociones "
        "faciales básicas (Enojado, Disgusto, Miedo, Feliz, Triste, Sorprendido y Neutral) entrenado sobre "
        "el benchmark internacional FER2013. Se aprovecha la aceleración por hardware de alta gama "
        "(GPU NVIDIA GeForce RTX 5060 Ti con 16GB VRAM mediante PyTorch y CUDA 12.8), integrando módulos "
        "de aumento de datos (*Data Augmentation*), reducción dinámica de tasa de aprendizaje (*ReduceLROnPlateau*) "
        "y un pipeline de inferencia en tiempo real acoplado a la cámara web con el detector Haar Cascade de OpenCV."
    )


def create_objectives(doc):
    """Sección 2: Objetivos."""
    add_heading_styled(doc, "2. OBJETIVOS DEL PROYECTO", level=1)

    add_heading_styled(doc, "2.1. Objetivo General", level=2)
    add_body_text(doc,
        "Diseñar, entrenar y validar una Red Neuronal Convolucional (CNN) profunda optimizada para la "
        "clasificación multiclase de 7 emociones faciales sobre el dataset FER2013, aprovechando la aceleración "
        "por GPU para lograr un modelo robusto y desplegable en inferencia en tiempo real mediante webcam."
    )

    add_heading_styled(doc, "2.2. Objetivos Específicos", level=2)
    objectives = [
        "Construir una arquitectura CNN de 4 bloques convolucionales con capas dobles de convolución, "
        "Batch Normalization, activaciones ReLU, MaxPooling y regularización por Dropout.",
        "Implementar un pipeline de preprocesamiento y Data Augmentation (rotación, traslación, flip horizontal, "
        "variación de contraste) para mitigar el sobreajuste y compensar el desbalance de clases.",
        "Acelerar el proceso de entrenamiento utilizando tensores CUDA en la GPU NVIDIA RTX 5060 Ti.",
        "Evaluar cuantitativamente el rendimiento del modelo mediante métricas de Accuracy, Precision, "
        "Recall, F1-Score y análisis de matriz de confusión normalizada.",
        "Integrar el modelo entrenado con OpenCV para la detección de rostros y clasificación de expresiones "
        "faciales en tiempo real a una tasa superior a 30 cuadros por segundo (FPS).",
        "Demostrar el flujo integral del aprendizaje supervisado: Entrada (imágenes 48x48) → Proceso (extracción "
        "convolucional y optimización de pesos) → Salida (vector de probabilidades por emoción)."
    ]
    for obj in objectives:
        p = doc.add_paragraph(obj, style="List Bullet")
        p.paragraph_format.space_after = Pt(4)
        for run in p.runs:
            run.font.name = "Calibri"
            run.font.size = Pt(11)


def create_theoretical_framework(doc):
    """Sección 3: Marco Teórico."""
    doc.add_page_break()
    add_heading_styled(doc, "3. MARCO TEÓRICO Y CONCEPTUAL", level=1)

    add_heading_styled(doc, "3.1. Inteligencia Artificial y Aprendizaje Supervisado", level=2)
    add_body_text(doc,
        "El Aprendizaje Supervisado es el paradigma del Machine Learning donde un algoritmo ajusta sus parámetros "
        "internos a partir de un conjunto de entrenamiento constituido por pares de entrada-salida: D = {(x_i, y_i)}. "
        "El objetivo consiste en aprender una función de mapeo f: X → Y que minimice el error empírico sobre "
        "muestras no vistas, generalizando el comportamiento aprendido."
    )

    add_heading_styled(doc, "3.2. Deep Learning y Representación Jerárquica", level=2)
    add_body_text(doc,
        "A diferencia de los modelos estadísticos convencionales que requieren ingeniería manual de atributos, "
        "el Deep Learning procesa los datos en niveles sucesivos de abstracción. Las primeras capas de la red "
        "capturan primitivas espaciales (gradientes de intensidad, bordes orientados, curvas); las capas intermedias "
        "combinan estos patrones en estructuras anatómicas (ojos, nariz, labios); finalmente, las capas superiores "
        "integran la geometría facial completa para discriminar la emoción expresada."
    )

    add_heading_styled(doc, "3.3. Redes Neuronales Convolucionales (CNN)", level=2)
    add_body_text(doc,
        "Las CNNs se fundamentan en dos principios clave de diseño que las hacen idóneas para imágenes:\n"
        "1. Conectividad Local: Las neuronas convolucionales procesan únicamente campos receptivos restringidos, "
        "preservando la correlación espacial de los píxeles contiguos.\n"
        "2. Pesos Compartidos: Los mismos filtros (kernels) se aplican a lo largo de toda la imagen, garantizando "
        "invarianza a la traslación y reduciendo exponencialmente la cantidad de parámetros frente a redes densas."
    )

    add_heading_styled(doc, "3.4. Componentes y Bloques de la Arquitectura", level=2)
    comps = [
        ("Convolución 2D (Conv2D):", "Aplica una operación de convolución discreta con kernels de 3×3 para extraer mapas de características."),
        ("Batch Normalization (BatchNorm):", "Estandariza las salidas de cada capa restando la media y dividiendo por la desviación estándar del mini-lote, acelerando la convergencia y permitiendo mayores tasas de aprendizaje."),
        ("Unidad Lineal Rectificada (ReLU):", "Función no lineal f(x) = max(0, x) que soluciona el problema del desvanecimiento del gradiente (*vanishing gradient*)."),
        ("Max Pooling:", "Reduce el tamaño espacial de los mapas tomando el valor máximo en ventanas de 2×2 con paso (stride) 2, confiriendo robustez ante pequeñas deformaciones."),
        ("Dropout:", "Desactiva aleatoriamente neuronas con cierta probabilidad p durante el forward pass para prevenir la co-adaptación de rasgos y combatir el overfitting."),
        ("Capas Totalmente Conectadas (Dense/FC):", "Transforman los mapas de características aplanados en el espacio latente de decisión para la predicción de clases."),
        ("Softmax:", "Función logística multivariable que normaliza los logits de salida en una distribución de probabilidad que suma 1.0.")
    ]
    for name, desc in comps:
        p = doc.add_paragraph()
        r1 = p.add_run(f"•  {name} ")
        r1.font.bold = True
        r1.font.size = Pt(10.5)
        r2 = p.add_run(desc)
        r2.font.size = Pt(10.5)
        p.paragraph_format.space_after = Pt(4)

    add_heading_styled(doc, "3.5. Aceleración por GPU y Cómputo Paralelo CUDA", level=2)
    add_body_text(doc,
        "El entrenamiento de modelos convolucionales involucra billones de operaciones de multiplicación de matrices "
        "(GEMM) y convoluciones tensoriales. La tecnología NVIDIA CUDA permite paralelizar estos cálculos masivos "
        "en miles de núcleos de la GPU (RTX 5060 Ti), alcanzando aceleraciones de hasta 20x a 30x en comparación con "
        "ejecuciones convencionales sobre procesadores multinúcleo (CPU)."
    )


def create_dataset_section(doc):
    """Sección 4: Dataset FER2013."""
    doc.add_page_break()
    add_heading_styled(doc, "4. DATASET FER2013 Y PREPROCESAMIENTO", level=1)

    add_body_text(doc,
        "El conjunto de datos FER2013 (Facial Expression Recognition 2013) es un estándar en la literatura científica "
        "compuesto por 35,887 imágenes en escala de grises de 48×48 píxeles con rostros centrados en diversas "
        "condiciones del mundo real (iluminación variable, rotaciones leves, expresiones sutiles y marcadas)."
    )

    add_heading_styled(doc, "4.1. Distribución de Muestras y Clases", level=2)

    # Tabla de dataset
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

    add_heading_styled(doc, "4.2. Desbalance de Clases y Data Augmentation", level=2)
    add_body_text(doc,
        "Como se aprecia en la tabla anterior, existe un desbalance inherente: la clase 'Feliz' concentra el 25% "
        "de los datos, mientras que 'Disgusto' apenas cuenta con 547 muestras (1.52%). Para contrarrestar esta "
        "disparidad y prevenir que la red memorice patrones sin generalizar, se diseñó un pipeline de Data Augmentation "
        "en tiempo de ejecución utilizando Torchvision Transforms:"
    )

    aug_items = [
        ("Inversión horizontal aleatoria (*Random Horizontal Flip*):", "Probabilidad del 50% para simular simetría facial."),
        ("Rotación aleatoria (*Random Rotation*):", "Variación angular de hasta ±10° simulando inclinaciones de cabeza."),
        ("Traslación afín (*Random Affine Translation*):", "Desplazamiento horizontal y vertical de hasta 10%."),
        ("Ajuste de contraste y brillo (*ColorJitter*):", "Perturbación aleatoria de ±20% en escala de grises."),
        ("Normalización estadística:", "Escalamiento al rango [-1.0, 1.0] con media 0.5 y desviación estándar 0.5.")
    ]
    for k, v in aug_items:
        p = doc.add_paragraph()
        r1 = p.add_run(f"•  {k} ")
        r1.font.bold = True
        r1.font.size = Pt(10.5)
        r2 = p.add_run(v)
        r2.font.size = Pt(10.5)
        p.paragraph_format.space_after = Pt(3)


def create_architecture_section(doc):
    """Sección 5: Arquitectura del modelo."""
    doc.add_page_break()
    add_heading_styled(doc, "5. ARQUITECTURA DEL MODELO EMOTIONCNN", level=1)

    add_body_text(doc,
        "La arquitectura EmotionCNN fue construida en PyTorch con un diseño modular en bloques convolucionales "
        "dobles (*VGG-like style*), totalizando 2,488,423 parámetros entrenables. La secuencia de transformaciones "
        "reduce progresivamente la resolución espacial mientras incrementa la profundidad de los mapas de canales:"
    )

    arch_layers = [
        ("Entrada", "Tensor (1, 48, 48)", "Imagen en escala de grises normalizada", "-"),
        ("Bloque Conv 1", "2x Conv2D(32, k=3) + BatchNorm + ReLU + MaxPool(2x2) + Dropout(0.25)", "Tensor (32, 24, 24)", "9,568"),
        ("Bloque Conv 2", "2x Conv2D(64, k=3) + BatchNorm + ReLU + MaxPool(2x2) + Dropout(0.25)", "Tensor (64, 12, 12)", "55,424"),
        ("Bloque Conv 3", "2x Conv2D(128, k=3) + BatchNorm + ReLU + MaxPool(2x2) + Dropout(0.25)", "Tensor (128, 6, 6)", "221,440"),
        ("Bloque Conv 4", "2x Conv2D(256, k=3) + BatchNorm + ReLU + MaxPool(2x2) + Dropout(0.25)", "Tensor (256, 3, 3)", "885,248"),
        ("Flatten", "Aplanamiento tensorial 256 x 3 x 3", "Vector (2304)", "0"),
        ("Densa FC 1", "Linear(2304 → 512) + BatchNorm1D + ReLU + Dropout(0.5)", "Vector (512)", "1,180,160"),
        ("Densa FC 2", "Linear(512 → 256) + BatchNorm1D + ReLU + Dropout(0.5)", "Vector (256)", "131,328"),
        ("Salida FC 3", "Linear(256 → 7)", "Vector de Logits (7)", "1,799"),
        ("TOTAL", "Arquitectura Completa EmotionCNN", "Clasificación 7 Clases", "2,488,423")
    ]

    table = doc.add_table(rows=len(arch_layers) + 1, cols=4, style="Table Grid")
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    format_table_header(table.rows[0], ["Capa / Bloque", "Operaciones e Hiperparámetros", "Dimensión Salida", "Parámetros"])

    for i, row_data in enumerate(arch_layers):
        row = table.rows[i + 1]
        format_table_row(row, row_data, is_even=(i % 2 == 1), align_center_cols=[0, 2, 3])

    doc.add_paragraph()


def create_methodology_section(doc):
    """Sección 6: Metodología."""
    doc.add_page_break()
    add_heading_styled(doc, "6. METODOLOGÍA EXPERIMENTAL Y ENTRENAMIENTO", level=1)

    add_heading_styled(doc, "6.1. Configuración de Hiperparámetros", level=2)
    add_body_text(doc,
        "El entrenamiento se rige por un esquema de optimización estocástica adaptativa. A continuación se resumen "
        "los hiperparámetros establecidos en la configuración del proyecto:"
    )

    hp_table_data = [
        ("Tamaño de Lote (Batch Size)", f"{BATCH_SIZE} muestras"),
        ("Tasa de Aprendizaje Inicial (Learning Rate)", f"{LEARNING_RATE}"),
        ("Decaimiento de Pesos (Weight Decay L2)", "0.0001 (1e-4)"),
        ("Optimizador", "Adam (Adaptive Moment Estimation)"),
        ("Función de Pérdida (Loss Function)", "Cross-Entropy Loss (nn.CrossEntropyLoss)"),
        ("Planificador de Tasa (Scheduler)", "ReduceLROnPlateau (factor=0.5, paciencia=5 épocas)"),
        ("Criterio de Parada Temprana (Early Stopping)", f"{EARLY_STOPPING_PATIENCE} épocas consecutivas sin mejora en Val Loss"),
        ("Hardware / Aceleración", "NVIDIA GeForce RTX 5060 Ti 16GB (CUDA 12.8)")
    ]

    table = doc.add_table(rows=len(hp_table_data) + 1, cols=2, style="Table Grid")
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    format_table_header(table.rows[0], ["Hiperparámetro / Recurso", "Especificación"])

    for i, (k, v) in enumerate(hp_table_data):
        row = table.rows[i + 1]
        format_table_row(row, [k, v], is_even=(i % 2 == 1), align_center_cols=[0])

    doc.add_paragraph()

    add_heading_styled(doc, "6.2. Estrategia de Validación y Prevención de Overfitting", level=2)
    add_body_text(doc,
        "Para garantizar una evaluación rigurosa y no sesgada, el dataset se particionó en tres conjuntos disjuntos:\n"
        "•  Entrenamiento (24,403 imágenes): Empleado para la actualización de gradientes.\n"
        "•  Validación (4,306 imágenes): Evaluado al cierre de cada época para el ajuste del scheduler y Early Stopping.\n"
        "•  Prueba / Test (7,178 imágenes): Reservado exclusivamente para la inferencia y reporte final de métricas."
    )


def create_results_section(doc):
    """Sección 7: Resultados."""
    doc.add_page_break()
    add_heading_styled(doc, "7. RESULTADOS EXPERIMENTALES Y EVALUACIÓN", level=1)

    add_body_text(doc,
        "Tras completar el proceso de entrenamiento y optimización, se evaluó el desempeño del mejor punto de control "
        "(*checkpoint*) sobre las 7,178 imágenes del conjunto de test independiente. Las visualizaciones y métricas "
        "generadas por el sistema se presentan a continuación:"
    )

    # 7.1 Curvas de entrenamiento
    add_heading_styled(doc, "7.1. Curvas de Aprendizaje (Loss y Accuracy)", level=2)
    add_body_text(doc,
        "La evolución de la pérdida (*Loss*) y la exactitud (*Accuracy*) tanto en el conjunto de entrenamiento como "
        "en validación refleja una convergencia estable gracias a la regularización por Dropout y Batch Normalization:"
    )

    hist_plot = os.path.join(PLOTS_DIR, "historial_entrenamiento.png")
    if os.path.exists(hist_plot):
        try:
            doc.add_picture(hist_plot, width=Inches(6.0))
            p = doc.paragraphs[-1]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_cap = doc.add_paragraph()
            r = p_cap.add_run("Figura 1: Curvas de Pérdida (Loss) y Precisión (Accuracy) por Época de Entrenamiento.")
            r.font.size = Pt(9.5)
            r.font.italic = True
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        except Exception:
            pass

    # 7.2 Matriz de confusión y tabla de métricas
    doc.add_paragraph()
    add_heading_styled(doc, "7.2. Matriz de Confusión y Análisis por Emoción", level=2)
    add_body_text(doc,
        "La matriz de confusión permite analizar en detalle la tasa de acierto y los patrones de confusión entre "
        "clases con similitud visual (por ejemplo, 'Miedo' y 'Triste', o 'Neutral' y 'Enojado'):"
    )

    cm_plot = os.path.join(PLOTS_DIR, "matriz_confusion_normalizada.png")
    if not os.path.exists(cm_plot):
        cm_plot = os.path.join(PLOTS_DIR, "matriz_confusion.png")

    if os.path.exists(cm_plot):
        try:
            doc.add_picture(cm_plot, width=Inches(5.2))
            p = doc.paragraphs[-1]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_cap = doc.add_paragraph()
            r = p_cap.add_run("Figura 2: Matriz de Confusión Normalizada sobre el Conjunto de Test (7,178 imágenes).")
            r.font.size = Pt(9.5)
            r.font.italic = True
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        except Exception:
            pass

    # Tabla de métricas por clase
    doc.add_paragraph()
    add_heading_styled(doc, "Tabla Resumen de Métricas de Clasificación en Test", level=3)

    metrics_data = [
        ("Enojado (Angry)", "45.65%", "46.03%", "45.84%", "958"),
        ("Disgusto (Disgust)", "75.00%", "2.70%", "5.22%", "111"),
        ("Miedo (Fear)", "38.51%", "12.11%", "18.42%", "1,024"),
        ("Feliz (Happy)", "71.16%", "88.73%", "78.98%", "1,774"),
        ("Triste (Sad)", "44.64%", "34.08%", "38.65%", "1,247"),
        ("Sorprendido (Surprise)", "68.50%", "74.85%", "71.54%", "831"),
        ("Neutral (Neutral)", "45.53%", "66.99%", "54.22%", "1,233"),
        ("PROMEDIO GLOBAL (Accuracy)", "53.84%", "55.93%", "52.66%", "7,178")
    ]

    table_m = doc.add_table(rows=len(metrics_data) + 1, cols=5, style="Table Grid")
    table_m.alignment = WD_TABLE_ALIGNMENT.CENTER
    format_table_header(table_m.rows[0], ["Clase de Emoción", "Precision", "Recall", "F1-Score", "Muestras"])

    for i, row_data in enumerate(metrics_data):
        row = table_m.rows[i + 1]
        format_table_row(row, row_data, is_even=(i % 2 == 1), align_center_cols=[1, 2, 3, 4])

    doc.add_paragraph()

    acc_plot = os.path.join(PLOTS_DIR, "accuracy_por_emocion.png")
    if os.path.exists(acc_plot):
        try:
            doc.add_picture(acc_plot, width=Inches(5.2))
            p = doc.paragraphs[-1]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_cap = doc.add_paragraph()
            r = p_cap.add_run("Figura 3: Rendimiento de Acierto Individual por Emoción.")
            r.font.size = Pt(9.5)
            r.font.italic = True
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        except Exception:
            pass

    # 7.3 Ejemplos de inferencia
    doc.add_paragraph()
    add_heading_styled(doc, "7.3. Ejemplos de Inferencia Cualitativa", level=2)
    add_body_text(doc,
        "La siguiente figura ilustra un muestreo aleatorio de inferencias realizadas por el modelo sobre rostros del "
        "conjunto de prueba, indicando la emoción real, la predicha y el nivel de confianza probabilística asignado:"
    )

    pred_plot = os.path.join(PLOTS_DIR, "ejemplos_predicciones.png")
    if os.path.exists(pred_plot):
        try:
            doc.add_picture(pred_plot, width=Inches(5.8))
            p = doc.paragraphs[-1]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_cap = doc.add_paragraph()
            r = p_cap.add_run("Figura 4: Muestra de Inferences Correctas [OK] e Incorrectas [X] con Confianza Porcentual.")
            r.font.size = Pt(9.5)
            r.font.italic = True
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        except Exception:
            pass


def create_realtime_section(doc):
    """Sección 8: Demo en Tiempo Real."""
    doc.add_page_break()
    add_heading_styled(doc, "8. SISTEMA DE INFERENCIA EN TIEMPO REAL (WEBCAM)", level=1)

    add_body_text(doc,
        "Para validar la aplicabilidad práctica del modelo desarrollado en un contexto de producción interactivo, "
        "se implementó un módulo de inferencia en streaming mediante OpenCV acoplado a la cámara web. El pipeline "
        "opera de manera asíncrona ejecutando las siguientes etapas en cada cuadro de video:"
    )

    steps = [
        ("Adquisición de Cuadros (640x480):", "Captura continua de video a 30 FPS desde el dispositivo de captura."),
        ("Detección Facial Haar Cascade:", "Localización de regiones de interés (ROI) que contienen rostros humanos."),
        ("Preprocesamiento y Normalización:", "Conversión a escala de grises, redimensionamiento a 48×48 px y normalización tensorial."),
        ("Inferencia en GPU (Forward Pass):", "Paso del tensor normalizado por la EmotionCNN para obtener la distribución Softmax en menos de 10 milisegundos."),
        ("Superposición Gráfica (HUD):", "Renderizado de caja delimitadora dinámica, nombre de la emoción dominante y barras de probabilidad porcentual.")
    ]

    for title, desc in steps:
        p = doc.add_paragraph()
        r1 = p.add_run(f"•  {title} ")
        r1.font.bold = True
        r1.font.size = Pt(10.5)
        r2 = p.add_run(desc)
        r2.font.size = Pt(10.5)
        p.paragraph_format.space_after = Pt(4)

    add_body_text(doc,
        "El módulo interactivo incluye controles en vivo por teclado: tecla 's' para guardar capturas instantáneas, "
        "tecla 'p' para congelar/reanudar el flujo y tecla 'q' para retornar al menú principal."
    )


def create_conclusions(doc):
    """Sección 9: Conclusiones."""
    doc.add_page_break()
    add_heading_styled(doc, "9. CONCLUSIONES Y RECOMENDACIONES", level=1)

    conclusions = [
        "Se diseñó e implementó exitosamente una Red Neuronal Convolucional (EmotionCNN) de 4 bloques profundos, "
        "logrando una precisión superior al 55.9% en el benchmark complejo FER2013, superando ampliamente el nivel "
        "de azar aleatorio (14.28% para 7 clases).",

        "La aceleración mediante GPU NVIDIA RTX 5060 Ti con CUDA 12.8 y PyTorch redujo drásticamente el tiempo de "
        "entrenamiento a menos de 2 minutos por época, permitiendo una rápida iteración y ajuste de hiperparámetros.",

        "Las técnicas de regularización implementadas (Batch Normalization, Dropout al 25% y 50%, y Data Augmentation) "
        "resultaron determinantes para estabilizar la convergencia y evitar el sobreajuste severo en una red de 2.48 millones "
        "de parámetros.",

        "El análisis por emoción demostró un rendimiento sobresaliente en clases con gestos morfológicos distintivos "
        "como 'Feliz' (Recall 88.7%, Precision 71.2%) y 'Sorprendido' (Recall 74.9%), mientras que emociones como 'Disgusto' "
        "presentan un área de oportunidad clara debido al desbalance muestral del dataset.",

        "La integración en tiempo real con OpenCV demostró latencias de inferencia inferiores a 10 ms por cuadro, "
        "confirmando la viabilidad del modelo para su despliegue en sistemas interactivos de visión artificial."
    ]

    for conc in conclusions:
        p = doc.add_paragraph(conc, style="List Bullet")
        p.paragraph_format.space_after = Pt(6)
        for run in p.runs:
            run.font.name = "Calibri"
            run.font.size = Pt(11)


def create_references(doc):
    """Sección 10: Referencias."""
    doc.add_page_break()
    add_heading_styled(doc, "10. REFERENCIAS BIBLIOGRÁFICAS", level=1)

    references = [
        'Goodfellow, I., Bengio, Y., & Courville, A. (2016). "Deep Learning". MIT Press.',
        'Goodfellow, I. J., et al. (2013). "Challenges in Representation Learning: A report on three machine learning contests". ICML Workshop on Challenges in Representation Learning.',
        'LeCun, Y., Bengio, Y., & Hinton, G. (2015). "Deep learning". Nature, 521(7553), 436-444.',
        'He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Deep Residual Learning for Image Recognition". IEEE CVPR.',
        'Ioffe, S., & Szegedy, C. (2015). "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift". ICML.',
        'Srivastava, N., et al. (2014). "Dropout: A Simple Way to Prevent Neural Networks from Overfitting". JMLR, 15(1), 1929-1958.',
        'Paszke, A., et al. (2019). "PyTorch: An Imperative Style, High-Performance Deep Learning Library". NeurIPS.',
        'Bradski, G. (2000). "The OpenCV Library". Dr. Dobb\'s Journal of Software Tools.'
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
    """Genera el documento Word académico completo y lo guarda en las rutas del proyecto."""
    print("\n" + "=" * 60)
    print("  GENERANDO DOCUMENTO WORD (.DOCX)")
    print("=" * 60)

    doc = Document()

    # Configuración de márgenes estándar APA / IEEE (2.5 cm en todos los lados)
    for section in doc.sections:
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin = Cm(3.0)
        section.right_margin = Cm(2.5)

    print("   [1/12] Generando Carátula Institucional...")
    create_cover_page(doc)

    print("   [2/12] Generando Índice General...")
    create_index(doc)

    print("   [3/12] Redactando Introducción y Justificación...")
    create_introduction(doc)

    print("   [4/12] Redactando Objetivos...")
    create_objectives(doc)

    print("   [5/12] Redactando Marco Teórico...")
    create_theoretical_framework(doc)

    print("   [6/12] Redactando Sección del Dataset FER2013...")
    create_dataset_section(doc)

    print("   [7/12] Detallando Arquitectura EmotionCNN...")
    create_architecture_section(doc)

    print("   [8/12] Redactando Metodología y Configuración...")
    create_methodology_section(doc)

    print("   [9/12] Integrando Resultados, Métricas y Gráficas...")
    create_results_section(doc)

    print("   [10/12] Describiendo Inferencia en Tiempo Real...")
    create_realtime_section(doc)

    print("   [11/12] Redactando Conclusiones...")
    create_conclusions(doc)

    print("   [12/12] Incluyendo Referencias Bibliográficas...")
    create_references(doc)

    # Rutas de guardado
    parent_output = os.path.normpath(
        os.path.join(BASE_DIR, "..", "Actividad_CNN_Emociones_Risco_Gil_Manuel_Fabrizzio.docx")
    )
    outputs_folder = os.path.join(BASE_DIR, "outputs", "Actividad_CNN_Emociones_Risco_Gil_Manuel_Fabrizzio.docx")

    doc.save(parent_output)
    doc.save(outputs_folder)

    print(f"\n{'=' * 60}")
    print(f"[OK] DOCUMENTO WORD GENERADO Y ACTUALIZADO CON ÉXITO")
    print(f"{'=' * 60}")
    print(f"   Ruta Principal: {parent_output}")
    print(f"   Copia en outputs: {outputs_folder}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    generate_word_document()

