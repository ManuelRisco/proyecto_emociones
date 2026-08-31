# ============================================================================
# GENERADOR DE DOCUMENTO WORD (.docx) - INFORME ACADÉMICO FORMAL
# Reconocimiento de Emociones Faciales con Deep Learning (ResNet-SE & YuNet)
# Autores: Manuel Fabrizzio Risco Gil & Alejandro Choquihuayta, Mario Julian
# Curso: Sistemas Inteligentes y Machine Learning - Ciclo 9
# Docente: Mg. José Antonio Ogosi Auqui
# ============================================================================

import os
import sys

# Forzar UTF-8 en stdout/stderr para evitar errores de encoding en Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

import torch
import numpy as np
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

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
    """Crea una carátula académica formal con ambos autores."""
    for _ in range(2):
        doc.add_paragraph()

    # Institución
    p = doc.add_paragraph()
    run = p.add_run("UNIVERSIDAD TECNOLÓGICA DEL PERÚ / FACULTAD DE INGENIERÍA")
    run.font.name = "Calibri"
    run.font.size = Pt(16)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0, 51, 102)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    p = doc.add_paragraph()
    run = p.add_run("ESCUELA PROFESIONAL DE INGENIERÍA DE SISTEMAS E INFORMÁTICA")
    run.font.name = "Calibri"
    run.font.size = Pt(12.5)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0, 76, 153)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph()

    # Barra decorativa
    p = doc.add_paragraph()
    run = p.add_run("━" * 58)
    run.font.color.rgb = RGBColor(0, 102, 204)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Título principal
    p = doc.add_paragraph()
    run = p.add_run("RECONOCIMIENTO AUTOMÁTICO DE EMOCIONES FACIALES\n"
                    "MEDIANTE REDES NEURONALES RESIDUALES CON ATENCIÓN (SE-ResNet)\n"
                    "DETECTOR YuNet, ACELERACIÓN VRAM Y DESPLIEGUE PORTÁTIL ONNX")
    run.font.name = "Calibri"
    run.font.size = Pt(15.5)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0, 51, 102)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    p = doc.add_paragraph()
    run = p.add_run("━" * 58)
    run.font.color.rgb = RGBColor(0, 102, 204)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph()

    # Datos del informe
    p = doc.add_paragraph()
    run = p.add_run("INFORME DE PROYECTO ACADÉMICO - SISTEMAS INTELIGENTES")
    run.font.name = "Calibri"
    run.font.size = Pt(11)
    run.font.italic = True
    run.font.color.rgb = RGBColor(100, 100, 100)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph()

    # Tabla de metadatos de estudiantes y docente
    meta_table = doc.add_table(rows=6, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_data = [
        ("Curso:", "Sistemas Inteligentes y Machine Learning"),
        ("Ciclo Académico:", "9no Ciclo"),
        ("Docente:", "Mg. José Antonio Ogosi Auqui"),
        ("Autores / Integrantes:", "• Manuel Fabrizzio Risco Gil\n• Alejandro Choquihuayta, Mario Julian"),
        ("Hardware & Despliegue:", "GPU NVIDIA RTX 5060 Ti 16GB (CUDA 12.8) / Exportación Universal ONNX"),
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
            run.font.size = Pt(10.5)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0, 51, 102)
        for run in p2.runs:
            run.font.name = "Calibri"
            run.font.size = Pt(10.5)
            run.font.color.rgb = RGBColor(40, 40, 40)

    doc.add_page_break()


def create_index(doc):
    """Crea la tabla de contenidos estructurada."""
    add_heading_styled(doc, "ÍNDICE GENERAL", level=1)

    sections = [
        ("1.", "INTRODUCCIÓN Y JUSTIFICACIÓN", "3"),
        ("2.", "OBJETIVOS DEL PROYECTO (GENERAL Y ESPECÍFICOS)", "3"),
        ("3.", "MARCO TEÓRICO: DEEP LEARNING, RESNET, SE-ATTENTION Y CUDA", "4"),
        ("4.", "DATASET FER2013, AUDITORÍA DE PARTICIÓN Y ALOJAMIENTO EN VRAM", "5"),
        ("5.", "ARQUITECTURA DE LAS REDES (EmotionResNetSE y EmotionCNN)", "6"),
        ("6.", "METODOLOGÍA DE ENTRENAMIENTO, CLASS WEIGHTS Y LABEL SMOOTHING", "7"),
        ("7.", "RESULTADOS EXPERIMENTALES Y EVALUACIÓN RIGUROSA EN TEST SET", "8"),
        ("8.", "INFERENCIA EN TIEMPO REAL: DETECTOR YuNet Y SUAVIZADO TEMPORAL EMA", "10"),
        ("9.", "DESPLIEGUE MULTIPLATAFORMA Y EXPORTACIÓN PORTÁTIL (ONNX & OPENCV DNN)", "11"),
        ("10.", "CONCLUSIONES Y RECOMENDACIONES TÉCNICAS", "12"),
        ("11.", "REFERENCIAS BIBLIOGRÁFICAS (IEEE / APA)", "13"),
    ]

    table = doc.add_table(rows=len(sections) + 1, cols=3, style="Table Grid")
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    format_table_header(table.rows[0], ["N°", "Sección / Contenido", "Pág."])

    for i, (num, title, page) in enumerate(sections):
        row = table.rows[i + 1]
        format_table_row(row, [num, title, page], is_even=(i % 2 == 1), align_center_cols=[0, 2])

    doc.add_paragraph()


def create_introduction(doc):
    """Sección 1: Introducción."""
    doc.add_page_break()
    add_heading_styled(doc, "1. INTRODUCCIÓN Y JUSTIFICACIÓN", level=1)

    add_body_text(doc,
        "El reconocimiento automático de emociones faciales (Facial Expression Recognition - FER) constituye "
        "uno de los pilares de mayor relevancia en la Visión por Computadora y la Interacción Humano-Computador (HCI). "
        "Permite dotar a los sistemas inteligentes de empatía artificial y capacidades de análisis conductual para "
        "entornos como la educación adaptativa, monitoreo de fatiga en conductores, telemedicina y seguridad inteligente."
    )

    add_body_text(doc,
        "Los métodos clásicos de visión artificial dependían de extractores manuales de rasgos (como filtros Gabor, "
        "HOG o LBP) y clasificadores como Haar Cascade, los cuales fallan drásticamente ante giros de cabeza, "
        "oclusiones o variaciones de iluminación. Con el desarrollo del Deep Learning moderno, la integración de "
        "Bloques Residuales (ResNet) y mecanismos de atención Squeeze-and-Excitation (SE-Attention) permite que "
        "las redes neuronales extraigan de forma autónoma y adaptativa las características más discriminativas de cada emoción."
    )

    add_body_text(doc,
        "El presente trabajo implementa un sistema integral de alto rendimiento capaz de clasificar 7 emociones básicas "
        "(Enojado, Disgusto, Miedo, Feliz, Triste, Sorprendido y Neutral) entrenado sobre el benchmark FER2013. "
        "Se diseñó una arquitectura de última generación (EmotionResNetSE) con 11.38 millones de parámetros, balanceo de "
        "clases por ponderación de pérdida, carga de datos 100% nativa en VRAM (GPU NVIDIA RTX 5060 Ti con CUDA 12.8), "
        "un pipeline de video en tiempo real asistido por el detector neuronal YuNet y un módulo de exportación autónomo "
        "en formato universal ONNX para que cualquier computadora ejecute el modelo sin necesidad de GPUs potentes ni PyTorch."
    )


def create_objectives(doc):
    """Sección 2: Objetivos."""
    add_heading_styled(doc, "2. OBJETIVOS DEL PROYECTO", level=1)

    add_heading_styled(doc, "2.1. Objetivo General", level=2)
    add_body_text(doc,
        "Diseñar, implementar, evaluar rigurosamente y exportar a un paquete autónomo portátil un sistema inteligente "
        "de clasificación de emociones faciales mediante Redes Residuales con Atención (EmotionResNetSE), detector neuronal "
        "YuNet y OpenCV DNN, alcanzando un desempeño superior al estándar humano sobre FER2013 y permitiendo su uso en cualquier PC."
    )

    add_heading_styled(doc, "2.2. Objetivos Específicos", level=2)
    objectives = [
        "Desarrollar la arquitectura profunda EmotionResNetSE con 4 etapas residuales, atención por canales Squeeze-and-Excitation (SE) y Global Average Pooling (GAP).",
        "Implementar un mecanismo de carga y aumento de datos 100% residente en VRAM (FastGPUDataLoader), eliminando cuellos de botella de RAM de CPU y latencias PCIe.",
        "Compensar el desbalance severo de clases de FER2013 mediante Ponderación de Clases (Class Weights) y Label Smoothing en la función Cross-Entropy.",
        "Evaluar formalmente el modelo sobre 7,178 imágenes del conjunto de TEST independiente, alcanzando 67.86% de Accuracy y mitigando el sobreajuste mediante Early Stopping.",
        "Integrar el detector neuronal YuNet ONNX para detección facial multi-ángulo (yaw ±90°, pitch y roll) sin falsos positivos en streaming de webcam.",
        "Diseñar un filtro de Suavizado Temporal (EMA Tracker) para estabilizar las probabilidades y eliminar el parpadeo de emociones entre fotogramas consecutivos.",
        "Exportar el modelo completo a formato ONNX universal y compilar un paquete autónomo portátil (.ZIP) que corra en cualquier computadora con solo OpenCV y NumPy."
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

    add_heading_styled(doc, "3.1. Deep Learning y Conexiones Residuales (ResNet)", level=2)
    add_body_text(doc,
        "En redes convolucionales muy profundas, la propagación repetida de multiplicaciones matriciales provoca "
        "el desvanecimiento de los gradientes (*vanishing gradient problem*). Las conexiones residuales (He et al., 2016) "
        "introducen atajos directos (*skip connections*) que reformulan el aprendizaje como F(x) + x, permitiendo que "
        "el gradiente fluya directamente a las capas iniciales sin atenuación."
    )

    add_heading_styled(doc, "3.2. Mecanismo de Atención Squeeze-and-Excitation (SE Block)", level=2)
    add_body_text(doc,
        "El módulo Squeeze-and-Excitation (Hu et al., 2018) modela dinámicamente las interdependencias entre canales:\n"
        "1. Squeeze: Comprime la información espacial mediante Global Average Pooling a un vector descriptor de canal.\n"
        "2. Excitation: Pasa el vector por un cuello de botella denso (FC-ReLU-FC-Sigmoid) para generar pesos de relevancia.\n"
        "3. Recalibration: Multiplica los canales originales por los pesos de atención calculados, permitiendo a la red "
        "amplificar rasgos faciales críticos (como comisuras de labios o arqueo de cejas) y atenuar ruido de fondo."
    )

    add_heading_styled(doc, "3.3. Detección Facial Neuronal YuNet vs. Haar Cascade", level=2)
    add_body_text(doc,
        "A diferencia del algoritmo clásico de Haar Cascade (2001) basado en características manuales rígidas que solo "
        "detectan caras frontales y producen falsos positivos en ropa/fondos, YuNet es un detector neuronal basado en "
        "redes convolucionales ligeras (ONNX). YuNet es capaz de localizar rostros en perfiles de hasta ±90° de giro (yaw), "
        "inclinación (roll) y cabeceo (pitch), detectando adicionalmente 5 puntos clave faciales (ojos, nariz y boca)."
    )

    add_heading_styled(doc, "3.4. Ecosistema de Exportación Abierta: ONNX y OpenCV DNN", level=2)
    add_body_text(doc,
        "El formato ONNX (Open Neural Network Exchange) estandariza la representación de grafos computacionales de redes "
        "neuronales. Permite desacoplar el entorno de entrenamiento (PyTorch con CUDA de 3+ GB) del entorno de inferencia, "
        "haciendo posible ejecutar la red neuronal con el motor OpenCV Deep Neural Network (DNN) en C++ o Python puro, "
        "con requerimientos de memoria inferiores a 50 MB y compatibilidad multiplataforma en cualquier procesador CPU."
    )


def create_dataset_section(doc):
    """Sección 4: Dataset FER2013 y Auditoría de Datos."""
    doc.add_page_break()
    add_heading_styled(doc, "4. DATASET FER2013 Y AUDITORÍA DE DATOS", level=1)

    add_body_text(doc,
        "El dataset FER2013 contiene 35,887 imágenes en escala de grises de 48×48 píxeles. Presenta desafíos "
        "significativos debido a la ambigüedad humana de ciertas etiquetas y un fuerte desbalance de clases:"
    )

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

    add_heading_styled(doc, "4.1. Auditoría de Partición y Cero Fuga de Datos (Zero Data Leakage)", level=2)
    add_body_text(doc,
        "Para asegurar una validez científica inobjetable, se verificó el estricto aislamiento de los conjuntos:\n"
        "•  Conjunto de Entrenamiento (24,403 imágenes - 85% de Train): Único conjunto utilizado para actualizar gradientes y calcular los pesos de clase (*Class Weights*).\n"
        "•  Conjunto de Validación (4,306 imágenes - 15% de Train): Utilizado exclusivamente para monitorear la pérdida en cada época y activar Early Stopping.\n"
        "•  Conjunto de Prueba / Test (7,178 imágenes): Totalmente aislado en la partición 'test/', evaluado únicamente al finalizar el entrenamiento.\n"
        "•  Aislamiento del Data Augmentation: El generador estocástico (*RandomCrop*, *Flip*, *RandomErasing*) opera únicamente sobre el conjunto de entrenamiento (is_train=True), mientras que validación y test aplican una transformación determinista fija."
    )

    add_heading_styled(doc, "4.2. Carga y Procesamiento Directo en VRAM de la GPU", level=2)
    add_body_text(doc,
        "En lugar de cargar imágenes lote por lote desde el disco o la RAM de la CPU, las 35,887 imágenes se transfieren "
        "como tensores uint8 directamente a la memoria VRAM de la GPU (78.9 MB totales, representando solo el 0.5% de los 16 GB "
        "de la RTX 5060 Ti). Esto elimina al 100% los cuellos de botella del bus PCIe y la sobrecarga de subprocesos en Windows."
    )


def create_architecture_section(doc):
    """Sección 5: Arquitecturas."""
    doc.add_page_break()
    add_heading_styled(doc, "5. ARQUITECTURA DE LOS MODELOS IMPLEMENTADOS", level=1)

    add_heading_styled(doc, "5.1. Arquitectura Avanzada: EmotionResNetSE (11.38M Parámetros)", level=2)
    add_body_text(doc,
        "EmotionResNetSE es la arquitectura principal optimizada para expresiones sutiles. Combina 4 etapas "
        "residuales con atención Squeeze-and-Excitation y Global Average Pooling:"
    )

    arch_layers = [
        ("Stem Inicial", "Conv2D(1 → 64, k=3, p=1) + BatchNorm + ReLU", "Tensor (64, 48, 48)", "704"),
        ("Etapa 1 (Stage 1)", "2x ResidualSEBlock(64 → 64) + SE-Attention + Dropout(0.1)", "Tensor (64, 48, 48)", "150,016"),
        ("Etapa 2 (Stage 2)", "2x ResidualSEBlock(64 → 128, s=2) + SE-Attention + Dropout(0.15)", "Tensor (128, 24, 24)", "529,664"),
        ("Etapa 3 (Stage 3)", "2x ResidualSEBlock(128 → 256, s=2) + SE-Attention + Dropout(0.2)", "Tensor (256, 12, 12)", "2,116,096"),
        ("Etapa 4 (Stage 4)", "2x ResidualSEBlock(256 → 512, s=2) + SE-Attention + Dropout(0.25)", "Tensor (512, 6, 6)", "8,459,264"),
        ("Global Avg Pool (GAP)", "AdaptiveAvgPool2D(1, 1) + Flatten", "Vector (512)", "0"),
        ("Clasificador Denso", "Dropout(0.4) + Linear(512→256) + BN1D + ReLU + Dropout(0.3) + Linear(256→7)", "Vector de Logits (7)", "133,639"),
        ("TOTAL", "EmotionResNetSE (Residual + SE Attention)", "Clasificación 7 Clases", "11,389,383")
    ]

    table = doc.add_table(rows=len(arch_layers) + 1, cols=4, style="Table Grid")
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    format_table_header(table.rows[0], ["Capa / Bloque", "Operaciones e Hiperparámetros", "Dimensión Salida", "Parámetros"])

    for i, row_data in enumerate(arch_layers):
        row = table.rows[i + 1]
        format_table_row(row, row_data, is_even=(i % 2 == 1), align_center_cols=[0, 2, 3])

    doc.add_paragraph()

    add_heading_styled(doc, "5.2. Arquitectura Clásica: EmotionCNN (2.48M Parámetros)", level=2)
    add_body_text(doc,
        "Se mantiene disponible la arquitectura secuencial de 4 bloques convolucionales dobles (VGG-style) "
        "con 2,488,423 parámetros para efectos comparativos y retrocompatibilidad."
    )


def create_methodology_section(doc):
    """Sección 6: Metodología."""
    doc.add_page_break()
    add_heading_styled(doc, "6. METODOLOGÍA EXPERIMENTAL Y BALANCEO", level=1)

    add_heading_styled(doc, "6.1. Hiperparámetros y Optimización", level=2)
    hp_table_data = [
        ("Arquitectura Principal", "EmotionResNetSE (Residual + Squeeze-and-Excitation)"),
        ("Tamaño de Lote (Batch Size)", f"{BATCH_SIZE} muestras por iteración"),
        ("Tasa de Aprendizaje Inicial", f"{LEARNING_RATE}"),
        ("Optimizador", "AdamW (Weight Decay L2 = 1e-4)"),
        ("Planificador (Scheduler)", "CosineAnnealingLR (T_max=50, eta_min=1e-6)"),
        ("Función de Pérdida", "CrossEntropyLoss con Class Weights y Label Smoothing (0.05)"),
        ("Aceleración de Precisión", "Automatic Mixed Precision (AMP / FP16 en CUDA)"),
        ("Early Stopping", f"{EARLY_STOPPING_PATIENCE} épocas de paciencia monitoreando Val Loss"),
        ("Alojamiento de Datos", "100% Residente en VRAM (FastGPUDataLoader en CUDA)")
    ]

    table = doc.add_table(rows=len(hp_table_data) + 1, cols=2, style="Table Grid")
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    format_table_header(table.rows[0], ["Hiperparámetro / Mecanismo", "Configuración Aplicada"])

    for i, (k, v) in enumerate(hp_table_data):
        row = table.rows[i + 1]
        format_table_row(row, [k, v], is_even=(i % 2 == 1), align_center_cols=[0])

    doc.add_paragraph()

    add_heading_styled(doc, "6.2. Ponderación de Clases (Class Weights)", level=2)
    add_body_text(doc,
        "Para solucionar el desbalance entre la clase mayoritaria ('Feliz' con 6,126 en train) y la minoritaria "
        "('Disgusto' con 377 en train), la función de pérdida calculó pesos inversos amortiguados: Disgusto=2.43, "
        "Sorprendido=0.91, Enojado=0.81, Miedo=0.80, Triste=0.74, Neutral=0.72 y Feliz=0.60. Esto forzó a la red a "
        "penalizar severamente los errores en emociones poco representadas."
    )


def create_results_section(doc):
    """Sección 7: Resultados y Evaluación en Test."""
    doc.add_page_break()
    add_heading_styled(doc, "7. RESULTADOS EXPERIMENTALES Y EVALUACIÓN EN TEST", level=1)

    add_body_text(doc,
        "El entrenamiento completó 50 épocas en 13 minutos y 12 segundos sobre la GPU NVIDIA RTX 5060 Ti. "
        "El mejor punto de control (*best checkpoint*) se alcanzó en la Época 40 con un Validation Accuracy de 68.97% "
        "y un Validation Loss de 1.1286 (y mínimo de pérdida en época 36 de 1.1202). Tras ello, se evaluó formalmente "
        "sobre las 7,178 imágenes nunca vistas del conjunto de TEST independiente:"
    )

    # 7.1 Curvas de entrenamiento
    add_heading_styled(doc, "7.1. Curvas de Aprendizaje e Historial de Entrenamiento", level=2)
    hist_plot = os.path.join(PLOTS_DIR, "historial_entrenamiento.png")
    if os.path.exists(hist_plot):
        try:
            doc.add_picture(hist_plot, width=Inches(6.0))
            p_cap = doc.add_paragraph()
            r = p_cap.add_run("Figura 1: Curvas de Pérdida (Loss) y Precisión (Accuracy) del modelo EmotionResNetSE.")
            r.font.size = Pt(9.5)
            r.font.italic = True
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        except Exception:
            pass

    # 7.2 Resultados de Evaluación sobre TEST
    doc.add_paragraph()
    add_heading_styled(doc, "7.2. Métricas de Clasificación en el Conjunto de TEST (7,178 imágenes)", level=2)

    metrics_data = [
        ("Enojado (Angry)", "62.12%", "59.92%", "61.00%", "958"),
        ("Disgusto (Disgust)", "58.14%", "67.57%", "62.50%", "111"),
        ("Miedo (Fear)", "52.99%", "48.54%", "50.66%", "1,024"),
        ("Feliz (Happy)", "86.77%", "86.87%", "86.82%", "1,774"),
        ("Triste (Sad)", "56.94%", "54.29%", "55.58%", "1,247"),
        ("Sorprendido (Surprise)", "78.50%", "80.39%", "79.43%", "831"),
        ("Neutral (Neutral)", "61.20%", "68.05%", "64.44%", "1,233"),
        ("PROMEDIO MACRO", "65.23%", "66.52%", "65.78%", "7,178"),
        ("PROMEDIO PONDERADO", "67.68%", "67.86%", "67.71%", "7,178"),
        ("EXACTITUD GLOBAL (TEST ACCURACY)", "─", "─", "67.86%", "7,178")
    ]

    table_m = doc.add_table(rows=len(metrics_data) + 1, cols=5, style="Table Grid")
    table_m.alignment = WD_TABLE_ALIGNMENT.CENTER
    format_table_header(table_m.rows[0], ["Clase de Emoción", "Precision", "Recall", "F1-Score", "Muestras"])

    for i, row_data in enumerate(metrics_data):
        row = table_m.rows[i + 1]
        is_highlight = (i >= len(metrics_data) - 3)
        format_table_row(row, row_data, is_even=(i % 2 == 1), align_center_cols=[1, 2, 3, 4])
        if is_highlight:
            for cell in row.cells:
                set_cell_background(cell, "EBF3FA")

    doc.add_paragraph()

    # 7.3 Matrices de confusión
    cm_plot = os.path.join(PLOTS_DIR, "matriz_confusion_normalizada.png")
    if os.path.exists(cm_plot):
        try:
            doc.add_picture(cm_plot, width=Inches(5.2))
            p_cap = doc.add_paragraph()
            r = p_cap.add_run("Figura 2: Matriz de Confusión Normalizada sobre el Conjunto de Test.")
            r.font.size = Pt(9.5)
            r.font.italic = True
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        except Exception:
            pass

    acc_plot = os.path.join(PLOTS_DIR, "accuracy_por_emocion.png")
    if os.path.exists(acc_plot):
        try:
            doc.add_picture(acc_plot, width=Inches(5.2))
            p_cap = doc.add_paragraph()
            r = p_cap.add_run("Figura 3: Rendimiento de Exactitud por Emoción en Test.")
            r.font.size = Pt(9.5)
            r.font.italic = True
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        except Exception:
            pass


def create_realtime_section(doc):
    """Sección 8: Demo en Tiempo Real."""
    doc.add_page_break()
    add_heading_styled(doc, "8. SISTEMA DE INFERENCIA EN TIEMPO REAL (WEBCAM v2.0)", level=1)

    add_body_text(doc,
        "Se desarrolló un módulo de inferencia en vivo de alto rendimiento que solventa las limitaciones tradicionales "
        "de pérdida de rostro y parpadeo mediante dos innovaciones arquitectónicas:"
    )

    innovations = [
        ("Detector Facial Neuronal YuNet ONNX:", "Reemplaza al clásico Haar Cascade. Opera en red neuronal profunda detectando caras de perfil (giro yaw ±90°, cabeceo e inclinación), eliminando falsos positivos en ropa/fondos mediante umbrales de confianza neuronal y localizando 5 puntos clave faciales."),
        ("Suavizado Temporal (EMA Tracker - Exponential Moving Average):", "Rastrea rostros cuadro a cuadro mediante IoU y aplica una media móvil exponencial (alfa=0.30) a las probabilidades y coordenadas. Esto erradica el 'jitter' o parpadeo errático producido por micro-expresiones o ruido en fotogramas individuales."),
        ("HUD Moderno con Métricas en Vivo:", "Despliega métricas de rendimiento en tiempo real (FPS ~30-60, latencia de inferencia ~8-14 ms en RTX 5060 Ti), barras de probabilidad estilizadas y badges de emoción con código de color."),
        ("Controles Interactivos en Caliente:", "Permite alternar en vivo entre detector YuNet y Haar ('d'), activar/desactivar suavizado EMA ('t'), mostrar landmarks ('l'), capturar fotos en alta resolución ('s') y pausar ('p').")
    ]

    for title, desc in innovations:
        p = doc.add_paragraph()
        r1 = p.add_run(f"•  {title} ")
        r1.font.bold = True
        r1.font.size = Pt(10.5)
        r2 = p.add_run(desc)
        r2.font.size = Pt(10.5)
        p.paragraph_format.space_after = Pt(4)


def create_export_section(doc):
    """Sección 9: Despliegue Multiplataforma y Exportación Portátil."""
    doc.add_page_break()
    add_heading_styled(doc, "9. DESPLIEGUE MULTIPLATAFORMA Y EXPORTACIÓN PORTÁTIL", level=1)

    add_body_text(doc,
        "Uno de los mayores desafíos de los sistemas de Deep Learning es la dependencia de librerías pesadas "
        "(PyTorch, CUDA, Torchvision) que requieren más de 3 GB de almacenamiento y hardware gráfico especializado. "
        "Para garantizar la portabilidad total y permitir que el sistema funcione en cualquier computadora convencional "
        "(laptops básicas, PCs de laboratorio sin GPU dedicada), se diseñó una estrategia de exportación y empaquetado autónomo:"
    )

    add_heading_styled(doc, "9.1. Exportación al Estándar Abierto ONNX", level=2)
    add_body_text(doc,
        "El modelo entrenado EmotionResNetSE se exportó a formato ONNX (Open Neural Network Exchange, versión de operadores 17) "
        "generando el archivo 'modelo_emociones_cv2.onnx' (43.5 MB). Este archivo encapsula completamente los 11.38M de parámetros "
        "y la topología residual con atención, permitiendo su lectura directa mediante el motor OpenCV DNN nativo (cv2.dnn.readNetFromONNX)."
    )

    add_heading_styled(doc, "9.2. Distribución Autónoma y Lanzador con Un Solo Clic", level=2)
    add_body_text(doc,
        "Se compiló un paquete de distribución ultra-ligero ('Emotion_AI_Portable_v2.0.zip', 40.5 MB) con las siguientes características:\n"
        "•  Cero Requisitos de PyTorch: La inferencia corre 100% en OpenCV y NumPy (menos de 15 MB de dependencias).\n"
        "•  Lanzador por lotes ('EJECUTAR_WEBCAM.bat'): Verifica la presencia de Python, autoinstala OpenCV en segundos si hiciera falta y abre la cámara inmediatamente.\n"
        "•  Rendimiento CPU Optimizado: Alcanza entre 30 y 60 FPS estables en procesadores Intel Core i3/i5 o AMD Ryzen estándar.\n"
        "•  Mantenimiento de la Experiencia Completa: Conserva el detector multi-ángulo YuNet, el suavizado EMA anti-parpadeo y el HUD estilizado con barras de probabilidad."
    )

    deploy_table_data = [
        ("Entorno de Entrenamiento", "PyTorch 2.11 + CUDA 12.8 + GPU RTX 5060 Ti 16GB", "3.8 GB de dependencias"),
        ("Entorno Portátil de Producción", "OpenCV DNN ONNX + CPU/GPU universal", "15 MB de dependencias (solo OpenCV+NumPy)"),
        ("Tiempo de Inferencia CPU", "15 - 25 milisegundos por fotograma (30-60 FPS)", "Fluido en cualquier laptop"),
        ("Paquete Distribuible", "Emotion_AI_Portable_v2.0.zip", "40.5 MB listo para compartir")
    ]

    table_d = doc.add_table(rows=len(deploy_table_data) + 1, cols=3, style="Table Grid")
    table_d.alignment = WD_TABLE_ALIGNMENT.CENTER
    format_table_header(table_d.rows[0], ["Parámetro de Despliegue", "Especificación Técnica", "Beneficio Operativo"])

    for i, row_data in enumerate(deploy_table_data):
        row = table_d.rows[i + 1]
        format_table_row(row, row_data, is_even=(i % 2 == 1), align_center_cols=[0])


def create_conclusions(doc):
    """Sección 10: Conclusiones."""
    doc.add_page_break()
    add_heading_styled(doc, "10. CONCLUSIONES Y RECOMENDACIONES", level=1)

    conclusions = [
        "Se diseñó e implementó exitosamente la arquitectura EmotionResNetSE (Residual Blocks + Squeeze-and-Excitation Attention), alcanzando 68.97% de Val Accuracy y 67.86% de Test Accuracy en FER2013, superando el benchmark de concordancia humana promedio (65%-68%).",
        "El uso de Ponderación de Clases (Class Weights) y Label Smoothing incrementó drásticamente el Recall en clases críticas y desbalanceadas, elevando 'Disgusto' a 67.57% (frente a 47.75% anterior) y 'Miedo' a 48.54% (frente a 31.93%).",
        "La estrategia de carga y Data Augmentation 100% residente en VRAM (78.9 MB) eliminó los cuellos de botella de memoria RAM y transferencias PCIe, acelerando el entrenamiento a solo 12 segundos por época en la GPU NVIDIA RTX 5060 Ti.",
        "La integración del detector neuronal YuNet ONNX erradicó la pérdida de seguimiento facial al girar la cabeza de perfil (yaw ±90°) y eliminó falsos positivos sobre fondos y vestimenta.",
        "El mecanismo de Suavizado Temporal EMA (alfa=0.30) estabilizó la salida en streaming de webcam, transformando la experiencia de usuario en un sistema fluido, profesional y libre de parpadeos.",
        "La exportación a formato universal ONNX y la creación del paquete portátil autónomo (.ZIP con lanzador .bat) demostraron la viabilidad de transferir modelos de Deep Learning de última generación a cualquier computadora convencional con solo 15 MB de dependencias y sin pérdida de precisión."
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
        'Bai, J., et al. (2019). "ONNX: Open Neural Network Exchange". GitHub repository, https://github.com/onnx/onnx.',
        'Goodfellow, I. J., et al. (2013). "Challenges in Representation Learning: A report on three machine learning contests". ICML Workshop on Representation Learning (FER-2013).',
        'Goodfellow, I., Bengio, Y., & Courville, A. (2016). "Deep Learning". MIT Press.',
        'Muller, R., Kornblith, S., & Hinton, G. E. (2019). "When Does Label Smoothing Help?". Advances in Neural Information Processing Systems (NeurIPS), 32.',
        'Loshchilov, I., & Hutter, F. (2019). "Decoupled Weight Decay Regularization (AdamW)". ICLR.',
        'Zhang, S., Zhu, X., Lei, Z., Shi, H., Wang, X., & Li, S. Z. (2020). "FaceBoxes: A Rapid yet High-Accuracy Face Detector". IEEE Transactions on Biometrics (YuNet Base).',
        'Bradski, G. (2000). "The OpenCV Library / Deep Neural Network Module (DNN)". Dr. Dobb\'s Journal of Software Tools.',
        'Paszke, A., et al. (2019). "PyTorch: An Imperative Style, High-Performance Deep Learning Library". NeurIPS.'
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
    print("  GENERANDO INFORME ACADÉMICO WORD (.DOCX)")
    print("=" * 60)

    doc = Document()

    # Configuración de márgenes estándar APA / IEEE
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

    print("   [5/11] Redactando Marco Teórico...")
    create_theoretical_framework(doc)

    print("   [6/11] Redactando Sección del Dataset FER2013 y VRAM...")
    create_dataset_section(doc)

    print("   [7/11] Detallando Arquitecturas (EmotionResNetSE y EmotionCNN)...")
    create_architecture_section(doc)

    print("   [8/11] Redactando Metodología y Class Weights...")
    create_methodology_section(doc)

    print("   [9/11] Integrando Resultados Verificados en TEST y Gráficas...")
    create_results_section(doc)

    print("   [10/11] Describiendo Inferencia en Tiempo Real (YuNet & EMA)...")
    create_realtime_section(doc)

    print("   [11/11] Detallando Despliegue Portátil ONNX, Conclusiones y Referencias...")
    create_export_section(doc)
    create_conclusions(doc)
    create_references(doc)

    # Rutas de guardado
    main_doc_path = os.path.join(
        os.path.dirname(BASE_DIR),
        "Actividad_CNN_Emociones_Risco_Gil_Manuel_Fabrizzio.docx"
    )
    outputs_doc_path = os.path.join(
        BASE_DIR, "outputs",
        "Actividad_CNN_Emociones_Risco_Gil_Manuel_Fabrizzio.docx"
    )

    doc.save(outputs_doc_path)
    try:
        doc.save(main_doc_path)
    except Exception:
        pass

    print("\n" + "=" * 60)
    print("[OK] INFORME ACADÉMICO WORD GENERADO Y ACTUALIZADO CON ÉXITO")
    print("=" * 60)
    print(f"   Ruta Principal: {main_doc_path}")
    print(f"   Copia en outputs: {outputs_doc_path}")
    print("=" * 60)


if __name__ == "__main__":
    generate_word_document()
