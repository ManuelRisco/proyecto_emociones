# ============================================================================
# SCRIPT PARA GENERAR GUÍA EXPLICATIVA COMPLETA EN WORD (.DOCX)
# Proyecto: Reconocimiento de Emociones Faciales con CNN en PyTorch
# Autor: Manuel Fabrizzio Risco Gil
# ============================================================================

import os
import sys
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
DOC_PATH = os.path.join(OUTPUT_DIR, "Guia_Explicativa_Completa_Proyecto_Emociones_CNN.docx")
SEMANA1_DIR = os.path.dirname(BASE_DIR)
DOC_PATH_SEMANA1 = os.path.join(SEMANA1_DIR, "Guia_Explicativa_Completa_Proyecto_Emociones_CNN.docx")

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
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
    heading = doc.add_heading(text, level=level)
    for run in heading.runs:
        run.font.name = "Calibri"
        if level == 1:
            run.font.size = Pt(16)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0, 51, 102)  # Azul oscuro institucional
        elif level == 2:
            run.font.size = Pt(13)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0, 80, 160)  # Azul medio
        elif level == 3:
            run.font.size = Pt(11)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0, 120, 215) # Azul acento
    heading.paragraph_format.space_before = Pt(12)
    heading.paragraph_format.space_after = Pt(4)
    return heading

def add_body_text(doc, text, bold=False, italic=False, alignment=WD_ALIGN_PARAGRAPH.JUSTIFY):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.name = "Calibri"
    run.font.size = Pt(11)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = RGBColor(40, 40, 40)
    p.alignment = alignment
    pf = p.paragraph_format
    pf.space_after = Pt(6)
    pf.line_spacing = 1.15
    if alignment == WD_ALIGN_PARAGRAPH.JUSTIFY:
        pf.first_line_indent = Cm(0.75)
    return p

def add_bullet(doc, title, desc):
    p = doc.add_paragraph(style='List Bullet')
    r1 = p.add_run(title + ": ")
    r1.font.name = "Calibri"
    r1.font.size = Pt(10.5)
    r1.font.bold = True
    r1.font.color.rgb = RGBColor(0, 51, 102)
    r2 = p.add_run(desc)
    r2.font.name = "Calibri"
    r2.font.size = Pt(10.5)
    r2.font.color.rgb = RGBColor(40, 40, 40)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.15
    return p

def add_callout_box(doc, title, text):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    cell = table.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, "EBF3FA")
    set_cell_margins(cell, top=140, bottom=140, left=180, right=180)
    
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    r_title = p.add_run("💡 " + title + "\n")
    r_title.font.name = "Calibri"
    r_title.font.size = Pt(11)
    r_title.font.bold = True
    r_title.font.color.rgb = RGBColor(0, 51, 102)
    
    r_text = p.add_run(text)
    r_text.font.name = "Calibri"
    r_text.font.size = Pt(10)
    r_text.font.color.rgb = RGBColor(30, 30, 30)
    p.paragraph_format.space_after = Pt(2)
    
    p_space = doc.add_paragraph()
    p_space.paragraph_format.space_before = Pt(4)
    p_space.paragraph_format.space_after = Pt(4)

def format_table_header(row, col_names, bg_color="003366"):
    for idx, name in enumerate(col_names):
        cell = row.cells[idx]
        cell.text = name
        set_cell_background(cell, bg_color)
        set_cell_margins(cell, top=100, bottom=100, left=120, right=120)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in p.runs:
            run.font.name = "Calibri"
            run.font.size = Pt(10)
            run.font.bold = True
            run.font.color.rgb = RGBColor(255, 255, 255)

def format_table_row(row, values, is_even=False, align_center_cols=None):
    bg_color = "F4F7FA" if is_even else "FFFFFF"
    align_center_cols = align_center_cols or []
    for idx, val in enumerate(values):
        cell = row.cells[idx]
        cell.text = str(val)
        set_cell_background(cell, bg_color)
        set_cell_margins(cell, top=70, bottom=70, left=100, right=100)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if idx in align_center_cols else WD_ALIGN_PARAGRAPH.LEFT
        for run in p.runs:
            run.font.name = "Calibri"
            run.font.size = Pt(9.5)
            run.font.color.rgb = RGBColor(40, 40, 40)

def generate_comprehensive_guide():
    print("=" * 60)
    print("   GENERANDO GUÍA EXPLICATIVA EN WORD (.DOCX)...")
    print("=" * 60)
    
    doc = Document()
    
    # Configurar márgenes estándar (2.5 cm)
    for section in doc.sections:
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2.5)
    
    # =========================================================================
    # PORTADA / ENCABEZADO
    # =========================================================================
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(30)
    p_title.paragraph_format.space_after = Pt(6)
    r_title = p_title.add_run("GUÍA EXPLICATIVA COMPLETA Y DETALLADA")
    r_title.font.name = "Calibri"
    r_title.font.size = Pt(22)
    r_title.font.bold = True
    r_title.font.color.rgb = RGBColor(0, 51, 102)
    
    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(20)
    r_sub = p_sub.add_run("Sistema de Reconocimiento de Emociones Faciales con Redes Neuronales Convolucionales (CNN) en PyTorch")
    r_sub.font.name = "Calibri"
    r_sub.font.size = Pt(13)
    r_sub.font.italic = True
    r_sub.font.color.rgb = RGBColor(70, 70, 70)
    
    # Cuadro informativo del autor
    table_meta = doc.add_table(rows=4, cols=2)
    table_meta.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_info = [
        ("Autor / Estudiante:", "Manuel Fabrizzio Risco Gil"),
        ("Curso:", "Sistemas Inteligentes y Machine Learning (Ciclo 9)"),
        ("Tecnologías Principales:", "Python 3.10+, PyTorch, OpenCV, CUDA, NumPy, Scikit-Learn"),
        ("Dataset Empleado:", "FER2013 (Facial Expression Recognition 2013 - 35,887 imágenes)")
    ]
    for idx, (label, val) in enumerate(meta_info):
        row = table_meta.rows[idx]
        c1, c2 = row.cells[0], row.cells[1]
        c1.text = label
        c2.text = val
        set_cell_background(c1, "EAEFF5")
        set_cell_background(c2, "F8FAFC")
        set_cell_margins(c1, 60, 60, 100, 100)
        set_cell_margins(c2, 60, 60, 100, 100)
        c1.paragraphs[0].runs[0].font.bold = True
        c1.paragraphs[0].runs[0].font.size = Pt(10)
        c2.paragraphs[0].runs[0].font.size = Pt(10)
    
    doc.add_page_break()
    
    # =========================================================================
    # SECCIÓN 1: INTRODUCCIÓN Y CONCEPTOS BÁSICOS
    # =========================================================================
    add_heading_styled(doc, "1. ¿Qué es este proyecto y qué problema resuelve?", level=1)
    
    add_body_text(doc, 
        "Este proyecto consiste en un sistema inteligente de visión por computadora capaz de analizar rostros humanos "
        "en imágenes o video en tiempo real (webcam) y determinar de forma automática cuál de las 7 emociones universales "
        "está expresando la persona: Enojado (Angry), Disgusto (Disgust), Miedo (Fear), Feliz (Happy), Triste (Sad), "
        "Sorprendido (Surprise) o Neutral."
    )
    add_body_text(doc,
        "El reconocimiento de emociones faciales (conocido en la literatura científica como Facial Emotion Recognition o FER) "
        "es una tarea no trivial debido a que dos personas pueden expresar la misma emoción de formas sutilmente distintas, "
        "la iluminación del entorno varía, existen giros o inclinaciones del rostro, y la resolución de las cámaras puede ser baja. "
        "Para solucionar este desafío, se recurre a algoritmos de Aprendizaje Profundo (Deep Learning)."
    )
    
    add_callout_box(doc, "Idea Clave: ¿Cómo piensa la IA en este proyecto?",
        "En lugar de que un programador escriba reglas manuales (como 'si la boca está curvada hacia arriba, entonces está feliz'), "
        "el modelo recibe decenas de miles de fotos etiquetadas y aprende por sí mismo qué combinaciones de píxeles, líneas y sombras "
        "representan cada emoción mediante una Red Neuronal Convolucional (CNN)."
    )
    
    # =========================================================================
    # SECCIÓN 2: MAPA CONCEPTUAL DE IA
    # =========================================================================
    add_heading_styled(doc, "2. Fundamentos Teóricos: ¿Red Neuronal, Convolucional o qué cosa es?", level=1)
    
    add_body_text(doc,
        "Para entender con precisión qué modelo se utiliza, es fundamental comprender la jerarquía de las disciplinas:"
    )
    
    add_bullet(doc, "Inteligencia Artificial (IA)", "Es el campo general de la ciencia que busca que las máquinas simulen el comportamiento inteligente humano.")
    add_bullet(doc, "Machine Learning (ML)", "Es una rama de la IA donde las máquinas aprenden patrones a partir de datos históricos sin ser programadas explícitamente.")
    add_bullet(doc, "Deep Learning (DL)", "Es un subcampo del Machine Learning basado en Redes Neuronales Artificiales profundas (múltiples capas ocultas) que aprenden representaciones jerárquicas.")
    add_bullet(doc, "Red Neuronal Convolucional (CNN)", "Es un tipo especializado de Red Neuronal Profunda diseñada específicamente para procesar datos con estructura de cuadrícula, como imágenes.")
    
    add_body_text(doc,
        "Por lo tanto, este proyecto utiliza un modelo de Deep Learning, que es una Red Neuronal, y específicamente una "
        "Red Neuronal Convolucional (CNN).", bold=True
    )
    
    add_heading_styled(doc, "¿Por qué una CNN y no una Red Neuronal tradicional (MLP)?", level=2)
    add_body_text(doc,
        "En una red neuronal tradicional completamente conectada (Multi-Layer Perceptron o MLP), cada píxel de la imagen se conecta a cada neurona. "
        "Si aplanamos una imagen, se pierde la relación espacial 2D (es decir, qué píxel está al lado de cuál). Además, si una persona mueve su cara "
        "unos centímetros, la red MLP no sabría reconocerla porque los píxeles cambiaron de posición."
    )
    add_body_text(doc,
        "Las CNN resuelven este problema mediante dos propiedades fundamentales:\n"
        "1. Invariancia a la traslación: Pueden detectar un ojo o una sonrisa sin importar en qué parte de la imagen se encuentre.\n"
        "2. Compartición de pesos (Weight Sharing): Usan filtros pequeños (kernels de 3x3) que se deslizan por toda la imagen, reduciendo drásticamente la cantidad de parámetros a calcular y evitando el sobreajuste."
    )
    
    # =========================================================================
    # SECCIÓN 3: EL DATASET FER2013
    # =========================================================================
    add_heading_styled(doc, "3. El Dataset: FER2013 (Facial Expression Recognition)", level=1)
    
    add_body_text(doc,
        "El entrenamiento y evaluación del modelo se realiza sobre el dataset FER2013, recopilado por Pierre-Luc Carrier y Aaron Courville, "
        "presentado en la conferencia ICML 2013. Este dataset contiene 35,887 imágenes faciales en escala de grises con una resolución de 48×48 píxeles."
    )
    
    table_emotions = doc.add_table(rows=8, cols=4)
    table_emotions.alignment = WD_TABLE_ALIGNMENT.CENTER
    format_table_header(table_emotions.rows[0], ["ID Clase", "Emoción (Español)", "Emoción (Inglés)", "Emoji Representativo"])
    
    emotions_data = [
        ("0", "Enojado", "Angry", "😠"),
        ("1", "Disgusto", "Disgust", "🤢"),
        ("2", "Miedo", "Fear", "😨"),
        ("3", "Feliz", "Happy", "😃"),
        ("4", "Triste", "Sad", "😢"),
        ("5", "Sorprendido", "Surprise", "😲"),
        ("6", "Neutral", "Neutral", "😐")
    ]
    for i, row_data in enumerate(emotions_data):
        format_table_row(table_emotions.rows[i+1], row_data, is_even=(i % 2 == 1), align_center_cols=[0, 3])
        
    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_before = Pt(4)
    
    add_heading_styled(doc, "Aumento de Datos (Data Augmentation)", level=2)
    add_body_text(doc,
        "Para evitar que el modelo simplemente 'memorice' las fotos de entrenamiento (overfitting) y para aumentar su robustez, "
        "el módulo src/dataset.py aplica transformaciones aleatorias a las imágenes en cada época de entrenamiento:"
    )
    add_bullet(doc, "RandomHorizontalFlip (p=0.5)", "Espejea el rostro horizontalmente. Un rostro enojado sigue estando enojado si mira a la izquierda o a la derecha.")
    add_bullet(doc, "RandomRotation (±10°)", "Simula inclinaciones naturales de la cabeza del usuario.")
    add_bullet(doc, "RandomAffine (traslación y escala)", "Desplaza levemente el rostro y modifica la escala (zoom in/out de ±10%).")
    add_bullet(doc, "ColorJitter (brillo y contraste)", "Altera sutilmente el contraste y brillo para simular distintas condiciones de iluminación.")
    add_bullet(doc, "Normalización estándar", "Convierte los valores de píxel de [0, 255] al rango [-1, 1], facilitando la convergencia matemática del optimizador.")
    
    # =========================================================================
    # SECCIÓN 4: ARQUITECTURA DE LA RED NEURONAL (EmotionCNN)
    # =========================================================================
    add_heading_styled(doc, "4. Arquitectura Detallada de la Red: EmotionCNN", level=1)
    
    add_body_text(doc,
        "La arquitectura del modelo está implementada en el archivo src/model.py bajo la clase EmotionCNN heredando de nn.Module de PyTorch. "
        "Consta de 4 bloques convolucionales secuenciales seguidos de un bloque clasificador con 2 capas completamente conectadas (Fully Connected)."
    )
    
    table_arch = doc.add_table(rows=7, cols=4)
    table_arch.alignment = WD_TABLE_ALIGNMENT.CENTER
    format_table_header(table_arch.rows[0], ["Capa / Bloque", "Operaciones Internas", "Dimensiones de Salida", "Propósito Funcional"])
    
    arch_data = [
        ("Entrada", "Tensor Grayscale 1 canal", "Batch × 1 × 48 × 48", "Imagen facial de entrada"),
        ("Bloque Conv 1", "2x Conv2d(3x3, 32) + BatchNorm + ReLU + MaxPool2d(2x2) + Dropout(0.25)", "Batch × 32 × 24 × 24", "Detección de bordes, gradientes y texturas"),
        ("Bloque Conv 2", "2x Conv2d(3x3, 64) + BatchNorm + ReLU + MaxPool2d(2x2) + Dropout(0.25)", "Batch × 64 × 12 × 12", "Detección de formas simples (ojos, cejas)"),
        ("Bloque Conv 3", "2x Conv2d(3x3, 128) + BatchNorm + ReLU + MaxPool2d(2x2) + Dropout(0.25)", "Batch × 128 × 6 × 6", "Detección de partes faciales (boca, nariz)"),
        ("Bloque Conv 4", "2x Conv2d(3x3, 256) + BatchNorm + ReLU + MaxPool2d(2x2) + Dropout(0.25)", "Batch × 256 × 3 × 3", "Detección de expresiones y rasgos globales"),
        ("Clasificador FC", "Flatten (2304) → Linear(512) → ReLU → Dropout(0.5) → Linear(256) → ReLU → Dropout(0.5) → Linear(7)", "Batch × 7", "Cálculo de logits para las 7 emociones")
    ]
    for i, row_data in enumerate(arch_data):
        format_table_row(table_arch.rows[i+1], row_data, is_even=(i % 2 == 1), align_center_cols=[0, 2])
        
    p_sp2 = doc.add_paragraph()
    p_sp2.paragraph_format.space_before = Pt(4)
    
    add_heading_styled(doc, "Explicación de cada capa matemática de la red:", level=2)
    add_bullet(doc, "Convolución 2D (Conv2D)", "Aplica una matriz de pesos (filtro de 3x3) que multiplica y suma píxeles locales para resaltar patrones visuales clave.")
    add_bullet(doc, "Normalización por Lote (BatchNorm2d)", "Normaliza las activaciones de la capa anterior restando la media y dividiendo por la desviación estándar. Esto evita el desvanecimiento de gradientes y permite entrenar con tasas de aprendizaje más altas.")
    add_bullet(doc, "Función de Activación ReLU (Rectified Linear Unit)", "Aplica f(x) = max(0, x). Convierte valores negativos en 0 e introduce no linealidad en la red, permitiendo aprender funciones altamente complejas.")
    add_bullet(doc, "Max Pooling (MaxPool2d 2x2)", "Toma el valor máximo de cada ventana de 2x2 píxeles, reduciendo el tamaño a la mitad (48→24→12→6→3). Esto reduce el costo computacional y provee invariancia espacial.")
    add_bullet(doc, "Dropout (0.25 en convolución, 0.5 en densas)", "Apaga aleatoriamente un porcentaje de neuronas durante cada paso de entrenamiento. Esto obliga a la red a no depender de una sola neurona, aprendiendo representaciones redundantes y robustas.")
    
    # =========================================================================
    # SECCIÓN 5: ENTRENAMIENTO Y OPTIMIZACIÓN
    # =========================================================================
    add_heading_styled(doc, "5. Proceso de Entrenamiento y Optimización", level=1)
    
    add_body_text(doc,
        "El entrenamiento se lleva a cabo en el script src/train.py mediante el siguiente flujo cíclico:"
    )
    
    add_bullet(doc, "1. Propagación hacia adelante (Forward Pass)", "La imagen entra a la red y genera 7 números llamados 'logits' (puntuaciones no normalizadas).")
    add_bullet(doc, "2. Cálculo de la Pérdida (Loss - CrossEntropyLoss)", "Compara los logits predichos con la etiqueta real utilizando Entropía Cruzada Categórica. Si el modelo se equivoca con alta confianza, el castigo numérico es muy alto.")
    add_bullet(doc, "3. Retropropagación (Backward Pass / Backpropagation)", "Calcula la derivada parcial (gradiente) del error respecto a cada uno de los pesos de la red mediante la regla de la cadena.")
    add_bullet(doc, "4. Actualización de Pesos (Optimizador AdamW)", "Ajusta los millones de parámetros de la red en la dirección contraria al gradiente. AdamW incorpora decaimiento de pesos (Weight Decay = 1e-4) para evitar que los pesos crezcan descontroladamente.")
    add_bullet(doc, "5. Ajuste de Tasa de Aprendizaje (ReduceLROnPlateau)", "Si la pérdida en validación no mejora durante 3 épocas consecutivas, la tasa de aprendizaje se reduce automáticamente a la mitad (factor 0.5) para afinar la búsqueda del mínimo global.")
    add_bullet(doc, "6. Parada Temprana (Early Stopping)", "Si el modelo no mejora su rendimiento en 10 épocas seguidas (patience=10), el entrenamiento se detiene automáticamente y se conserva el mejor checkpoint.")
    add_bullet(doc, "7. Aceleración con GPU y Mixed Precision (AMP)", "Utiliza torch.cuda.amp.autocast para realizar operaciones en precisión flotante de 16 bits (FP16) combinadas con 32 bits (FP32), duplicando la velocidad de entrenamiento en GPUs modernas (como NVIDIA RTX) sin perder precisión.")
    
    # =========================================================================
    # SECCIÓN 6: SISTEMA EN TIEMPO REAL (WEBCAM + OPENCV)
    # =========================================================================
    add_heading_styled(doc, "6. Sistema de Inferencia en Tiempo Real (Webcam)", level=1)
    
    add_body_text(doc,
        "El módulo src/realtime.py conecta la cámara web del usuario con el modelo de Deep Learning en un pipeline fluido:"
    )
    
    add_bullet(doc, "Paso 1: Captura de Cuadro", "OpenCV (cv2.VideoCapture) captura el fotograma de la cámara en vivo a 30 FPS.")
    add_bullet(doc, "Paso 2: Detección Facial (Haar Cascade)", "El clasificador en cascada haarcascade_frontalface_default.xml localiza las coordenadas (x, y, ancho, alto) de cualquier rostro presente en la imagen.")
    add_bullet(doc, "Paso 3: Recorte y Normalización", "Se extrae únicamente el área del rostro, se convierte a escala de grises, se redimensiona a 48×48 píxeles y se convierte en tensor de PyTorch.")
    add_bullet(doc, "Paso 4: Inferencia con la CNN", "El tensor se envía al modelo (en modo evaluación model.eval() y sin calcular gradientes torch.no_grad()). Se aplica la función Softmax para obtener las probabilidades porcentuales de cada emoción.")
    add_bullet(doc, "Paso 5: Renderizado Gráfico", "Se dibuja un recuadro verde alrededor del rostro, se escribe el nombre de la emoción dominante con su probabilidad (ej: 'Feliz: 94.2%') y se añade el emoji correspondiente en pantalla.")
    
    # =========================================================================
    # SECCIÓN 7: ESTRUCTURA DEL CÓDIGO FUENTE
    # =========================================================================
    add_heading_styled(doc, "7. Guía y Estructura del Código Fuente", level=1)
    
    add_body_text(doc, "A continuación se resume el rol de cada archivo dentro del proyecto:")
    
    table_files = doc.add_table(rows=9, cols=3)
    table_files.alignment = WD_TABLE_ALIGNMENT.CENTER
    format_table_header(table_files.rows[0], ["Archivo", "Módulo / Rol", "Descripción de Funcionalidad"])
    
    files_data = [
        ("main.py", "Punto de Entrada", "Menú interactivo por consola (1: Train, 2: Eval, 3: Predict, 4: Webcam, 5: Word, 6: Arch)."),
        ("src/config.py", "Configuración", "Hiperparámetros (LR, Batch Size, Épocas), rutas, etiquetas y detección de GPU NVIDIA CUDA."),
        ("src/dataset.py", "Procesamiento de Datos", "Carga del dataset FER2013, data augmentation y DataLoaders optimizados en RAM."),
        ("src/model.py", "Modelo PyTorch", "Definición de la clase EmotionCNN con 4 bloques convolucionales y capas densas."),
        ("src/train.py", "Entrenamiento", "Bucle de entrenamiento con AdamW, CrossEntropy, Scheduler, Early Stopping y AMP."),
        ("src/evaluate.py", "Evaluación", "Cálculo de métricas (Accuracy, Precision, Recall, F1) y generación de la Matriz de Confusión."),
        ("src/predict.py", "Predicción Individual", "Inferencia sobre una sola foto estática seleccionada por el usuario."),
        ("src/realtime.py", "Inferencia en Vivo", "Detección facial con Haar Cascades y clasificación de emociones en tiempo real con webcam.")
    ]
    for i, row_data in enumerate(files_data):
        format_table_row(table_files.rows[i+1], row_data, is_even=(i % 2 == 1), align_center_cols=[0])
        
    p_sp3 = doc.add_paragraph()
    p_sp3.paragraph_format.space_before = Pt(4)
    
    # =========================================================================
    # SECCIÓN 8: PREGUNTAS CLAVE PARA SUSTENTACIÓN / EXPOSICIÓN
    # =========================================================================
    add_heading_styled(doc, "8. Preguntas Clave para Sustentación o Examen", level=1)
    
    qa_list = [
        ("¿Qué tipo de problema de Machine Learning resuelve este proyecto?",
         "Es un problema de Aprendizaje Supervisado de Clasificación Multiclase (7 clases excluyentes)."),
        ("¿Por qué se utiliza CrossEntropyLoss como función de pérdida?",
         "Porque es la función estándar para problemas de clasificación multiclase donde cada muestra pertenece a una única categoría. Mide la discrepancia entre la distribución de probabilidad predicha por la red y la etiqueta real one-hot."),
        ("¿Qué función cumple la capa Batch Normalization?",
         "Acelera el entrenamiento y estabiliza los pesos al normalizar la distribución de las activaciones intermedias, mitigando el problema del desvanecimiento del gradiente."),
        ("¿Cuál es la diferencia entre los filtros convolucionales iniciales y los finales?",
         "Los filtros iniciales (Bloque 1, 32 filtros) capturan características de bajo nivel como bordes y contrastes. Los filtros más profundos (Bloque 4, 256 filtros) combinan esos bordes para reconocer formas complejas como bocas fruncidas, cejas levantadas o sonrisas."),
        ("¿Cómo se evita el sobreajuste (overfitting) en este modelo?",
         "Se aplican tres técnicas complementarias: 1) Data Augmentation (modificaciones aleatorias a las imágenes), 2) Dropout (apagado aleatorio de neuronas en 25% y 50%), y 3) Early Stopping (detención automática al no observar mejoras).")
    ]
    
    for q, a in qa_list:
        add_bullet(doc, "P: " + q, "\nR: " + a)
        
    # Guardar documento en ambas ubicaciones (en outputs/ y en la carpeta de la semana)
    doc.save(DOC_PATH)
    try:
        doc.save(DOC_PATH_SEMANA1)
    except Exception:
        pass
        
    print(f"\n[OK] Documento generado exitosamente en:")
    print(f"     1. {DOC_PATH}")
    print(f"     2. {DOC_PATH_SEMANA1}")

if __name__ == "__main__":
    generate_comprehensive_guide()
