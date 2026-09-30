# ============================================================================
# GENERADOR DE DOCUMENTO WORD (.docx) EN EL ESCRITORIO
# Documentación Completa: Alcance, Restricciones, Criterios, Modelos,
# Métricas Reales y Pseudocódigo Formal
# ============================================================================

import os
import sys
import shutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from src.utils import setup_utf8
setup_utf8()

import docx
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls


def set_cell_background(cell, fill_hex):
    """Aplica color de fondo a una celda de tabla."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)


def set_cell_margins(cell, top=100, bottom=100, left=140, right=140):
    """Ajusta márgenes internos de celda en dxa."""
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
    """Añade un encabezado con estilo institucional UTP."""
    heading = doc.add_heading(text, level=level)
    for run in heading.runs:
        run.font.name = "Calibri"
        if level == 1:
            run.font.size = Pt(14)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0, 51, 102)  # Azul UTP
        elif level == 2:
            run.font.size = Pt(12)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0, 76, 153)
        elif level == 3:
            run.font.size = Pt(10.5)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0, 102, 204)
    heading.paragraph_format.space_before = Pt(10)
    heading.paragraph_format.space_after = Pt(4)
    return heading


def add_body_text(doc, text, bold=False, italic=False, alignment=WD_ALIGN_PARAGRAPH.JUSTIFY):
    """Añade un párrafo de texto justificado estándar."""
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.name = "Calibri"
    run.font.size = Pt(10.5)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = RGBColor(35, 35, 35)

    p.alignment = alignment
    pf = p.paragraph_format
    pf.space_after = Pt(5)
    pf.line_spacing = 1.15
    if alignment == WD_ALIGN_PARAGRAPH.JUSTIFY:
        pf.first_line_indent = Cm(0.6)
    return p


def add_bullet_point(doc, title, text):
    """Añade una viñeta académica con título en negrita."""
    p = doc.add_paragraph(style="List Bullet")
    r1 = p.add_run(title + " ")
    r1.font.name = "Calibri"
    r1.font.size = Pt(10)
    r1.font.bold = True
    r1.font.color.rgb = RGBColor(0, 51, 102)
    r2 = p.add_run(text)
    r2.font.name = "Calibri"
    r2.font.size = Pt(10)
    r2.font.color.rgb = RGBColor(40, 40, 40)
    p.paragraph_format.space_after = Pt(3)


def format_table_header(row, col_names, bg_color="003366"):
    """Formatea la cabecera de una tabla."""
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
            run.font.size = Pt(9.5)
            run.font.bold = True
            run.font.color.rgb = RGBColor(255, 255, 255)


def format_table_row(row, values, is_even=False, align_center_cols=None):
    """Formatea una fila de datos con fondo alternado."""
    bg_color = "F0F4F8" if is_even else "FFFFFF"
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
            run.font.size = Pt(9)
            run.font.color.rgb = RGBColor(40, 40, 40)


def add_code_block(doc, code_text):
    """Añade un bloque de pseudocódigo en tabla sombreada con fuente monoespaciada."""
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    set_cell_background(cell, "F8F9FA")
    set_cell_margins(cell, top=140, bottom=140, left=180, right=180)
    
    # Borde sutil
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="6" w:space="0" w:color="D0D7DE"/>'
        f'<w:left w:val="single" w:sz="18" w:space="0" w:color="003366"/>'
        f'<w:bottom w:val="single" w:sz="6" w:space="0" w:color="D0D7DE"/>'
        f'<w:right w:val="single" w:sz="6" w:space="0" w:color="D0D7DE"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(tcBorders)

    lines = code_text.strip().split("\n")
    cell.text = "" # limpiar
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.05

    for i, line in enumerate(lines):
        if i > 0:
            p = cell.add_paragraph()
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.05
        
        run = p.add_run(line)
        run.font.name = "Consolas"
        run.font.size = Pt(8.5)
        
        # Color coding básico para palabras clave
        stripped = line.strip()
        if stripped.startswith("//") or stripped.startswith("/*"):
            run.font.color.rgb = RGBColor(106, 115, 125) # Gris comentario
            run.font.italic = True
        elif stripped.startswith(("ALGORITMO", "ENTRADA:", "SALIDA:", "INICIO:", "FIN")):
            run.font.color.rgb = RGBColor(0, 51, 102) # Azul oscuro
            run.font.bold = True
        elif any(stripped.startswith(k) for k in ["PARA ", "MIENTRAS ", "SI ", "SINO:", "FIN SI", "FIN PARA", "FIN MIENTRAS", "RETORNAR"]):
            run.font.color.rgb = RGBColor(160, 20, 50) # Rojo control
            run.font.bold = True
        else:
            run.font.color.rgb = RGBColor(36, 41, 46)


def build_document():
    doc = Document()

    # Márgenes de página (2 cm por lado)
    for section in doc.sections:
        section.top_margin = Cm(2.0)
        section.bottom_margin = Cm(2.0)
        section.left_margin = Cm(2.0)
        section.right_margin = Cm(2.0)

    # ─────────────────────────────────────────────────────────────────────────
    # CARÁTULA ACADÉMICA FORMAL
    # ─────────────────────────────────────────────────────────────────────────
    p_inst = doc.add_paragraph()
    p_inst.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r1 = p_inst.add_run("UNIVERSIDAD TECNOLÓGICA DEL PERÚ\n")
    r1.font.name = "Calibri"
    r1.font.size = Pt(16)
    r1.font.bold = True
    r1.font.color.rgb = RGBColor(0, 51, 102)

    r2 = p_inst.add_run("FACULTAD DE INGENIERÍA DE SISTEMAS Y SOFTWARE\n")
    r2.font.name = "Calibri"
    r2.font.size = Pt(12.5)
    r2.font.bold = True
    r2.font.color.rgb = RGBColor(60, 60, 60)

    r3 = p_inst.add_run("CURSO: SISTEMAS INTELIGENTES Y MACHINE LEARNING (CICLO 9)\n")
    r3.font.name = "Calibri"
    r3.font.size = Pt(11.5)
    r3.font.italic = True
    r3.font.color.rgb = RGBColor(80, 80, 80)
    p_inst.paragraph_format.space_after = Pt(20)

    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(15)
    p_title.paragraph_format.space_after = Pt(10)
    r_t = p_title.add_run("DOCUMENTO TÉCNICO Y EXPERIMENTAL:\nDELIMITACIÓN, MODELOS, PSEUDOCÓDIGOS Y MÉTRICAS REALES DEL SISTEMA DE RECONOCIMIENTO DE EMOCIONES FACIALES")
    r_t.font.name = "Calibri"
    r_t.font.size = Pt(15)
    r_t.font.bold = True
    r_t.font.color.rgb = RGBColor(0, 51, 102)

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(25)
    r_s = p_sub.add_run("Arquitectura EmotionResNetSE (11.39M Parámetros) con Detector Facial YuNet ONNX, Invarianza Lumínica CLAHE/Z-Score y Filtro Temporal EMA")
    r_s.font.name = "Calibri"
    r_s.font.size = Pt(11)
    r_s.font.italic = True
    r_s.font.color.rgb = RGBColor(70, 70, 70)

    # Tabla de metadatos
    tbl_meta = doc.add_table(rows=4, cols=2)
    tbl_meta.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_info = [
        ("AUTORES:", "• Manuel Fabrizzio Risco Gil\n• Alejandro Choquihuayta, Mario Julian"),
        ("DOCENTE:", "Mg. José Antonio Ogosi Auqui"),
        ("CICLO / ASIGNATURA:", "9no Ciclo — Sistemas Inteligentes y Machine Learning"),
        ("FECHA / CIUDAD:", "Septiembre 2026 — Lima, Perú")
    ]
    for idx, (label, val) in enumerate(meta_info):
        row = tbl_meta.rows[idx]
        c1, c2 = row.cells[0], row.cells[1]
        c1.text = label
        c2.text = val
        set_cell_background(c1, "EBF3FA")
        set_cell_background(c2, "F7FAFC")
        set_cell_margins(c1, 70, 70, 100, 100)
        set_cell_margins(c2, 70, 70, 100, 100)
        c1.paragraphs[0].runs[0].font.bold = True
        c1.paragraphs[0].runs[0].font.size = Pt(9.5)
        c2.paragraphs[0].runs[0].font.size = Pt(9.5)

    doc.add_page_break()

    # ─────────────────────────────────────────────────────────────────────────
    # SECCIÓN 1: DELIMITACIÓN DEL ALCANCE
    # ─────────────────────────────────────────────────────────────────────────
    add_heading_styled(doc, "1. DELIMITACIÓN DEL ALCANCE DEL PROYECTO (SCOPE)", level=1)
    
    add_heading_styled(doc, "1.1 Qué Incluye el Proyecto (In-Scope)", level=2)
    add_bullet_point(doc, "Clasificación de 7 Emociones Universales:", 
                     "Categorización bajo la taxonomía de Paul Ekman: Enojado, Disgusto, Miedo, Feliz, Triste, Sorprendido y Neutral, entregando vectores de probabilidad Softmax continuos.")
    add_bullet_point(doc, "Detección Facial Neuronal y Extracción de 5 Landmarks:", 
                     "Integración del modelo convolucional YuNet (ONNX) para localización de recuadros y coordenadas sub-píxel de ojos, nariz y comisuras bucales.")
    add_bullet_point(doc, "Alineación Afín Geométrica de Cabeza:", 
                     "Rotación y re-encuadre afín automático en base a la línea interpupilar, neutralizando inclinaciones de cabeza de hasta ±45°.")
    add_bullet_point(doc, "Invarianza Lumínica Fotométrica:", 
                     "Preprocesamiento en cascada mediante ecualización adaptativa local CLAHE (cuadrícula 4x4, clip limit 2.0) y estandarización canónica Z-Score (μ=0, σ=1), garantizando estabilidad total ante sombras o focos directos.")
    add_bullet_point(doc, "Estabilidad Temporal y Anti-Flickering:", 
                     "Filtro Exponencial Móvil (EMA) de régimen dual (α=0.18 en reposo, α=0.48 en movimientos rápidos) y ventanas de debounce por clase de 2 a 5 frames.")
    add_bullet_point(doc, "Seguimiento Multi-Rostro y Memoria Biométrica Face Re-ID:", 
                     "Tracking espacial IoU simultáneo y banco de plantillas biométricas 28x28 normalizadas por correlación cruzada (umbral 0.62) para retener la identidad temporal del sujeto.")
    add_bullet_point(doc, "Reportabilidad Independiente por Sujeto:", 
                     "Generación de N gráficas de distribución de probabilidades independientes para N rostros en imágenes estáticas y capturas limpias de pantalla sin menús.")
    add_bullet_point(doc, "Despliegue Portable Autónomo:", 
                     "Exportación a grafo ONNX universal y paquete distribuible .zip (40.5 MB) ejecutable en CPU x86_64 con OpenCV DNN sin requerir PyTorch ni GPU.")

    add_heading_styled(doc, "1.2 Qué NO Incluye el Proyecto (Out-of-Scope)", level=2)
    add_bullet_point(doc, "Reconocimiento de Identidad Civil o Nominal:", 
                     "No asocia caras a nombres reales, DNI ni bases de datos de identidad ciudadana; el seguimiento es estrictamente anónimo y temporal.")
    add_bullet_point(doc, "Emociones Complejas o Secundarias:", 
                     "No clasifica ironía, celos, orgullo, desprecio aislado, aburrimiento ni estados psicopatológicos clínicos.")
    add_bullet_point(doc, "Análisis de Unidades de Acción Continuas (FACS Profundo):", 
                     "No genera regresión de micro-músculos faciales individuales continuos; clasifica la expresión de forma holística.")
    add_bullet_point(doc, "Modalidades Acústicas o Corporales:", 
                     "No procesa señales de audio/voz, lenguaje natural ni postura corporal.")
    add_bullet_point(doc, "Oclusiones Severas (>50%):", 
                     "No opera sobre rostros cubiertos con mascarillas integrales, pasamontañas ni perfiles extremos (yaw > 75°).")
    add_bullet_point(doc, "Infraestructura Cloud Distribuida:", 
                     "No requiere servidores remotos ni microservicios WebRTC; es una arquitectura puramente local en el borde (Edge).")

    # ─────────────────────────────────────────────────────────────────────────
    # SECCIÓN 2: RESTRICCIONES TÉCNICAS Y DE DATOS
    # ─────────────────────────────────────────────────────────────────────────
    add_heading_styled(doc, "2. RESTRICCIONES TÉCNICAS Y DE DATOS", level=1)
    
    add_body_text(doc, "2.1 Restricciones de Datos (Dataset FER-2013):", bold=True)
    add_bullet_point(doc, "Baja Resolución Espacial:", "Imágenes de entrada en escala de grises de solo 48×48 píxeles, eliminando detalles de textura fina dérmica.")
    add_bullet_point(doc, "Desbalance Extremo de Clases:", "La clase Disgusto cuenta con apenas 436 imágenes originales (~1.5% del total) frente a más de 7,200 de Feliz. Se requirió oversampling a 2,000 muestras y Focal Loss.")
    add_bullet_point(doc, "Ruido de Etiquetado Humano:", "Entre 10% y 15% de muestras en FER-2013 presentan ambigüedades intrínsecas o marcas de agua, estableciendo un techo teórico en la concordancia humana de 65% - 68%.")

    add_body_text(doc, "2.2 Restricciones Técnicas y de Hardware:", bold=True)
    add_bullet_point(doc, "Latencia Límite de Tiempo Real:", "El ciclo completo de procesamiento no debe superar los 33.3 ms por fotograma para asegurar ≥30 FPS.")
    add_bullet_point(doc, "Hardware de Entrenamiento:", "Requiere GPU NVIDIA con soporte CUDA y memoria VRAM dedicada para soportar Automatic Mixed Precision (AMP FP16).")
    add_bullet_point(doc, "Entorno de Despliegue Portable:", "Debe ejecutarse en CPU x86_64 bajo OpenCV DNN con un consumo de memoria RAM inferior a 500 MB (operando realmente en ~240 MB).")

    # ─────────────────────────────────────────────────────────────────────────
    # SECCIÓN 3: CRITERIOS DE ÉXITO MEDIBLES
    # ─────────────────────────────────────────────────────────────────────────
    add_heading_styled(doc, "3. CRITERIOS DE ÉXITO MEDIBLES (KPIS)", level=1)
    
    tbl_kpi = doc.add_table(rows=6, cols=4, style="Table Grid")
    tbl_kpi.alignment = WD_TABLE_ALIGNMENT.CENTER
    format_table_header(tbl_kpi.rows[0], ["Criterio de Evaluación", "Meta Requerida", "Valor Obtenido", "Estado"])
    kpi_data = [
        ("Exactitud Global en Test (Top-1 Accuracy)", "≥ 68.0%", "68.14% (Directo) / 69.45% (TTA)", "SUPERADO"),
        ("Macro F1-Score en Test", "≥ 0.62", "0.6684", "SUPERADO"),
        ("Tasa de Cuadros por Segundo (FPS)", "≥ 30 FPS", "45 - 60 FPS (CPU) / 60 FPS (GPU)", "SUPERADO"),
        ("Latencia Integral End-to-End", "≤ 33.3 ms", "21.6 ms - 24.5 ms", "SUPERADO"),
        ("Invarianza Lumínica (Variación ΔP)", "≤ 5.0%", "ΔP = 0.2% (CLAHE + Z-Score)", "SUPERADO")
    ]
    for idx, row_val in enumerate(kpi_data):
        row = tbl_kpi.rows[idx + 1]
        format_table_row(row, row_val, is_even=(idx % 2 == 1), align_center_cols=[1, 2, 3])
        if row_val[3] == "SUPERADO":
            row.cells[3].paragraphs[0].runs[0].font.bold = True
            row.cells[3].paragraphs[0].runs[0].font.color.rgb = RGBColor(0, 128, 0)

    doc.add_page_break()

    # ─────────────────────────────────────────────────────────────────────────
    # SECCIÓN 4: MODELOS DE IA UTILIZADOS
    # ─────────────────────────────────────────────────────────────────────────
    add_heading_styled(doc, "4. MODELOS DE INTELIGENCIA ARTIFICIAL EMPLEADOS", level=1)
    add_body_text(doc, 
        "El sistema se fundamenta en una arquitectura en cascada integrada por tres modelos complementarios:")
    
    add_bullet_point(doc, "1. EmotionResNetSE (Clasificador de Emociones):", 
                     "Red Neuronal Convolucional Residual Profunda con 11,389,383 parámetros entrenables (~11.39 M). Integra 4 etapas residuales con bloques ResidualSEBlock dotados de mecanismos de atención por canales (Squeeze-and-Excitation con ratio r=16) y clasificador Global Average Pooling (GAP) con regularización Dropout dual (0.4 y 0.3).")
    add_bullet_point(doc, "2. YuNet ONNX (Detector Facial Profundo):", 
                     "Detector convolucional ligero en una sola etapa (OpenCV Zoo). Predice cajas delimitadoras y las coordenadas sub-píxel de 5 puntos faciales biométricos (ojos, nariz y comisuras de la boca) en menos de 12 ms, permitiendo el cálculo del ángulo de inclinación para la rotación afín.")
    add_bullet_point(doc, "3. Face Re-ID Engine (Memoria Biométrica Temporal):", 
                     "Módulo de emparejamiento biométrico que calcula la correlación cruzada normalizada de media cero (ZNCC) sobre plantillas faciales 28×28 px estandarizadas. Con un umbral ≥0.62, reconoce si un rostro reingresa al encuadre conservando su ID ('SUJETO #01', etc.).")

    # ─────────────────────────────────────────────────────────────────────────
    # SECCIÓN 5: RESULTADOS EXPERIMENTALES Y DATOS REALES
    # ─────────────────────────────────────────────────────────────────────────
    add_heading_styled(doc, "5. RESULTADOS EXPERIMENTALES Y DATOS REALES", level=1)

    add_heading_styled(doc, "5.1 Métricas Reales del Modelo en el Conjunto de Test (7,178 Imágenes)", level=2)
    tbl_test = doc.add_table(rows=9, cols=5, style="Table Grid")
    tbl_test.alignment = WD_TABLE_ALIGNMENT.CENTER
    format_table_header(tbl_test.rows[0], ["Emoción Objetivo", "Precisión", "Sensibilidad (Recall)", "F1-Score", "Muestras (Support)"])
    test_metrics = [
        ("Enojado (Angry)", "0.6300", "0.5616", "0.5938", "958"),
        ("Disgusto (Disgust)", "0.6750", "0.7297", "0.7013", "111"),
        ("Miedo (Fear)", "0.5458", "0.4941", "0.5187", "1,024"),
        ("Feliz (Happy)", "0.8719", "0.8782", "0.8750", "1,774"),
        ("Triste (Sad)", "0.5600", "0.5317", "0.5455", "1,247"),
        ("Sorprendido (Surprise)", "0.7609", "0.8424", "0.7995", "831"),
        ("Neutral (Neutral)", "0.6097", "0.6853", "0.6453", "1,233"),
        ("PROMEDIO GLOBAL", "0.6780", "0.6814", "0.6785", "7,178")
    ]
    for idx, row_val in enumerate(test_metrics):
        row = tbl_test.rows[idx + 1]
        is_last = (idx == len(test_metrics) - 1)
        format_table_row(row, row_val, is_even=(idx % 2 == 1), align_center_cols=[1, 2, 3, 4])
        if is_last:
            set_cell_background(row.cells[0], "EBF3FA")
            row.cells[0].paragraphs[0].runs[0].font.bold = True
            for c in row.cells[1:]:
                set_cell_background(c, "EBF3FA")
                c.paragraphs[0].runs[0].font.bold = True

    add_heading_styled(doc, "5.2 Equipo y Hardware Utilizado en los Experimentos", level=2)
    add_bullet_point(doc, "GPU:", "NVIDIA GeForce RTX 5060 Ti (8 GB VRAM dedicada, Driver 32.0.16.1692, Tensor Cores activos con AMP FP16).")
    add_bullet_point(doc, "CPU:", "AMD Ryzen 7 5700X 8-Core Processor (16 hilos lógicos, 3.4 GHz base / 4.6 GHz Turbo).")
    add_bullet_point(doc, "Memoria RAM del Sistema:", "32.0 GB DDR4.")
    add_bullet_point(doc, "Sistema Operativo:", "Microsoft Windows 11 Pro 64-bit.")
    add_bullet_point(doc, "Sensor de Video:", "Webcam USB 2.0 (640×480 px, modo MJPG, búfer hardware = 1).")
    add_bullet_point(doc, "Software:", "Python 3.11, PyTorch 2.x CUDA 12.x, OpenCV 4.10+ DNN, ONNX Opset 17.")

    add_heading_styled(doc, "5.3 Latencia y Rendimiento Cronometrado", level=2)
    add_bullet_point(doc, "Detector Facial YuNet (OpenCV DNN):", "11.58 ms por frame (86.3 FPS en frame 640×480).")
    add_bullet_point(doc, "EmotionResNetSE en GPU (PyTorch CUDA):", "10.00 ms por rostro (100.0 FPS en lote unitario).")
    add_bullet_point(doc, "EmotionResNetSE en CPU (ONNX Portátil):", "9.41 ms por rostro (106.2 FPS en CPU AMD Ryzen).")
    add_bullet_point(doc, "Pipeline Completo End-to-End:", "21.60 ms a 24.50 ms por cuadro (~45 a 60 FPS efectivos en pantalla).")

    add_heading_styled(doc, "5.4 Consumo de Memoria", level=2)
    add_bullet_point(doc, "VRAM GPU en Inferencia:", "186.6 MB dedicados (pico máximo de contexto CUDA: 1,278.2 MB).")
    add_bullet_point(doc, "VRAM GPU en Entrenamiento:", "~57 MB ocupados por el dataset completo en formato uint8.")
    add_bullet_point(doc, "Memoria RAM en Modo Portátil (CPU):", "180 a 240 MB de RAM continua (sin runtime de PyTorch ni CUDA).")
    add_bullet_point(doc, "Archivos de Distribución:", "Checkpoint PyTorch: 26.8 MB | Grafo ONNX: 43.5 MB | ZIP Portátil Autónomo: 40.5 MB.")

    add_heading_styled(doc, "5.5 Umbrales Calibrados del Sistema", level=2)
    add_bullet_point(doc, "Umbrales de Confianza por Emoción:", "Enojado: 0.30 | Disgusto: 0.18 | Miedo: 0.35 | Feliz: 0.28 | Triste: 0.35 | Sorprendido: 0.18 | Neutral: 0.24.")
    add_bullet_point(doc, "Boosts de Sensibilidad Dinámica:", "Enojado: ×1.55 (+55%) | Disgusto: ×1.55 (+55%) | Sorprendido: ×1.55 (+55%) | Miedo/Triste: ×1.05 (+5%) | Feliz: ×1.00 | Neutral: ×0.85 (-15%).")
    add_bullet_point(doc, "Ventana de Debounce Antirruido:", "Enojado y Sorprendido: 2 frames (~33 ms) | Disgusto y Feliz: 3 frames (~50 ms) | Miedo: 4 frames (~66 ms) | Triste y Neutral: 5 frames (~83 ms).")
    add_bullet_point(doc, "Filtro Temporal EMA:", "α=0.18 en reposo y α=0.48 reactivo cuando ΔP > 0.15.")
    add_bullet_point(doc, "Biometría Re-ID y Detección:", "Similitud ZNCC ≥ 0.62 | IoU ≥ 0.30 | YuNet Score ≥ 0.65, NMS = 0.30.")

    add_heading_styled(doc, "5.6 Escenario Concreto de Aplicación", level=2)
    add_body_text(doc, 
        "El proyecto se implementa para el escenario: 'Sistema de Cómputo en el Borde (Edge Computing) para la Monitorización y Diagnóstico del Clima Emocional en Aulas Universitarias y Laboratorios de Cómputo de Educación Superior (UTP)'. "
        "Permite a los docentes supervisar en tiempo real los niveles agregados de frustración/enojo, confusión/sorpresa o desinterés de los estudiantes sin vulnerar su privacidad (cumpliendo Privacy by Design: sin nombres ni transmisión de video a la nube). "
        "Gracias a CLAHE y Z-Score opera con invarianza ante la luz tenue de proyectores o reflejos de ventanas, y gracias a su empaquetado portable ONNX (40 MB, <240 MB RAM) se ejecuta en cualquier computadora convencional de laboratorio sin requerir tarjetas gráficas aceleradoras dedicadas.")

    doc.add_page_break()

    # ─────────────────────────────────────────────────────────────────────────
    # SECCIÓN 6: PSEUDOCÓDIGOS FORMALES DEL PROYECTO
    # ─────────────────────────────────────────────────────────────────────────
    add_heading_styled(doc, "6. PSEUDOCÓDIGOS FORMALES DEL SISTEMA", level=1)
    add_body_text(doc, 
        "A continuación se especifican los dos algoritmos fundamentales que componen el ciclo de vida del proyecto: "
        "el algoritmo de optimización y entrenamiento profundo, y el algoritmo de inferencia y seguimiento biométrico en tiempo real.")

    add_heading_styled(doc, "6.1 Algoritmo 1: Entrenamiento y Optimización del Modelo EmotionResNetSE", level=2)
    
    pseudo_train = """ALGORITMO 1: Entrenamiento_EmotionResNetSE
-----------------------------------------------------------------------------------------
ENTRADA:
    D_train, D_val      : Conjuntos particionados de FER-2013 (imágenes 48x48)
    E_max               : Número máximo de épocas (E_max = 86)
    B                   : Tamaño de lote (B = 256)
    η_base              : Tasa de aprendizaje inicial (η = 0.0005)
    γ                   : Factor de foco para Focal Loss (γ = 2.0)
    α_mix               : Hiperparámetro de interpolación Mixup (α = 0.1)
    Paciencia           : Criterio de Early Stopping (Paciencia = 15)

SALIDA:
    M_optimo            : Pesos del mejor modelo entrenado guardado en disco (.pth)
    Historial           : Curvas de pérdida y exactitud por época

INICIO:
    // 1. Balanceo de datos y preparación en VRAM
    w_clases ← CalcularPesosInversos(D_train)
    D_train_bal ← AplicarOversampling(D_train, muestras_minimas = 2000)
    CargarDatasetCompletoEnVRAM(D_train_bal, formato = UINT8)
    
    // 2. Inicialización del modelo y optimizadores
    M ← Inicializar_EmotionResNetSE(num_clases = 7, canales_entrada = 1)
    Optimizador ← AdamW(M.parametros, lr = η_base, weight_decay = 1e-4)
    Planificador ← SequentialLR(WarmupLineal(8 epocas) + CosenoAnnealing(E_max - 8))
    EscaladorAMP ← GradScaler(precision = FP16)
    
    mejor_loss_val ← +∞
    contador_paciencia ← 0

    // 3. Bucle de épocas de entrenamiento
    PARA epoca ← 1 HASTA E_max HACER:
        M.modo_entrenamiento()
        loss_acumulada ← 0.0

        PARA CADA lote (X, y) EN D_train_bal HACER:
            // Data Augmentation masivo ejecutado en núcleos CUDA
            X_aug ← FlipHorizontalAleatorio(X)
            X_aug ← RotacionAfinAleatoria(X_aug, angulo = [-12°, +12°])
            X_aug ← RandomErasing(X_aug, prob = 0.2)
            
            // Regularización Mixup
            SI aleatorio(0, 1) < 0.5 ENTONCES:
                X_in, y_a, y_b, λ ← AplicarMixup(X_aug, y, α_mix)
            SINO:
                X_in, y_a, y_b, λ ← X_aug, y, y, 1.0
            FIN SI

            // Forward Pass con Precisión Mixta Automática (AMP)
            CON PrecisionMixta(EscaladorAMP) HACER:
                logits ← M.forward(X_in)
                loss ← λ * FocalLoss(logits, y_a, γ, w_clases) + 
                       (1 - λ) * FocalLoss(logits, y_b, γ, w_clases)
            FIN CON

            // Backward Pass y optimización de gradientes
            Optimizador.cero_gradientes()
            EscaladorAMP.escalar(loss).retropropagar()
            EscaladorAMP.desescalar(Optimizador)
            RecortarNormaGradientes(M.parametros, max_norma = 1.0)
            EscaladorAMP.paso(Optimizador)
            EscaladorAMP.actualizar()

            loss_acumulada ← loss_acumulada + loss.valor
        FIN PARA

        Planificador.paso()

        // 4. Evaluación en conjunto de validación
        loss_val, acc_val ← Evaluar(M, D_val)
        Historial.registrar(epoca, loss_acumulada, loss_val, acc_val)

        // 5. Criterio de guardado y Early Stopping
        SI loss_val < mejor_loss_val ENTONCES:
            mejor_loss_val ← loss_val
            M_optimo ← M.copiar_pesos()
            GuardarCheckpoint(M_optimo, "outputs/models/mejor_modelo_emociones.pth")
            contador_paciencia ← 0
        SINO:
            contador_paciencia ← contador_paciencia + 1
            SI contador_paciencia ≥ Paciencia ENTONCES:
                Imprimir("Parada temprana ejecutada en época:", epoca)
                ROMPER BUCLE
            FIN SI
        FIN SI
    FIN PARA

    RETORNAR M_optimo, Historial
FIN"""
    add_code_block(doc, pseudo_train)

    add_heading_styled(doc, "6.2 Algoritmo 2: Inferencia en Tiempo Real, Estabilización Temporal y Memoria Biométrica", level=2)
    
    pseudo_infer = """ALGORITMO 2: Inferencia_TiempoReal_MultiRostro
-----------------------------------------------------------------------------------------
ENTRADA:
    FlujoCamara         : Video continuo desde la Webcam (640x480 @ 60 FPS)
    Red_YuNet           : Detector neuronal facial en formato ONNX
    Red_EmotionCNN      : Clasificador EmotionResNetSE (ONNX / PyTorch)
    Umbrales            : Umbrales adaptativos [0.30, 0.18, 0.35, 0.28, 0.35, 0.18, 0.24]
    Boosts              : Ponderación reactiva [1.55, 1.55, 1.05, 1.00, 1.05, 1.55, 0.85]

SALIDA:
    Pantalla_HUD        : Interfaz gráfica interactiva con recuadros, probabilidades y Re-ID
    Capturas            : Imágenes limpias exportadas sin superposición de menú

INICIO:
    BancoMemoria_ReID ← DiccionarioVacio() // {ID: (Plantilla_28x28, Historial_EMA)}
    contador_sujetos ← 0
    ConfigurarCamara(FlujoCamara, codec = "MJPG", buffer_size = 1)

    MIENTRAS VentanaAbierta() HACER:
        frame_actual ← FlujoCamara.capturar_cuadro()
        SI frame_actual ES NULO ENTONCES:
            CONTINUAR
        FIN SI

        // Detección facial neuronal con YuNet
        rostros_detectados ← Red_YuNet.detectar(frame_actual, score_thresh = 0.65, nms_thresh = 0.30)

        PARA CADA rostro EN rostros_detectados HACER:
            bbox ← rostro.caja_delimitadora // (x, y, w, h)
            landmarks ← rostro.cinco_puntos // Ojo_izq, Ojo_der, Nariz, Boca_izq, Boca_der

            // 1. Alineación Afín Geométrica (Nivelación interpupilar)
            dx ← landmarks.Ojo_der.x - landmarks.Ojo_izq.x
            dy ← landmarks.Ojo_der.y - landmarks.Ojo_izq.y
            angulo_cabeza ← ArcoTangente2(dy, dx) * (180.0 / π)
            centro_ojos ← ((landmarks.Ojo_izq.x + landmarks.Ojo_der.x) / 2, 
                           (landmarks.Ojo_izq.y + landmarks.Ojo_der.y) / 2)
            
            MatrizAfin ← CalcularMatrizRotacionEscala(centro_ojos, angulo_cabeza, escala = 1.0)
            rostro_alineado ← AplicarTransformacionAfin(frame_actual, MatrizAfin, bbox)

            // 2. Preprocesamiento Fotométrico Invariante a la Iluminación
            parche_gris ← ConvertirEscalaGrises(rostro_alineado)
            parche_48x48 ← Redimensionar(parche_gris, (48, 48))
            parche_clahe ← AplicarCLAHE(parche_48x48, clip_limit = 2.0, grid = 4x4)
            parche_zscore ← (parche_clahe - Media(parche_clahe)) / (DesviacionEstandar(parche_clahe) + 1e-7)

            // 3. Inferencia con Test-Time Augmentation (TTA)
            parche_flip ← EspejoHorizontal(parche_zscore)
            prob_original ← Softmax(Red_EmotionCNN.inferir(parche_zscore))
            prob_flip     ← Softmax(Red_EmotionCNN.inferir(parche_flip))
            prob_ensamble ← (prob_original + prob_flip) / 2.0

            prob_calibrada ← NormalizarL1(prob_ensamble * Boosts)

            // 4. Memoria Biométrica Face Re-ID (Identificación Multi-Rostro)
            plantilla_actual ← RedimensionarYNormalizar(parche_clahe, (28, 28))
            sujeto_id ← BuscarSujetoPorSimilitud(plantilla_actual, BancoMemoria_ReID, umbral = 0.62)

            SI sujeto_id NO ENCONTRADO ENTONCES:
                contador_sujetos ← contador_sujetos + 1
                sujeto_id ← "SUJETO #" + contador_sujetos
                BancoMemoria_ReID[sujeto_id] ← {
                    plantilla: plantilla_actual,
                    ema_prob: prob_calibrada,
                    contador_debounce: 0,
                    emocion_fijada: "Neutral"
                }
            SINO:
                BancoMemoria_ReID[sujeto_id].plantilla ← 0.8 * Anterior + 0.2 * plantilla_actual
            FIN SI

            // 5. Suavizado Temporal Dinámico (EMA) y Anti-Flickering
            prob_anterior ← BancoMemoria_ReID[sujeto_id].ema_prob
            delta_prob ← NormaInfinita(prob_calibrada - prob_anterior)
            
            α ← (delta_prob > 0.15) ? 0.48 : 0.18
            prob_suavizada ← (α * prob_calibrada) + ((1.0 - α) * prob_anterior)
            BancoMemoria_ReID[sujeto_id].ema_prob ← prob_suavizada

            // 6. Decisión Categórica con Debounce
            candidato_emocion ← ArgMaxConUmbrales(prob_suavizada, Umbrales)
            SI candidato_emocion ≠ BancoMemoria_ReID[sujeto_id].emocion_fijada ENTONCES:
                BancoMemoria_ReID[sujeto_id].contador_debounce += 1
                limite_debounce ← ObtenerDebounce(candidato_emocion)
                SI BancoMemoria_ReID[sujeto_id].contador_debounce ≥ limite_debounce ENTONCES:
                    BancoMemoria_ReID[sujeto_id].emocion_fijada ← candidato_emocion
                    BancoMemoria_ReID[sujeto_id].contador_debounce ← 0
                FIN SI
            SINO:
                BancoMemoria_ReID[sujeto_id].contador_debounce ← 0
            FIN SI

            // 7. Renderizado en el HUD
            color_sujeto ← AsignarPaletaNeon(sujeto_id)
            DibujarBBoxYLandmarks(frame_actual, bbox, landmarks, color_sujeto)
            DibujarBarrasProbabilidad(frame_actual, bbox, prob_suavizada, BancoMemoria_ReID[sujeto_id].emocion_fijada)
        FIN PARA

        SuperponerFPSYMetricas(frame_actual)
        MostrarVentana("Emotion AI - Inferencia en Tiempo Real", frame_actual)

        tecla ← LeerTecla(1 milisegundo)
        SI tecla == 'C' ENTONCES:
            ExportarCapturaLimpia(frame_actual, "outputs/plots/captura_limpia.png")
        SINO SI tecla == 'ESC' O tecla == 'Q' ENTONCES:
            ROMPER BUCLE
        FIN SI
    FIN MIENTRAS

    LiberarCamara(FlujoCamara)
    CerrarTodasLasVentanas()
FIN"""
    add_code_block(doc, pseudo_infer)

    # ─────────────────────────────────────────────────────────────────────────
    # GUARDAR DOCUMENTO EN EL ESCRITORIO
    # ─────────────────────────────────────────────────────────────────────────
    file_name = "Documentacion_Tecnica_Pseudocodigo_Emociones_IA.docx"
    
    # Ruta principal: OneDrive Desktop
    desktop_onedrive = r"C:\Users\manue\OneDrive\Desktop"
    desktop_local = r"C:\Users\manue\Desktop"
    outputs_dir = os.path.join(BASE_DIR, "outputs")

    target_onedrive = os.path.join(desktop_onedrive, file_name)
    target_local = os.path.join(desktop_local, file_name)
    target_outputs = os.path.join(outputs_dir, file_name)

    doc.save(target_onedrive)
    print(f"[OK] Documento guardado con éxito en: {target_onedrive}")

    if os.path.exists(desktop_local):
        shutil.copy2(target_onedrive, target_local)
        print(f"[OK] Copia espejo guardada en: {target_local}")

    shutil.copy2(target_onedrive, target_outputs)
    print(f"[OK] Copia de respaldo en repositorio: {target_outputs}")


if __name__ == "__main__":
    build_document()
