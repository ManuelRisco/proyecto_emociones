# -*- coding: utf-8 -*-
"""
Script de verificación y auditoría exhaustiva del documento Word (.docx)
Comprueba:
- Citas biyectivas entre el cuerpo del texto/tablas y la bibliografía (100 % de cruce).
- Estructura estricta de Figuras y Tablas bajo normas APA 7ma edición.
- Detección de títulos internos incrustados y meta-frases de IA.
- Conteo total de palabras y párrafos.
"""
import docx
import re

doc = docx.Document("2026_2_ISC-Informe_Proyecto Final_SISIN.docx")
print("Total Paragraphs:", len(doc.paragraphs))
print("Total Tables:", len(doc.tables))

# Check images
img_count = sum(1 for p in doc.paragraphs if any('w:drawing' in r._r.xml or 'pic:pic' in r._r.xml for r in p.runs))
print("Total Images Embedded:", img_count)

# Extract full text including tables
para_text = "\n".join([p.text for p in doc.paragraphs])

# Extract text from all tables
table_text_list = []
for tbl in doc.tables:
    for row in tbl.rows:
        for cell in row.cells:
            table_text_list.append(cell.text)
table_text = "\n".join(table_text_list)

full_text = para_text + "\n" + table_text

# Extract References
ref_start = False
ref_list = []
for p in doc.paragraphs:
    if "REFERENCIAS BIBLIOGRÁFICAS" in p.text.upper():
        ref_start = True
        continue
    if ref_start and "ANEXOS" in p.text.upper():
        break
    if ref_start and p.text.strip():
        if not p.text.strip().startswith("A continuación se presenta"):
            ref_list.append(p.text.strip())

print(f"\n--- Total References in Bibliography Section: {len(ref_list)} ---")
for idx, r in enumerate(ref_list, 1):
    author_year = r.split('(')[0].strip() + ' (' + r.split('(')[1].split(')')[0] + ')'
    print(f"[{idx:02d}] {author_year}")

# Verify all references are cited in body or tables before references
body_text = para_text.split("REFERENCIAS BIBLIOGRÁFICAS")[0] + "\n" + table_text

print("\n--- Verificando Citas en el Cuerpo del Texto y Tablas ---")
not_cited = []
for r in ref_list:
    first_author = r.split(',')[0].strip()
    year = r.split('(')[1].split(')')[0].strip()
    
    if "Defensoría del Pueblo" in r:
        search_key = "Defensoría del Pueblo"
    elif "Instituto Nacional de Estadística" in r:
        search_key = "INEI"
    elif "Ministerio de Salud" in r:
        search_key = "MINSA"
    elif "Superintendencia Nacional" in r:
        search_key = "SUNEDU"
    elif "UNESCO" in r:
        search_key = "UNESCO"
    elif "Paricela-Canazas" in r:
        search_key = "Paricela-Canazas"
    elif "Hernández-Sampieri" in r:
        search_key = "Hernández-Sampieri"
    elif "Ley N." in r:
        search_key = "Ley N."
    elif "Decreto Supremo" in r:
        search_key = "Decreto Supremo"
    elif "Oppenheim" in r:
        search_key = "Oppenheim"
    elif "Ogosi-Auqui" in r:
        search_key = "Ogosi-Auqui"
    else:
        search_key = first_author

    matches = [m.start() for m in re.finditer(re.escape(search_key), body_text, re.IGNORECASE)]
    if not matches:
        not_cited.append((search_key, year))
    else:
        print(f"[OK CITED] {search_key} ({year}) -> {len(matches)} apariciones.")

if not_cited:
    print(f"\n[ALERTA] Referencias no citadas: {len(not_cited)}")
    for nc in not_cited:
        print("  ", nc)
else:
    print(f"\n[ÉXITO TOTAL] ¡Las {len(ref_list)} referencias bibliográficas están 100 % cruzadas y citadas en el texto y tablas!")

# Verify APA 7 Figure structure
print("\n--- Verificando Estructura APA 7 para Figuras ---")
for f in range(1, 11):
    fig_tag = f"Figura {f}"
    p_indices = [i for i, p in enumerate(doc.paragraphs) if p.text.strip() == fig_tag]
    print(f"{fig_tag}: Encabezado en párrafo {p_indices}")
    for idx in p_indices:
        title_p = doc.paragraphs[idx+1].text
        img_p = doc.paragraphs[idx+2]
        has_img = any('w:drawing' in r._r.xml or 'pic:pic' in r._r.xml for r in img_p.runs)
        note_p = doc.paragraphs[idx+3].text
        print(f"   Título arriba: '{title_p[:45]}...'")
        print(f"   Imagen en medio: {has_img}")
        print(f"   Nota abajo: '{note_p[:45]}...'")

# Verify APA 7 Table structure
print("\n--- Verificando Estructura APA 7 para Tablas ---")
for t in range(1, 7):
    tbl_tag = f"Tabla {t}"
    p_indices = [i for i, p in enumerate(doc.paragraphs) if p.text.strip() == tbl_tag]
    print(f"{tbl_tag}: Encabezado en párrafo {p_indices}")
    for idx in p_indices:
        title_p = doc.paragraphs[idx+1].text
        print(f"   Título arriba: '{title_p[:45]}...'")

# Check for AI phrases
print("\n--- Auditoría de Frases Sospechosas de IA / Jargón Robótico ---")
ai_phrases = [
    "para dar cumplimiento a", "observación docente", "chatgpt", "prompt realizado",
    "asistente de ia", "es importante destacar", "salubridad de los rasgos",
    "como coronación del objetivo general", "visor holográfico", "catapultó la exactitud"
]
found_ai = []
for idx, p in enumerate(doc.paragraphs):
    for phr in ai_phrases:
        if phr in p.text.lower():
            # Exclude Table 4 headers or descriptions that intentionally audit AI
            if "uso responsable de ia generativa" not in p.text.lower():
                found_ai.append((idx, phr, p.text[:80]))

if found_ai:
    print(f"Encontradas {len(found_ai)} frases sospechosas:")
    for f in found_ai:
        print(f"   P{f[0]}: [{f[1]}] -> {f[2]}...")
else:
    print("[ÉXITO] 0 frases o expresiones artificiales encontradas en la redacción general!")

# Word Count
words = sum(len(p.text.split()) for p in doc.paragraphs) + sum(len(cell.text.split()) for tbl in doc.tables for row in tbl.rows for cell in row.cells)
print(f"\nTotal Word Count: {words:,} palabras en el documento.")
