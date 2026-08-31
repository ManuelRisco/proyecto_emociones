# 📁 Dataset FER2013 - Instrucciones de Descarga

## ¿Qué es FER2013?

FER2013 (Facial Expression Recognition 2013) es un dataset público de referencia para el reconocimiento de emociones faciales. Contiene **35,887 imágenes** en escala de grises de **48x48 píxeles**, clasificadas en 7 emociones.

## Clases del Dataset

| ID | Emoción | Cantidad Aprox. |
|----|---------|-----------------|
| 0 | Angry (Enojado) | 4,953 |
| 1 | Disgust (Disgusto) | 547 |
| 2 | Fear (Miedo) | 5,121 |
| 3 | Happy (Feliz) | 8,989 |
| 4 | Sad (Triste) | 6,077 |
| 5 | Surprise (Sorprendido) | 4,002 |
| 6 | Neutral | 6,198 |

## Opción 1: Descargar como CSV (Recomendado)

1. Ve a: https://www.kaggle.com/datasets/deadskull7/fer2013
2. Inicia sesión o crea una cuenta en Kaggle (gratuito)
3. Haz clic en **"Download"**
4. Extrae el archivo `fer2013.csv`
5. Coloca el archivo `fer2013.csv` en esta carpeta (`data/`)

## Opción 2: Descargar como imágenes organizadas en carpetas

1. Ve a: https://www.kaggle.com/datasets/msambare/fer2013
2. Inicia sesión o crea una cuenta en Kaggle (gratuito)
3. Haz clic en **"Download"**
4. Extrae las carpetas `train/` y `test/` dentro de esta carpeta (`data/`)

La estructura debe quedar así:
```
data/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── sad/
│   ├── surprise/
│   └── neutral/
├── test/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── sad/
│   ├── surprise/
│   └── neutral/
└── README.md (este archivo)
```

## Opción 3: Descargar con la API de Kaggle

```bash
pip install kaggle
kaggle datasets download -d msambare/fer2013
unzip fer2013.zip -d data/
```

## Nota

El proyecto soporta **ambos formatos** (CSV y carpetas de imágenes). El script `dataset.py` detectará automáticamente qué formato tienes disponible.
