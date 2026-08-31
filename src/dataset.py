# ============================================================================
# MODULO DE CARGA Y PREPROCESAMIENTO DEL DATASET FER2013
# Reconocimiento de Emociones Faciales con CNN
# Autor: Manuel Fabrizzio Risco Gil
# ============================================================================
# Este modulo se encarga de:
# 1. Cargar el dataset FER2013 (formato CSV o carpetas de imagenes)
# 2. Aplicar Data Augmentation para mejorar la generalizacion
# 3. Crear DataLoaders de PyTorch para entrenamiento eficiente con GPU
# ============================================================================

import os
import sys
import io
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Forzar UTF-8 en Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

# Asegurar que el directorio raiz del proyecto este en sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    DATA_DIR, FER2013_CSV_PATH, IMG_SIZE,
    BATCH_SIZE, NUM_WORKERS, EMOTION_LABELS, DEVICE
)


class FER2013DatasetCSV(Dataset):
    """
    Dataset personalizado para cargar FER2013 desde archivo CSV con pre-parsing en RAM.
    """

    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.labels = self.dataframe["emotion"].values.astype(np.int64)

        print(f"   [DATA] Pre-parseando {len(self.dataframe)} imagenes del CSV a RAM...")
        pixels_list = [np.fromstring(p, dtype=np.uint8, sep=" ") for p in self.dataframe["pixels"]]
        self.data = np.vstack(pixels_list).reshape(-1, IMG_SIZE, IMG_SIZE)
        print(f"   [OK] Pre-parsing completado ({self.data.nbytes / (1024*1024):.1f} MB en RAM)")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.fromarray(self.data[idx], mode="L")

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        label = int(self.labels[idx])
        return image, label


class FER2013DatasetFolder(Dataset):
    """
    Dataset personalizado para cargar FER2013 desde carpetas de imagenes.
    Optimizado: Pre-carga todas las imagenes en memoria RAM (uint8) para eliminar el cuello de botella de disco.
    """

    FOLDER_TO_LABEL = {
        "angry": 0,
        "disgust": 1,
        "fear": 2,
        "happy": 3,
        "sad": 4,
        "surprise": 5,
        "neutral": 6,
    }

    def __init__(self, root_dir, transform=None, preload=True):
        self.root_dir = root_dir
        self.transform = transform
        self.preload = preload
        self.image_paths = []
        self.labels = []

        for emotion_name, label in self.FOLDER_TO_LABEL.items():
            emotion_dir = os.path.join(root_dir, emotion_name)
            if os.path.isdir(emotion_dir):
                for img_name in os.listdir(emotion_dir):
                    img_path = os.path.join(emotion_dir, img_name)
                    if img_path.lower().endswith((".jpg", ".jpeg", ".png")):
                        self.image_paths.append(img_path)
                        self.labels.append(label)

        self.labels = np.array(self.labels, dtype=np.int64)
        print(f"   [DATA] Detectadas {len(self.image_paths)} imagenes en {root_dir}")

        if self.preload and len(self.image_paths) > 0:
            print(f"   [DATA] Pre-cargando imagenes en memoria RAM...")
            self.data = np.zeros((len(self.image_paths), IMG_SIZE, IMG_SIZE), dtype=np.uint8)
            for idx, img_path in enumerate(self.image_paths):
                img = Image.open(img_path).convert("L")
                if img.size != (IMG_SIZE, IMG_SIZE):
                    img = img.resize((IMG_SIZE, IMG_SIZE))
                self.data[idx] = np.array(img, dtype=np.uint8)
            print(f"   [OK] Pre-carga completada ({self.data.nbytes / (1024*1024):.1f} MB en RAM)")
        else:
            self.data = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.data is not None:
            image = Image.fromarray(self.data[idx], mode="L")
        else:
            img_path = self.image_paths[idx]
            image = Image.open(img_path).convert("L")
            if image.size != (IMG_SIZE, IMG_SIZE):
                image = image.resize((IMG_SIZE, IMG_SIZE))

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        label = int(self.labels[idx])
        return image, label


class CustomSubset(Dataset):
    """
    Subconjunto personalizado para aplicar transformaciones especificas a splits de datos.
    """
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        if self.dataset.data is not None:
            image = Image.fromarray(self.dataset.data[actual_idx], mode="L")
        else:
            img_path = self.dataset.image_paths[actual_idx]
            image = Image.open(img_path).convert("L")
            if image.size != (IMG_SIZE, IMG_SIZE):
                image = image.resize((IMG_SIZE, IMG_SIZE))

        if self.transform:
            image = self.transform(image)
        elif self.dataset.transform:
            image = self.dataset.transform(image)
        else:
            image = transforms.ToTensor()(image)

        label = int(self.dataset.labels[actual_idx])
        return image, label


def get_transforms():
    """
    Define las transformaciones para entrenamiento y evaluacion.
    """
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    return train_transform, test_transform


def load_data_csv():
    """
    Carga el dataset FER2013 desde el archivo CSV.
    """
    print("\n[DATA] Cargando dataset FER2013 desde CSV...")

    if not os.path.exists(FER2013_CSV_PATH):
        raise FileNotFoundError(
            f"\n[!] No se encontro el archivo: {FER2013_CSV_PATH}\n"
            f"   Por favor descarga el dataset FER2013 de Kaggle y coloca\n"
            f"   el archivo 'fer2013.csv' en la carpeta 'data/'.\n"
            f"   URL: https://www.kaggle.com/datasets/deadskull7/fer2013"
        )

    df = pd.read_csv(FER2013_CSV_PATH)
    print(f"   Total de imagenes: {len(df)}")

    train_df = df[df["Usage"] == "Training"]
    val_df = df[df["Usage"] == "PublicTest"]
    test_df = df[df["Usage"] == "PrivateTest"]

    print(f"   Entrenamiento: {len(train_df)} imagenes")
    print(f"   Validacion:    {len(val_df)} imagenes")
    print(f"   Test:          {len(test_df)} imagenes")

    print("\n   Distribucion de clases (entrenamiento):")
    for label_id, label_name in EMOTION_LABELS.items():
        count = len(train_df[train_df["emotion"] == label_id])
        bar = "=" * (count // 200)
        print(f"   {label_name:>12}: {count:>5} {bar}")

    train_transform, test_transform = get_transforms()

    train_dataset = FER2013DatasetCSV(train_df, transform=train_transform)
    val_dataset = FER2013DatasetCSV(val_df, transform=test_transform)
    test_dataset = FER2013DatasetCSV(test_df, transform=test_transform)

    is_cuda = (DEVICE.type == "cuda")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=is_cuda,
        persistent_workers=(NUM_WORKERS > 0 and is_cuda),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=is_cuda,
        persistent_workers=(NUM_WORKERS > 0 and is_cuda),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=is_cuda,
        persistent_workers=(NUM_WORKERS > 0 and is_cuda),
    )

    print("\n[OK] Dataset cargado exitosamente!")
    return train_loader, val_loader, test_loader


def load_data_folders():
    """
    Carga el dataset FER2013 desde carpetas de imagenes con pre-carga en RAM.
    """
    print("\n[DATA] Cargando dataset FER2013 desde carpetas de imagenes...")

    train_dir = os.path.join(DATA_DIR, "train")
    test_dir = os.path.join(DATA_DIR, "test")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"\n[!] No se encontro la carpeta: {train_dir}\n"
            f"   Por favor descarga el dataset FER2013 de Kaggle y extrae\n"
            f"   las carpetas 'train/' y 'test/' dentro de 'data/'.\n"
            f"   URL: https://www.kaggle.com/datasets/msambare/fer2013"
        )

    train_transform, test_transform = get_transforms()

    # Pre-cargar imágenes base en memoria RAM (sin transform para permitir split limpio)
    train_dataset_full = FER2013DatasetFolder(train_dir, transform=None, preload=True)

    total_train = len(train_dataset_full)
    val_size = int(0.15 * total_train)
    train_size = total_train - val_size

    # Split reproducible
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(total_train, generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = CustomSubset(train_dataset_full, train_indices, transform=train_transform)
    val_dataset = CustomSubset(train_dataset_full, val_indices, transform=test_transform)
    test_dataset = FER2013DatasetFolder(test_dir, transform=test_transform, preload=True)

    print(f"\n   Entrenamiento: {train_size} imagenes")
    print(f"   Validacion:    {val_size} imagenes")
    print(f"   Test:          {len(test_dataset)} imagenes")

    is_cuda = (DEVICE.type == "cuda")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=is_cuda,
        persistent_workers=(NUM_WORKERS > 0 and is_cuda),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=is_cuda,
        persistent_workers=(NUM_WORKERS > 0 and is_cuda),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=is_cuda,
        persistent_workers=(NUM_WORKERS > 0 and is_cuda),
    )

    print("\n[OK] Dataset cargado exitosamente!")
    return train_loader, val_loader, test_loader


def load_data():
    """
    Funcion principal que detecta automaticamente el formato del dataset.
    """
    train_dir = os.path.join(DATA_DIR, "train")
    if os.path.isdir(train_dir):
        return load_data_folders()

    if os.path.exists(FER2013_CSV_PATH):
        return load_data_csv()

    raise FileNotFoundError(
        "\n" + "=" * 60 +
        "\n[!] DATASET NO ENCONTRADO"
        "\n" + "=" * 60 +
        "\nPor favor descarga el dataset FER2013 de Kaggle:"
        "\n"
        "\n  Opcion 1 (Carpetas de imagenes - Recomendado):"
        "\n  https://www.kaggle.com/datasets/msambare/fer2013"
        "\n  -> Extraer 'train/' y 'test/' en la carpeta 'data/'"
        "\n"
        "\n  Opcion 2 (Archivo CSV):"
        "\n  https://www.kaggle.com/datasets/deadskull7/fer2013"
        "\n  -> Colocar 'fer2013.csv' en la carpeta 'data/'"
        "\n" + "=" * 60
    )
