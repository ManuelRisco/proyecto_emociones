# ============================================================================
# MÓDULO DE CARGA Y PREPROCESAMIENTO DIRECTO EN VRAM (GPU NATIVE)
# Reconocimiento de Emociones Faciales con CNN
# ============================================================================
# Características de Alto Rendimiento:
#   1. Carga el dataset completo en VRAM al inicio (elimina cuellos PCIe)
#   2. Oversampling de clases minoritarias (Disgust: 360→2000+) en CPU/NumPy
#   3. Data Augmentation 100% en CUDA (Flip, Crop, Rotación, Erasing)
#   4. Mixup opcional con alpha reducido (0.1) para no diluir minoritarias
#   5. Fallback transparente a CPU DataLoader si no hay GPU disponible
# ============================================================================

import math
import os
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.utils import ensure_root_in_path, mixup_batch
ensure_root_in_path()

from src.config import (
    DATA_DIR, FER2013_CSV_PATH, IMG_SIZE,
    BATCH_SIZE, NUM_WORKERS, EMOTION_LABELS, NUM_CLASSES,
    DEVICE, LOAD_TO_VRAM, USE_MIXUP, MIXUP_ALPHA, MIN_SAMPLES_PER_CLASS,
)


# ──────────────────────────────────────────────────────────────────────────────
# 1. DATALOADER NATIVO EN VRAM (GPU ULTRA-FAST)
# ──────────────────────────────────────────────────────────────────────────────

class FastGPUDataLoader:
    """
    DataLoader de ultra-alto rendimiento que almacena y procesa todo el dataset
    directamente en VRAM de la GPU.

    Ventajas frente al DataLoader estándar de PyTorch:
    - Transferencia PCIe = 0 durante el entrenamiento (datos ya en GPU).
    - Data Augmentation ejecutado con núcleos CUDA sin overhead de CPU.
    - >12,000 imágenes/segundo en RTX 5060 Ti.

    Augmentation en GPU (solo modo train):
        1. Random Horizontal Flip (p=0.5)
        2. Random Crop +4 px padding — POR IMAGEN (no por batch completo)
        3. Random Rotation ±12° — POR IMAGEN con bilinear interpolation
        4. Random Erasing / Cutout (p=0.25)
    """

    def __init__(
        self,
        images_tensor: torch.Tensor,
        labels_tensor: torch.Tensor,
        batch_size: int = BATCH_SIZE,
        shuffle: bool = True,
        is_train: bool = True,
        use_mixup: bool = False,
        mixup_alpha: float = MIXUP_ALPHA,
    ):
        self.images     = images_tensor
        self.labels     = labels_tensor
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.is_train   = is_train
        self.use_mixup  = use_mixup and is_train
        self.mixup_alpha = mixup_alpha
        self.n_samples  = images_tensor.size(0)

    def __len__(self) -> int:
        return (self.n_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        device = self.images.device

        if self.shuffle:
            indices = torch.randperm(self.n_samples, device=device)
        else:
            indices = torch.arange(self.n_samples, device=device)

        for i in range(0, self.n_samples, self.batch_size):
            batch_idx = indices[i : i + self.batch_size]
            x = self.images[batch_idx]
            y = self.labels[batch_idx]

            # Normalización: uint8 [0, 255] → float32 [-1.0, 1.0]
            x = (x.float() / 255.0 - 0.5) / 0.5

            if self.is_train:
                x = self._augment(x, device)

            if self.use_mixup:
                x, y = mixup_batch(x, y, alpha=self.mixup_alpha, num_classes=NUM_CLASSES)

            yield x, y

    def _augment(self, x: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Aplica augmentation CUDA sobre el batch."""
        b = x.size(0)

        # 1. Random Horizontal Flip por imagen
        flip_mask = torch.rand(b, 1, 1, 1, device=device) < 0.5
        x = torch.where(flip_mask, torch.flip(x, dims=[-1]), x)

        # 2. Random Crop +4 px — POR IMAGEN
        padded = F.pad(x, (4, 4, 4, 4), mode="reflect")  # (B, 1, 56, 56)
        crop_xs = torch.randint(0, 8, (b,), device=device)
        crop_ys = torch.randint(0, 8, (b,), device=device)
        crops = torch.stack([
            padded[i, :, crop_ys[i]: crop_ys[i] + IMG_SIZE, crop_xs[i]: crop_xs[i] + IMG_SIZE]
            for i in range(b)
        ])
        x = crops

        # 3. Random Rotation ±12° — POR IMAGEN con affine grid (bilinear, CUDA)
        max_angle = 12.0
        angles = (torch.rand(b, device=device) * 2 - 1) * max_angle  # [-12, 12]
        angles_rad = angles * (math.pi / 180.0)
        cos_a = torch.cos(angles_rad)
        sin_a = torch.sin(angles_rad)

        # Matriz de rotación 2D afín: shape (B, 2, 3)
        theta = torch.zeros(b, 2, 3, device=device)
        theta[:, 0, 0] = cos_a
        theta[:, 0, 1] = -sin_a
        theta[:, 1, 0] = sin_a
        theta[:, 1, 1] = cos_a
        # theta[:, :, 2] = 0  → sin traslación

        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)

        # 4. Random Erasing / Cutout (25% de probabilidad por batch)
        if torch.rand(1, device=device).item() < 0.25:
            eh = torch.randint(6, 14, (1,)).item()
            ew = torch.randint(6, 14, (1,)).item()
            y0 = torch.randint(0, IMG_SIZE - eh, (1,)).item()
            x0 = torch.randint(0, IMG_SIZE - ew, (1,)).item()
            x[:, :, y0:y0+eh, x0:x0+ew] = torch.randn(
                (b, 1, eh, ew), device=device
            ) * 0.5

        return x


# ──────────────────────────────────────────────────────────────────────────────
# 2. DATASET Y TRANSFORMS CLÁSICOS (Fallback CPU)
# ──────────────────────────────────────────────────────────────────────────────

class FER2013DatasetFolder(Dataset):
    """Dataset que lee imágenes desde carpetas train/test organizadas por emoción."""

    FOLDER_TO_LABEL = {
        "angry": 0, "disgust": 1, "fear": 2, "happy": 3,
        "sad": 4, "surprise": 5, "neutral": 6,
    }

    def __init__(self, root_dir: str, transform=None):
        self.root_dir  = root_dir
        self.transform = transform
        self.image_paths: list[str] = []
        self.labels: list[int] = []

        for emotion_name, label in self.FOLDER_TO_LABEL.items():
            emotion_dir = os.path.join(root_dir, emotion_name)
            if os.path.isdir(emotion_dir):
                for img_name in os.listdir(emotion_dir):
                    img_path = os.path.join(emotion_dir, img_name)
                    if img_path.lower().endswith((".jpg", ".jpeg", ".png")):
                        self.image_paths.append(img_path)
                        self.labels.append(label)

        self._labels_arr = np.array(self.labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self._labels_arr)

    def __getitem__(self, idx: int):
        image = Image.open(self.image_paths[idx]).convert("L")
        if image.size != (IMG_SIZE, IMG_SIZE):
            image = image.resize((IMG_SIZE, IMG_SIZE))
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        return image, int(self._labels_arr[idx])


def get_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """Retorna los transforms de entrenamiento y evaluación para modo CPU."""
    train_transform = transforms.Compose([
        transforms.RandomCrop(IMG_SIZE, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=12),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.92, 1.08)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), value="random"),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    return train_transform, test_transform


# ──────────────────────────────────────────────────────────────────────────────
# 3. OVERSAMPLING DE CLASES MINORITARIAS (CPU / NumPy)
# ──────────────────────────────────────────────────────────────────────────────

def _oversample_minority(
    images: np.ndarray,
    labels: np.ndarray,
    min_samples: int = MIN_SAMPLES_PER_CLASS,
    num_classes: int = NUM_CLASSES,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Duplica imágenes de clases con menos de `min_samples` muestras.

    No crea imágenes sintéticas: repite las existentes con índices aleatorios.
    La variedad proviene de la augmentation GPU que aplica distintos crops,
    flips y rotaciones en cada época.

    Ejemplo con min_samples=2000:
        Disgust:  360 → 2,160  (duplicación con mezcla aleatoria)
        Fear:   3,495 → 3,495  (sin cambio, ya supera el mínimo)
        Happy:  6,104 → 6,104  (sin cambio)
    """
    rng = np.random.default_rng(seed)
    extra_imgs: list[np.ndarray] = []
    extra_lbls: list[np.ndarray] = []

    counts_before = np.bincount(labels, minlength=num_classes)
    print("\n   [OVERSAMPLE] Distribución antes / después del oversampling:")

    for cls in range(num_classes):
        count = counts_before[cls]
        name  = EMOTION_LABELS[cls]
        if count < min_samples:
            n_extra = min_samples - count
            cls_idx = np.where(labels == cls)[0]
            if len(cls_idx) == 0:
                print(f"     {name:>12}:     0  (sin muestras, omitido)")
                continue
            extra_idx = rng.choice(cls_idx, size=n_extra, replace=True)
            extra_imgs.append(images[extra_idx])
            extra_lbls.append(labels[extra_idx])
            print(f"     {name:>12}: {count:>5} → {count + n_extra:>5}  (+{n_extra})")
        else:
            print(f"     {name:>12}: {count:>5}  (sin cambio)")

    if extra_imgs:
        images = np.concatenate([images] + extra_imgs, axis=0)
        labels = np.concatenate([labels] + extra_lbls, axis=0)
        # Mezclar para no dejar todas las repeticiones al final
        perm   = rng.permutation(len(labels))
        images = images[perm]
        labels = labels[perm]

    return images, labels


# ──────────────────────────────────────────────────────────────────────────────
# 4. PESOS DE CLASE (Para FocalLoss / CrossEntropyLoss)
# ──────────────────────────────────────────────────────────────────────────────

def compute_class_weights(
    labels: torch.Tensor | np.ndarray,
    num_classes: int = NUM_CLASSES,
    smooth_factor: float = 0.5,
) -> torch.Tensor:
    """
    Calcula pesos inversamente proporcionales a la frecuencia de cada emoción.

    Nota: se calcula ANTES del oversampling para que los pesos reflejen
    el desbalance real del dataset (no el balanceado artificialmente).
    """
    labels_arr = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else np.array(labels)

    class_counts = np.bincount(labels_arr, minlength=num_classes)
    weights = len(labels_arr) / (num_classes * (class_counts + 1e-6))
    weights = np.power(weights, smooth_factor)
    weights = weights / np.mean(weights)

    print("\n   [PESOS] Ponderación de clases para FocalLoss:")
    for i, (name, count) in enumerate(zip(EMOTION_LABELS.values(), class_counts)):
        print(f"     {name:>12}: {count:>5} muestras | Peso: {weights[i]:.2f}")

    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)


# ──────────────────────────────────────────────────────────────────────────────
# 5. FUNCIONES DE CARGA OPTIMIZADA A VRAM
# ──────────────────────────────────────────────────────────────────────────────

def _read_folder_to_numpy(root_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Lee una carpeta de imágenes y retorna (images uint8, labels int64)."""
    images: list[np.ndarray] = []
    labels: list[int] = []

    for emotion_name, label in FER2013DatasetFolder.FOLDER_TO_LABEL.items():
        emotion_dir = os.path.join(root_dir, emotion_name)
        if os.path.isdir(emotion_dir):
            for img_name in os.listdir(emotion_dir):
                img_path = os.path.join(emotion_dir, img_name)
                if img_path.lower().endswith((".jpg", ".jpeg", ".png")):
                    img = Image.open(img_path).convert("L")
                    if img.size != (IMG_SIZE, IMG_SIZE):
                        img = img.resize((IMG_SIZE, IMG_SIZE))
                    images.append(np.array(img, dtype=np.uint8))
                    labels.append(label)

    return np.array(images, dtype=np.uint8), np.array(labels, dtype=np.int64)


def _numpy_to_vram(
    data: np.ndarray,
    labels: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transfiere arrays numpy a VRAM como uint8 (mínimo uso de memoria)."""
    x = torch.from_numpy(data).unsqueeze(1).to(DEVICE, dtype=torch.uint8)
    y = torch.from_numpy(labels).to(DEVICE, dtype=torch.int64)
    return x, y


def _split_train_val(
    images: np.ndarray,
    labels: np.ndarray,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split reproducible entrenamiento/validación."""
    n = len(labels)
    val_size = int(val_fraction * n)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=generator).numpy()
    train_idx, val_idx = indices[val_size:], indices[:val_size]
    return images[train_idx], labels[train_idx], images[val_idx], labels[val_idx]


def _make_gpu_loaders(
    train_x_raw: np.ndarray,
    train_y_raw: np.ndarray,
    val_x:   np.ndarray,
    val_y:   np.ndarray,
    test_x:  np.ndarray,
    test_y:  np.ndarray,
    return_weights: bool,
):
    """
    Aplica oversampling al conjunto de entrenamiento, transfiere a VRAM
    y crea FastGPUDataLoaders.

    El oversampling ocurre en NumPy (CPU) ANTES de transferir a GPU,
    por lo que no consume VRAM extra permanente — los tensores en VRAM
    ya tienen el tamaño final balanceado.
    """
    # Calcular pesos con distribución ORIGINAL (antes de oversampling)
    class_weights = compute_class_weights(train_y_raw) if return_weights else None

    # Oversampling: balancear clases minoritarias
    train_x_os, train_y_os = _oversample_minority(train_x_raw, train_y_raw)
    del train_x_raw, train_y_raw  # liberar memoria original

    total_mb = (
        train_x_os.nbytes + val_x.nbytes + test_x.nbytes
    ) / (1024 * 1024)

    print(
        f"\n   [VRAM] Transfiriendo tensores a GPU..."
    )
    train_img_t, train_lbl_t = _numpy_to_vram(train_x_os, train_y_os)
    val_img_t,   val_lbl_t   = _numpy_to_vram(val_x,  val_y)
    test_img_t,  test_lbl_t  = _numpy_to_vram(test_x, test_y)
    del train_x_os, val_x, test_x, train_y_os, val_y, test_y

    print(
        f"   [OK] {train_img_t.size(0) + val_img_t.size(0) + test_img_t.size(0):,} imágenes en VRAM "
        f"({total_mb:.1f} MB raw) | "
        f"Train:{train_img_t.size(0):,} Val:{val_img_t.size(0):,} Test:{test_img_t.size(0):,}"
    )

    train_loader = FastGPUDataLoader(
        train_img_t, train_lbl_t, shuffle=True,  is_train=True,
        use_mixup=USE_MIXUP, mixup_alpha=MIXUP_ALPHA,
    )
    val_loader   = FastGPUDataLoader(val_img_t,  val_lbl_t,  shuffle=False, is_train=False)
    test_loader  = FastGPUDataLoader(test_img_t, test_lbl_t, shuffle=False, is_train=False)

    if return_weights:
        return train_loader, val_loader, test_loader, class_weights
    return train_loader, val_loader, test_loader


def load_data_folders(return_weights: bool = False):
    """Carga el dataset FER-2013 desde carpetas train/test a VRAM."""
    train_dir = os.path.join(DATA_DIR, "train")
    test_dir  = os.path.join(DATA_DIR, "test")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"\n[!] No se encontró: {train_dir}\n"
            f"   Extrae 'train/' y 'test/' dentro de la carpeta 'data/'."
        )

    print("\n[DATA] Leyendo imágenes de disco...")
    train_imgs_raw, train_lbls_raw = _read_folder_to_numpy(train_dir)
    test_imgs_raw,  test_lbls_raw  = _read_folder_to_numpy(test_dir)

    train_data, train_labels, val_data, val_labels = _split_train_val(
        train_imgs_raw, train_lbls_raw
    )
    del train_imgs_raw, train_lbls_raw

    if LOAD_TO_VRAM and DEVICE.type == "cuda":
        return _make_gpu_loaders(
            train_data, train_labels,
            val_data,   val_labels,
            test_imgs_raw, test_lbls_raw,
            return_weights,
        )

    # ── Fallback CPU ──────────────────────────────────────────────────────────
    train_tf, test_tf = get_transforms()
    train_ds = FER2013DatasetFolder(train_dir, transform=train_tf)
    val_ds   = FER2013DatasetFolder(train_dir, transform=test_tf)
    test_ds  = FER2013DatasetFolder(test_dir,  transform=test_tf)

    kw = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **kw)

    if return_weights:
        return train_loader, val_loader, test_loader, compute_class_weights(train_labels)
    return train_loader, val_loader, test_loader


def load_data_csv(return_weights: bool = False):
    """Carga el dataset FER-2013 desde el archivo fer2013.csv a VRAM."""
    import pandas as pd

    if not os.path.exists(FER2013_CSV_PATH):
        raise FileNotFoundError(f"[!] No se encontró: {FER2013_CSV_PATH}")

    print("\n[DATA] Leyendo CSV FER-2013...")
    df = pd.read_csv(FER2013_CSV_PATH)

    def _parse(sub_df: "pd.DataFrame"):
        pix = [np.fromstring(p, dtype=np.uint8, sep=" ") for p in sub_df["pixels"]]
        x = np.vstack(pix).reshape(-1, IMG_SIZE, IMG_SIZE)
        y = sub_df["emotion"].values.astype(np.int64)
        return x, y

    train_data, train_labels = _parse(df[df["Usage"] == "Training"].reset_index(drop=True))
    val_data,   val_labels   = _parse(df[df["Usage"] == "PublicTest"].reset_index(drop=True))
    test_data,  test_labels  = _parse(df[df["Usage"] == "PrivateTest"].reset_index(drop=True))
    del df

    if LOAD_TO_VRAM and DEVICE.type == "cuda":
        return _make_gpu_loaders(
            train_data, train_labels,
            val_data,   val_labels,
            test_data,  test_labels,
            return_weights,
        )

    return load_data_folders(return_weights=return_weights)


def load_data(return_weights: bool = False):
    """
    Punto de entrada principal para carga de datos.
    Detecta automáticamente si el dataset está en carpetas (train/test)
    o en formato CSV (fer2013.csv) y lo transfiere a VRAM con oversampling.
    """
    if os.path.isdir(os.path.join(DATA_DIR, "train")):
        return load_data_folders(return_weights=return_weights)
    if os.path.exists(FER2013_CSV_PATH):
        return load_data_csv(return_weights=return_weights)
    raise FileNotFoundError(
        "Dataset no encontrado. Coloca 'train/' y 'test/' en data/ "
        "o el archivo 'fer2013.csv' en data/."
    )
