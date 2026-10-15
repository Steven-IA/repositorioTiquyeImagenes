"""Constantes, utilidades comunes y construcción del modelo de OmniVision AI.

Este módulo es la única fuente de verdad para:
  - rutas del proyecto,
  - tamaño de imagen y semilla,
  - construcción de la única CNN (MobileNetV2 + cabeza propia),
  - carga del modelo y de los metadatos de clases.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
CONFIG_DIR = PROJECT_ROOT / "config"

CATALOG_CSV = DATA_PROCESSED / "catalog.csv"
CLASS_NAMES_JSON = MODELS_DIR / "class_names.json"
CLASS_METADATA_JSON = MODELS_DIR / "class_metadata.json"
MODEL_PATH = MODELS_DIR / "omnivision_cnn.keras"
HISTORY_CSV = RESULTS_DIR / "history.csv"
THRESHOLD_JSON = RESULTS_DIR / "threshold_analysis.json"

LABEL_MAPPING_JSON = CONFIG_DIR / "label_mapping.json"
BALANCE_REPORT_JSON = RESULTS_DIR / "balance_report.json"

# ---------------------------------------------------------------------------
# Hiperparámetros
# ---------------------------------------------------------------------------
IMG_SIZE = 128
BATCH_SIZE = 64
SEED = 42

# Tope de imágenes por clase (entrenamiento+validación). El conjunto de prueba
# se conserva completo. configurable en config/label_mapping.json ("max_per_class").
DEFAULT_MAX_PER_CLASS = 600


def set_global_seed(seed: int = SEED) -> None:
    """Fija semillas de Python y NumPy (TensorFlow se fija en train.py)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dirs() -> None:
    for d in (DATA_RAW, DATA_PROCESSED, MODELS_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_class_names() -> list[str]:
    with open(CLASS_NAMES_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def load_class_metadata() -> dict:
    with open(CLASS_METADATA_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def load_label_mapping() -> dict:
    with open(LABEL_MAPPING_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def build_model(num_classes: int, img_size: int = IMG_SIZE, dropout: float = 0.3):
    """Construye la única CNN final: MobileNetV2 (ImageNet) + cabeza densa.

    El modelo incluye internamente:
      - redimensionado a img_size x img_size,
      - aumentación de datos (solo activa durante entrenamiento),
      - preprocesamiento propio de MobileNetV2,
      - clasificador softmax con `num_classes` salidas.

    De este modo, en inferencia basta con alimentar el modelo con tensores
    RGB en el rango [0, 255]; el preprocesado es idéntico al de entrenamiento.
    """
    import tensorflow as tf
    from tensorflow.keras import layers

    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name="imagen")

    x = layers.RandomFlip("horizontal", name="aug_flip")(inputs)
    x = layers.RandomRotation(0.08, name="aug_rotacion")(x)
    x = layers.RandomZoom(0.1, name="aug_zoom")(x)
    x = layers.RandomTranslation(0.05, 0.05, name="aug_traslacion")(x)
    x = layers.RandomContrast(0.1, name="aug_contraste")(x)
    x = layers.RandomBrightness(0.1, name="aug_brillo")(x)

    base = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    # Preprocesado propio de MobileNetV2 (x/127.5 - 1) como Rescaling,
    # para que el modelo se serialice sin capas Lambda (Keras 3 safe_mode).
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="preprocess_mobilenetv2")(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    salida = layers.Dense(num_classes, activation="softmax", name="salida")(x)

    model = tf.keras.Model(inputs, salida, name="omnivision_cnn")
    return model, base


def load_trained_model(model_path: Path | str = MODEL_PATH):
    """Carga el único modelo CNN guardado, sin reentrenar."""
    import tensorflow as tf

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {model_path}. "
            "Ejecuta primero: py -3.10 -m src.train (o el notebook)."
        )
    return tf.keras.models.load_model(model_path, compile=False)
