"""
Módulo de entrenamiento del modelo BiLSTM para SpamShield AI.

Arquitectura:
  TextVectorization → Embedding → Bidirectional LSTM → Dropout → Dense → Dense(sigmoid)
"""

import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# --- Rutas ---
ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

# --- Hiperparámetros ---
VOCAB_SIZE = 10000
SEQUENCE_LENGTH = 200
EMBEDDING_DIM = 128
LSTM_UNITS = 64
DROPOUT_RATE_1 = 0.3
DROPOUT_RATE_2 = 0.2
BATCH_SIZE = 64
EPOCHS = 20
PATIENCE = 3
RANDOM_SEED = 42


def set_reproducibility() -> None:
    """Fija semillas para reproducibilidad."""
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)


def load_splits() -> dict:
    """Carga los subconjuntos train/val/test desde data/processed/."""
    splits = {}
    for name in ["train", "val", "test"]:
        path = PROCESSED_DIR / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"No se encontró {path}. Ejecuta primero data_preprocessing.py"
            )
        splits[name] = pd.read_csv(path)
    return splits


def build_model() -> models.Model:
    """Construye el modelo BiLSTM con TextVectorization integrada."""
    # Capa de vectorización de texto (se adapta solo con datos de entrenamiento)
    vectorize_layer = layers.TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=SEQUENCE_LENGTH,
        standardize="lower_and_strip_punctuation",
    )

    model = models.Sequential([
        vectorize_layer,
        layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, mask_zero=True),
        layers.Bidirectional(layers.LSTM(LSTM_UNITS, return_sequences=False)),
        layers.Dropout(DROPOUT_RATE_1),
        layers.Dense(32, activation="relu"),
        layers.Dropout(DROPOUT_RATE_2),
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train_model() -> None:
    """Entrena el modelo y guarda el mejor checkpoint."""
    set_reproducibility()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    splits = load_splits()

    X_train = splits["train"]["message"].values
    y_train = splits["train"]["label"].values
    X_val = splits["val"]["message"].values
    y_val = splits["val"]["label"].values

    print(f"\nEntrenamiento: {len(X_train)} ejemplos")
    print(f"Validación:   {len(X_val)} ejemplos")

    model = build_model()

    # Adaptar TextVectorization SOLO con datos de entrenamiento
    model.layers[0].adapt(X_train)
    print(f"Vocabulario adaptado: {model.layers[0].vocabulary_size()} tokens")

    model.summary()

    # Callbacks
    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )

    checkpoint_path = MODELS_DIR / "spamshield_lstm.keras"
    model_checkpoint = callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )

    # Entrenamiento
    print("\nIniciando entrenamiento...")
    start_time = time.time()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping, model_checkpoint],
        verbose=0,
    )

    elapsed = time.time() - start_time
    print(f"\nEntrenamiento completado en {elapsed:.1f} segundos")
    print(f"Épocas ejecutadas: {len(history.history['loss'])}")

    # Guardar historial de entrenamiento para las gráficas
    history_path = RESULTS_DIR / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history.history, f, indent=2)
    print(f"Historial guardado: {history_path}")

    # Guardar configuración
    config = {
        "vocab_size": VOCAB_SIZE,
        "sequence_length": SEQUENCE_LENGTH,
        "embedding_dim": EMBEDDING_DIM,
        "lstm_units": LSTM_UNITS,
        "epochs_trained": len(history.history["loss"]),
        "training_time_seconds": round(elapsed, 2),
        "final_train_loss": round(history.history["loss"][-1], 4),
        "final_val_loss": round(history.history["val_loss"][-1], 4),
        "final_train_accuracy": round(history.history["accuracy"][-1], 4),
        "final_val_accuracy": round(history.history["val_accuracy"][-1], 4),
        "best_val_loss": round(min(history.history["val_loss"]), 4),
        "best_val_accuracy": round(max(history.history["val_accuracy"]), 4),
        "best_epoch": int(np.argmin(history.history["val_loss"])) + 1,
    }

    config_path = MODELS_DIR / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Configuración guardada: {config_path}")
    print(f"Modelo guardado: {checkpoint_path}")


if __name__ == "__main__":
    train_model()
