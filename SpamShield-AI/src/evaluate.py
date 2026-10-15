"""
Módulo de evaluación del modelo SpamShield AI.

Genera:
- Métricas de clasificación (accuracy, precision, recall, F1)
- Matriz de confusión gráfica
- Gráficas de entrenamiento (loss y accuracy)
"""

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# --- Rutas ---
ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"


def load_model_and_data():
    """Carga el modelo entrenado y el conjunto de prueba."""
    model_path = MODELS_DIR / "spamshield_lstm.keras"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontró {model_path}. Ejecuta primero train.py"
        )

    model = tf.keras.models.load_model(model_path)

    test_path = PROCESSED_DIR / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(
            f"No se encontró {test_path}. Ejecuta primero data_preprocessing.py"
        )

    test_df = pd.read_csv(test_path)
    return model, test_df


def plot_training_history() -> None:
    """Genera gráficas de pérdida y accuracy durante el entrenamiento."""
    # Cargar historial desde el config (guardado por train.py)
    config_path = MODELS_DIR / "config.json"
    if not config_path.exists():
        print("No se encontró config.json — omitiendo gráficas de entrenamiento")
        return

    with open(config_path) as f:
        config = json.load(f)

    # Las gráficas de entrenamiento se generan desde el historial completo
    # que se guarda en results/training_history.json durante train.py
    history_path = RESULTS_DIR / "training_history.json"
    if not history_path.exists():
        print("No se encontró training_history.json — omitiendo gráficas")
        return

    with open(history_path) as f:
        history = json.load(f)

    epochs = range(1, len(history["loss"]) + 1)

    # Gráfica de pérdida
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history["loss"], "b-", label="Pérdida entrenamiento", linewidth=2)
    ax.plot(epochs, history["val_loss"], "r-", label="Pérdida validación", linewidth=2)
    ax.set_title("Pérdida de entrenamiento vs validación", fontsize=14)
    ax.set_xlabel("Época", fontsize=12)
    ax.set_ylabel("Pérdida (binary crossentropy)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "training_loss.png", dpi=150)
    plt.close()
    print("Guardada: results/training_loss.png")

    # Gráfica de accuracy
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history["accuracy"], "b-", label="Accuracy entrenamiento", linewidth=2)
    ax.plot(epochs, history["val_accuracy"], "r-", label="Accuracy validación", linewidth=2)
    ax.set_title("Accuracy de entrenamiento vs validación", fontsize=14)
    ax.set_xlabel("Época", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "training_accuracy.png", dpi=150)
    plt.close()
    print("Guardada: results/training_accuracy.png")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Genera y guarda la matriz de confusión."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Ham (0)", "Spam (1)"],
        yticklabels=["Ham (0)", "Spam (1)"],
        ax=ax,
    )
    ax.set_title("Matriz de confusión — Conjunto de prueba", fontsize=14)
    ax.set_xlabel("Etiqueta predicha", fontsize=12)
    ax.set_ylabel("Etiqueta real", fontsize=12)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()
    print("Guardada: results/confusion_matrix.png")


def evaluate() -> None:
    """Evalúa el modelo con el conjunto de prueba."""
    print("=" * 60)
    print("SpamShield AI — Evaluación del modelo")
    print("=" * 60)

    model, test_df = load_model_and_data()

    X_test = test_df["message"].values
    y_true = test_df["label"].values

    # Predicciones
    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    # Métricas globales
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\n--- Métricas globales ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    # Reporte por clase
    print(f"\n--- Reporte de clasificación ---")
    print(classification_report(
        y_true, y_pred,
        target_names=["Ham (0)", "Spam (1)"],
    ))

    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"--- Matriz de confusión ---")
    print(f"True Negatives (ham correcto):  {tn}")
    print(f"False Positives (ham → spam):    {fp}")
    print(f"False Negatives (spam → ham):    {fn}")
    print(f"True Positives (spam correcto):  {tp}")

    # Generar gráficas
    plot_confusion_matrix(y_true, y_pred)
    plot_training_history()

    # Guardar métricas en JSON
    metrics = {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }
    metrics_path = RESULTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMétricas guardadas: {metrics_path}")


if __name__ == "__main__":
    evaluate()
