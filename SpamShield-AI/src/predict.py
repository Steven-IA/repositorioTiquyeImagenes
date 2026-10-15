"""
Módulo de predicción reutilizable para SpamShield AI.

Proporciona una función para cargar el modelo y clasificar mensajes.
"""

from pathlib import Path
from typing import Union

import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "spamshield_lstm.keras"

# Mapa de etiquetas
LABEL_MAP = {0: "ham", 1: "spam"}
VALID_LABELS = {"ham", "spam"}


def load_model():
    """Carga el modelo entrenado desde disco."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {MODEL_PATH}. "
            "Ejecuta primero: python src/train.py"
        )
    return tf.keras.models.load_model(str(MODEL_PATH))


def predict_message(model, message: str) -> dict:
    """
    Clasifica un mensaje de texto.

    Args:
        model: Modelo Keras cargado.
        message: Texto del mensaje SMS.

    Returns:
        Diccionario con:
            - 'label': 'ham' o 'spam'
            - 'probability': probabilidad de spam (0.0 a 1.0)
            - 'confidence': nivel de confianza ('alta', 'media', 'baja')
    """
    # Validación de entrada
    if message is None or not isinstance(message, str):
        raise ValueError("El mensaje debe ser una cadena de texto no nula.")

    message = message.strip()
    if len(message) == 0:
        raise ValueError("El mensaje no puede estar vacío.")

    # Predicción (Keras 3.x requiere tf.constant con dtype string)
    probability = float(
        model.predict(tf.constant([message], dtype=tf.string), verbose=0).flatten()[0]
    )

    # Etiqueta predicha
    label = "spam" if probability >= 0.5 else "ham"

    # Nivel de confianza
    if probability >= 0.8 or probability <= 0.2:
        confidence = "alta"
    elif probability >= 0.65 or probability <= 0.35:
        confidence = "media"
    else:
        confidence = "baja"

    return {
        "label": label,
        "probability": round(probability, 4),
        "confidence": confidence,
    }


def predict_batch(model, messages: list) -> list:
    """
    Clasifica un lote de mensajes.

    Args:
        model: Modelo Keras cargado.
        messages: Lista de cadenas de texto.

    Returns:
        Lista de diccionarios con las predicciones.
    """
    if not isinstance(messages, (list, tuple)):
        raise ValueError("messages debe ser una lista o tupla de cadenas.")

    results = []
    for msg in messages:
        results.append(predict_message(model, msg))
    return results


if __name__ == "__main__":
    # Ejemplo de uso
    model = load_model()
    test_messages = [
        "Congratulations! You've won a $1000 gift card. Click here to claim.",
        "Hey, are we still meeting for lunch tomorrow?",
        "URGENT: Your account has been compromised. Call now!",
        "Thanks for the birthday wishes!",
    ]

    print("SpamShield AI — Predicciones de ejemplo")
    print("=" * 60)
    for msg in test_messages:
        result = predict_message(model, msg)
        print(f"Mensaje: {msg[:60]}...")
        print(f"  → {result['label'].upper()} "
              f"(prob={result['probability']}, confianza={result['confidence']})")
        print()
