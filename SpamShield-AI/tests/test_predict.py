"""
Pruebas unitarias para el módulo de predicción de SpamShield AI.

Ejecutar con: pytest tests/test_predict.py -v
"""

import sys
from pathlib import Path

import pytest

# Añadir src/ al path para importar el módulo
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import predict


@pytest.fixture(scope="module")
def model():
    """Carga el modelo una vez para todas las pruebas."""
    return predict.load_model()


def test_model_loads(model):
    """El modelo se puede cargar correctamente."""
    assert model is not None
    assert hasattr(model, "predict")


def test_valid_message_prediction(model):
    """Un mensaje válido produce una predicción."""
    result = predict.predict_message(model, "Free money! Click here now!")
    assert "label" in result
    assert "probability" in result
    assert "confidence" in result


def test_probability_in_range(model):
    """La probabilidad está en el intervalo [0, 1]."""
    result = predict.predict_message(model, "Hello, how are you?")
    assert 0.0 <= result["probability"] <= 1.0


def test_label_is_valid(model):
    """La etiqueta devuelta es 'ham' o 'spam'."""
    result = predict.predict_message(model, "Meeting at 3pm tomorrow")
    assert result["label"] in predict.VALID_LABELS


def test_empty_message_raises_error(model):
    """Un mensaje vacío lanza ValueError."""
    with pytest.raises(ValueError):
        predict.predict_message(model, "")


def test_whitespace_only_raises_error(model):
    """Un mensaje con solo espacios lanza ValueError."""
    with pytest.raises(ValueError):
        predict.predict_message(model, "   ")


def test_none_message_raises_error(model):
    """Un mensaje None lanza ValueError."""
    with pytest.raises(ValueError):
        predict.predict_message(model, None)


def test_non_string_raises_error(model):
    """Un tipo no string lanza ValueError."""
    with pytest.raises(ValueError):
        predict.predict_message(model, 12345)


def test_spam_message_detected(model):
    """Un mensaje claramente spam es detectado como spam."""
    spam_msg = "Congratulations! You won $1000000! Call now to claim your prize!"
    result = predict.predict_message(model, spam_msg)
    assert result["label"] == "spam"
    assert result["probability"] > 0.5


def test_ham_message_detected(model):
    """Un mensaje claramente legítimo es detectado como ham."""
    ham_msg = "Hey, want to grab lunch tomorrow at the usual place?"
    result = predict.predict_message(model, ham_msg)
    assert result["label"] == "ham"
    assert result["probability"] < 0.5


def test_batch_prediction(model):
    """La predicción por lote funciona correctamente."""
    messages = [
        "Free iPhone! Click here!",
        "See you at the meeting tomorrow",
        "URGENT: Claim your reward now!",
    ]
    results = predict.predict_batch(model, messages)
    assert len(results) == 3
    for r in results:
        assert r["label"] in predict.VALID_LABELS
        assert 0.0 <= r["probability"] <= 1.0
