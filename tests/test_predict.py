"""Pruebas de OmniVision AI: metadatos, catálogo, umbral, modelo e inferencia.

Ejecución (PowerShell, desde la raíz):
    .venv\\Scripts\\python -m pytest tests -q
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.common import (
    CATALOG_CSV,
    CLASS_METADATA_JSON,
    CLASS_NAMES_JSON,
    IMG_SIZE,
    MODEL_PATH,
    RESULTS_DIR,
    THRESHOLD_JSON,
    load_class_metadata,
    load_class_names,
)


# ---------------------------------------------------------------------------
# Metadatos y catálogo
# ---------------------------------------------------------------------------
def test_class_names_and_metadata_coherent():
    names = load_class_names()
    meta = load_class_metadata()
    assert len(names) == 67
    assert names == sorted(names), "class_names.json debe estar en orden alfabético"
    assert meta["num_classes"] == 67
    for i, name in enumerate(names):
        cls = meta["classes"][name]
        assert cls["index"] == i
        assert cls["label"] == name
        assert cls["group"] in {"scene", "person", "dinosaur", "insect",
                                "animal", "mosquito", "plant_disease"}


def test_catalog_integrity():
    import csv
    names = set(load_class_names())
    with open(CATALOG_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 40000
    md5s = [r["md5"] for r in rows]
    assert len(md5s) == len(set(md5s)), "no puede haber MD5 repetidos"
    path_splits: dict[str, set[str]] = {}
    labels = set()
    splits_per_label: dict[str, set[str]] = {}
    for r in rows:
        assert r["unified_label"] in names, r["unified_label"]
        assert r["split"] in {"train", "val", "test"}
        path_splits.setdefault(r["path"], set()).add(r["split"])
        labels.add(r["unified_label"])
        splits_per_label.setdefault(r["unified_label"], set()).add(r["split"])
    assert all(len(s) == 1 for s in path_splits.values()), \
        "una imagen no puede estar en dos particiones"
    assert labels == names
    assert all("test" in s for s in splits_per_label.values()), \
        "toda clase debe tener imágenes de prueba"


def test_threshold_derived_from_validation():
    data = json.loads(THRESHOLD_JSON.read_text(encoding="utf-8"))
    sel = data["selected"]
    assert 0.0 < sel["threshold"] < 1.0
    assert sel["precision"] >= 0.85
    assert sel["coverage"] > 0.05
    assert len(data["curve"]) > 50


# ---------------------------------------------------------------------------
# Modelo e inferencia
# ---------------------------------------------------------------------------
def test_build_model_softmax_sums_to_one():
    import tensorflow as tf
    from src.common import build_model

    model, base = build_model(num_classes=7, img_size=IMG_SIZE)
    x = tf.random.uniform((3, IMG_SIZE, IMG_SIZE, 3), maxval=255.0)
    y = model.predict(x, verbose=0)
    assert y.shape == (3, 7)
    np.testing.assert_allclose(y.sum(axis=1), 1.0, rtol=1e-4)
    assert not base.trainable, "la fase 1 debe empezar con el backbone congelado"


@pytest.fixture(scope="module")
def predictor():
    from src.predict import Predictor
    if not MODEL_PATH.exists():
        pytest.skip("modelo no entrenado")
    return Predictor()


def test_predictor_structure(predictor):
    from PIL import Image
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, (90, 60, 3), dtype=np.uint8), "RGB")
    r = predictor.predict(img, top_k=5)
    assert r["best_label"] in predictor.class_names
    probs = [t["probability"] for t in r["top"]]
    assert probs == sorted(probs, reverse=True)
    assert all(t["name_es"] and t["group"] for t in r["top"])
    assert isinstance(r["acepta"], bool)


def test_predict_from_file(tmp_path: Path, predictor):
    from PIL import Image
    p = tmp_path / "sintetica.png"
    Image.new("RGB", (50, 50), color=(30, 120, 200)).save(p)
    r = predictor.predict(str(p), top_k=3)
    assert len(r["top"]) == 3
    assert 0.0 <= r["best_probability"] <= 1.0


def test_model_file_exists_and_loads():
    if not MODEL_PATH.exists():
        pytest.skip("modelo no entrenado")
    import tensorflow as tf
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    assert model.input_shape[1:] == (IMG_SIZE, IMG_SIZE, 3)
    assert model.output_shape[-1] == 67


def test_reports_exist_after_evaluation():
    for name in ("test_report.json", "per_class_metrics.csv",
                 "confusion_matrix.csv", "confusion_matrix.png"):
        assert (RESULTS_DIR / name).exists(), f"falta {name}"
