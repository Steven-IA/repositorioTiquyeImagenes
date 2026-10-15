"""Evaluación completa del modelo OmniVision AI sobre el conjunto de prueba.

Calcula:
  - accuracy global y top-5 en test,
  - precisión/recall/F1 por clase y matriz de confusión,
  - exactitud por fuente de datos (CIFAR-100, Intel, PlantVillage, MosqVision),
  - análisis de umbral de confianza en VALIDACIÓN (no se inventa el umbral:
    se deriva de datos reales para alcanzar ~90% de precisión aceptada),
  - pares de clases más confundidas.

Salidas:
  - results/test_report.json
  - results/per_class_metrics.csv
  - results/confusion_matrix.csv
  - results/threshold_analysis.json   (umbral derivado en validación)
  - results/confusion_matrix.png
  - results/coverage_vs_threshold.png
  - results/top_confusions.png

Uso:
    .venv\\Scripts\\python -m src.evaluate
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from src.common import (
    CATALOG_CSV,
    MODEL_PATH,
    RESULTS_DIR,
    THRESHOLD_JSON,
    ensure_dirs,
    load_class_names,
    load_class_metadata,
)
from src.train import load_catalog, make_dataset

TARGET_PRECISION = 0.90
BATCH = 64


def predict_split(model, paths: list[str], labels: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve probabilidades (N, C) e etiquetas verdaderas (N,)."""
    import tensorflow as tf

    ds = make_dataset(paths, labels, training=False, batch_size=BATCH)
    probs = model.predict(ds, verbose=2)
    return probs, np.asarray(labels)


def choose_threshold(val_probs: np.ndarray, val_labels: np.ndarray) -> dict:
    """Deriva el umbral mínimo de confianza en validación.

    Se busca el umbral más bajo cuya precisión condicional (entre las
    imágenes aceptadas) alcance TARGET_PRECISION, priorizando cobertura.
    """
    preds = val_probs.argmax(axis=1)
    conf = val_probs.max(axis=1)
    correct = preds == val_labels
    rows = []
    chosen = None
    for t in np.arange(0.30, 0.96, 0.01):
        acc_mask = conf >= t
        cov = float(acc_mask.mean())
        prec = float(correct[acc_mask].mean()) if acc_mask.any() else 1.0
        rows.append({"threshold": round(float(t), 2),
                     "coverage": round(cov, 4),
                     "precision": round(prec, 4)})
        if chosen is None and prec >= TARGET_PRECISION and cov > 0.05:
            chosen = rows[-1]
    if chosen is None:  # por si la curva no alcanza el objetivo
        chosen = max(rows, key=lambda r: (r["precision"], r["coverage"]))
    return {"target_precision": TARGET_PRECISION, "selected": chosen, "curve": rows}


def per_class_metrics(cm: np.ndarray, class_names: list[str]) -> list[dict]:
    """Precisión, recall, F1 y soporte a partir de la matriz de confusión."""
    rows = []
    for i, name in enumerate(class_names):
        tp = float(cm[i, i])
        support = float(cm[i].sum())
        predicted = float(cm[:, i].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        rows.append({
            "class": name, "support": int(support),
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4),
        })
    return rows


def plot_confusion(cm: np.ndarray, class_names: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=5)
    ax.set_yticklabels(class_names, fontsize=5)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Verdadero")
    ax.set_title("Matriz de confusión (67 clases, conjunto de prueba)")
    fig.colorbar(im, shrink=0.7)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=130)
    plt.close(fig)


def plot_coverage(curve: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ts = [r["threshold"] for r in curve]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ts, [r["coverage"] for r in curve], label="cobertura")
    ax.plot(ts, [r["precision"] for r in curve], label="precisión aceptada")
    ax.axhline(TARGET_PRECISION, color="r", ls="--", lw=1, label="objetivo 90%")
    ax.set_xlabel("Umbral de confianza")
    ax.set_ylabel("Proporción")
    ax.set_title("Cobertura y precisión vs umbral (validación)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "coverage_vs_threshold.png", dpi=130)
    plt.close(fig)


def plot_top_confusions(cm: np.ndarray, class_names: list[str], n: int = 20) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                pairs.append((float(cm[i, j]), class_names[i], class_names[j]))
    pairs.sort(reverse=True)
    top = pairs[:n]
    fig, ax = plt.subplots(figsize=(10, 7))
    labels = [f"{a} → {b}" for _, a, b in top]
    ax.barh(range(len(top)), [v for v, _, _ in top], color="#c05621")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Imágenes mal clasificadas")
    ax.set_title("Pares de clases más confundidas (test)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "top_confusions.png", dpi=130)
    plt.close(fig)
    return top


def main() -> None:
    ensure_dirs()
    import tensorflow as tf
    from sklearn.metrics import confusion_matrix

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No hay modelo en {MODEL_PATH}; ejecuta src.train primero.")

    t0 = time.time()
    class_names = load_class_names()
    meta = load_class_metadata()
    num_classes = len(class_names)

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print(f"Modelo cargado: {MODEL_PATH} ({num_classes} clases)")

    # --- Umbral derivado en VALIDACIÓN ------------------------------------
    va_p, va_l = load_catalog("val")
    print(f"\nValidación: {len(va_p)} imágenes")
    va_probs, va_labels = predict_split(model, va_p, va_l)
    thr = choose_threshold(va_probs, va_labels)
    THRESHOLD_JSON.write_text(json.dumps(thr, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    print(f"Umbral seleccionado: {thr['selected']}")

    # --- Evaluación en TEST ------------------------------------------------
    te_p, te_l = load_catalog("test")
    print(f"\nPrueba: {len(te_p)} imágenes")
    te_probs, te_labels = predict_split(model, te_p, te_l)
    preds = te_probs.argmax(axis=1)
    conf = te_probs.max(axis=1)
    correct = preds == te_labels
    acc = float(correct.mean())
    top5 = float(np.mean([
        te_labels[i] in np.argsort(te_probs[i])[-5:] for i in range(len(te_labels))
    ]))
    t = thr["selected"]["threshold"]
    accepted = conf >= t
    acc_cond = float(correct[accepted].mean()) if accepted.any() else 0.0
    cov = float(accepted.mean())

    print(f"Accuracy test: {acc:.4f} | top-5: {top5:.4f}")
    print(f"Con umbral {t}: cobertura={cov:.3f} precisión aceptada={acc_cond:.4f}")

    # --- Matriz de confusión y métricas por clase ---------------------------
    cm = confusion_matrix(te_labels, preds, labels=list(range(num_classes)))
    per_class = per_class_metrics(cm, class_names)

    with open(RESULTS_DIR / "confusion_matrix.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + class_names)
        for i, name in enumerate(class_names):
            w.writerow([name] + cm[i].tolist())

    with open(RESULTS_DIR / "per_class_metrics.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["class", "support", "precision",
                                          "recall", "f1"])
        w.writeheader()
        w.writerows(per_class)

    # --- Exactitud por fuente ----------------------------------------------
    src_acc: dict[str, list[bool]] = {}
    with open(CATALOG_CSV, encoding="utf-8") as f:
        test_rows = [r for r in csv.DictReader(f) if r["split"] == "test"]
    path_to_src = {str(Path(r["path"]).resolve()): r["source"] for r in test_rows}
    for p, ok in zip(te_p, correct):
        src_acc.setdefault(path_to_src.get(p, "?"), []).append(bool(ok))
    per_source = {s: round(float(np.mean(v)), 4) for s, v in sorted(src_acc.items())}

    # --- Gráficas -----------------------------------------------------------
    print("Generando gráficas...")
    plot_confusion(cm, class_names)
    plot_coverage(thr["curve"])
    top_pairs = plot_top_confusions(cm, class_names)

    # --- Reporte ------------------------------------------------------------
    f1s = [r["f1"] for r in per_class]
    report = {
        "model": str(MODEL_PATH),
        "num_classes": num_classes,
        "test_images": len(te_p),
        "val_images": len(va_p),
        "accuracy": round(acc, 4),
        "top5_accuracy": round(top5, 4),
        "macro_f1": round(float(np.mean(f1s)), 4),
        "threshold": t,
        "test_coverage_at_threshold": round(cov, 4),
        "test_precision_accepted": round(acc_cond, 4),
        "per_source_accuracy": per_source,
        "worst_10_classes": sorted(per_class, key=lambda r: r["f1"])[:10],
        "best_10_classes": sorted(per_class, key=lambda r: r["f1"],
                                   reverse=True)[:10],
        "top_confusions": [{"true": a, "pred": b, "count": int(v)}
                           for v, a, b in top_pairs[:10]],
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seconds": round(time.time() - t0, 1),
    }
    (RESULTS_DIR / "test_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n== Resultados (test) ==")
    print(f"  accuracy={acc:.4f}  top5={top5:.4f}  macro-F1={report['macro_f1']}")
    print(f"  por fuente: {per_source}")
    print(f"  5 peores clases: {[r['class'] for r in report['worst_10_classes'][:5]]}")
    print(f"\nReportes en {RESULTS_DIR}")


if __name__ == "__main__":
    main()
