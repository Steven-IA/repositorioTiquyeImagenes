"""Entrenamiento del único clasificador CNN de OmniVision AI.

Fases:
  1. MobileNetV2 congelado + cabeza densa (Adam 1e-3).
  2. Fine-tune de las últimas capas de MobileNetV2 (Adam 1e-4).

Salidas:
  - models/omnivision_cnn.keras       (mejor modelo por val_accuracy)
  - results/history.csv               (una fila por época, con segundos/época)
  - results/training_log.json         (resumen reproducible de la ejecución)

Uso:
    .venv\\Scripts\\python -m src.train                      # entrenamiento completo
    .venv\\Scripts\\python -m src.train --benchmark         # 1 época rápida para medir CPU
    .venv\\Scripts\\python -m src.train --epochs1 8 --epochs2 4
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import Counter
from pathlib import Path

from src.common import (
    BATCH_SIZE,
    CLASS_NAMES_JSON,
    CATALOG_CSV,
    HISTORY_CSV,
    IMG_SIZE,
    MODEL_PATH,
    RESULTS_DIR,
    SEED,
    build_model,
    ensure_dirs,
    load_class_names,
    set_global_seed,
)

# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
def load_catalog(split: str) -> tuple[list[str], list[int]]:
    """Rutas absolutas y etiquetas (índices de class_names.json) de un split."""
    class_names = load_class_names()
    index = {c: i for i, c in enumerate(class_names)}
    paths: list[str] = []
    labels: list[int] = []
    with open(CATALOG_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] != split:
                continue
            lab = row["unified_label"]
            if lab not in index:
                raise KeyError(f"Etiqueta {lab!r} no está en class_names.json")
            paths.append(str(Path(row["path"]).resolve()))
            labels.append(index[lab])
    if not paths:
        raise ValueError(f"El split {split!r} está vacío")
    return paths, labels


def make_dataset(paths: list[str], labels: list[int], *, training: bool,
                 batch_size: int = BATCH_SIZE):
    """tf.data: decodifica JPEG/PNG a RGB float32 [0,255] y agrupa en lotes.

    La aumentación y el preprocesado viven dentro del modelo (src/common.py),
    por lo que aquí solo se decodifica y redimensiona.
    """
    import tensorflow as tf

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(buffer_size=4096, seed=SEED, reshuffle_each_iteration=True)

    def decode(path, label):
        img = tf.io.read_file(path)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], antialias=True)
        return tf.cast(img, tf.float32), label

    ds = ds.map(decode, num_parallel_calls=8, deterministic=False)
    ds = ds.batch(batch_size, drop_remainder=False)
    return ds.prefetch(tf.data.AUTOTUNE)


def balanced_class_weights(labels: list[int], num_classes: int) -> dict[int, float]:
    counts = Counter(labels)
    n = len(labels)
    return {c: (n / (num_classes * counts[c])) for c in counts}


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------
def append_history(row: dict, fieldnames: list[str]) -> None:
    exists = HISTORY_CSV.exists()
    with open(HISTORY_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow(row)


def compile_model(model, lr: float) -> None:
    import tensorflow as tf
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5"),
        ],
    )


def make_callbacks(phase: int, limit_steps: int | None) -> list:
    """Callbacks de Keras: early stopping, mejor checkpoint, log y tiempo/época."""
    import tensorflow as tf

    class HistoryRow(tf.keras.callbacks.Callback):
        """Escribe una fila por época en results/history.csv con los segundos."""

        def __init__(self, phase_: int):
            self.phase = phase_
            self.t0 = 0.0

        def on_epoch_begin(self, epoch, logs=None):
            self.t0 = time.perf_counter()

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            append_history(
                {"epoch": epoch + 1,
                 "phase": self.phase,
                 "epoch_seconds": round(time.perf_counter() - self.t0, 1),
                 "accuracy": logs.get("accuracy", ""),
                 "val_accuracy": logs.get("val_accuracy", ""),
                 "top5": logs.get("top5", ""),
                 "val_top5": logs.get("val_top5", ""),
                 "loss": logs.get("loss", ""),
                 "val_loss": logs.get("val_loss", "")},
                ["epoch", "phase", "epoch_seconds", "accuracy", "val_accuracy",
                 "top5", "val_top5", "loss", "val_loss"])

    callbacks: list = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=3,
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            str(RESULTS_DIR / f"phase{phase}.keras"),
            monitor="val_accuracy", save_best_only=True, verbose=0),
        tf.keras.callbacks.TerminateOnNaN(),
        HistoryRow(phase),
    ]
    if limit_steps:

        class StopAfterBatches(tf.keras.callbacks.Callback):
            def __init__(self, limit: int):
                self.limit = limit
                self.seen = 0

            def on_train_batch_end(self, batch, logs=None):
                self.seen += 1
                if self.seen >= self.limit:
                    self.model.stop_training = True

        callbacks.append(StopAfterBatches(limit_steps))
    return callbacks


def train_phase(model, base, phase: int, epochs: int, train_ds, val_ds,
                class_weight: dict[int, float], lr: float,
                unfreeze_last: int = 0, limit_steps: int | None = None) -> dict:
    """Entrena una fase y devuelve {'best_val_accuracy', 'epochs_run', 'seconds'}."""
    if phase == 2:
        base.trainable = True
        for layer in base.layers[:-unfreeze_last]:
            layer.trainable = False
    else:
        base.trainable = False

    compile_model(model, lr)
    callbacks = make_callbacks(phase, limit_steps)

    t0 = time.perf_counter()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )
    total = time.perf_counter() - t0

    vals = history.history.get("val_accuracy", [0.0])
    return {
        "best_val_accuracy": round(float(max(vals)), 4),
        "epochs_run": len(history.history.get("loss", [])),
        "seconds": round(total, 1),
    }


def run_benchmark(train_ds, num_classes: int, batches: int = 50) -> None:
    """Mide el tiempo real de los lotes de entrenamiento en CPU."""
    import tensorflow as tf

    print(f"== Benchmark: {batches} lotes de la fase 1 (sin validación ni guardado) ==")
    model, _ = build_model(num_classes)
    compile_model(model, 1e-3)

    class Measure(tf.keras.callbacks.Callback):
        def __init__(self, limit: int):
            self.limit = limit
            self.seen = 0
            self.train_seconds = 0.0
            self.t_last = 0.0

        def on_train_begin(self, logs=None):
            self.t_last = time.perf_counter()

        def on_train_batch_end(self, batch, logs=None):
            now = time.perf_counter()
            self.train_seconds += now - self.t_last
            self.t_last = now
            self.seen += 1
            if self.seen >= self.limit:
                self.model.stop_training = True

    m = Measure(batches)
    model.fit(train_ds, epochs=1, callbacks=[m], verbose=2)

    per_batch = m.train_seconds / max(m.seen, 1)
    n_train = len(load_catalog("train")[0])
    est_epoch = per_batch * (n_train / BATCH_SIZE)
    print(f"\nLotes medidos: {m.seen} | tiempo de entrenamiento: {m.train_seconds:.1f}s")
    print(f"Tiempo por lote: {per_batch:.2f}s  ({BATCH_SIZE} imágenes/lote)")
    print(f"Época estimada (train completo): {est_epoch/60:.1f} min")
    print(f"Presupuesto estimado 10+5 épocas (fase 1+2): {est_epoch*15/3600:.1f} h "
          "(sin contar early stopping)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena el clasificador OmniVision AI")
    parser.add_argument("--epochs1", type=int, default=10, help="épocas de la fase 1")
    parser.add_argument("--epochs2", type=int, default=5, help="épocas de la fase 2")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--benchmark", action="store_true",
                        help="mide el tiempo de CPU con ~50 lotes y sale")
    parser.add_argument("--no-finetune", action="store_true", help="omite la fase 2")
    parser.add_argument("--unfreeze-last", type=int, default=30,
                        help="capas finales de MobileNetV2 a descongelar (fase 2)")
    parser.add_argument("--limit-steps", type=int, default=None,
                        help="lotes por época (debug)")
    args = parser.parse_args()

    ensure_dirs()
    set_global_seed(SEED)
    import tensorflow as tf
    tf.keras.utils.set_random_seed(SEED)
    print("GPU:", tf.config.list_physical_devices("GPU") or "ninguna (CPU)")

    class_names = load_class_names()
    num_classes = len(class_names)
    print(f"Clases: {num_classes} | IMG_SIZE={IMG_SIZE} | BATCH={args.batch_size}")

    tr_p, tr_l = load_catalog("train")
    va_p, va_l = load_catalog("val")
    print(f"train={len(tr_p)}  val={len(va_p)}")

    train_ds = make_dataset(tr_p, tr_l, training=True, batch_size=args.batch_size)
    val_ds = make_dataset(va_p, va_l, training=False, batch_size=args.batch_size)
    class_weight = balanced_class_weights(tr_l, num_classes)

    if args.benchmark:
        run_benchmark(train_ds, num_classes)
        return

    HISTORY_CSV.unlink(missing_ok=True)  # historial limpio de esta ejecución
    if MODEL_PATH.exists():
        print(f"Nota: ya existe {MODEL_PATH}; se sobrescribirá al final.")

    # Fase 1
    print("\n=== Fase 1: backbone congelado (Adam 1e-3) ===")
    model, base = build_model(num_classes)
    f1 = train_phase(model, base, 1, args.epochs1, train_ds, val_ds,
                     class_weight, lr=1e-3, limit_steps=args.limit_steps)
    print(f"Fase 1 -> mejor val_accuracy={f1['best_val_accuracy']} "
          f"({f1['epochs_run']} épocas, {f1['seconds']:.0f}s)")
    best = {"val_accuracy": f1["best_val_accuracy"], "phase": 1}
    best_weights = model.get_weights()

    # Fase 2
    f2 = None
    if not args.no_finetune:
        print(f"\n=== Fase 2: fine-tune últimas {args.unfreeze_last} capas (Adam 1e-4) ===")
        f2 = train_phase(model, base, 2, args.epochs2, train_ds, val_ds,
                         class_weight, lr=1e-4, unfreeze_last=args.unfreeze_last,
                         limit_steps=args.limit_steps)
        print(f"Fase 2 -> mejor val_accuracy={f2['best_val_accuracy']} "
              f"({f2['epochs_run']} épocas, {f2['seconds']:.0f}s)")
        if f2["best_val_accuracy"] >= best["val_accuracy"]:
            best = {"val_accuracy": f2["best_val_accuracy"], "phase": 2}
            best_weights = model.get_weights()
        else:
            print("La fase 1 superó a la fase 2 en validación: se conserva la fase 1.")

    # Guarda el mejor modelo (pesos en memoria -> estructura final intacta)
    model.set_weights(best_weights)
    tf.keras.utils.set_random_seed(SEED)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"\nModelo guardado: {MODEL_PATH}  (mejor val_accuracy={best['val_accuracy']})")

    # Historial y resumen
    summary = {
        "model": str(MODEL_PATH.relative_to(MODEL_PATH.parents[1])),
        "num_classes": num_classes,
        "img_size": IMG_SIZE,
        "batch_size": args.batch_size,
        "seed": SEED,
        "device": "CPU (sin GPU disponible en Windows nativo)",
        "train_images": len(tr_p),
        "val_images": len(va_p),
        "phase1": f1,
        "phase2": f2,
        "best_phase": best,
        "tf_version": tf.__version__,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (RESULTS_DIR / "training_log.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Resumen: {RESULTS_DIR / 'training_log.json'}")
    print(f"Historial: {HISTORY_CSV}")


if __name__ == "__main__":
    main()
