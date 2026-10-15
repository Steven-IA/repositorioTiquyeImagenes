"""Predicción con el modelo entrenado de OmniVision AI (CLI y API reutilizable).

Uso:
    .venv\\Scripts\\python -m src.predict ruta/a/imagen.jpg [otra.png ...]
    .venv\\Scripts\\python -m src.predict foto.jpg --top-k 3
    .venv\\Scripts\\python -m src.predict foto.jpg --sin-umbral

API para la app Streamlit:
    from src.predict import Predictor
    p = Predictor()                       # carga modelo + metadatos una vez
    resultado = p.predict(ruta_o_bytes)   # {'top': [...], 'acepta': bool, ...}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.common import (
    IMG_SIZE,
    MODEL_PATH,
    RESULTS_DIR,
    load_class_names,
    load_class_metadata,
)

THRESHOLD_JSON = RESULTS_DIR / "threshold_analysis.json"


def load_threshold() -> float | None:
    """Umbral derivado en validación (results/threshold_analysis.json)."""
    if not THRESHOLD_JSON.exists():
        return None
    data = json.loads(THRESHOLD_JSON.read_text(encoding="utf-8"))
    return float(data["selected"]["threshold"])


class Predictor:
    """Carga el modelo una sola vez y expone predicciones con nombres en español."""

    def __init__(self, model_path: Path | str = MODEL_PATH):
        import tensorflow as tf

        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.class_names = load_class_names()
        self.metadata = load_class_metadata()
        self.threshold = load_threshold()

    def _load_image(self, source) -> "np.ndarray":
        """Imagen (bytes, ruta o PIL) -> tensor float32 [0,255] (1, H, W, 3)."""
        import tensorflow as tf
        from PIL import Image
        import io

        if isinstance(source, (str, Path)):
            img_bytes = Path(source).read_bytes()
        elif isinstance(source, bytes):
            img_bytes = source
        elif isinstance(source, Image.Image):
            img_bytes = _pil_to_png_bytes(source)
        else:
            raise TypeError(f"Entrada no soportada: {type(source)}")

        t = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
        t.set_shape([None, None, 3])
        t = tf.image.resize(t, [IMG_SIZE, IMG_SIZE], antialias=True)
        return tf.cast(t, tf.float32)[tf.newaxis, ...]

    def predict(self, source, top_k: int = 5, threshold: float | None = None) -> dict:
        """Devuelve las top_k clases con probabilidad, nombre ES y aceptación."""
        thr = self.threshold if threshold is None else threshold
        x = self._load_image(source)
        probs = self.model.predict(x, verbose=0)[0]
        order = probs.argsort()[::-1][:top_k]
        top = []
        for i in order:
            label = self.class_names[int(i)]
            cls = self.metadata["classes"].get(label, {})
            top.append({
                "label": label,
                "name_es": cls.get("name_es", label),
                "group": cls.get("group", "unknown"),
                "probability": round(float(probs[int(i)]), 4),
            })
        best_p = float(probs.max())
        return {
            "top": top,
            "best_label": top[0]["label"] if top else None,
            "best_probability": round(best_p, 4),
            "threshold": thr,
            "acepta": (thr is not None and best_p >= thr),
        }


def _pil_to_png_bytes(image) -> bytes:
    import io
    from PIL import Image

    buf = io.BytesIO()
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(buf, format="PNG")
    return buf.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clasifica imágenes con el modelo OmniVision AI")
    parser.add_argument("imagenes", nargs="+", help="rutas de las imágenes")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--sin-umbral", action="store_true",
                        help="ignora el umbral derivado en validación")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit(f"No hay modelo en {MODEL_PATH}. Ejecuta: py -3.10 -m src.train")

    predictor = Predictor()
    thr = None if args.sin_umbral else predictor.threshold
    if thr is not None:
        print(f"Umbral de confianza (derivado en validación): {thr}\n")

    for path in args.imagenes:
        r = predictor.predict(path, top_k=args.top_k, threshold=thr)
        print(f"== {path} ==")
        for i, item in enumerate(r["top"], 1):
            mark = " <-- mejor" if i == 1 else ""
            print(f"  {i}. {item['name_es']} ({item['label']}, {item['group']}) "
                  f"{item['probability']*100:.1f}%{mark}")
        estado = "aceptada" if r["acepta"] else "RECHAZADA (baja confianza)"
        print(f"  -> {r['best_label']} [{estado}]\n")


if __name__ == "__main__":
    main()
