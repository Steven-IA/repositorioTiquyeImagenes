"""Interfaz web de OmniVision AI (Streamlit).

Clasifica una imagen en una de las 67 categorías unificadas de los cuatro
datasets locales del proyecto. El umbral de confianza no se inventa: se deriva
del conjunto de validación (results/threshold_analysis.json).

Ejecución (PowerShell, desde la raíz del proyecto):
    .venv\\Scripts\\streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.common import CLASS_METADATA_JSON, MODEL_PATH, RESULTS_DIR, load_class_names

st.set_page_config(
    page_title="OmniVision AI",
    page_icon=":material/visibility:",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Caché: el modelo se carga una sola vez
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando el modelo CNN…")
def load_predictor():
    from src.predict import Predictor
    return Predictor()


@st.cache_data(show_spinner=False)
def load_report() -> dict | None:
    p = RESULTS_DIR / "test_report.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


@st.cache_data(show_spinner=False)
def load_classes() -> pd.DataFrame:
    metadata = json.loads(CLASS_METADATA_JSON.read_text(encoding="utf-8"))
    rows = [
        {"clase": v["label"], "nombre": v["name_es"], "grupo": v["group"],
         "total": sum(v["counts"].values())}
        for v in metadata["classes"].values()
    ]
    return pd.DataFrame(rows).sort_values(["grupo", "clase"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("OmniVision AI")
st.caption("Un único clasificador CNN para escenas, personas, dinosaurios, "
           "insectos, animales, mosquitos y enfermedades de plantas "
           "(67 categorías de cuatro datasets).")

if not MODEL_PATH.exists():
    st.error(f"No se encontró el modelo en `{MODEL_PATH}`. "
             "Entrenalo primero: `.venv\\Scripts\\python -m src.train`")
    st.stop()

report = load_report()
classes_df = load_classes()
predictor = load_predictor()
thr = predictor.threshold

with st.sidebar:
    st.header("Modelo")
    st.markdown(
        f"- **CNN:** MobileNetV2 (ImageNet) + cabeza densa\n"
        f"- **Categorías:** {len(predictor.class_names)}\n"
        f"- **Entrada:** 128×128 RGB\n"
        f"- **Umbral (derivado en val.):** {thr if thr is not None else '—'}"
    )
    if report:
        st.header("Resultados en prueba")
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{report['accuracy']*100:.1f}%")
        c2.metric("Top-5", f"{report['top5_accuracy']*100:.1f}%")
        c3, c4 = st.columns(2)
        c3.metric("Macro-F1", f"{report['macro_f1']:.3f}")
        c4.metric("Imágenes", f"{report['test_images']:,}".replace(",", "."))
        with st.expander("Exactitud por fuente"):
            st.table(pd.Series(report["per_source_accuracy"], name="accuracy"))
    top_k = st.slider("Predicciones a mostrar (top-k)", 3, 10, 5)
    with st.expander("Acerca de los datos"):
        st.markdown(
            "- **CIFAR-100** (escenas, personas, dinosaurios, insectos, animales)\n"
            "- **Intel Image Classification** (escenas urbanas y naturales)\n"
            "- **PlantVillage** (38 enfermedades de plantas)\n"
            "- **MosqVision-3K** (aedes, anopheles, culex)\n\n"
            "Equivalencias fusionadas (p. ej. *forest* CIFAR ↔ Intel) y el "
            "tope de 600 imágenes por clase están documentados en "
            "`config/label_mapping.json` y `README.md`."
        )

uploaded = st.file_uploader(
    "Subí una imagen (JPG o PNG)",
    type=["jpg", "jpeg", "png"],
)

if uploaded is not None:
    col_img, col_res = st.columns(2)
    with col_img:
        st.subheader("Imagen")
        st.image(uploaded, width="stretch",
                 alt=f"Imagen subida: {uploaded.name}")
    with col_res:
        st.subheader("Predicción")
        with st.spinner("Clasificando…"):
            r = predictor.predict(uploaded.getvalue(), top_k=top_k)

        best = r["top"][0]
        if r["acepta"]:
            st.success(f"**{best['name_es']}** — confianza "
                       f"{best['probability']*100:.1f}% (aceptada)")
        else:
            st.warning(f"**{best['name_es']}** con solo "
                       f"{best['probability']*100:.1f}% de confianza: por debajo "
                       f"del umbral {thr}. Tratala como «sin confianza».")

        chart_df = pd.DataFrame({
            "categoría": [f"{t['name_es']}" for t in r["top"]],
            "probabilidad": [t["probability"] for t in r["top"]],
        }).set_index("categoría")
        st.bar_chart(chart_df, horizontal=True, alt="Probabilidades top-k")

        with st.container(border=True):
            det = pd.DataFrame(r["top"])[["name_es", "label", "group", "probability"]]
            det.columns = ["Categoría (ES)", "Etiqueta", "Grupo", "Probabilidad"]
            st.dataframe(det, hide_index=True, width="stretch")

        with st.expander("Detalle de metadatos"):
            meta = json.loads(CLASS_METADATA_JSON.read_text(encoding="utf-8"))
            cls = meta["classes"][best["label"]]
            st.json({k: cls[k] for k in
                     ("name_es", "group", "sources", "counts", "species",
                      "health_status") if k in cls})

st.divider()
st.subheader("Catálogo de las 67 categorías")
grupo = st.pills("Filtrar por grupo", sorted(classes_df["grupo"].unique()),
                 selection_mode="multi", key="pills_grupo")
filtrado = classes_df if not grupo else classes_df[classes_df["grupo"].isin(grupo)]
st.dataframe(filtrado, hide_index=True, width="stretch",
             column_config={
                 "total": st.column_config.NumberColumn("Imágenes (total)"),
             })
