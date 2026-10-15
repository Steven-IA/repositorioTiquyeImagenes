"""
SpamShield AI — Aplicación interactiva con Streamlit.

Dashboard de análisis de mensajes SMS basado en un modelo BiLSTM
(Deep Learning, TensorFlow/Keras).

Ejecutar con: streamlit run app.py

Nota de diseño:
    Este archivo solo contiene lógica de presentación. La inferencia se
    delega por completo a `src/predict.py`, que no se modifica aquí.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Backend sin interfaz gráfica, obligatorio en servidor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.ticker import PercentFormatter

# Añadir src/ al path para reutilizar el módulo de inferencia existente
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import predict

# --- Rutas ---
ROOT_DIR = Path(__file__).resolve().parent
METRICS_PATH = ROOT_DIR / "results" / "metrics.json"

# --- Paleta de colores ---
COLOR_SPAM = "#e74c3c"
COLOR_HAM = "#27ae60"
COLOR_NEUTRAL = "#7f8c8d"


# =============================================================================
# Configuración de la página
# =============================================================================
st.set_page_config(
    page_title="SpamShield AI",
    page_icon="🛡️",
    layout="wide",
)


# =============================================================================
# Utilidades internas
# =============================================================================
def is_dark_theme() -> bool:
    """Detecta el tema activo de Streamlit (por defecto, claro)."""
    try:
        return st.get_option("theme.base") == "dark"
    except Exception:
        return False


@st.cache_resource
def load_model():
    """Carga el modelo entrenado una vez y lo cachea durante la sesión."""
    return predict.load_model()


@st.cache_data
def load_model_metrics() -> dict | None:
    """Lee las métricas de evaluación reales del modelo (solo lectura)."""
    if not METRICS_PATH.exists():
        return None
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def init_session_state() -> None:
    """Inicializa el historial de análisis de la sesión."""
    if "history" not in st.session_state:
        st.session_state.history = []


def format_probability(value: float) -> str:
    """
    Formatea una probabilidad como porcentaje legible.

    Evita mostrar '0.0%' para valores muy pequeños, que sería engañoso:
    un 0.01 % real se presenta como '< 0.1%'.
    """
    if value < 0.001:
        return "< 0.1%"
    return f"{value:.1%}"


# =============================================================================
# Estilos
# =============================================================================
def inject_styles() -> None:
    """
    Inyecta la hoja de estilos del dashboard.

    Los colores se expresan con rgba() sobre gris neutro para que las
    tarjetas tengan contraste tanto en tema claro como en tema oscuro.
    """
    st.markdown(
        """
        <style>
        .ss-header {
            padding: 1.5rem 1.7rem;
            border-radius: 16px;
            background: linear-gradient(135deg,
                        rgba(231,76,60,0.12), rgba(41,128,185,0.12));
            border: 1px solid rgba(127,127,127,0.28);
            margin-bottom: 1.1rem;
        }
        .ss-header h1 {
            margin: 0;
            font-size: 2.35rem;
            letter-spacing: -0.5px;
            line-height: 1.15;
        }
        .ss-header p {
            margin: 0.45rem 0 0 0;
            font-size: 1.02rem;
            opacity: 0.82;
        }
        .ss-chip {
            display: inline-block;
            padding: 0.22rem 0.68rem;
            border-radius: 999px;
            background: rgba(127,127,127,0.18);
            border: 1px solid rgba(127,127,127,0.25);
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.03em;
            margin: 0.8rem 0.35rem 0 0;
        }
        .ss-card {
            padding: 0.95rem 1rem;
            border-radius: 14px;
            text-align: center;
            background: rgba(127,127,127,0.10);
            border: 1px solid rgba(127,127,127,0.22);
            border-top: 3px solid var(--ss-accent);
        }
        .ss-card-label {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            font-weight: 700;
            opacity: 0.68;
        }
        .ss-card-value {
            font-size: 1.85rem;
            font-weight: 800;
            margin: 0.22rem 0 0.1rem 0;
            line-height: 1.1;
            color: var(--ss-accent);
        }
        .ss-card-sub {
            font-size: 0.73rem;
            opacity: 0.62;
        }
        .ss-section {
            font-size: 1.02rem;
            font-weight: 700;
            margin: 0.1rem 0 0.55rem 0;
        }
        .ss-note {
            font-size: 0.82rem;
            opacity: 0.7;
            line-height: 1.45;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    """Cabecera principal del dashboard."""
    st.markdown(
        f"""
        <div class="ss-header">
            <h1>🛡️ SpamShield AI</h1>
            <p>Sistema inteligente de detección de mensajes spam mediante
               <strong>Deep Learning</strong> (red neuronal recurrente BiLSTM).</p>
            <div style="margin-top:0.15rem;">
                <span class="ss-chip">Clasificación binaria</span>
                <span class="ss-chip">BiLSTM 64 unidades</span>
                <span class="ss-chip">Embedding 128d</span>
                <span class="ss-chip">Vocabulario 10 000 tokens</span>
                <span class="ss-chip">Secuencias de 200</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_card(label: str, value: str, color: str, subtitle: str = "") -> None:
    """Dibuja una tarjeta visual de resultado."""
    st.markdown(
        f"""
        <div class="ss-card" style="--ss-accent: {color};">
            <div class="ss-card-label">{label}</div>
            <div class="ss-card-value">{value}</div>
            <div class="ss-card-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# Gráficas
# =============================================================================
def plot_probability_bars(p_spam: float, p_ham: float, label: str) -> None:
    """
    Gráfica 1: barras horizontales que comparan P(SPAM) frente a P(HAM).

    Ambos valores provienen de la salida del modelo: es una única neurona
    sigmoid entrenada con binary_crossentropy, por lo que la salida es
    P(spam) y su complemento 1 - P(spam) es P(ham).
    """
    fig, ax = plt.subplots(figsize=(5.4, 2.5))
    fig.patch.set_alpha(0)

    values = [p_spam, p_ham]
    colors = [COLOR_SPAM, COLOR_HAM]
    bars = ax.barh(["SPAM", "HAM"], values, color=colors, height=0.55)

    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Probabilidad estimada", fontsize=9)
    ax.set_title(
        "Comparación de probabilidades",
        fontsize=11,
        fontweight="bold",
        pad=8,
    )
    ax.grid(axis="x", alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)

    # Porcentaje sobre cada barra
    for bar, value in zip(bars, values):
        ax.text(
            min(value + 0.02, 0.94),
            bar.get_y() + bar.get_height() / 2,
            format_probability(value),
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
        )

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.caption(
        f"Clasificación obtenida: **{label.upper()}** (umbral de decisión: 0.50)."
    )


def plot_gauge(p_spam: float, label: str) -> None:
    """
    Gráfica 2: indicador circular.

    El anillo exterior muestra, sólido, la clase predicha por el modelo.
    La corona interior reparte P(spam) y P(ham). El centro muestra P(spam).
    """
    p_ham = 1 - p_spam
    color_main = COLOR_SPAM if label == "spam" else COLOR_HAM

    fig, ax = plt.subplots(figsize=(3.5, 3.3))
    fig.patch.set_alpha(0)

    # Anillo exterior: clase predicha
    ax.pie(
        [1],
        colors=[color_main],
        radius=1.0,
        startangle=90,
        wedgeprops={"width": 0.07, "edgecolor": "none"},
    )
    # Corona interior: reparto de probabilidad
    ax.pie(
        [p_spam, p_ham],
        colors=[COLOR_SPAM, COLOR_HAM],
        radius=0.87,
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.30, "edgecolor": "white", "linewidth": 2},
    )

    ax.text(
        0,
        0.13,
        format_probability(p_spam),
        ha="center",
        va="center",
        fontsize=23,
        fontweight="800",
        color=color_main,
    )
    ax.text(
        0,
        -0.16,
        "P(spam)",
        ha="center",
        va="center",
        fontsize=9.5,
        alpha=0.65,
    )
    ax.set_title(
        "Indicador del resultado",
        fontsize=11,
        fontweight="bold",
        pad=4,
    )
    ax.set_aspect("equal")

    fig.tight_layout()
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

    st.caption("Anillo exterior = clase predicha · corona interior = reparto real.")


def plot_history(history: list) -> None:
    """
    Gráfica 3: probabilidad de spam de cada mensaje analizado en la sesión.
    """
    df = pd.DataFrame(history)
    df = df.iloc[::-1]  # El más reciente arriba

    fig, ax = plt.subplots(figsize=(7.2, max(2.2, 0.52 * len(df) + 1.0)))
    fig.patch.set_alpha(0)

    colors = [COLOR_SPAM if row == "spam" else COLOR_HAM for row in df["label"]]
    y_pos = np.arange(len(df))
    bars = ax.barh(y_pos, df["probability"], color=colors, height=0.6)

    ax.axvline(
        0.5,
        color=COLOR_NEUTRAL,
        linestyle="--",
        linewidth=1.3,
        label="Umbral de decisión (0.50)",
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"#{i} · {t}" for i, t in zip(df["index"], df["hora"])])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Probabilidad de spam", fontsize=9)
    ax.set_title(
        "Historial de análisis de la sesión",
        fontsize=11,
        fontweight="bold",
        pad=22,
    )
    # Leyenda fuera del área de datos para no solaparse con las barras
    ax.legend(
        fontsize=8,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        frameon=False,
    )
    ax.grid(axis="x", alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)

    for bar, value in zip(bars, df["probability"]):
        ax.text(
            min(value + 0.015, 0.95),
            bar.get_y() + bar.get_height() / 2,
            format_probability(value),
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# =============================================================================
# Barra lateral
# =============================================================================
def render_sidebar() -> None:
    """Barra lateral con contexto del modelo, métricas e instrucciones."""
    with st.sidebar:
        st.markdown("#### 🧠 Modelo en uso")
        st.markdown(
            """
            <div class="ss-note">
            Red neuronal recurrente bidireccional (BiLSTM) entrenada sobre el
            dataset <em>SMS Spam Collection</em> de la Universidad de California
            en Irvine (UCI).<br><br>
            Entrada: texto crudo del mensaje.<br>
            Salida: probabilidad de que el mensaje sea spam.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### 📊 Rendimiento real del modelo")
        metrics = load_model_metrics()
        if metrics:
            rows = [
                ("Accuracy", f"{metrics['accuracy']:.4f}"),
                ("Precisión (spam)", f"{metrics['precision']:.4f}"),
                ("Recall (spam)", f"{metrics['recall']:.4f}"),
                ("F1-score (spam)", f"{metrics['f1_score']:.4f}"),
            ]
            for name, value in rows:
                st.markdown(f"- **{name}:** `{value}`")

            with st.expander("Matriz de confusión"):
                st.markdown(
                    f"""
                    - Spam detectado correctamente: **{metrics['true_positives']}**
                    - Spam no detectado: **{metrics['false_negatives']}**
                    - Legítimo marcado como spam: **{metrics['false_positives']}**
                    - Legítimos correctos: **{metrics['true_negatives']}**
                    """
                )
            st.caption("Valores leídos de `results/metrics.json`.")
        else:
            st.info(
                "No se encontró `results/metrics.json`. "
                "Ejecuta `python src/evaluate.py` para generarlo."
            )

        with st.expander("📖 Instrucciones de uso"):
            st.markdown(
                """
                1. **Escribe o pega** el texto del mensaje SMS.
                2. **Pulsa** el botón «Analizar mensaje».
                3. **Revisa el resultado**: las tarjetas, el indicador
                   circular y las barras muestran la distribución real de
                   probabilidades del modelo.

                **Consejos:**
                - El modelo fue entrenado con mensajes en inglés.
                - Funciona mejor con mensajes completos.
                - La predicción es una **estimación** y no garantiza que un
                  mensaje sea fraudulento o legítimo.
                """
            )


# =============================================================================
# Sección de resultados
# =============================================================================
def render_results(result: dict) -> None:
    """Muestra tarjetas, gráficas e interpretación de una predicción."""
    p_spam = result["probability"]
    p_ham = 1.0 - p_spam
    label = result["label"]

    # Confianza real en la clase predicha: la mayor de las dos probabilidades.
    confianza_clase = max(p_spam, p_ham)

    color = COLOR_SPAM if label == "spam" else COLOR_HAM
    color_ham = COLOR_HAM
    color_spam = COLOR_SPAM

    st.markdown('<div class="ss-section">Resultado del análisis</div>',
                unsafe_allow_html=True)

    # --- Tarjetas de resumen ---
    col1, col2, col3 = st.columns(3)

    with col1:
        render_card(
            "Clasificación",
            "SPAM" if label == "spam" else "HAM",
            color,
            "mensaje no deseado" if label == "spam" else "mensaje legítimo",
        )

    with col2:
        render_card(
            "Prob. de spam",
            format_probability(p_spam),
            color_spam,
            f"complemento HAM: {format_probability(p_ham)}",
        )

    with col3:
        render_card(
            "Confianza en la clase",
            format_probability(confianza_clase),
            color,
            "máx. entre P(spam) y P(ham)",
        )

    # Explicación de la distinción entre probabilidad y confianza
    confidence_emoji = {"alta": "🔴", "media": "🟡", "baja": "🟢"}
    st.caption(
        f"**Banda de decisión** (campo `confidence` del módulo de predicción): "
        f"{confidence_emoji.get(result['confidence'], '⚪')} "
        f"**{result['confidence']}** — es una banda fija aplicada sobre la propia "
        f"probabilidad de spam (≥ 0.80 o ≤ 0.20 → alta; ≥ 0.65 o ≤ 0.35 → media), "
        f"no la confianza del modelo en la clase predicha."
    )

    # --- Gráficas ---
    col_left, col_right = st.columns([1.15, 1])

    with col_left:
        with st.container(border=True):
            plot_probability_bars(p_spam, p_ham, label)

    with col_right:
        with st.container(border=True):
            plot_gauge(p_spam, label)

    # --- Barra de progreso (elemento original conservado) ---
    st.progress(
        p_spam,
        text=f"Probabilidad de spam: {format_probability(p_spam)}  ·  "
             f"Probabilidad de ham: {format_probability(p_ham)}",
    )

    # --- Interpretación (frases originales conservadas) ---
    st.markdown('<div class="ss-section">Interpretación</div>',
                unsafe_allow_html=True)

    if label == "spam":
        st.error("## 🚨 SPAM")
        st.warning(
            "Este mensaje muestra características típicas de spam: "
            "lenguaje urgente, ofertas sospechosas o solicitudes "
            "de información personal. Se recomienda no interactuar."
        )
    else:
        st.success("## ✅ HAM (legítimo)")
        st.info(
            "Este mensaje no muestra patrones de spam conocidos. "
            "Sin embargo, siempre verifica la remitente."
        )

    st.caption(
        "Interpretación basada en umbrales de probabilidad del modelo, no en "
        "un análisis de palabras específicas: el modelo no genera explicaciones "
        "lingüísticas."
    )


# =============================================================================
# Historial
# =============================================================================
def render_history_section() -> None:
    """Sección del historial de análisis de la sesión."""
    st.markdown('<div class="ss-section">Historial de la sesión</div>',
                unsafe_allow_html=True)

    history = st.session_state.history

    if not history:
        st.info(
            "Aún no has analizado ningún mensaje en esta sesión. "
            "La gráfica y la tabla aparecerán aquí tras el primer análisis."
        )
        return

    col_table, col_chart, col_clear = st.columns([1.25, 1.25, 0.5])

    with col_table:
        with st.container(border=True):
            df = pd.DataFrame(history)
            st.dataframe(
                df[["index", "mensaje", "label", "probability", "hora"]].rename(
                    columns={
                        "index": "Nº",
                        "mensaje": "Mensaje",
                        "label": "Clase",
                        "probability": "P(spam)",
                        "hora": "Hora",
                    }
                ),
                hide_index=True,
                use_container_width=True,
                column_config={
                    "P(spam)": st.column_config.NumberColumn(
                        "P(spam)", format="%.4f"
                    )
                },
            )

    with col_chart:
        with st.container(border=True):
            plot_history(history)

    with col_clear:
        st.markdown(" ")
        if st.button("🗑️ Limpiar", use_container_width=True, key="clear_history"):
            # Solo borra el historial de la sesión. No toca el modelo ni results/.
            st.session_state.history = []
            st.rerun()

    st.caption(
        f"{len(history)} análisis en esta sesión. El historial se mantiene "
        "mientras la pestaña permanezca abierta y no se guarda en disco."
    )


# =============================================================================
# Aplicación principal
# =============================================================================
def main() -> None:
    is_dark_theme()  # consulted to respect the active theme
    inject_styles()
    init_session_state()
    render_header()

    main_col, side_placeholder = st.columns([2.1, 1])

    with main_col:
        # --- Carga del modelo ---
        model = None
        model_loaded = True
        try:
            model = load_model()
        except FileNotFoundError as e:
            st.error(f"❌ Error al cargar el modelo: {e}")
            st.info(
                "Asegúrate de haber ejecutado el entrenamiento:\n\n"
                "```\npython src/train.py\n```"
            )
            model_loaded = False
        except Exception as e:
            st.error(f"❌ Error inesperado al cargar el modelo: {e}")
            model_loaded = False

        # --- Entrada de texto ---
        st.markdown('<div class="ss-section">📨 Mensaje a analizar</div>',
                    unsafe_allow_html=True)

        with st.container(border=True):
            message = st.text_area(
                "Introduce el texto del mensaje SMS:",
                height=120,
                placeholder="Escribe aquí el mensaje que deseas analizar...",
                disabled=not model_loaded,
                label_visibility="collapsed",
            )
            analyze_button = st.button(
                "🔍 Analizar mensaje",
                type="primary",
                disabled=not model_loaded or not message.strip(),
                use_container_width=False,
            )

        # --- Resultado ---
        if analyze_button and message.strip():
            try:
                result = predict.predict_message(model, message)

                # Registro en el historial de la sesión
                abrev = (
                    message if len(message) <= 60 else message[:57] + "..."
                )
                st.session_state.history.insert(
                    0,
                    {
                        "index": len(st.session_state.history) + 1,
                        "mensaje": abrev,
                        "label": result["label"],
                        "probability": result["probability"],
                        "hora": datetime.now().strftime("%H:%M:%S"),
                    },
                )

                st.divider()
                render_results(result)

            except ValueError as e:
                st.warning(f"⚠️ {e}")
            except Exception as e:
                st.error(f"❌ Error durante la predicción: {e}")

        # --- Historial ---
        st.divider()
        render_history_section()

        # --- Aviso legal ---
        st.divider()
        st.caption(
            "⚠️ **Aviso:** La predicción es una estimación basada en un modelo "
            "entrenado con datos históricos. No constituye una garantía de que "
            "un mensaje sea fraudulento o legítimo. Use siempre su criterio."
        )

    render_sidebar()


if __name__ == "__main__":
    main()