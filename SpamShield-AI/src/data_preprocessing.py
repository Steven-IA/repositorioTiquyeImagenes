"""
Módulo de carga y preprocesamiento de datos para SpamShield AI.

Flujo:
1. Cargar el dataset SMS Spam Collection desde data/raw/.
2. Limpiar textos (normalización, eliminación de duplicados).
3. Codificar etiquetas (ham=0, spam=1).
4. Dividir en train/val/test estratificado (70/15/15).
5. Guardar los subconjuntos en data/processed/.
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
from sklearn.model_selection import train_test_split

# --- Rutas relativas a la raíz del proyecto ---
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

DATASET_FILENAME = "SMSSpamCollection"
RANDOM_SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15


def find_dataset_file() -> Path:
    """Busca el archivo del dataset en data/raw/."""
    candidates = [
        RAW_DIR / DATASET_FILENAME,
        RAW_DIR / "SMSSpamCollection.txt",
        RAW_DIR / "smsspamcollection.tsv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No se encontró el dataset en {RAW_DIR}. "
        "Descárgalo desde https://archive.ics.uci.edu/dataset/228/sms+spam+collection"
    )


def load_dataset() -> pd.DataFrame:
    """Carga el dataset y devuelve un DataFrame con columnas ['label', 'message']."""
    dataset_path = find_dataset_file()
    df = pd.read_csv(
        dataset_path,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="latin-1",
    )
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia el dataset: elimina nulos, duplicados y normaliza textos."""
    initial_rows = len(df)

    # Eliminar filas con valores nulos
    df = df.dropna(subset=["label", "message"])

    # Convertir mensaje a string
    df["message"] = df["message"].astype(str)

    # Normalizar etiquetas a minúsculas
    df["label"] = df["label"].str.strip().str.lower()

    # Eliminar mensajes vacíos o solo espacios
    df = df[df["message"].str.strip() != ""]

    # Eliminar espacios múltiples
    df["message"] = df["message"].str.replace(r"\s+", " ", regex=True).str.strip()

    # Eliminar duplicados
    df = df.drop_duplicates(subset=["message"]).reset_index(drop=True)

    final_rows = len(df)
    print(f"Limpieza: {initial_rows} → {final_rows} registros "
          f"({initial_rows - final_rows} eliminados)")

    return df


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Codifica etiquetas: ham=0, spam=1."""
    label_map = {"ham": 0, "spam": 1}
    df["label_encoded"] = df["label"].map(label_map)
    return df


def split_dataset(df: pd.DataFrame) -> dict:
    """Divide el dataset en train/val/test estratificado (70/15/15)."""
    X = df["message"].values
    y = df["label_encoded"].values

    # Primera separación: train (70%) vs temp (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(TEST_SIZE + VAL_SIZE),
        random_state=RANDOM_SEED, stratify=y,
    )

    # Segunda separación: val (15%) vs test (15%) — ajuste proporcional
    val_ratio = VAL_SIZE / (TEST_SIZE + VAL_SIZE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_ratio),
        random_state=RANDOM_SEED, stratify=y_temp,
    )

    splits = {
        "train": pd.DataFrame({"message": X_train, "label": y_train}),
        "val": pd.DataFrame({"message": X_val, "label": y_val}),
        "test": pd.DataFrame({"message": X_test, "label": y_test}),
    }

    for name, split_df in splits.items():
        spam_count = (split_df["label"] == 1).sum()
        ham_count = (split_df["label"] == 0).sum()
        print(f"{name:5s}: {len(split_df):4d} ejemplos "
              f"(ham={ham_count}, spam={spam_count})")

    return splits


def save_splits(splits: dict) -> None:
    """Guarda los subconjuntos en data/processed/."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, split_df in splits.items():
        output_path = PROCESSED_DIR / f"{name}.csv"
        split_df.to_csv(output_path, index=False)
        print(f"Guardado: {output_path}")


def main() -> None:
    """Ejecuta el flujo completo de preprocesamiento."""
    print("=" * 60)
    print("SpamShield AI — Preprocesamiento de datos")
    print("=" * 60)

    df = load_dataset()
    print(f"\nDataset cargado: {len(df)} registros")
    print(f"Columnas: {list(df.columns)}")
    print(f"Distribución de clases:\n{df['label'].value_counts()}")

    df = clean_dataset(df)
    df = encode_labels(df)

    print(f"\nDistribución tras limpieza:\n{df['label'].value_counts()}")

    splits = split_dataset(df)
    save_splits(splits)

    print("\nPreprocesamiento completado.")


if __name__ == "__main__":
    main()
