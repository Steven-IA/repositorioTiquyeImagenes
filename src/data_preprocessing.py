"""Preparación local de los cuatro datasets de OmniVision AI.

Lee exclusivamente de la carpeta `dataset/` (nunca descarga), extrae
CIFAR-100 a `data/extracted/` si hace falta, valida cada imagen, detecta
duplicados, construye las particiones, aplica el tope por clase y genera:

  - data/processed/catalog.csv        (catálogo unificado, una fila por imagen)
  - models/class_names.json           (orden exacto de las salidas del modelo)
  - models/class_metadata.json        (grupos, etiquetas originales, equivalencias)
  - results/preprocessing_report.json (conteos, corruptos, duplicados, fuentes)
  - results/balance_report.json       (balance de clases)
  - results/distribution_*.png        (gráficas de distribución)

Uso:
    .venv\\Scripts\\python -m src.data_preprocessing            # reutiliza si existe
    .venv\\Scripts\\python -m src.data_preprocessing --force   # reconstruye
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import pickle
import random
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from src.common import (
    BALANCE_REPORT_JSON,
    CATALOG_CSV,
    CLASS_METADATA_JSON,
    CLASS_NAMES_JSON,
    CONFIG_DIR,
    DATA_PROCESSED,
    IMG_SIZE,
    LABEL_MAPPING_JSON,
    PROJECT_ROOT,
    RESULTS_DIR,
    SEED,
    ensure_dirs,
    load_label_mapping,
)

DATASET_DIR = PROJECT_ROOT / "dataset"
EXTRACTED_DIR = PROJECT_ROOT / "data" / "extracted"

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".ppm"}

# Rutas reales detectadas por src/inspect_dataset.py (documentadas en README)
PATH_CIFAR_TAR = DATASET_DIR / "General" / "cifar-100-python.tar.gz"
PATH_CIFAR_DIR = EXTRACTED_DIR / "cifar100" / "cifar-100-python"
PATH_CIFAR_IMAGES = EXTRACTED_DIR / "cifar100" / "images"
PATH_INTEL = DATASET_DIR / "Paisajes"
PATH_PLANT = DATASET_DIR / "Plantas" / "raw" / "color"
PATH_MOSQ = DATASET_DIR / "Mosquito_dataset"

GROUP_PLANT = "plant_disease"


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def md5_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def unify_plant_label(original: str) -> str:
    """'Tomato___Bacterial_spot' -> 'tomato_bacterial_spot' (determinista)."""
    out = []
    for ch in original.lower():
        out.append(ch if ch.isalnum() else "_")
    label = "".join(out)
    while "__" in label:
        label = label.replace("__", "_")
    return label.strip("_")


def plant_species_health(original: str) -> tuple[str, str]:
    if "___" in original:
        species, health = original.split("___", 1)
    else:
        species, health = original, ""
    return species, health.replace("_", " ").strip()


def stratified_split(labels: list[str], fracs: tuple[float, float, float],
                     seed: int = SEED) -> list[str]:
    """División estratificada 70/15/15 (o las fracciones indicadas) por etiqueta."""
    rng = random.Random(seed)
    by_label: dict[str, list[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        by_label[lab].append(i)
    result = [""] * len(labels)
    for lab, idxs in by_label.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        n = len(idxs)
        n_test = round(fracs[2] * n)
        n_val = round(fracs[1] * n)
        test = idxs[:n_test]
        val = idxs[n_test:n_test + n_val]
        train = idxs[n_test + n_val:]
        # Garantiza al menos una imagen en train cuando la clase es mínima.
        if not train and (test or val):
            train = [val.pop() if val else test.pop()]
        for i in train:
            result[i] = "train"
        for i in val:
            result[i] = "val"
        for i in test:
            result[i] = "test"
    return result


def validate_image(data: bytes) -> tuple[bool, tuple[int, int, str], str]:
    """Verifica los bytes de una imagen. Devuelve (ok, (w, h, formato), error)."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
        with Image.open(io.BytesIO(data)) as im:
            im.convert("RGB")
            w, h = im.size
            fmt = (im.format or "").lower()
        if w <= 0 or h <= 0:
            return False, (0, 0, ""), "dimensiones nulas"
        return True, (w, h, fmt), ""
    except Exception as e:  # noqa: BLE001 - cualquier archivo dañado se excluye
        return False, (0, 0, ""), f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Fuentes
# ---------------------------------------------------------------------------
def ensure_cifar_extracted() -> None:
    """Extrae el tarball oficial a data/extracted/ (sin tocar dataset/)."""
    if not PATH_CIFAR_TAR.exists():
        raise FileNotFoundError(f"No se encontró CIFAR-100 en {PATH_CIFAR_TAR}")
    if (PATH_CIFAR_DIR / "train").exists():
        return
    PATH_CIFAR_DIR.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(PATH_CIFAR_TAR, "r:gz") as tf:
        tf.extractall(PATH_CIFAR_DIR.parent)
    if not (PATH_CIFAR_DIR / "train").exists():
        raise RuntimeError("La extracción de CIFAR-100 no produjo los archivos esperados")


def export_cifar_images(mapping: dict) -> int:
    """Escribe como PNG las imágenes de CIFAR-100 seleccionadas.

    Solo se exportan las clases elegidas en config/label_mapping.json
    (aproximadamente 16.000 imágenes de 32x32). Idempotente.
    """
    selection: dict[str, str] = mapping["cifar100_selection"]
    meta = pickle.load(open(PATH_CIFAR_DIR / "meta", "rb"), encoding="latin1")
    fine = meta["fine_label_names"]
    n_written = 0
    for split_file, split_name in (("train", "train"), ("test", "test")):
        data = pickle.load(open(PATH_CIFAR_DIR / split_file, "rb"), encoding="latin1")
        per_class_counter: dict[str, int] = defaultdict(int)
        for arr, label_idx in zip(data["data"], data["fine_labels"]):
            name = fine[label_idx]
            if name not in selection:
                continue
            i = per_class_counter[name]
            per_class_counter[name] += 1
            out_dir = PATH_CIFAR_IMAGES / split_name / name
            out_path = out_dir / f"{i:05d}.png"
            if out_path.exists():
                n_written += 1
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            r = arr[:1024].reshape(32, 32)
            g = arr[1024:2048].reshape(32, 32)
            b = arr[2048:].reshape(32, 32)
            img = Image.merge("RGB", (Image.fromarray(r), Image.fromarray(g), Image.fromarray(b)))
            img.save(out_path)
            n_written += 1
    return n_written


def collect_cifar(mapping: dict) -> list[dict]:
    selection: dict[str, str] = mapping["cifar100_selection"]
    records: list[dict] = []
    for split_name in ("train", "test"):
        base = PATH_CIFAR_IMAGES / split_name
        if not base.exists():
            continue
        for fine_dir in sorted(base.iterdir()):
            if not fine_dir.is_dir() or fine_dir.name not in selection:
                continue
            official_split = "train" if split_name == "train" else "test"
            for p in sorted(fine_dir.glob("*.png")):
                records.append({
                    "path": str(p.relative_to(PROJECT_ROOT)),
                    "source": "cifar100",
                    "original_label": fine_dir.name,
                    "unified_label": selection[fine_dir.name],
                    "official_split": official_split,
                    "species": "",
                    "health_status": "",
                })
    return records


def collect_intel(mapping: dict) -> list[dict]:
    intel_map: dict[str, str] = mapping["intel_map"]
    records: list[dict] = []
    specs = (("seg_train", "train"), ("seg_test", "test"))
    for part, official in specs:
        base = PATH_INTEL / part / part  # estructura real: seg_train/seg_train/<clase>
        if not base.exists():
            raise FileNotFoundError(f"No se encontró {base}")
        for cls_dir in sorted(base.iterdir()):
            if not cls_dir.is_dir():
                continue
            cls = cls_dir.name.lower()
            if cls not in intel_map:
                raise ValueError(f"Clase de Intel sin mapear: {cls_dir.name}")
            for p in sorted(cls_dir.iterdir()):
                if p.suffix.lower() not in IMG_EXT:
                    continue
                records.append({
                    "path": str(p.relative_to(PROJECT_ROOT)),
                    "source": "intel",
                    "original_label": cls,
                    "unified_label": intel_map[cls],
                    "official_split": official,
                    "species": "",
                    "health_status": "",
                })
    return records


def collect_plantvillage() -> list[dict]:
    if not PATH_PLANT.exists():
        raise FileNotFoundError(f"No se encontró PlantVillage en {PATH_PLANT}")
    records: list[dict] = []
    for cls_dir in sorted(PATH_PLANT.iterdir()):
        if not cls_dir.is_dir():
            continue
        species, health = plant_species_health(cls_dir.name)
        for p in sorted(cls_dir.iterdir()):
            if p.suffix.lower() not in IMG_EXT:
                continue
            records.append({
                "path": str(p.relative_to(PROJECT_ROOT)),
                "source": "plantvillage",
                "original_label": cls_dir.name,
                "unified_label": unify_plant_label(cls_dir.name),
                "official_split": "",
                "species": species,
                "health_status": health,
            })
    return records


def collect_mosquito(mapping: dict) -> list[dict]:
    mosq_map: dict[str, str] = mapping["mosquito_map"]
    mosq_map_ci = {k.lower(): v for k, v in mosq_map.items()}
    if not PATH_MOSQ.exists():
        raise FileNotFoundError(f"No se encontró MosqVision-3K en {PATH_MOSQ}")
    records: list[dict] = []
    for cls_dir in sorted(PATH_MOSQ.iterdir()):
        if not cls_dir.is_dir():
            continue
        key = cls_dir.name
        unified = mosq_map_ci.get(key.lower())
        if unified is None:
            raise ValueError(f"Carpeta de mosquitos sin mapear: {key}")
        for p in sorted(cls_dir.iterdir()):
            if p.suffix.lower() not in IMG_EXT:
                continue
            records.append({
                "path": str(p.relative_to(PROJECT_ROOT)),
                "source": "mosqvision",
                "original_label": key,
                    "unified_label": unified,
                "official_split": "",
                "species": key.capitalize(),
                "health_status": "",
            })
    return records


# ---------------------------------------------------------------------------
# Validación, deduplicación, splits y tope por clase
# ---------------------------------------------------------------------------
def validate_and_hash(records: list[dict]) -> tuple[list[dict], list[dict]]:
    valid: list[dict] = []
    corrupt: list[dict] = []
    for rec in records:
        fp = PROJECT_ROOT / rec["path"]
        try:
            data = fp.read_bytes()
        except OSError as e:
            corrupt.append({**rec, "error": f"OSError: {e}"})
            continue
        ok, (w, h, fmt), err = validate_image(data)
        if not ok:
            corrupt.append({**rec, "error": err})
            continue
        rec["width"], rec["height"], rec["format"] = w, h, fmt
        rec["md5"] = md5_bytes(data)
        rec["bytes"] = len(data)
        valid.append(rec)
    return valid, corrupt


def deduplicate(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Elimina imágenes idénticas (mismo MD5). Prioridad: test > val > train.

    Así una duplicada no puede aparecer nunca a la vez en train y test
    (evita fuga de información), y el conjunto de prueba se conserva.
    """
    prio = {"test": 0, "val": 1, "train": 2}
    seen: dict[str, dict] = {}
    kept: list[dict] = []
    removed: list[dict] = []
    for rec in sorted(records, key=lambda r: (prio.get(r["split"], 3), r["path"])):
        digest = rec["md5"]
        if digest in seen:
            removed.append({
                "path": rec["path"],
                "source": rec["source"],
                "unified_label": rec["unified_label"],
                "split": rec["split"],
                "duplicate_of": seen[digest]["path"],
            })
            continue
        seen[digest] = rec
        kept.append(rec)
    return kept, removed


def apply_splits(records: list[dict], mapping: dict) -> None:
    """Asigna split respetando particiones oficiales."""
    fracs = mapping["splits"]
    by_source: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_source[r["source"]].append(r)

    for source, recs in by_source.items():
        if source in ("intel", "cifar100"):
            # Partición oficial de test intacta; de train se extrae val (15%).
            train_recs = [r for r in recs if r["official_split"] == "train"]
            test_recs = [r for r in recs if r["official_split"] == "test"]
            labels = [r["unified_label"] for r in train_recs]
            assign = stratified_split(labels, (0.85, fracs["val"], 0.0), SEED)
            for r, s in zip(train_recs, assign):
                r["split"] = "train" if s == "train" else "val"
            for r in test_recs:
                r["split"] = "test"
        else:
            # Sin partición oficial: 70/15/15 estratificado con semilla fija.
            labels = [r["unified_label"] for r in recs]
            assign = stratified_split(
                labels, (fracs["train"], fracs["val"], fracs["test"]), SEED)
            for r, s in zip(recs, assign):
                r["split"] = s


def apply_class_cap(records: list[dict], max_per_class: int) -> tuple[list[dict], dict]:
    """Tope de imágenes por clase en train+val (el test se conserva completo).

    Se mantiene la proporción train/val original de cada clase.
    """
    by_label: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if r["split"] in ("train", "val"):
            by_label[r["unified_label"]].append(r)

    rng = random.Random(SEED)
    keep_paths: set[str] = set()
    capped: dict[str, dict] = {}
    for label, recs in by_label.items():
        n_train = sum(1 for r in recs if r["split"] == "train")
        n_val = sum(1 for r in recs if r["split"] == "val")
        total = n_train + n_val
        if total <= max_per_class:
            continue
        prop_val = n_val / total
        target_val = max(5, round(max_per_class * prop_val)) if n_val else 0
        target_train = max_per_class - target_val
        trains = [r for r in recs if r["split"] == "train"]
        vals = [r for r in recs if r["split"] == "val"]
        rng.shuffle(trains)
        rng.shuffle(vals)
        chosen = trains[:target_train] + vals[:target_val]
        keep_paths.update(r["path"] for r in chosen)
        capped[label] = {
            "antes_train": n_train, "antes_val": n_val,
            "despues_train": min(target_train, len(trains)),
            "despues_val": min(target_val, len(vals)),
        }

    out = []
    for r in records:
        if r["split"] == "test":
            out.append(r)                      # el test nunca se recorta
        elif r["unified_label"] in capped:
            if r["path"] in keep_paths:
                out.append(r)                  # clase capada: solo los elegidos
        else:
            out.append(r)                      # clase por debajo del tope
    return out, capped


# ---------------------------------------------------------------------------
# Salidas
# ---------------------------------------------------------------------------
def build_class_metadata(records: list[dict], mapping: dict) -> tuple[list[str], dict]:
    labels = sorted({r["unified_label"] for r in records})
    name_es = {k: v["name_es"] for k, v in mapping["unified"].items()}
    group = {k: v["group"] for k, v in mapping["unified"].items()}

    counts: dict[str, dict[str, int]] = {l: {"train": 0, "val": 0, "test": 0} for l in labels}
    sources: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    extras: dict[str, dict[str, set]] = defaultdict(lambda: {"species": set(), "health": set()})
    for r in records:
        counts[r["unified_label"]][r["split"]] += 1
        sources[r["unified_label"]][r["source"]].add(r["original_label"])
        if r["species"]:
            extras[r["unified_label"]]["species"].add(r["species"])
        if r["health_status"]:
            extras[r["unified_label"]]["health"].add(r["health_status"])

    # PlantVillage: nombre en español desde config con fallback.
    pv_names = mapping.get("plantvillage_name_es", {})
    metadata = {
        "img_size": IMG_SIZE,
        "seed": SEED,
        "num_classes": len(labels),
        "equivalencias_justificadas": mapping.get("equivalencias_justificadas", []),
        "classes": {},
    }
    for i, lab in enumerate(labels):
        meta = {
            "index": i,
            "label": lab,
            "name_es": name_es.get(lab) or pv_names.get(lab) or lab.replace("_", " "),
            "group": group.get(lab) or (
                GROUP_PLANT if lab in pv_names or lab.startswith(
                    tuple(f"{s}_" for s in
                          ["apple", "blueberry", "cherry", "corn", "grape", "orange",
                           "peach", "pepper", "potato", "raspberry", "soybean",
                           "squash", "strawberry", "tomato"]))
                else "unknown"),
            "sources": {s: sorted(v) for s, v in sources.get(lab, {}).items()},
            "counts": counts[lab],
        }
        sp = sorted(extras[lab]["species"])
        he = sorted(extras[lab]["health"])
        if sp:
            meta["species"] = sp
        if he:
            meta["health_status"] = he
        metadata["classes"][lab] = meta
    return labels, metadata


def save_plots(records: list[dict], labels: list[str], class_groups: dict[str, str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Distribución por clase (train+val+test)
    fig, ax = plt.subplots(figsize=(14, 7))
    c = Counter(r["unified_label"] for r in records)
    vals = [c[l] for l in labels]
    ax.bar(range(len(labels)), vals, color="#2b6cb0")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("Número de imágenes")
    ax.set_title("Distribución de imágenes por categoría unificada (total)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "distribution_per_class.png", dpi=130)
    plt.close(fig)

    # 2) Distribución por grupo semántico
    fig, ax = plt.subplots(figsize=(8, 5))
    groups = Counter(class_groups[r["unified_label"]] for r in records)
    ax.bar(groups.keys(), groups.values(), color="#2f855a")
    ax.set_ylabel("Imágenes")
    ax.set_title("Distribución por grupo semántico")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "distribution_per_group.png", dpi=130)
    plt.close(fig)

    # 3) Distribución por fuente y partición
    fig, ax = plt.subplots(figsize=(8, 5))
    src_split = Counter((r["source"], r["split"]) for r in records)
    sources = sorted({s for s, _ in src_split})
    splits = ["train", "val", "test"]
    x = range(len(sources))
    width = 0.25
    for k, sp in enumerate(splits):
        ax.bar([i + width * k for i in x],
               [src_split[(s, sp)] for s in sources], width, label=sp)
    ax.set_xticks([i + width for i in x])
    ax.set_xticklabels(sources)
    ax.set_ylabel("Imágenes")
    ax.set_title("Imágenes por fuente y partición")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "distribution_source_split.png", dpi=130)
    plt.close(fig)

    # 4) Cruce fuente x grupo (heatmap)
    fig, ax = plt.subplots(figsize=(8, 5))
    sg = Counter((r["source"], class_groups[r["unified_label"]]) for r in records)
    srcs = sorted({s for s, _ in sg})
    grps = sorted({g for _, g in sg})
    mat = np.array([[sg[(s, g)] for g in grps] for s in srcs])
    im = ax.imshow(mat, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(grps)))
    ax.set_xticklabels(grps, rotation=20, ha="right")
    ax.set_yticks(range(len(srcs)))
    ax.set_yticklabels(srcs)
    for i in range(len(srcs)):
        for j in range(len(grps)):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center", fontsize=8)
    ax.set_title("Imágenes por fuente y grupo semántico")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "distribution_source_group.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------
def run(force: bool = False) -> dict:
    ensure_dirs()
    mapping = load_label_mapping()

    if CATALOG_CSV.exists() and not force:
        report_path = RESULTS_DIR / "preprocessing_report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["reused"] = True
            print(f"Catálogo existente reutilizado: {CATALOG_CSV} "
                  "(usa --force para reconstruir)")
            return report

    print("[1/8] Extrayendo CIFAR-100 (si hace falta)...")
    ensure_cifar_extracted()
    n = export_cifar_images(mapping)
    print(f"      imágenes CIFAR-100 disponibles: {n}")

    print("[2/8] Recopilando registros de las cuatro fuentes...")
    recs_c = collect_cifar(mapping)
    recs_i = collect_intel(mapping)
    recs_p = collect_plantvillage()
    recs_m = collect_mosquito(mapping)
    counts_raw = {
        "cifar100": len(recs_c), "intel": len(recs_i),
        "plantvillage": len(recs_p), "mosqvision": len(recs_m),
    }
    print(f"      {counts_raw}")

    # seg_pred (Intel) está etiquetado como no utilizado: 0 etiquetas.
    n_pred = len(list((PATH_INTEL / "seg_pred" / "seg_pred").glob("*"))) \
        if (PATH_INTEL / "seg_pred" / "seg_pred").exists() else 0

    all_recs = recs_c + recs_i + recs_p + recs_m

    print("[3/8] Validando imágenes y calculando MD5...")
    valid, corrupt = validate_and_hash(all_recs)
    print(f"      válidas: {len(valid)} | corruptas/excluidas: {len(corrupt)}")

    print("[4/8] Asignando particiones...")
    apply_splits(valid, mapping)

    print("[5/8] Detectando y eliminando duplicados...")
    deduped, dupes = deduplicate(valid)
    print(f"      duplicados eliminados: {len(dupes)}")

    print("[6/8] Aplicando tope por clase (train+val)...")
    max_per = int(mapping.get("max_per_class", 600))
    final, capped = apply_class_cap(deduped, max_per)
    print(f"      clases ajustadas: {len(capped)} | registros finales: {len(final)}")

    print("[7/8] Construyendo metadatos y escribiendo catálogo...")
    labels, metadata = build_class_metadata(final, mapping)
    metadata["sources"] = {
        "cifar100": {
            "archivo": str(PATH_CIFAR_TAR.relative_to(PROJECT_ROOT)),
            "detalle": "extraído a data/extracted/cifar100 (imágenes seleccionadas 32x32)",
            "web": "https://www.cs.toronto.edu/~kriz/cifar.html",
            "seleccionadas": sorted(set(mapping["cifar100_selection"])),
        },
        "intel": {
            "archivo": str(PATH_INTEL.relative_to(PROJECT_ROOT)),
            "detalle": "seg_train/seg_test con particiones oficiales; seg_pred sin etiquetas no se usa",
            "web": "https://huggingface.co/datasets/sfarrukhm/intel-image-classification",
            "seg_pred_excluidas": n_pred,
        },
        "plantvillage": {
            "archivo": str(PATH_PLANT.relative_to(PROJECT_ROOT)),
            "detalle": "raw/color con 38 clases; metadatos en dataset/Plantas/metadata",
            "web": "https://www.tensorflow.org/datasets/catalog/plant_village",
        },
        "mosqvision": {
            "archivo": str(PATH_MOSQ.relative_to(PROJECT_ROOT)),
            "detalle": "AEDES/ANOPHELES/CULEX con 1000 imágenes cada una",
            "web": "https://zenodo.org/records/14238701",
        },
    }

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    cols = ["path", "source", "original_label", "unified_label", "group", "split",
            "species", "health_status", "width", "height", "format", "md5", "bytes"]
    with open(CATALOG_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(final, key=lambda x: x["path"]):
            row = {c: r.get(c, "") for c in cols}
            row["group"] = metadata["classes"][r["unified_label"]]["group"]
            w.writerow(row)

    CLASS_NAMES_JSON.write_text(json.dumps(labels, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
    CLASS_METADATA_JSON.write_text(json.dumps(metadata, indent=2, ensure_ascii=False),
                                   encoding="utf-8")

    print("[8/8] Generando reportes y gráficas...")
    split_counts = Counter(r["split"] for r in final)
    per_source_split = Counter((r["source"], r["split"]) for r in final)
    report = {
        "reused": False,
        "img_size_entrenamiento": IMG_SIZE,
        "seed": SEED,
        "raw_counts": counts_raw,
        "seg_pred_intel_excluidas_sin_etiqueta": n_pred,
        "validas": len(valid),
        "corruptas": len(corrupt),
        "corruptos_detalle": corrupt[:200],
        "duplicados_eliminados": len(dupes),
        "duplicados_detalle": dupes[:200],
        "clases_capadas": capped,
        "max_per_class_train_val": max_per,
        "final_counts": dict(split_counts),
        "final_por_fuente_y_split": {f"{s}|{sp}": c for (s, sp), c in per_source_split.items()},
        "num_clases": len(labels),
        "class_names": labels,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "preprocessing_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    balance = {
        "por_clase": {l: metadata["classes"][l]["counts"] for l in labels},
        "total_por_clase": {l: sum(metadata["classes"][l]["counts"].values()) for l in labels},
        "por_grupo": dict(Counter(metadata["classes"][l]["group"] for l in labels)),
        "max_per_class_train_val": max_per,
        "clases_capadas": {k: v for k, v in capped.items()},
    }
    BALANCE_REPORT_JSON.write_text(json.dumps(balance, indent=2, ensure_ascii=False),
                                    encoding="utf-8")

    class_groups = {l: metadata["classes"][l]["group"] for l in labels}
    save_plots(final, labels, class_groups)

    print("\nListo.")
    print(f"  Catálogo:  {CATALOG_CSV}  ({len(final)} registros)")
    print(f"  Clases:    {len(labels)} -> {CLASS_NAMES_JSON}")
    print(f"  Splits:    {dict(split_counts)}")
    print(f"  Reportes:  {RESULTS_DIR / 'preprocessing_report.json'}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepara los datasets locales de OmniVision AI")
    parser.add_argument("--force", action="store_true", help="reconstruye el catálogo")
    args = parser.parse_args()
    run(force=args.force)


if __name__ == "__main__":
    main()
