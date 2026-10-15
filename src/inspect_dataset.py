"""Inspección compacta de la carpeta dataset/ (solo lectura)."""
from __future__ import annotations

import json
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".ppm", ".gif"}


def summarize_dir(path: Path, depth: int = 0, max_depth: int = 3) -> None:
    if depth > max_depth:
        return
    entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    dirs = [e for e in entries if e.is_dir()]
    files = [e for e in entries if e.is_file()]
    if files:
        imgs = [f for f in files if f.suffix.lower() in IMG_EXT]
        other = [f for f in files if f.suffix.lower() not in IMG_EXT]
        size_mb = sum(f.stat().st_size for f in files) / 1e6
        print("  " * depth + f"[{len(files)} archivos, {size_mb:.1f} MB, {len(imgs)} imágenes]"
              + (f" otros: {[o.name for o in other[:6]]}" if other else ""))
    for d in dirs:
        print("  " * depth + f"{d.name}/")
        summarize_dir(d, depth + 1, max_depth)


print("=" * 70)
print("RAIZ DEL PROYECTO:")
for p in sorted(ROOT.iterdir()):
    print("  ", p.name + ("/" if p.is_dir() else ""))

print("=" * 70)
print("DATASET/ (estructura de carpetas con conteo de imágenes):")
for top in sorted(DATASET.iterdir()):
    print("-" * 70)
    print(top.name + "/")
    summarize_dir(top, depth=1, max_depth=4)

# Archives comprimidos
print("=" * 70)
print("ARCHIVOS COMPRIMIDOS:")
zips = [p for p in DATASET.rglob("*") if p.suffix.lower() in {".zip", ".tar", ".gz", ".7z", ".rar"}]
if not zips:
    print("  (ninguno)")
for z in zips:
    print(f"  {z.relative_to(DATASET)}  {z.stat().st_size/1e6:.1f} MB")
    if z.suffix.lower() == ".zip":
        try:
            with zipfile.ZipFile(z) as zf:
                names = zf.namelist()
                print(f"     -> {len(names)} entradas; primeras: {names[:5]}")
        except Exception as e:
            print("     -> ERROR al leer zip:", e)

# Datos especiales CIFAR u otros
print("=" * 70)
print("ARCHIVOS DE DATOS ESPECIALES (pickle/parquet/csv/json/txt):")
special = [p for p in DATASET.rglob("*")
           if p.suffix.lower() in {".pkl", ".pickle", ".parquet", ".csv", ".json", ".txt", ".md", ".mat"}]
for s in special[:60]:
    print(f"  {s.relative_to(DATASET)}  {s.stat().st_size/1e3:.1f} KB")
print(f"  (total: {len(special)})")

# Conteo por clase: Intel
print("=" * 70)
print("INTEL (Paisajes): conteo por partición y clase")
for part in ["seg_train", "seg_test", "seg_pred"]:
    base = DATASET / "Paisajes" / part
    if not base.exists():
        continue
    # puede existir nivel extra homónimo
    candidates = [base] + [d for d in base.iterdir() if d.is_dir()]
    for c in candidates:
        classes = [d for d in c.iterdir() if d.is_dir()]
        if classes:
            counts = {d.name: len(list(d.glob("*"))) for d in sorted(classes)}
            print(f"  {part} en {c.relative_to(DATASET)}: {counts}")
            break

# PlantVillage
print("=" * 70)
print("PLANTVILLAGE (Plantas): conteo por clase")
pv = DATASET / "Plantas" / "raw" / "color"
if pv.exists():
    counts = {d.name: len(list(d.iterdir())) for d in sorted(pv.iterdir()) if d.is_dir()}
    print(f"  {len(counts)} clases, total {sum(counts.values())} imágenes")
    for k, v in counts.items():
        print(f"   {v:5d}  {k}")

# Mosquitos
print("=" * 70)
print("MOSQUITOS: conteo por clase")
mos = DATASET / "Mosquito_dataset"
if mos.exists():
    for d in sorted(mos.iterdir()):
        if d.is_dir():
            files = list(d.iterdir())
            exts = Counter(f.suffix.lower() for f in files if f.is_file())
            print(f"  {d.name}: {len(files)} archivos {dict(exts)}")
        else:
            print(f"  archivo: {d.name} {d.stat().st_size/1e3:.1f} KB")

# General (CIFAR)
print("=" * 70)
print("GENERAL (¿CIFAR-100?):")
gen = DATASET / "General"
if gen.exists():
    for p in sorted(gen.rglob("*")):
        if p.is_file() and p.suffix.lower() not in IMG_EXT:
            print(f"  {p.relative_to(DATASET)}  {p.stat().st_size/1e6:.2f} MB")
