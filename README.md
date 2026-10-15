# OmniVision AI — un único clasificador para cuatro datasets

Clasificador de imágenes con **TensorFlow/Keras** (MobileNetV2 preentrenado en
ImageNet) que integra **cuatro datasets locales** en **una sola red** con
**67 categorías**: escenas, personas, dinosaurios, insectos, animales,
mosquitos y enfermedades de plantas.

| Métrica (conjunto de prueba, 13 261 imágenes) | Valor |
|---|---|
| Accuracy | **87,96 %** |
| Top-5 | **98,73 %** |
| Macro-F1 | 0,845 |
| Umbral de confianza (derivado en validación) | 0,47 → ~90 % de precisión aceptada |

## Datasets (ya descargados en `dataset/`, nunca se re-descargan)

| Fuente | Ruta local | Clases usadas | Notas |
|---|---|---|---|
| CIFAR-100 | `dataset/General/cifar-100-python.tar.gz` | 26 de 100 etiquetas finas | se extrae a `data/extracted/` (imágenes 32×32) |
| Intel Image Classification | `dataset/Paisajes/` | 6 | usan las particiones oficiales `seg_train`/`seg_test`; `seg_pred` (sin etiquetas) se excluye |
| PlantVillage | `dataset/Plantas/raw/color/` | 38 (especie + salud) | licencia CC0; metadatos en `dataset/Plantas/metadata/` |
| MosqVision-3K | `dataset/Mosquito_dataset/` | 3 (aedes, anopheles, culex) | 1000 imágenes por género |

**Fusiones justificadas** (ver `config/label_mapping.json`):

- *forest*, *mountain*, *sea* existen en CIFAR-100 **e** Intel → una sola clase.
- *baby/boy/girl/man/woman* (CIFAR) → una sola clase `person`.
- `road` (CIFAR) **no** se fusiona con `street` (Intel); `plain` se mantiene
  separada de `glacier`.
- Las enfermedades de plantas conservan especie + estado sanitario.

## Pipeline de datos

`src/data_preprocessing.py` (idempotente, con `--force` para reconstruir):

1. extrae CIFAR-100 y exporta solo las 26 clases seleccionadas;
2. valida cada imagen con PIL y calcula MD5 (0 corruptas en la ejecución real);
3. elimina **duplicados exactos** priorizando el conjunto de prueba
   (1 959 eliminados);
4. respeta las particiones oficiales de CIFAR/Intel y divide 70/15/15
   estratificado el resto (semilla 42);
5. aplica un **tope de 600 imágenes por clase** en train+val (el test queda
   completo) → 47 694 registros: train 28 627 / val 5 806 / test 13 261.

Salidas: `data/processed/catalog.csv`, `models/class_names.json` (orden
alfabético de las 67 salidas), `models/class_metadata.json` (nombre en
español, grupo, fuentes), `results/*` (reportes y gráficas).

## Reproducción completa (PowerShell, desde la raíz)

```powershell
# 1) Entorno (ya creado en este equipo: Python 3.10 + .venv)
py -3.10 -m venv .venv
.venv\Scripts\pip install -r requirements.txt

# 2) Preparar datos (no descarga nada: usa dataset/)
.venv\Scripts\python -m src.data_preprocessing --force

# 3) Entrenar (~35 min en CPU de 16 hilos; sin GPU en Windows nativo)
.venv\Scripts\python -m src.train --epochs1 15 --epochs2 8
# opcional: .venv\Scripts\python -m src.train --benchmark   # mide tiempo/época

# 4) Evaluar (accuracy, matriz de confusión, umbral derivado en val)
.venv\Scripts\python -m src.evaluate

# 5) Probar una imagen
.venv\Scripts\python -m src.predict "ruta\a\imagen.jpg"

# 6) App web
.venv\Scripts\streamlit run app.py

# 7) Pruebas
.venv\Scripts\python -m pytest tests -q

# 8) Notebook académico (ejecuta todo el flujo con los artefactos existentes)
.venv\Scripts\jupyter nbconvert --to notebook --execute --inplace `
    notebooks\01_omnivision_classification.ipynb
```

## Modelo y entrenamiento

- **Arquitectura única**: aumentación (solo en entrenamiento) → preprocesado
  MobileNetV2 → MobileNetV2 congelado → GlobalAveragePooling → Dropout 0.3 →
  Dense(67, softmax). La aumentación y el preprocesado viven **dentro** del
  modelo, así la inferencia es idéntica al entrenamiento.
- **Fase 1**: backbone congelado, Adam 1e-3, 15 épocas (mejor val 86,5 %).
- **Fase 2**: fine-tune de las últimas 30 capas, Adam 1e-4, 8 épocas
  (mejor val **88,4 %**). EarlyStopping con paciencia 3; se conserva la
  mejor fase. `class_weight` balanceado. El modelo final es
  `models/omnivision_cnn.keras`.

## Estructura

```
app.py                          # interfaz Streamlit
config/label_mapping.json       # mapa reproducible de 67 clases y fusiones
src/common.py                   # rutas, hiperparámetros, build_model()
src/data_preprocessing.py       # integración de los 4 datasets
src/train.py                    # fase 1 + fine-tune, historial y reportes
src/evaluate.py                 # test completo + umbral derivado en val
src/predict.py                  # CLI y clase Predictor (la usa la app)
notebooks/01_omnivision_classification.ipynb
tests/test_predict.py           # 8 pruebas (pytest)
models/omnivision_cnn.keras     # único modelo final (+ class_names/metadata)
results/                        # métricas, matriz de confusión, umbral, gráficas
dataset/                        # datasets locales (no se sube al repo)
```

## Limitaciones documentadas

- **Sin GPU**: TensorFlow ≥ 2.11 no usa GPU en Windows nativo y la instalación
  de WSL2 no se completó; el entrenamiento se hizo en CPU (16 hilos, ~35 min).
- CIFAR-100 aporta imágenes de 32×32 (se reescalan a 128×128): es la fuente
  con menor exactitud (76,5 % en test) frente a PlantVillage (92,8 %),
  Intel (87,0 %) y MosqVision (84,7 %).
- `dataset/Plantas/leaf_grouping/leaf-map.json` no se pudo vincular a los
  nombres de archivo (claves distintas): no se usó para evitar fugas entre
  splits; los duplicados exactos se eliminaron por MD5.
- MosqVision-3K contiene imágenes transformadas cuyo posible grupo original
  no está disponible: la detección de duplicados por MD5 cubre solo
  coincidencias exactas.
- Existen descargas parciales corruptas previas en `data/raw/` (ignoradas y
  fuera de git); el pipeline nunca las lee.
- El modelo se serializa sin capas `Lambda` (preprocesado como `Rescaling`
  equivalente) para ser cargable con el `safe_mode` de Keras 3.
