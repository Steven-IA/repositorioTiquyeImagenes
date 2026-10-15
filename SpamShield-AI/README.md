# 🛡️ SpamShield AI

Sistema inteligente de detección de mensajes spam mediante Deep Learning (BiLSTM) con TensorFlow/Keras.

## Descripción

SpamShield AI analiza el texto de un mensaje SMS y lo clasifica como **ham** (legítimo) o **spam** (no deseado) utilizando una red neuronal recurrente Bidirectional LSTM entrenada con el dataset SMS Spam Collection de UCI.

## Dataset

- **Nombre:** SMS Spam Collection
- **Fuente:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- **Registros:** ~5,574 mensajes etiquetados
- **Clases:** ham (legítimo), spam (no deseado)

## Tecnologías

| Categoría | Tecnología |
|---|---|
| Lenguaje | Python 3.10 |
| Deep Learning | TensorFlow 2.16 / Keras |
| Datos | Pandas, NumPy |
| Visualización | Matplotlib, Seaborn |
| ML | Scikit-learn |
| Interfaz | Streamlit |
| Pruebas | Pytest |

## Estructura del proyecto

```
SpamShield-AI/
├── data/
│   ├── raw/                  → Dataset original (no versionado)
│   └── processed/            → Subconjuntos train/val/test
├── notebooks/
│   └── 01_spam_classification.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── models/
│   ├── spamshield_lstm.keras
│   └── config.json
├── results/
│   ├── class_distribution.png
│   ├── message_lengths.png
│   ├── training_loss.png
│   ├── training_accuracy.png
│   └── confusion_matrix.png
├── tests/
│   └── test_predict.py
├── app.py
├── requirements.txt
└── README.md
```

## Instalación (Windows PowerShell)

### 1. Verificar Python

```powershell
py -3.10 --version
```

> **Nota:** Se requiere Python 3.10 para compatibilidad con TensorFlow 2.16.

### 2. Crear entorno virtual

```powershell
cd SpamShield-AI
py -3.10 -m venv venv
```

### 3. Activar entorno virtual

```powershell
.\venv\Scripts\Activate.ps1
```

> Si aparece un error de ejecución de scripts, ejecuta primero:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### 4. Instalar dependencias

```powershell
pip install -r requirements.txt
```

### 5. Descargar el dataset

```powershell
# Crear directorio
New-Item -ItemType Directory -Force -Path data\raw

# Descargar
Invoke-WebRequest -Uri "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip" -OutFile "data\raw\smsspamcollection.zip"

# Extraer
Expand-Archive -Path "data\raw\smsspamcollection.zip" -DestinationPath "data\raw" -Force
```

## Uso

### Preprocesamiento

```powershell
python src/data_preprocessing.py
```

### Entrenamiento

```powershell
python src/train.py
```

### Evaluación

```powershell
python src/evaluate.py
```

### Pruebas

```powershell
pytest tests/test_predict.py -v
```

### Aplicación Streamlit

```powershell
streamlit run app.py
```

### Notebook

```powershell
jupyter notebook notebooks/01_spam_classification.ipynb
```

## Arquitectura del modelo

```
TextVectorization (vocab=10000, seq_len=200)
    ↓
Embedding (128 dimensiones)
    ↓
Bidirectional LSTM (64 unidades)
    ↓
Dropout (0.3)
    ↓
Dense (32, ReLU)
    ↓
Dropout (0.2)
    ↓
Dense (1, Sigmoid)
```

## Resultados obtenidos

Valores reales del conjunto de prueba, leídos de `results/metrics.json`.

| Métrica | Valor |
|---|---|
| Accuracy | 0.978 |
| Precision (spam) | 0.9438 |
| Recall (spam) | 0.875 |
| F1-score (spam) | 0.9081 |

**Matriz de confusión (conjunto de prueba):**

|  | Predicho Ham | Predicho Spam |
|---|---|---|
| **Real Ham** | 673 | 5 |
| **Real Spam** | 12 | 84 |

- **True Negatives:** 673 (ham correctamente clasificado)
- **False Positives:** 5 (ham clasificado como spam)
- **False Negatives:** 12 (spam clasificado como ham)
- **True Positives:** 84 (spam correctamente detectado)

## Limitaciones

- El modelo fue entrenado solo con mensajes en inglés.
- No detecta spam basado en imágenes o enlaces maliciosos (solo texto).
- La predicción es una estimación, no una garantía.
- Puede tener falsos positivos en mensajes legítimos con lenguaje similar al spam.

## Licencia

Proyecto educativo — uso académico.
