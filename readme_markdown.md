# 🚀 Smartwatch Knowledge System - Proyecto Capstone CSDS-352

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Database-orange.svg)](https://www.trychroma.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red.svg)](https://streamlit.io/)

## 📋 Descripción del Proyecto

Sistema inteligente end-to-end de gestión de conocimiento técnico para smartwatches que integra **embeddings semánticos**, **machine learning supervisado y no supervisado**, y **visualización interactiva**.

### 🎯 Características Principales

- ✅ **Pipeline de Ingesta Automatizado**: Procesamiento de PDFs → Chunking → Embeddings 384D
- ✅ **Búsqueda Semántica**: ChromaDB con búsqueda por similitud de coseno (< 0.5 segundos)
- ✅ **Clasificación de Calidad**: Logistic Regression con auto-etiquetado (97.6% accuracy)
- ✅ **Clustering Automático**: K-Means para descubrir grupos naturales en los datos
- ✅ **Visualización Interactiva**: PCA y t-SNE para exploración visual en 2D
- ✅ **Interfaz Web**: Streamlit app con búsqueda y visualización de clusters
- ✅ **Sistema Completo**: Un solo comando ejecuta todo el pipeline

### 🏆 Resultados Actuales

| Métrica | Valor |
|---------|-------|
| **Total de chunks procesados** | 1,942 |
| **Dimensionalidad de embeddings** | 384D |
| **Tiempo de búsqueda** | < 0.5 segundos |
| **Accuracy del clasificador** | 97.6% |
| **Número de clusters** | 2 (óptimo automático) |
| **Silhouette Score** | 0.150 |
| **Varianza explicada (PCA)** | 19.6% |

---

## 📁 Estructura del Proyecto
```
capstone/
├── main.py                           # 🎯 Pipeline principal COMPLETO
├── app.py                            # 🌐 Streamlit Web App
├── config.py                         # ⚙️ Configuración del sistema
├── requirements.txt                  # 📦 Dependencias Python
├── README.md                         # 📖 Este archivo
│
├── src/                              # 📂 Código fuente modular
│   ├── ingestion/
│   │   ├── document_processor.py     # 📄 Procesamiento PDFs (chunking)
│   │   └── embedding_generator.py    # 🧠 all-MiniLM-L6-v1 embeddings
│   │
│   ├── quality_control/
│   │   ├── auto_labeler.py          # 🏷️ Etiquetado automático
│   │   ├── feature_extractor.py     # 🔧 10 features extraídas
│   │   └── quality_classifier.py    # 🤖 Logistic Regression
│   │
│   ├── storage/
│   │   └── chroma_manager.py         # 💾 ChromaDB manager + search
│   │
│   ├── clustering/
│   │   └── cluster_manager.py        # 🎲 K-Means clustering + K óptimo
│   │
│   └── visualization/
│       └── visualizer.py             # 🎨 PCA + t-SNE (384D → 2D)
│
├── data/                             # 📊 Datos del sistema
│   ├── raw/                         # 📚 Documentos de entrenamiento
│   │   ├── apple_watch/             # 🍎 Manuales Apple Watch
│   │   ├── samsung/                 # 📱 Manuales Samsung
│   │   ├── garmin/                  # 🏃 Manuales Garmin
│   │   └── fitbit/                  # 💪 Manuales Fitbit
│   │
│   ├── new/                         # ✨ Documentos nuevos (demo)
│   │   ├── apple_watch/
│   │   ├── samsung/
│   │   ├── garmin/
│   │   └── fitbit/
│   │
│   └── processed/                   # 🔄 Datos procesados
│
├── models/                          # 🤖 Modelos entrenados (.joblib)
│   ├── quality_classifier.joblib   # 📈 Clasificador de calidad
│   ├── cluster_model.joblib        # 🎲 Modelo K-Means
│   └── visualization_cache.joblib  # 🎨 Cache PCA + t-SNE
│
├── logs/                            # 📝 Archivos de log
│   └── system.log
│
└── chroma-data/                     # 💽 ChromaDB persistente
    └── chroma.sqlite3
```

---

## 🔧 Requisitos Previos

### 1. **Docker** (para ChromaDB)
```bash
# Instalar Docker según tu sistema operativo
# Ubuntu/Debian:
sudo apt update && sudo apt install docker.io

# macOS:
brew install --cask docker

# Windows: Descargar Docker Desktop
# https://www.docker.com/products/docker-desktop
```

### 2. **Python 3.12+**
```bash
python --version  # Verificar versión (debe ser >= 3.12)
```

### 3. **Git** (para clonar el repositorio)
```bash
git --version
```

---

## 🚀 Instalación y Configuración

### Paso 1: Clonar el Repositorio
```bash
git clone https://github.com/ramirezdev26/Smartwatch_technical_system.git
cd Smartwatch_technical_system
```

### Paso 2: Crear Entorno Virtual
```bash
# Crear entorno virtual
python -m venv .venv

# Activar entorno virtual
# Linux/macOS:
source .venv/bin/activate

# Windows:
.venv\Scripts\activate
```

### Paso 3: Instalar Dependencias
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Dependencias principales:**
- `sentence-transformers`: Embeddings con all-MiniLM-L6-v1
- `chromadb`: Base de datos vectorial
- `scikit-learn`: Machine learning (Logistic Regression, K-Means, PCA)
- `streamlit`: Interfaz web interactiva
- `plotly`: Visualizaciones interactivas
- `loguru`: Sistema de logging
- `pypdf2`: Procesamiento de PDFs

### Paso 4: Crear Estructura de Directorios
```bash
# Ejecutar script de setup (crea todos los directorios necesarios)
mkdir -p data/{raw,new,processed}/{apple_watch,samsung,garmin,fitbit}
mkdir -p models logs chroma-data
```

### Paso 5: Configurar ChromaDB (Base de Datos Vectorial)
```bash
# Iniciar ChromaDB con Docker
docker run -d \
  --name chromadb \
  -v $(pwd)/chroma-data:/chroma/chroma \
  -p 8000:8000 \
  chromadb/chroma:latest

# Verificar que está funcionando
curl http://localhost:8000/api/v1/heartbeat
# Respuesta esperada: {"nanosecond heartbeat": ...}

# Ver logs (opcional)
docker logs chromadb

# Detener ChromaDB (cuando termines)
docker stop chromadb

# Reiniciar ChromaDB
docker start chromadb
```

---

## 📚 Preparar Datos de Entrenamiento

### Opción 1: Usar Datos Existentes
Si ya tienes documentos en el repositorio:
```bash
# Verificar archivos existentes
ls -la data/raw/*/
```

### Opción 2: Agregar Nuevos Documentos
```bash
# Colocar manuales PDF en los directorios correspondientes
cp path/to/apple_manual.pdf data/raw/apple_watch/
cp path/to/garmin_manual.pdf data/raw/garmin/
cp path/to/samsung_manual.pdf data/raw/samsung/
cp path/to/fitbit_manual.pdf data/raw/fitbit/
```

**Formatos soportados:** `.pdf`, `.txt`

**Fuentes recomendadas:**
- [Apple Watch User Guide](https://support.apple.com/guide/watch/welcome/watchos)
- [Garmin Support](https://support.garmin.com/)
- [Samsung Wearables](https://www.samsung.com/us/support/)
- [Fitbit Manuals](https://help.fitbit.com/)

---

## 🎯 Ejecución del Sistema

### Pipeline Completo (Recomendado)

Este comando ejecuta **TODO** el pipeline de forma automática:
```bash
python main.py
```

**¿Qué hace este comando?**

1. ✅ **Fase 1: Ingesta de Documentos**
   - Procesa todos los PDFs en `data/raw/`
   - Genera chunks de 150 palabras con 20 de overlap
   - Crea embeddings 384D con all-MiniLM-L6-v1

2. ✅ **Fase 2: Control de Calidad**
   - Entrena clasificador Logistic Regression
   - Etiqueta chunks como Relevante/Ambiguo/Irrelevante
   - Almacena en ChromaDB con metadata de calidad

3. ✅ **Fase 3: Machine Learning**
   - Ejecuta K-Means para encontrar K óptimo (Silhouette Score)
   - Genera visualizaciones PCA y t-SNE
   - Guarda modelos entrenados en `models/`

**Salida esperada:**
```
🚀 INICIANDO PIPELINE COMPLETO
============================================================

📥 FASE 1: INGESTA DE DOCUMENTOS Y EMBEDDINGS
   ✅ Procesados 1,942 chunks de 4 marcas

🎯 FASE 2: CONTROL DE CALIDAD
   ✅ Clasificador entrenado (Accuracy: 0.976)
   ✅ Chunks almacenados en ChromaDB

🎲 FASE 3: CLUSTERING Y VISUALIZACIÓN
   ✅ K óptimo encontrado: 2 clusters
   ✅ Silhouette Score: 0.150
   ✅ Visualizaciones generadas (PCA + t-SNE)

🎉 PIPELINE COMPLETADO EXITOSAMENTE
⏱️ Tiempo total: XX segundos
```

### Interfaz Web Interactiva (Streamlit)

Una vez que el pipeline ha ejecutado, lanza la aplicación web:
```bash
streamlit run app.py
```

**Características de la UI:**

📍 **Tab 1: Búsqueda Semántica**
- Campo de búsqueda con consultas en lenguaje natural
- Filtros por marca (apple_watch, garmin, samsung, fitbit)
- Top K resultados configurables (1-20)
- Priorización automática por calidad
- Resultados expandibles con metadata completa

📍 **Tab 2: Visualización de Clusters**
- Selector PCA vs t-SNE
- Coloreado por: Clusters, Marcas, Calidad
- Gráfico interactivo con Plotly (zoom, pan, hover)
- Estadísticas de clusters
- Información del sistema

**Ejemplos de búsqueda:**
```
"battery life issues"
"heart rate monitoring accuracy"
"GPS tracking outdoor activities"
"water resistance swimming"
"sleep tracking features"
```

---

## 📊 Resultados del Sistema

### Clustering (K-Means)

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **K óptimo** | 2 | Búsqueda automática (2-10) |
| **Silhouette Score** | 0.150 | Separación moderada (esperado para docs técnicos) |
| **Cluster 1** | 1,540 docs (79.3%) | Dominado por **Apple Watch** |
| **Cluster 0** | 402 docs (20.7%) | Dominado por **Garmin** |

**Interpretación:**
- El algoritmo descubrió **2 grupos naturales** sin etiquetas
- **Cluster 1**: Enfoque consumidor/lifestyle (Apple, Samsung, Fitbit)
- **Cluster 0**: Enfoque atlético/performance (Garmin)
- Score bajo es **esperado**: documentos técnicos similares entre sí

### Visualización (Dimensionality Reduction)

| Técnica | Varianza Explicada | Ventajas | Uso |
|---------|-------------------|----------|-----|
| **PCA** | 19.6% (PC1+PC2) | Rápida, determinística | Exploración inicial |
| **t-SNE** | N/A | Mejor separación visual | Visualización final |

### Clasificación de Calidad

| Clase | Criterios | % del Total |
|-------|-----------|-------------|
| **Relevante** | Longitud óptima + términos técnicos + alta similitud | ~60% |
| **Ambiguo** | Longitud media o pocos términos técnicos | ~30% |
| **Irrelevante** | Muy corto, sin términos técnicos, baja similitud | ~10% |

---


## 🔧 Configuración Avanzada

### Modificar Parámetros en `config.py`
```python
# Chunking
CHUNK_SIZE = 150          # Palabras por chunk
CHUNK_OVERLAP = 20        # Overlap en palabras
MIN_CHUNK_LENGTH = 50     # Mínimo de caracteres

# Embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v1"  # Modelo de Sentence Transformers
EMBEDDING_DIM = 384                    # Dimensiones del embedding

# Clustering
MIN_CLUSTERS = 2          # Mínimo K para búsqueda
MAX_CLUSTERS = 10         # Máximo K para búsqueda
KMEANS_MAX_ITER = 20      # Iteraciones K-Means

# Visualización
PCA_COMPONENTS = 2        # Componentes PCA
TSNE_PERPLEXITY = 30      # Perplexity t-SNE
TSNE_N_ITER = 1000        # Iteraciones t-SNE

# ChromaDB
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "smartwatch_docs"
```

### Re-entrenar Modelos
```bash
# Eliminar modelos existentes
rm models/*.joblib

# Re-ejecutar pipeline completo
python main.py
```

---

## 📖 Arquitectura Técnica

### Pipeline de Datos
```
PDF/TXT → Document Processor → Embedding Generator → ChromaDB
   ↓              ↓                     ↓                ↓
Chunks(150w)  Clean Text          384D Vectors    Indexed Storage
```

### Pipeline de Machine Learning
```
Embeddings 384D
    ├─→ Quality Classifier (Supervised)
    │   └─→ Logistic Regression
    │       └─→ Labels: Relevante/Ambiguo/Irrelevante
    │
    ├─→ Cluster Manager (Unsupervised)
    │   └─→ K-Means (K óptimo vía Silhouette)
    │       └─→ 2 clusters naturales
    │
    └─→ Visualizer (Dimensionality Reduction)
        ├─→ PCA (384D → 2D, 19.6% varianza)
        └─→ t-SNE (384D → 2D, mejor separación)
```

### Stack Tecnológico

| Componente | Tecnología | Versión |
|------------|-----------|---------|
| **Embeddings** | Sentence Transformers | all-MiniLM-L6-v1 |
| **Vector DB** | ChromaDB | latest |
| **Clustering** | scikit-learn K-Means | 1.3+ |
| **Classification** | scikit-learn Logistic Regression | 1.3+ |
| **Visualization** | PCA + t-SNE (sklearn) | 1.3+ |
| **Web App** | Streamlit | 1.28+ |
| **Plotting** | Plotly | 5.17+ |
| **Logging** | Loguru | 0.7+ |

---
