# 🚀 Smartwatch Knowledge System - Proyecto Capstone CSDS-352

## 📋 Descripción del Proyecto

Sistema inteligente de gestión de conocimiento técnico para smartwatches que integra:
- **Pipeline de ingesta** de documentos PDF/texto
- **Embeddings semánticos** con all-MiniLM-L6-v1  
- **Clasificador de calidad** automático
- **Base de datos vectorial** ChromaDB para búsqueda semántica
- **Control de calidad** con etiquetado automático

---

## 📁 Estructura del Proyecto

```
capstone/
├── main.py                           # 🎯 Pipeline principal
├── config.py                         # ⚙️ Configuración del sistema
├── requirements.txt                   # 📦 Dependencias Python
├── README.md                         # 📖 Este archivo
│
├── src/                              # 📂 Código fuente
│   ├── ingestion/
│   │   ├── document_processor.py     # 📄 Procesamiento de PDFs
│   │   └── embedding_generator.py    # 🧠 Generación de embeddings
│   ├── quality_control/
│   │   ├── auto_labeler.py          # 🏷️ Etiquetado automático
│   │   ├── feature_extractor.py     # 🔧 Extracción de características
│   │   └── quality_classifier.py    # 🤖 Clasificador de calidad
│   └── storage/
│       └── chroma_manager.py         # 💾 Gestor ChromaDB
│
├── data/                             # 📊 Datos del sistema
│   ├── raw/                         # 📚 Documentos de entrenamiento
│   │   ├── apple_watch/             # 🍎 Manuales Apple Watch
│   │   ├── samsung/                 # 📱 Manuales Samsung
│   │   ├── garmin/                  # 🏃 Manuales Garmin
│   │   └── fitbit/                  # 💪 Manuales Fitbit
│   ├── new/                         # ✨ Documentos nuevos (para demo)
│   │   ├── apple_watch/
│   │   ├── samsung/
│   │   ├── garmin/
│   │   └── fitbit/
│   └── processed/                   # 🔄 Datos procesados
│
├── models/                          # 🤖 Modelos entrenados
│   └── quality_classifier.joblib   # 📈 Clasificador guardado
│
├── logs/                            # 📝 Archivos de log
│   └── system.log
│
└── chroma-data/                     # 💽 Datos persistentes ChromaDB
```

---

## 🔧 Requisitos Previos

### 1. **Docker** (para ChromaDB)
```bash
# Instalar Docker según tu sistema operativo
# Ubuntu/Debian:
sudo apt update && sudo apt install docker.io

# macOS:
brew install docker

# Windows: Descargar Docker Desktop
```

### 2. **Python 3.12+**
```bash
python --version  # Verificar versión
```

### 3. **Git** (para clonar el repositorio)

---

## 🚀 Instalación y Configuración

### Paso 1: Crear Entorno Virtual
```bash
# Crear entorno virtual
python -m venv .venv

# Activar entorno virtual
# Linux/macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate
```

### Paso 2: Instalar Dependencias
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Paso 3: Crear Estructura de Directorios
```bash
# Crear directorios necesarios
mkdir -p data/raw/apple_watch
mkdir -p data/raw/samsung  
mkdir -p data/raw/garmin
mkdir -p data/raw/fitbit
mkdir -p data/new/apple_watch
mkdir -p data/new/samsung
mkdir -p data/new/garmin
mkdir -p data/new/fitbit
mkdir -p data/processed
mkdir -p models
mkdir -p logs
mkdir -p chroma-data
```

### Paso 4: Configurar ChromaDB (Base de Datos Vectorial)
```bash
# Iniciar ChromaDB con Docker
docker run -d \
  --name chromadb \
  -v $(pwd)/chroma-data:/chroma/chroma \
  -p 8000:8000 \
  chromadb/chroma:latest

# Verificar que está funcionando
curl http://localhost:8000/api/v1/heartbeat
# Debería responder: {"nanosecond heartbeat": ...}
```

---

## 📚 Preparar Datos de Entrenamiento

### Opción 1: Usar Datos Existentes del Repositorio
Si ya tienes documentos en tu repositorio:
```bash
# Verificar archivos existentes
ls -la data/raw/*/
```

### Opción 2: Agregar Nuevos Documentos
```bash
# Colocar manuales PDF en los directorios correspondientes
cp tu_manual_apple.pdf data/raw/apple_watch/
cp tu_manual_samsung.pdf data/raw/samsung/
cp tu_manual_garmin.pdf data/raw/garmin/
cp tu_manual_fitbit.pdf data/raw/fitbit/
```

**Formatos soportados:** `.pdf`

---

## 🎯 Ejecución del Sistema

### Paso 1: Ejecutar Pipeline Completo
```bash
# Ejecutar sistema completo
python main.py
```

### Paso 2: Para Demo con Documentos Nuevos (Opcional)
```bash
# 1. Copiar un documento a data/new/ para simular ingesta nueva
cp data/raw/apple_watch/algún_manual.pdf data/new/fitbit/manual_nuevo.pdf

# 2. Ejecutar pipeline nuevamente para ver clasificación
python main.py
```

---

## 📊 Qué Esperar Durante la Ejecución

### Fase 1: Conexión y Configuración (10 segundos)
```
🚀 === SMARTWATCH KNOWLEDGE SYSTEM CON CLASIFICADOR ===
🔌 PASO 1: Conexión con ChromaDB
✅ Conexión exitosa a Chroma DB
🗑️ Colección limpiada para demo fresca
⚙️ PASO 2: Inicializar Componentes
🧠 Modelo embedding: all-MiniLM-L6-v1
```

### Fase 2: Procesamiento de Datos de Entrenamiento
```
📥 PROCESANDO DOCUMENTOS: DATOS DE ENTRENAMIENTO  
📄 Documentos encontrados: 4
🧠 Generando embeddings...
✅ SAMSUNG: 256 chunks procesados
✅ FITBIT: 104 chunks procesados
```

### Fase 3: Entrenamiento del Clasificador
```
🎓 === ENTRENAMIENTO DEL CLASIFICADOR ===
🏷️ Auto-etiquetando 1481 chunks con REGLAS SIMPLES...
🎯 Accuracy en PRUEBA: 0.976 (97.6%)
💾 Modelo guardado en: models/quality_classifier.joblib
```

### Fase 4: Almacenamiento con Etiquetas
```
🏷️ PASO 4.5: Etiquetar Datos de Entrenamiento para Almacenamiento
💾 Almacenando documentos en Chroma DB...
✅ Datos de entrenamiento etiquetados almacenados: 1481 chunks
🎯 Relevantes: 1222 chunks
```

### Fase 5: Documentos Nuevos (si existen)
```
📥 PROCESANDO DOCUMENTOS: DOCUMENTOS NUEVOS
🔍 === CLASIFICANDO DOCUMENTOS NUEVOS ===
✅ FITBIT: alta_calidad (70/71 relevantes)
```

### Fase 6: Verificación Final
```
🔍 === VERIFICACIÓN FINAL ===
✅ Chunks relevantes encontrados en ChromaDB: 1222
🎯 ¡Sistema listo! Las búsquedas priorizarán chunks relevantes.
```

### Fase 7: Demo de Búsqueda (Interactivo)
```
¿Quieres probar el sistema de búsqueda mejorado? (y/n): y

🔍 === DEMO DE BÚSQUEDA MEJORADO ===
🎯 Los chunks RELEVANTES aparecen primero

🔍 Ingresa tu consulta: apple watch battery life

📄 RESULTADO #1
🏷️  🎯 RELEVANTE
🏢 Marca: APPLE_WATCH  
📊 Similitud: 0.456 - RELACIONADO ✅
📝 Contenido:
  Check the battery percentage, then turn off Low Power Mode...
```

---

## 🔍 Comandos de Verificación y Debug

### Verificar Estado de ChromaDB
```bash
# Estado del contenedor
docker ps | grep chromadb

# Logs de ChromaDB
docker logs chromadb

# Reiniciar ChromaDB si hay problemas
docker restart chromadb
```