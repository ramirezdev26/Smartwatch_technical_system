"""
Configuración actualizada CON integración Chroma DB
Reemplaza tu config.py actual
"""

import os
from pathlib import Path

# Paths del proyecto
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
NEW_DATA_DIR = DATA_DIR / "new"

# Configuración del modelo de embeddings
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v1"
EMBEDDING_DIMENSION = 384
MAX_SEQUENCE_LENGTH = 256

# Configuración de chunking optimizada
CHUNK_SIZE = 150  # palabras
CHUNK_OVERLAP = 20  # palabras
MIN_CHUNK_LENGTH = 50  # caracteres mínimos

# ========================================
# CONFIGURACIÓN CHROMA DB (NUEVO)
# ========================================

# Conexión con Chroma DB
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
CHROMA_COLLECTION_NAME = "smartwatch_docs"

# Configuración de almacenamiento en Chroma
CHROMA_BATCH_SIZE = 100  # Chunks por lote al almacenar
CHROMA_MAX_RETRIES = 3  # Reintentos en caso de error

# ========================================
# CONFIGURACIÓN EXISTENTE
# ========================================

# Configuración de la base de datos vectorial local (para backup)
CHROMA_DB_PATH = str(DATA_DIR / "chroma_db")

# Marcas de smartwatches soportadas
SUPPORTED_BRANDS = ["apple_watch", "garmin", "fitbit", "samsung"]

# Tipos de documentos soportados
SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".md"]

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = PROJECT_ROOT / "logs" / "system.log"

# Configuración para debugging
DEBUG_PDF_EXTRACTION = True
MAX_CHUNKS_PER_DOC = 1000  # Límite de seguridad

# ========================================
# CONFIGURACIÓN FASE 2 (PRÓXIMA)
# ========================================

# Control de calidad - Clasificador de relevancia
RELEVANCE_CLASSIFIER_MODEL = "logistic_regression"
RELEVANCE_THRESHOLD = 0.7  # Umbral para considerar relevante

# Detección de anomalías
ANOMALY_DETECTOR_MODEL = "isolation_forest"
ANOMALY_CONTAMINATION = 0.1  # Porcentaje esperado de anomalías

# Búsqueda semántica
DEFAULT_SEARCH_RESULTS = 5  # Número default de resultados
MAX_SEARCH_RESULTS = 20  # Máximo permitido

# ========================================
# CREAR DIRECTORIOS
# ========================================

# Crear directorios si no existen
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, EMBEDDINGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

for brand in SUPPORTED_BRANDS:
    (RAW_DATA_DIR / brand).mkdir(exist_ok=True)

NEW_DATA_DIR.mkdir(exist_ok=True)
for brand in SUPPORTED_BRANDS:
    (NEW_DATA_DIR / brand).mkdir(exist_ok=True)

# Crear directorio de logs
LOG_FILE.parent.mkdir(exist_ok=True)
