# config.py - Configuración corregida

import os
from pathlib import Path

# Paths del proyecto
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

# Configuración del modelo de embeddings
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v1"
EMBEDDING_DIMENSION = 384
MAX_SEQUENCE_LENGTH = 256  # Máximo del modelo

# ⚠️ CONFIGURACIÓN CORREGIDA PARA CHUNKS
# Calcular chunk size basado en tokens, no palabras
# Relación aproximada: 1 token ≈ 0.75 palabras en español
CHUNK_SIZE_TOKENS = 200  # Dejar margen para el modelo (256 max)
CHUNK_SIZE_WORDS = int(CHUNK_SIZE_TOKENS * 0.75)  # ~150 palabras
CHUNK_OVERLAP_WORDS = 20  # Overlap más pequeño

# Configuración de la base de datos vectorial
CHROMA_DB_PATH = str(DATA_DIR / "chroma_db")
COLLECTION_NAME = "smartwatch_docs"

# Marcas de smartwatches soportadas
SUPPORTED_BRANDS = ["apple_watch", "garmin", "fitbit", "samsung"]

# Tipos de documentos soportados
SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".md"]

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = PROJECT_ROOT / "logs" / "system.log"

# Configuración para debugging de PDFs
DEBUG_PDF_EXTRACTION = True
MIN_CHUNK_LENGTH = 50  # Mínimo de caracteres para considerar un chunk válido
MAX_CHUNKS_PER_DOC = 1000  # Límite de seguridad

# Crear directorios si no existen
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, EMBEDDINGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

for brand in SUPPORTED_BRANDS:
    (RAW_DATA_DIR / brand).mkdir(exist_ok=True)