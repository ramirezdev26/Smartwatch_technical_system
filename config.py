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
MAX_SEQUENCE_LENGTH = 256

# Configuración de la base de datos vectorial
CHROMA_DB_PATH = str(DATA_DIR / "chroma_db")
COLLECTION_NAME = "smartwatch_docs"

# Configuración de procesamiento
CHUNK_SIZE = 512  # Tamaño de chunks para documentos largos
CHUNK_OVERLAP = 50  # Overlap entre chunks

# Marcas de smartwatches soportadas
SUPPORTED_BRANDS = ["apple_watch", "garmin", "fitbit", "samsung"]

# Tipos de documentos soportados
SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".md"]

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = PROJECT_ROOT / "logs" / "system.log"

# Crear directorios si no existen
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, EMBEDDINGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

for brand in SUPPORTED_BRANDS:
    (RAW_DATA_DIR / brand).mkdir(exist_ok=True)