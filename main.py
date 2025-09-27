"""
Script principal para probar el pipeline de ingesta
"""
import sys
from pathlib import Path
import pandas as pd
from loguru import logger

# Añadir src al path
sys.path.append(str(Path(__file__).parent / "src"))

from config import *
from src.ingestion.document_processor import DocumentProcessor
from src.ingestion.embedding_generator import EmbeddingGenerator


def setup_logging():
    """Configura el sistema de logging"""
    logger.remove()  # Remover handler por defecto
    logger.add(
        sys.stdout,
        level=LOG_LEVEL,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )

    # Crear directorio de logs
    LOG_FILE.parent.mkdir(exist_ok=True)
    logger.add(LOG_FILE, level=LOG_LEVEL, rotation="10 MB")


def test_pipeline():
    """Función de prueba del pipeline completo"""
    logger.info("Iniciando test del pipeline de ingesta")

    # Inicializar componentes
    processor = DocumentProcessor(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    embedding_generator = EmbeddingGenerator(EMBEDDING_MODEL_NAME)

    # Mostrar info del modelo
    model_info = embedding_generator.get_model_info()
    logger.info(f"Modelo cargado: {model_info}")

    # Buscar documentos en el directorio raw
    documents_found = []
    for brand_dir in RAW_DATA_DIR.iterdir():
        if brand_dir.is_dir():
            for file_path in brand_dir.glob("*"):
                if file_path.suffix in SUPPORTED_EXTENSIONS:
                    documents_found.append(file_path)

    logger.info(f"Documentos encontrados: {len(documents_found)}")

    if not documents_found:
        logger.warning("No se encontraron documentos para procesar")
        logger.info(f"Asegúrate de tener documentos en: {RAW_DATA_DIR}")
        logger.info(f"Estructura esperada: {RAW_DATA_DIR}/[marca]/[archivo.pdf|.txt]")
        return

    # Procesar cada documento
    all_processed_docs = []

    for doc_path in documents_found[:2]:  # Procesar solo 2 docs para prueba
        logger.info(f"Procesando: {doc_path}")

        # Procesar según el tipo de archivo
        if doc_path.suffix == ".pdf":
            result = processor.process_pdf(doc_path)
        else:
            result = processor.process_text_file(doc_path)

        if result:
            # Generar embeddings
            enhanced_chunks = embedding_generator.generate_embeddings(result["chunks"])

            result["chunks"] = enhanced_chunks
            all_processed_docs.append(result)

            logger.info(f"Documento procesado: {len(enhanced_chunks)} chunks generados")

    logger.info(f"Pipeline completado. {len(all_processed_docs)} documentos procesados")

    # Mostrar estadísticas
    total_chunks = sum(len(doc["chunks"]) for doc in all_processed_docs)
    logger.info(f"Total de chunks con embeddings: {total_chunks}")

    return all_processed_docs


if __name__ == "__main__":
    setup_logging()

    logger.info("=== SMARTWATCH KNOWLEDGE SYSTEM ===")
    logger.info("Iniciando pipeline de ingesta")

    try:
        processed_docs = test_pipeline()
        logger.info("Test completado exitosamente")

    except Exception as e:
        logger.error(f"Error en el pipeline: {e}")
        raise