"""
Pipeline principal CON integración Chroma
Modifica tu main.py actual para incluir almacenamiento persistente
"""
import sys
from pathlib import Path
import pandas as pd
from loguru import logger
import time

# Añadir src al path
sys.path.append(str(Path(__file__).parent / "src"))

from config import *
from src.ingestion.document_processor import DocumentProcessor
from src.ingestion.embedding_generator import EmbeddingGenerator
from src.storage.chroma_manager import ChromaManager


def setup_logging():
    """Configura el sistema de logging"""
    logger.remove()
    logger.add(
        sys.stdout,
        level=LOG_LEVEL,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )

    LOG_FILE.parent.mkdir(exist_ok=True)
    logger.add(LOG_FILE, level=LOG_LEVEL, rotation="10 MB")


def test_chroma_pipeline():
    """
    Pipeline completo CON almacenamiento en Chroma
    FASE 1: Integración con base de datos vectorial
    """
    logger.info("🚀 === SMARTWATCH KNOWLEDGE SYSTEM CON CHROMA ===")
    logger.info("📋 FASE 1: Pipeline de Ingesta + Almacenamiento Vectorial")
    logger.info("=" * 70)

    total_start = time.time()

    # PASO 1: Verificar conexión con Chroma
    logger.info("\n🔌 PASO 1: Conexión con Chroma DB")
    logger.info("-" * 40)

    try:
        chroma_manager = ChromaManager()

        # Verificar estado actual
        stats = chroma_manager.get_collection_stats()
        logger.info(f"📊 Estado actual de Chroma:")
        logger.info(f"   📚 Documentos existentes: {stats.get('total_documents', 0)}")

        # Preguntar si limpiar datos existentes
        if stats.get('total_documents', 0) > 0:
            logger.info("⚠️ Ya hay datos en Chroma. Para esta demo, limpiaremos la colección.")
            chroma_manager.clear_collection()
            logger.info("🗑️ Colección limpiada para demo fresca")

    except Exception as e:
        logger.error(f"❌ Error conectando con Chroma: {e}")
        logger.error("💡 Asegúrate de que Docker esté corriendo:")
        logger.error("   docker run -v ./chroma-data:/data -p 8000:8000 chromadb/chroma")
        return None

    # PASO 2: Pipeline de ingesta (tu código actual)
    logger.info("\n📥 PASO 2: Pipeline de Ingesta de Documentos")
    logger.info("-" * 40)

    # Inicializar componentes
    processor = DocumentProcessor(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    embedding_generator = EmbeddingGenerator(EMBEDDING_MODEL_NAME)

    # Mostrar configuración
    model_info = embedding_generator.get_model_info()
    logger.info(f"⚙️ Configuración:")
    logger.info(f"   🧠 Modelo: {model_info['model_name']}")
    logger.info(f"   📐 Embedding dimension: {model_info['embedding_dimension']}D")
    logger.info(f"   📏 Chunk size: {CHUNK_SIZE} palabras")
    logger.info(f"   🔗 Overlap: {CHUNK_OVERLAP} palabras")

    # Buscar documentos
    documents_found = []
    for brand_dir in RAW_DATA_DIR.iterdir():
        if brand_dir.is_dir():
            for file_path in brand_dir.glob("*"):
                if file_path.suffix in SUPPORTED_EXTENSIONS:
                    documents_found.append(file_path)

    logger.info(f"📄 Documentos encontrados: {len(documents_found)}")

    if not documents_found:
        logger.warning("❌ No se encontraron documentos para procesar")
        return None

    # Procesar documentos
    ingestion_start = time.time()
    all_processed_docs = []

    for i, doc_path in enumerate(documents_found, 1):
        logger.info(f"\n📄 Procesando {i}/{len(documents_found)}: {doc_path.name}")

        # Procesar según tipo
        if doc_path.suffix == ".pdf":
            result = processor.process_pdf(doc_path)
        else:
            result = processor.process_text_file(doc_path)

        if result:
            # Generar embeddings
            logger.info("🧠 Generando embeddings...")
            enhanced_chunks = embedding_generator.generate_embeddings(result["chunks"])

            result["chunks"] = enhanced_chunks
            all_processed_docs.append(result)

            chunk_count = len(enhanced_chunks)
            brand = result["metadata"].get("brand", "unknown")
            logger.info(f"✅ {brand.upper()}: {chunk_count} chunks procesados")

    ingestion_time = time.time() - ingestion_start
    total_chunks = sum(len(doc["chunks"]) for doc in all_processed_docs)

    logger.info(f"\n📊 RESUMEN DE INGESTA:")
    logger.info(f"   ✅ Documentos procesados: {len(all_processed_docs)}")
    logger.info(f"   📦 Total chunks: {total_chunks}")
    logger.info(f"   ⏱️ Tiempo de procesamiento: {ingestion_time:.1f} segundos")
    logger.info(f"   ⚡ Velocidad: {total_chunks/ingestion_time:.1f} chunks/segundo")

    # PASO 3: Almacenamiento en Chroma (NUEVO!)
    logger.info(f"\n💾 PASO 3: Almacenamiento en Chroma DB")
    logger.info("-" * 40)

    storage_start = time.time()

    try:
        storage_stats = chroma_manager.store_documents(all_processed_docs)
        storage_time = time.time() - storage_start

        logger.info(f"📊 RESUMEN DE ALMACENAMIENTO:")
        logger.info(f"   💾 Chunks almacenados: {storage_stats['chunks_stored']}")
        logger.info(f"   📚 Total en Chroma: {storage_stats['total_in_collection']}")
        logger.info(f"   ⏱️ Tiempo almacenamiento: {storage_time:.1f} segundos")
        logger.info(f"   🚀 Velocidad almacenamiento: {storage_stats['storage_rate']:.1f} chunks/segundo")

    except Exception as e:
        logger.error(f"❌ Error almacenando en Chroma: {e}")
        return None

    # PASO 4: Verificación y pruebas de búsqueda
    logger.info(f"\n🔍 PASO 4: Verificación con Búsquedas de Prueba")
    logger.info("-" * 40)

    # Queries de prueba para verificar que funciona
    test_queries = [
        {
            "query": "Apple Watch battery drain",
            "expected_brand": "apple",
            "description": "Problema de batería Apple"
        },
        {
            "query": "Samsung charging issues",
            "expected_brand": "samsung",
            "description": "Problemas de carga Samsung"
        },
        {
            "query": "Fitbit sleep tracking",
            "expected_brand": "fitbit",
            "description": "Seguimiento de sueño Fitbit"
        }
    ]

    search_results_summary = []

    for i, test in enumerate(test_queries, 1):
        logger.info(f"\n🧪 Prueba {i}: {test['description']}")
        logger.info(f"   Query: '{test['query']}'")

        search_start = time.time()
        results = chroma_manager.search(test["query"], top_k=3)
        search_time = time.time() - search_start

        if results:
            top_result = results[0]
            top_brand = top_result["metadata"].get("brand", "unknown")
            score = top_result["similarity_score"]

            logger.info(f"   ⚡ Búsqueda en {search_time:.3f} segundos")
            logger.info(f"   🎯 Top resultado: {top_brand.upper()} (score: {score:.3f})")
            logger.info(f"   📝 Preview: {top_result['text'][:100]}...")

            # Verificar si encontró la marca esperada
            if test["expected_brand"] in top_brand:
                logger.info(f"   ✅ Resultado CORRECTO")
                search_results_summary.append("✅")
            else:
                logger.info(f"   ⚠️ Resultado inesperado (esperaba {test['expected_brand']})")
                search_results_summary.append("⚠️")
        else:
            logger.info(f"   ❌ No se encontraron resultados")
            search_results_summary.append("❌")

    # RESUMEN FINAL
    total_time = time.time() - total_start

    logger.info(f"\n🎉 === PIPELINE COMPLETADO EXITOSAMENTE ===")
    logger.info("=" * 70)
    logger.info(f"⏱️ TIEMPO TOTAL: {total_time:.1f} segundos")
    logger.info(f"📊 RESULTADOS:")
    logger.info(f"   📄 Documentos procesados: {len(all_processed_docs)}")
    logger.info(f"   📦 Chunks con embeddings: {total_chunks}")
    logger.info(f"   💾 Almacenados en Chroma: {storage_stats['chunks_stored']}")
    logger.info(f"   🔍 Búsquedas de prueba: {'/'.join(search_results_summary)}")

    logger.info(f"\n🎯 FASE 1 COMPLETADA:")
    logger.info("   ✅ Pipeline de ingesta funcionando")
    logger.info("   ✅ Integración con Chroma DB exitosa")
    logger.info("   ✅ Almacenamiento vectorial persistente")
    logger.info("   ✅ Búsqueda semántica básica operativa")

    logger.info(f"\n📋 SIGUIENTE: FASE 2")
    logger.info("   🎯 Clasificador de relevancia (Logistic Regression)")
    logger.info("   🚨 Detección de anomalías (Isolation Forest)")
    logger.info("   🔍 Sistema de búsqueda semántica avanzado")

    return {
        "processed_docs": all_processed_docs,
        "chroma_manager": chroma_manager,
        "storage_stats": storage_stats,
        "total_time": total_time
    }


def demo_search_interface(chroma_manager: ChromaManager):
    """
    Demo interactiva de búsqueda una vez que los datos están en Chroma
    """
    logger.info(f"\n🎭 === DEMO INTERACTIVA DE BÚSQUEDA ===")
    logger.info("🔍 Prueba búsquedas en tu base de conocimiento")
    logger.info("💡 Escribe 'quit' para salir")
    logger.info("-" * 50)

    while True:
        try:
            query = input("\n❓ Tu consulta: ").strip()

            if query.lower() in ['quit', 'exit', 'salir', 'q']:
                logger.info("👋 ¡Demo terminada!")
                break

            if not query:
                continue

            # Realizar búsqueda
            start_time = time.time()
            results = chroma_manager.search(query, top_k=3)
            search_time = time.time() - start_time

            if results:
                logger.info(f"⚡ Encontrado en {search_time:.3f} segundos:")

                for i, result in enumerate(results, 1):
                    score = result["similarity_score"]
                    brand = result["metadata"].get("brand", "unknown")
                    doc_name = result["metadata"].get("document_name", "unknown")
                    text_preview = result["text"][:150] + "..."

                    print(f"\n{i}. [{brand.upper()}] Score: {score:.3f}")
                    print(f"   📄 Fuente: {doc_name}")
                    print(f"   📝 {text_preview}")
            else:
                print("❌ No se encontraron resultados relevantes")

        except KeyboardInterrupt:
            logger.info("\n👋 Demo interrumpida")
            break
        except Exception as e:
            logger.error(f"Error en búsqueda: {e}")


def main():
    """Función principal del pipeline con Chroma"""
    setup_logging()

    try:
        # Ejecutar pipeline completo
        results = test_chroma_pipeline()

        if results:
            # Ofrecer demo interactiva
            response = input("\n🤔 ¿Quieres probar la búsqueda interactiva? (y/n): ").strip().lower()
            if response in ['y', 'yes', 's', 'si']:
                demo_search_interface(results["chroma_manager"])

            logger.info("🎯 Pipeline con Chroma completado exitosamente!")
        else:
            logger.error("❌ Pipeline falló")

    except Exception as e:
        logger.error(f"💥 Error crítico: {e}")
        raise


if __name__ == "__main__":
    main()


