"""
Pipeline principal CON integración del clasificador de calidad
Flujo: data/raw/* (entrenar) → data/new/* (clasificar)
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
from src.quality_control.quality_classifier import SimpleQualityClassifier


def setup_logging():
    """Configura el sistema de logging"""
    logger.remove()
    logger.add(
        sys.stdout,
        level=LOG_LEVEL,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )

    LOG_FILE.parent.mkdir(exist_ok=True)
    logger.add(LOG_FILE, level=LOG_LEVEL, rotation="10 MB")


def process_documents_from_directory(
    processor, embedding_generator, base_dir, description
):
    """
    Procesa documentos de un directorio específico

    Args:
        processor: DocumentProcessor instance
        embedding_generator: EmbeddingGenerator instance
        base_dir: Path al directorio base (data/raw o data/new)
        description: Descripción para logging

    Returns:
        Lista de documentos procesados con chunks y embeddings
    """
    logger.info(f"\nPROCESANDO DOCUMENTOS: {description}")
    logger.info(f"Directorio: {base_dir}")
    logger.info("-" * 50)

    # Buscar documentos
    documents_found = []
    if base_dir.exists():
        for brand_dir in base_dir.iterdir():
            if brand_dir.is_dir():
                for file_path in brand_dir.glob("*"):
                    if file_path.suffix in SUPPORTED_EXTENSIONS:
                        documents_found.append(file_path)

    logger.info(f"Documentos encontrados: {len(documents_found)}")

    if not documents_found:
        logger.warning(f"❌ No se encontraron documentos en {base_dir}")
        return []

    # Procesar documentos
    processing_start = time.time()
    all_processed_docs = []

    for i, doc_path in enumerate(documents_found, 1):
        logger.info(f"\nProcesando {i}/{len(documents_found)}: {doc_path.name}")

        # Procesar según tipo
        if doc_path.suffix == ".pdf":
            result = processor.process_pdf(doc_path)
        else:
            result = processor.process_text_file(doc_path)

        if result:
            # Generar embeddings
            logger.info("Generando embeddings...")
            enhanced_chunks = embedding_generator.generate_embeddings(result["chunks"])

            result["chunks"] = enhanced_chunks
            all_processed_docs.append(result)

            chunk_count = len(enhanced_chunks)
            brand = result["metadata"].get("brand", "unknown")
            logger.info(f"✅ {brand.upper()}: {chunk_count} chunks procesados")

    processing_time = time.time() - processing_start
    total_chunks = sum(len(doc["chunks"]) for doc in all_processed_docs)

    logger.info(f"\n📊 RESUMEN {description}:")
    logger.info(f"   Documentos procesados: {len(all_processed_docs)}")
    logger.info(f"   Total chunks: {total_chunks}")
    logger.info(f"    Tiempo: {processing_time:.1f} segundos")

    return all_processed_docs


def train_quality_classifier(processed_docs):
    """
    Entrena el clasificador de calidad con los documentos procesados

    Args:
        processed_docs: Lista de documentos con chunks y embeddings

    Returns:
        Clasificador entrenado o None si falla
    """
    logger.info(f"\n === ENTRENAMIENTO DEL CLASIFICADOR ===")
    logger.info("-" * 50)

    # Preparar datos de entrenamiento (todos los chunks)
    all_chunks = []
    for doc in processed_docs:
        for chunk in doc["chunks"]:
            # Agregar metadatos del documento al chunk para el etiquetado
            chunk_with_metadata = chunk.copy()
            chunk_with_metadata["document_metadata"] = doc["metadata"]
            all_chunks.append(chunk_with_metadata)

    logger.info(f"Total chunks para entrenamiento: {len(all_chunks)}")

    if len(all_chunks) < 10:
        logger.warning("Muy pocos chunks para entrenar clasificador")
        return None

    try:
        # Crear y entrenar clasificador
        classifier = SimpleQualityClassifier()
        training_results = classifier.train(all_chunks)

        # Guardar modelo
        model_path = Path("models/quality_classifier.joblib")
        model_path.parent.mkdir(exist_ok=True)
        classifier.save_model(model_path)

        logger.info("CLASIFICADOR ENTRENADO EXITOSAMENTE!")
        logger.info(f"Precisión: {training_results.get('accuracy', 'N/A')}")
        logger.info(f"Modelo guardado en: {model_path}")

        return classifier

    except Exception as e:
        logger.error(f"Error entrenando clasificador: {e}")
        return None


def classify_new_documents(classifier, processed_docs):
    """
    Clasifica documentos nuevos usando el clasificador entrenado

    Args:
        classifier: SimpleQualityClassifier entrenado
        processed_docs: Lista de documentos a clasificar

    Returns:
        Lista de documentos con clasificación de calidad
    """
    if not classifier or not classifier.is_trained:
        logger.warning("Clasificador no disponible - saltando clasificación")
        return processed_docs

    logger.info(f"\n=== CLASIFICANDO DOCUMENTOS NUEVOS ===")
    logger.info("-" * 50)

    classified_docs = []
    quality_stats = {"relevante": 0, "ambiguo": 0, "irrelevante": 0}

    for doc in processed_docs:
        logger.info(f"Clasificando: {doc['metadata']['file_name']}")

        try:
            # Clasificar chunks del documento
            enhanced_chunks = classifier.predict(doc["chunks"])
            doc["chunks"] = enhanced_chunks

            # Análisis estadístico del documento
            doc_quality_stats = {"relevante": 0, "ambiguo": 0, "irrelevante": 0}
            for chunk in enhanced_chunks:
                label = chunk.get("quality_label", "ambiguo")
                doc_quality_stats[label] += 1
                quality_stats[label] += 1

                # IMPORTANTE: Agregar chunk_quality para búsquedas en ChromaDB
                chunk["chunk_quality"] = label

            total_chunks = len(enhanced_chunks)
            relevant_ratio = (
                doc_quality_stats["relevante"] / total_chunks if total_chunks > 0 else 0
            )

            # Determinar calidad general del documento
            if relevant_ratio >= 0.7:
                doc_quality = "alta_calidad"
            elif relevant_ratio >= 0.4:
                doc_quality = "calidad_media"
            else:
                doc_quality = "baja_calidad"

            # Agregar análisis a metadatos del documento
            doc["metadata"]["quality_analysis"] = {
                "document_quality": doc_quality,
                "relevant_chunks": doc_quality_stats["relevante"],
                "ambiguous_chunks": doc_quality_stats["ambiguo"],
                "irrelevant_chunks": doc_quality_stats["irrelevante"],
                "relevance_ratio": relevant_ratio,
            }

            brand = doc["metadata"].get("brand", "unknown")
            logger.info(
                f"{brand.upper()}: {doc_quality} ({doc_quality_stats['relevante']}/{total_chunks} relevantes)"
            )

            classified_docs.append(doc)

        except Exception as e:
            logger.error(f"Error clasificando documento: {e}")
            classified_docs.append(doc)  # Agregar sin clasificar

    logger.info(f"\n RESUMEN DE CLASIFICACIÓN:")
    logger.info(f"   Documentos clasificados: {len(classified_docs)}")
    logger.info(f"   Distribución de chunks:")
    for label, count in quality_stats.items():
        percentage = (
            (count / sum(quality_stats.values())) * 100
            if sum(quality_stats.values()) > 0
            else 0
        )
        logger.info(f"      {label}: {count} chunks ({percentage:.1f}%)")

    return classified_docs


def enhanced_chroma_pipeline():
    """
    Pipeline completo con entrenamiento y clasificación
    Flujo: data/raw/* (entrenar) → data/new/* (clasificar)
    """
    logger.info("=== SMARTWATCH KNOWLEDGE SYSTEM CON CLASIFICADOR ===")
    logger.info("Pipeline: Entrenamiento + Clasificación + Almacenamiento")
    logger.info("=" * 80)

    total_start = time.time()

    # PASO 1: Verificar conexión con Chroma
    logger.info("\nPASO 1: Conexión con ChromaDB")
    logger.info("-" * 40)

    try:
        chroma_manager = ChromaManager()

        # Verificar estado actual
        stats = chroma_manager.get_collection_stats()
        logger.info(f"Estado actual de Chroma:")
        logger.info(f"   Documentos existentes: {stats.get('total_documents', 0)}")

        # Limpiar colección como siempre (para midterm)
        if stats.get("total_documents", 0) > 0:
            logger.info("Limpiando colección para demo fresca...")
            chroma_manager.clear_collection()
            logger.info("Colección limpiada")

    except Exception as e:
        logger.error(f"Error conectando con Chroma: {e}")
        logger.error("Asegúrate de que Docker esté corriendo:")
        logger.error(
            "   docker run -v ./chroma-data:/data -p 8000:8000 chromadb/chroma"
        )
        return None

    # PASO 2: Inicializar componentes de procesamiento
    logger.info("\nPASO 2: Inicializar Componentes")
    logger.info("-" * 40)

    processor = DocumentProcessor(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    embedding_generator = EmbeddingGenerator(EMBEDDING_MODEL_NAME)

    # Mostrar configuración
    model_info = embedding_generator.get_model_info()
    logger.info(f"Modelo embedding: {model_info['model_name']}")
    logger.info(f"Dimensión: {model_info['embedding_dimension']}D")
    logger.info(f"Chunk size: {CHUNK_SIZE} palabras")

    # PASO 3: Procesar documentos de entrenamiento (data/raw/*)
    training_docs = process_documents_from_directory(
        processor, embedding_generator, RAW_DATA_DIR, "DATOS DE ENTRENAMIENTO"
    )

    if not training_docs:
        logger.error("No hay datos de entrenamiento disponibles")
        return None

    # PASO 4: Entrenar clasificador con datos de entrenamiento
    classifier = train_quality_classifier(training_docs)

    # PASO 4.5: Etiquetar datos de entrenamiento para almacenamiento
    logger.info(f"\nPASO 4.5: Etiquetar Datos de Entrenamiento para Almacenamiento")
    logger.info("-" * 50)

    # Importar el auto labeler
    from src.quality_control.auto_labeler import SimpleAutoLabeler

    auto_labeler = SimpleAutoLabeler()

    # Etiquetar todos los chunks de entrenamiento
    labeled_training_docs = []
    for doc in training_docs:
        labeled_doc = doc.copy()

        # Etiquetar chunks del documento
        logger.info(f"Etiquetando: {doc['metadata']['file_name']}")
        labeled_chunks = auto_labeler.auto_label_chunks(doc["chunks"])
        labeled_doc["chunks"] = labeled_chunks

        # Estadísticas del documento
        quality_stats = {"relevante": 0, "ambiguo": 0, "irrelevante": 0}
        for chunk in labeled_chunks:
            label = chunk.get("auto_label", "ambiguo")
            quality_stats[label] += 1
            # Agregar el label como chunk_quality para búsquedas
            chunk["chunk_quality"] = label

        brand = doc["metadata"].get("brand", "unknown")
        logger.info(
            f"{brand.upper()}: {quality_stats['relevante']} relevantes, {quality_stats['ambiguo']} ambiguos, {quality_stats['irrelevante']} irrelevantes"
        )

        labeled_training_docs.append(labeled_doc)

    # PASO 5: Almacenar datos de entrenamiento ETIQUETADOS en ChromaDB
    logger.info(f"\nPASO 5: Almacenar Datos de Entrenamiento Etiquetados")
    logger.info("-" * 50)

    storage_result = chroma_manager.store_documents(labeled_training_docs)
    logger.info(
        f"Datos de entrenamiento etiquetados almacenados: {storage_result['chunks_stored']} chunks"
    )

    # Verificar que se guardaron las etiquetas
    total_relevant = sum(
        len([c for c in doc["chunks"] if c.get("chunk_quality") == "relevante"])
        for doc in labeled_training_docs
    )
    total_ambiguous = sum(
        len([c for c in doc["chunks"] if c.get("chunk_quality") == "ambiguo"])
        for doc in labeled_training_docs
    )
    total_irrelevant = sum(
        len([c for c in doc["chunks"] if c.get("chunk_quality") == "irrelevante"])
        for doc in labeled_training_docs
    )

    logger.info(f"🔍 Etiquetas guardadas en ChromaDB:")
    logger.info(f"   Relevantes: {total_relevant} chunks")
    logger.info(f"   Ambiguos: {total_ambiguous} chunks")
    logger.info(f"   Irrelevantes: {total_irrelevant} chunks")

    # PASO 6: Procesar documentos nuevos (data/new/*)
    new_data_dir = DATA_DIR / "new"
    new_docs = process_documents_from_directory(
        processor, embedding_generator, new_data_dir, "DOCUMENTOS NUEVOS"
    )

    # PASO 7: Clasificar documentos nuevos (si hay clasificador y documentos)
    if new_docs and classifier:
        classified_new_docs = classify_new_documents(classifier, new_docs)

        # PASO 8: Almacenar documentos nuevos clasificados
        logger.info(f"\nPASO 8: Almacenar Documentos Nuevos Clasificados")
        logger.info("-" * 50)

        storage_result_new = chroma_manager.store_documents(classified_new_docs)
        logger.info(
            f"✅ Documentos nuevos almacenados: {storage_result_new['chunks_stored']} chunks"
        )

        # Verificar etiquetas de documentos nuevos
        new_relevant = sum(
            len([c for c in doc["chunks"] if c.get("chunk_quality") == "relevante"])
            for doc in classified_new_docs
        )
        new_ambiguous = sum(
            len([c for c in doc["chunks"] if c.get("chunk_quality") == "ambiguo"])
            for doc in classified_new_docs
        )
        new_irrelevant = sum(
            len([c for c in doc["chunks"] if c.get("chunk_quality") == "irrelevante"])
            for doc in classified_new_docs
        )

        logger.info(f"🔍 Documentos nuevos - Etiquetas guardadas:")
        logger.info(f"   Relevantes: {new_relevant} chunks")
        logger.info(f"   Ambiguos: {new_ambiguous} chunks")
        logger.info(f"   Irrelevantes: {new_irrelevant} chunks")

    elif new_docs:
        logger.info("\nDocumentos nuevos encontrados pero sin clasificador")
        logger.info("Almacenando sin clasificar...")
        storage_result_new = chroma_manager.store_documents(new_docs)
        logger.info(
            f"Documentos almacenados: {storage_result_new['chunks_stored']} chunks"
        )

    else:
        logger.info(f"\nNo se encontraron documentos nuevos en {new_data_dir}")
        storage_result_new = {"chunks_stored": 0}

    # RESUMEN FINAL
    total_time = time.time() - total_start
    final_stats = chroma_manager.get_collection_stats()

    logger.info(f"\n=== RESUMEN FINAL ===")
    logger.info("=" * 50)
    logger.info(f"Tiempo total: {total_time:.1f} segundos")
    logger.info(f"Documentos de entrenamiento: {len(training_docs)}")
    logger.info(f"Documentos nuevos: {len(new_docs) if new_docs else 0}")
    logger.info(
        f"Clasificador: {'✅ Entrenado' if classifier else '❌ No disponible'}"
    )
    logger.info(f"Total en ChromaDB: {final_stats.get('total_documents', 0)} chunks")
    logger.info(f"Chunks de entrenamiento: {storage_result['chunks_stored']}")
    logger.info(f"Chunks nuevos: {storage_result_new['chunks_stored']}")

    # Verificación final: ¿Puede el sistema encontrar chunks relevantes?
    logger.info(f"\n === VERIFICACIÓN FINAL ===")
    logger.info("Probando si el sistema puede encontrar chunks relevantes...")

    try:
        # Buscar chunks relevantes directamente en ChromaDB
        relevant_test = chroma_manager.collection.get(
            where={"chunk_quality": "relevante"}, limit=5
        )

        relevant_found = len(relevant_test["ids"]) if relevant_test["ids"] else 0
        logger.info(f"✅ Chunks relevantes encontrados en ChromaDB: {relevant_found}")

        if relevant_found > 0:
            logger.info(
                "¡Sistema listo! Las búsquedas priorizarán chunks relevantes."
            )
        else:
            logger.warning(
                "No se encontraron chunks relevantes. Verificar etiquetado."
            )

    except Exception as e:
        logger.warning(f"Error verificando chunks relevantes: {e}")

    return {
        "chroma_manager": chroma_manager,
        "classifier": classifier,
        "training_docs": len(training_docs),
        "new_docs": len(new_docs) if new_docs else 0,
        "total_chunks": final_stats.get("total_documents", 0),
    }


def demo_search_interface(chroma_manager):
    """Demo de búsqueda mejorado con texto completo y prioridades"""
    logger.info("\n🔍 === DEMO DE BÚSQUEDA MEJORADO ===")
    logger.info("Los chunks RELEVANTES aparecen primero")
    logger.info("Se muestra el texto completo de cada resultado")
    print("\n" + "=" * 80)
    print("SISTEMA DE BÚSQUEDA INTELIGENTE")
    print("Prioridad: Resultados RELEVANTES primero")
    print("Similitud negativa = contenido muy diferente a tu consulta")
    print("Similitud alta (>0.3) = muy relacionado con tu consulta")
    print("=" * 80)

    while True:
        try:
            print("\n" + "-" * 60)
            query = input("Ingresa tu consulta (o 'quit' para salir): ").strip()

            if query.lower() in ["quit", "exit", "salir", "q"]:
                break

            if not query:
                continue

            print(f"\nBuscando: '{query}'...")
            print("=" * 80)

            # Realizar búsqueda con priorización
            results = chroma_manager.search(query, top_k=5, prioritize_relevant=True)

            if not results:
                print("❌ No se encontraron resultados")
                continue

            for i, result in enumerate(results, 1):
                similarity = result["similarity_score"]
                text = result["text"]  # TEXTO COMPLETO (sin truncar)
                brand = result["metadata"].get("brand", "unknown").upper()
                doc_name = result["metadata"].get("document_name", "unknown")
                priority = result.get("priority", "⚪ SIN_CLASIFICAR")

                # Interpretación de similitud
                if similarity >= 0.5:
                    similarity_desc = "MUY RELACIONADO "
                elif similarity >= 0.3:
                    similarity_desc = "RELACIONADO "
                elif similarity >= 0.0:
                    similarity_desc = "ALGO RELACIONADO "
                else:
                    similarity_desc = "MUY DIFERENTE ❌"

                print(f"\nRESULTADO #{i}")
                print(f"🏷{priority}")
                print(f" Marca: {brand}")
                print(f"Documento: {doc_name}")
                print(f" Similitud: {similarity:.3f} - {similarity_desc}")
                print(f" Contenido:")
                print("-" * 60)

                # Mostrar texto completo con formato mejorado
                # Dividir en líneas de máximo 80 caracteres para legibilidad
                words = text.split()
                lines = []
                current_line = []
                current_length = 0

                for word in words:
                    if current_length + len(word) + 1 > 78:  # 78 chars + 2 for indent
                        if current_line:
                            lines.append("  " + " ".join(current_line))
                            current_line = [word]
                            current_length = len(word)
                    else:
                        current_line.append(word)
                        current_length += len(word) + 1

                if current_line:
                    lines.append("  " + " ".join(current_line))

                # Imprimir con indentación
                for line in lines:
                    print(line)

                print("-" * 80)

            # Resumen de la búsqueda
            total_relevant = sum(
                1 for r in results if "RELEVANTE" in r.get("priority", "")
            )
            total_results = len(results)
            avg_similarity = sum(r["similarity_score"] for r in results) / len(results)

            print(f"\n RESUMEN DE BÚSQUEDA:")
            print(f"   Similitud promedio: {avg_similarity:.3f}")
            print(f"   Resultados relevantes: {total_relevant}/{total_results}")
            print(f"   Tip: Similitudes >0.3 suelen ser muy útiles")

            # Sugerencia si todas las similitudes son muy bajas
            if avg_similarity < 0.0:
                print(f"\nSUGERENCIA:")
                print(f"   Las similitudes son bajas. Intenta:")
                print(f"   • Usar palabras más específicas del dominio de smartwatches")
                print(
                    f"   • Consultas más técnicas como 'battery life', 'heart rate', 'GPS'"
                )
                print(
                    f"   • Frases en inglés (los manuales pueden tener más contenido en inglés)"
                )

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error en búsqueda: {e}")


def main():
    """Función principal"""
    setup_logging()

    try:
        results = enhanced_chroma_pipeline()

        if results:
            logger.info("Pipeline completado exitosamente!")

            # Demo de búsqueda opcional
            response = (
                input("\n¿Quieres probar el sistema de búsqueda? (y/n): ")
                .strip()
                .lower()
            )
            if response in ["y", "yes", "s", "si"]:
                demo_search_interface(results["chroma_manager"])
        else:
            logger.error("❌ Pipeline falló")

    except Exception as e:
        logger.error(f"💥 Error crítico: {e}")
        raise


if __name__ == "__main__":
    main()
