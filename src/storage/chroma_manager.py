"""
ChromaManager - Maneja todas las operaciones con Chroma DB
Fase 1: Almacenamiento e integración con el pipeline existente
"""
import chromadb
from typing import List, Dict, Any, Optional
from loguru import logger
import uuid
import time
from pathlib import Path


class ChromaManager:
    """Gestor de base de datos vectorial Chroma para smartwatch docs"""

    def __init__(self, host: str = "localhost", port: int = 8000, collection_name: str = "smartwatch_docs"):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.client = None
        self.collection = None

        # Conectar automáticamente
        self.connect()

    def connect(self):
        """Establece conexión con Chroma DB"""
        try:
            logger.info(f"🔌 Conectando a Chroma DB en {self.host}:{self.port}")

            # Crear cliente HTTP
            self.client = chromadb.HttpClient(host=self.host, port=self.port)

            # Verificar conexión
            heartbeat = self.client.heartbeat()
            logger.info(f"✅ Conexión exitosa a Chroma DB: {heartbeat}")

            # Crear o obtener colección
            self._setup_collection()

        except Exception as e:
            logger.error(f"❌ Error conectando a Chroma DB: {e}")
            logger.error("💡 Asegúrate de que el contenedor Docker esté corriendo:")
            logger.error("   docker run -v ./chroma-data:/data -p 8000:8000 chromadb/chroma")
            raise

    def _setup_collection(self):
        """Configura la colección de documentos"""
        try:
            # Intentar obtener colección existente
            self.collection = self.client.get_collection(name=self.collection_name)

            # Verificar contenido existente
            count = self.collection.count()
            logger.info(f"📚 Colección '{self.collection_name}' encontrada con {count} documentos")

        except Exception:
            # Crear nueva colección si no existe
            logger.info(f"🆕 Creando nueva colección '{self.collection_name}'")

            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={
                    "description": "Smartwatch technical documentation embeddings",
                    "embedding_model": "all-MiniLM-L6-v1",
                    "embedding_dimension": 384,
                    "created_by": "smartwatch_knowledge_system"
                }
            )
            logger.info(f"✅ Colección '{self.collection_name}' creada exitosamente")

    def store_documents(self, processed_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Almacena documentos procesados en Chroma

        Args:
            processed_docs: Lista de documentos procesados con chunks y embeddings

        Returns:
            Estadísticas del almacenamiento
        """
        logger.info("💾 Almacenando documentos en Chroma DB...")
        start_time = time.time()

        all_ids = []
        all_embeddings = []
        all_documents = []
        all_metadatas = []

        total_chunks = 0

        for doc in processed_docs:
            doc_metadata = doc["metadata"]
            chunks = doc["chunks"]

            logger.info(f"📄 Procesando {doc_metadata['file_name']}: {len(chunks)} chunks")

            for i, chunk in enumerate(chunks):
                # Crear ID único
                chunk_id = self._generate_chunk_id(doc_metadata, i)

                # Preparar metadatos enriquecidos
                enriched_metadata = {
                    "brand": doc_metadata.get("brand", "unknown"),
                    "document_name": doc_metadata["file_name"],
                    "document_path": doc_metadata["file_path"],
                    "chunk_index": i,
                    "word_count": chunk.get("word_count", 0),
                    "char_count": chunk.get("char_count", 0),
                    "document_type": doc_metadata.get("document_type", "pdf"),
                    "processing_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "total_chunks_in_doc": len(chunks),
                    "embedding_model": chunk.get("embedding_model", "all-MiniLM-L6-v1"),
                    "embedding_dimension": len(chunk["embedding"])
                }
                if "quality_analysis" in doc_metadata:
                    enriched_metadata["document_quality"] = doc_metadata["quality_analysis"]["document_quality"]

                chunk_quality = None

                # Buscar calidad del chunk en diferentes campos (por compatibilidad)
                if "chunk_quality" in chunk:
                    chunk_quality = chunk["chunk_quality"]
                elif "quality_label" in chunk:
                    chunk_quality = chunk["quality_label"]
                elif "auto_label" in chunk:
                    chunk_quality = chunk["auto_label"]

                # Guardar calidad del chunk si existe
                if chunk_quality:
                    enriched_metadata["chunk_quality"] = chunk_quality
                    enriched_metadata["quality_confidence"] = chunk.get("quality_confidence",
                                                                        chunk.get("auto_confidence", 0.0))

                # Agregar a listas
                all_ids.append(chunk_id)
                all_embeddings.append(chunk["embedding"])
                all_documents.append(chunk["text"])
                all_metadatas.append(enriched_metadata)

                total_chunks += 1

        # Almacenar en lotes para eficiencia
        batch_size = 100  # Chroma recomienda lotes de ~100
        stored_count = 0

        for i in range(0, len(all_ids), batch_size):
            batch_end = min(i + batch_size, len(all_ids))

            batch_ids = all_ids[i:batch_end]
            batch_embeddings = all_embeddings[i:batch_end]
            batch_documents = all_documents[i:batch_end]
            batch_metadatas = all_metadatas[i:batch_end]

            try:
                self.collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    documents=batch_documents,
                    metadatas=batch_metadatas
                )

                stored_count += len(batch_ids)
                logger.info(f"   💾 Lote {i // batch_size + 1}: {len(batch_ids)} chunks almacenados")

            except Exception as e:
                logger.error(f"❌ Error almacenando lote {i // batch_size + 1}: {e}")
                raise

        storage_time = time.time() - start_time

        # Verificar almacenamiento
        final_count = self.collection.count()

        logger.info(f"✅ Almacenamiento completado:")
        logger.info(f"   📦 Chunks almacenados: {stored_count}")
        logger.info(f"   📚 Total en colección: {final_count}")
        logger.info(f"   ⏱️ Tiempo: {storage_time:.2f} segundos")
        logger.info(f"   ⚡ Velocidad: {stored_count / storage_time:.1f} chunks/segundo")

        return {
            "chunks_stored": stored_count,
            "total_in_collection": final_count,
            "storage_time": storage_time,
            "storage_rate": stored_count / storage_time,
            "success": True
        }

    def _generate_chunk_id(self, doc_metadata: Dict[str, Any], chunk_index: int) -> str:
        """Genera ID único para un chunk"""
        brand = doc_metadata.get("brand", "unknown")
        doc_name = Path(doc_metadata["file_name"]).stem  # Sin extensión

        # Crear ID legible y único
        chunk_id = f"{brand}_{doc_name}_chunk_{chunk_index:03d}"

        # Limpiar caracteres problemáticos
        chunk_id = chunk_id.replace(" ", "_").replace("-", "_").lower()

        return chunk_id

    def search(self, query: str, top_k: int = 5, brand_filter: str = None, prioritize_relevant: bool = True) -> List[
        Dict[str, Any]]:
        """
        Realiza búsqueda semántica en Chroma con priorización de chunks relevantes

        Args:
            query: Consulta en lenguaje natural
            top_k: Número de resultados a retornar
            brand_filter: Filtrar por marca específica
            prioritize_relevant: Si priorizar chunks marcados como "relevante"

        Returns:
            Lista de resultados con metadatos y scores, priorizados por relevancia
        """
        if not self.collection:
            raise ValueError("❌ No hay conexión a Chroma DB")

        logger.info(f"🔍 Buscando en Chroma: '{query}' (prioridad: {'relevantes' if prioritize_relevant else 'todos'})")
        start_time = time.time()

        # Preparar filtros base
        base_filter = {}
        if brand_filter:
            base_filter["brand"] = brand_filter.lower()

        processed_results = []

        try:
            if prioritize_relevant:
                # PASO 1: Buscar primero en chunks RELEVANTES
                relevant_filter = base_filter.copy()
                relevant_filter["chunk_quality"] = "relevante"

                logger.info("🎯 Buscando primero en chunks RELEVANTES...")
                relevant_results = self.collection.query(
                    query_texts=[query],
                    n_results=min(top_k * 2, 20),  # Buscar más para tener opciones
                    where=relevant_filter if relevant_filter else None,
                    include=["documents", "metadatas", "distances"]
                )

                # Procesar resultados relevantes
                if relevant_results["documents"] and relevant_results["documents"][0]:
                    for i in range(len(relevant_results["documents"][0])):
                        result = {
                            "text": relevant_results["documents"][0][i],
                            "metadata": relevant_results["metadatas"][0][i],
                            "distance": relevant_results["distances"][0][i],
                            "similarity_score": 1 - relevant_results["distances"][0][i],
                            "ranking": len(processed_results) + 1,
                            "priority": "🎯 RELEVANTE"
                        }
                        processed_results.append(result)

                # PASO 2: Si necesitamos más resultados, buscar en el resto
                remaining_needed = top_k - len(processed_results)

                if remaining_needed > 0:
                    logger.info(f"🔍 Buscando {remaining_needed} resultados adicionales en todos los chunks...")

                    # Buscar en todos (sin filtro de calidad)
                    all_results = self.collection.query(
                        query_texts=[query],
                        n_results=min(top_k * 3, 30),  # Buscar más para filtrar duplicados
                        where=base_filter if base_filter else None,
                        include=["documents", "metadatas", "distances"]
                    )

                    # Agregar resultados no duplicados
                    existing_ids = set()
                    for result in processed_results:
                        # Usar hash del texto para evitar duplicados
                        existing_ids.add(hash(result["text"]))

                    if all_results["documents"] and all_results["documents"][0]:
                        for i in range(len(all_results["documents"][0])):
                            if len(processed_results) >= top_k:
                                break

                            text = all_results["documents"][0][i]
                            text_hash = hash(text)

                            # Evitar duplicados
                            if text_hash not in existing_ids:
                                metadata = all_results["metadatas"][0][i]
                                quality = metadata.get("chunk_quality", "sin_clasificar")

                                # Marcar prioridad según calidad
                                if quality == "relevante":
                                    priority = "🎯 RELEVANTE"
                                elif quality == "ambiguo":
                                    priority = "🟡 AMBIGUO"
                                elif quality == "irrelevante":
                                    priority = "🔴 IRRELEVANTE"
                                else:
                                    priority = "⚪ SIN_CLASIFICAR"

                                result = {
                                    "text": text,
                                    "metadata": metadata,
                                    "distance": all_results["distances"][0][i],
                                    "similarity_score": 1 - all_results["distances"][0][i],
                                    "ranking": len(processed_results) + 1,
                                    "priority": priority
                                }
                                processed_results.append(result)
                                existing_ids.add(text_hash)

            else:
                # Búsqueda normal sin priorización
                logger.info("🔍 Buscando en todos los chunks...")
                results = self.collection.query(
                    query_texts=[query],
                    n_results=top_k,
                    where=base_filter if base_filter else None,
                    include=["documents", "metadatas", "distances"]
                )

                if results["documents"] and results["documents"][0]:
                    for i in range(len(results["documents"][0])):
                        metadata = results["metadatas"][0][i]
                        quality = metadata.get("chunk_quality", "sin_clasificar")

                        if quality == "relevante":
                            priority = "🎯 RELEVANTE"
                        elif quality == "ambiguo":
                            priority = "🟡 AMBIGUO"
                        elif quality == "irrelevante":
                            priority = "🔴 IRRELEVANTE"
                        else:
                            priority = "⚪ SIN_CLASIFICAR"

                        result = {
                            "text": results["documents"][0][i],
                            "metadata": metadata,
                            "distance": results["distances"][0][i],
                            "similarity_score": 1 - results["distances"][0][i],
                            "ranking": i + 1,
                            "priority": priority
                        }
                        processed_results.append(result)

            search_time = time.time() - start_time

            # Limitar a top_k resultados finales
            final_results = processed_results[:top_k]

            # Actualizar rankings finales
            for i, result in enumerate(final_results):
                result["ranking"] = i + 1

            logger.info(f"✅ Búsqueda completada en {search_time:.3f}s: {len(final_results)} resultados")

            # Log de distribución de calidad
            quality_count = {}
            for result in final_results:
                priority = result["priority"]
                quality_count[priority] = quality_count.get(priority, 0) + 1

            if quality_count:
                logger.info("📊 Distribución de resultados por calidad:")
                for priority, count in quality_count.items():
                    logger.info(f"   {priority}: {count} resultados")

            return final_results

        except Exception as e:
            logger.error(f"❌ Error en búsqueda: {e}")
            return []

    def get_collection_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de la colección"""
        if not self.collection:
            return {"error": "No connection to Chroma"}

        try:
            count = self.collection.count()

            # Obtener muestra para análisis
            sample = self.collection.get(limit=10, include=["metadatas"])

            # Analizar marcas
            brands = {}
            if sample["metadatas"]:
                for metadata in sample["metadatas"]:
                    brand = metadata.get("brand", "unknown")
                    brands[brand] = brands.get(brand, 0) + 1

            return {
                "total_documents": count,
                "collection_name": self.collection_name,
                "brands_sample": brands,
                "status": "connected"
            }

        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {e}")
            return {"error": str(e)}

    def clear_collection(self):
        """Limpia toda la colección (usar con cuidado)"""
        logger.warning("⚠️ Limpiando toda la colección...")

        if self.collection:
            # Nota: Chroma no tiene clear() directo, necesitamos recrear
            self.client.delete_collection(name=self.collection_name)
            self._setup_collection()
            logger.info("🗑️ Colección limpiada y recreada")

    def health_check(self) -> bool:
        """Verifica que Chroma DB esté funcionando correctamente"""
        try:
            heartbeat = self.client.heartbeat()
            collection_count = self.collection.count()

            logger.info(f"💚 Health Check OK:")
            logger.info(f"   🔌 Heartbeat: {heartbeat}")
            logger.info(f"   📚 Documentos: {collection_count}")

            return True

        except Exception as e:
            logger.error(f"💔 Health Check FAILED: {e}")
            return False


def test_chroma_connection():
    """Función de prueba para verificar conexión con Chroma"""
    logger.info("🧪 Probando conexión con Chroma DB...")

    try:
        # Crear manager
        manager = ChromaManager()

        # Health check
        if manager.health_check():
            logger.info("✅ Chroma DB funcionando correctamente")

            # Mostrar estadísticas
            stats = manager.get_collection_stats()
            logger.info(f"📊 Estadísticas: {stats}")

            return manager
        else:
            logger.error("❌ Chroma DB no está funcionando")
            return None

    except Exception as e:
        logger.error(f"💥 Error en test de conexión: {e}")
        return None


if __name__ == "__main__":
    # Test de conexión independiente
    import sys
    from loguru import logger

    logger.remove()
    logger.add(sys.stdout, level="INFO")

    manager = test_chroma_connection()