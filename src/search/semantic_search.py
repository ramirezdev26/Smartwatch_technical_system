"""
Sistema de búsqueda semántica para tu demo midterm
"""
import numpy as np
from typing import List, Dict, Any, Tuple
from pathlib import Path
import pickle
import json
from loguru import logger
from sentence_transformers import SentenceTransformer


class SmartWatchSemanticSearch:
    """Sistema de búsqueda semántica para documentos de smartwatches"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v1"):
        self.model_name = model_name
        self.model = None
        self.chunks_database = []
        self.embeddings_matrix = None

        # Cargar modelo
        self._load_model()

    def _load_model(self):
        """Carga el modelo de embeddings"""
        logger.info(f"Cargando modelo {self.model_name} para búsqueda...")
        self.model = SentenceTransformer(self.model_name)
        logger.info("✅ Modelo de búsqueda cargado")

    def load_processed_documents(self, processed_docs: List[Dict[str, Any]]):
        """
        Carga documentos procesados y crea base de datos para búsqueda

        Args:
            processed_docs: Lista de documentos procesados con chunks y embeddings
        """
        logger.info("📥 Cargando documentos procesados para búsqueda...")

        all_chunks = []
        all_embeddings = []

        for doc in processed_docs:
            for chunk in doc["chunks"]:
                # Agregar metadatos del documento al chunk
                enriched_chunk = {
                    **chunk,
                    "source_document": doc["metadata"]["file_name"],
                    "brand": doc["metadata"].get("brand", "unknown"),
                    "file_path": doc["metadata"]["file_path"]
                }
                all_chunks.append(enriched_chunk)
                all_embeddings.append(chunk["embedding"])

        self.chunks_database = all_chunks
        self.embeddings_matrix = np.array(all_embeddings)

        logger.info(f"✅ Base de conocimiento cargada: {len(all_chunks)} chunks de {len(processed_docs)} documentos")

        # Mostrar estadísticas por marca
        brands = {}
        for chunk in all_chunks:
            brand = chunk.get("brand", "unknown")
            brands[brand] = brands.get(brand, 0) + 1

        logger.info("📊 Distribución por marca:")
        for brand, count in brands.items():
            logger.info(f"   {brand}: {count} chunks")

    def search(self, query: str, top_k: int = 5, brand_filter: str = None) -> List[Dict[str, Any]]:
        """
        Realiza búsqueda semántica

        Args:
            query: Consulta en lenguaje natural
            top_k: Número de resultados a retornar
            brand_filter: Filtrar por marca específica (opcional)

        Returns:
            Lista de chunks más relevantes con scores
        """
        if not self.chunks_database:
            logger.error("❌ No hay documentos cargados")
            return []

        logger.info(f"🔍 Buscando: '{query}'")

        # Generar embedding de la consulta
        query_embedding = self.model.encode([query])[0]

        # Calcular similitudes coseno
        similarities = np.dot(self.embeddings_matrix, query_embedding) / (
                np.linalg.norm(self.embeddings_matrix, axis=1) * np.linalg.norm(query_embedding)
        )

        # Obtener índices ordenados por similitud
        sorted_indices = np.argsort(similarities)[::-1]

        # Aplicar filtro de marca si se especifica
        filtered_results = []
        for idx in sorted_indices:
            chunk = self.chunks_database[idx]
            similarity_score = similarities[idx]

            # Filtrar por marca si se especifica
            if brand_filter and chunk.get("brand", "").lower() != brand_filter.lower():
                continue

            result = {
                **chunk,
                "similarity_score": float(similarity_score),
                "ranking_position": len(filtered_results) + 1
            }
            filtered_results.append(result)

            # Parar cuando tengamos suficientes resultados
            if len(filtered_results) >= top_k:
                break

        logger.info(f"✅ Encontrados {len(filtered_results)} resultados relevantes")

        return filtered_results

    def demonstrate_search(self):
        """Demostración del sistema de búsqueda con queries típicas"""

        demo_queries = [
            {
                "query": "My Apple Watch battery drains fast",
                "description": "Problema común de batería",
                "brand_filter": None
            },
            {
                "query": "How to charge Samsung Galaxy Watch",
                "description": "Instrucciones de carga",
                "brand_filter": "samsung"
            },
            {
                "query": "Fitbit sleep tracking not working",
                "description": "Problemas con seguimiento de sueño",
                "brand_filter": "fitbit"
            },
            {
                "query": "Garmin GPS accuracy issues",
                "description": "Problemas de precisión GPS",
                "brand_filter": "garmin"
            },
            {
                "query": "smartwatch heart rate monitor",
                "description": "Monitor de frecuencia cardíaca general",
                "brand_filter": None
            }
        ]

        logger.info("🎭 DEMOSTRACIÓN DE BÚSQUEDA SEMÁNTICA")
        logger.info("=" * 60)

        for i, demo in enumerate(demo_queries, 1):
            logger.info(f"\n--- Demo {i}: {demo['description']} ---")
            logger.info(f"Query: '{demo['query']}'")
            if demo['brand_filter']:
                logger.info(f"Filtro de marca: {demo['brand_filter']}")

            results = self.search(
                query=demo['query'],
                top_k=3,
                brand_filter=demo['brand_filter']
            )

            if results:
                logger.info(f"📋 Top {len(results)} resultados:")
                for j, result in enumerate(results, 1):
                    score = result['similarity_score']
                    brand = result.get('brand', 'unknown')
                    text_preview = result['text'][:100] + "..."

                    logger.info(f"  {j}. [{brand.upper()}] Score: {score:.3f}")
                    logger.info(f"     {text_preview}")
                    logger.info(f"     Fuente: {result['source_document']}")
            else:
                logger.warning("   ❌ No se encontraron resultados")

        logger.info(f"\n🎉 Demostración completada exitosamente!")

    def get_search_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas del sistema de búsqueda"""
        if not self.chunks_database:
            return {"error": "No hay datos cargados"}

        brands = {}
        total_words = 0
        total_chars = 0

        for chunk in self.chunks_database:
            brand = chunk.get("brand", "unknown")
            brands[brand] = brands.get(brand, 0) + 1
            total_words += chunk.get("word_count", 0)
            total_chars += chunk.get("char_count", 0)

        return {
            "total_chunks": len(self.chunks_database),
            "total_words": total_words,
            "total_characters": total_chars,
            "brands_covered": list(brands.keys()),
            "chunks_per_brand": brands,
            "embedding_dimension": self.embeddings_matrix.shape[1] if self.embeddings_matrix is not None else 0,
            "model_used": self.model_name
        }


def main():
    """Función principal para probar el sistema de búsqueda"""
    # Simular carga de documentos procesados
    # (En tu caso real, cargarías los resultados de tu pipeline)

    logger.info("🚀 INICIANDO SISTEMA DE BÚSQUEDA SEMÁNTICA")

    # Inicializar sistema de búsqueda
    search_system = SmartWatchSemanticSearch()

    # Aquí cargarías tus documentos procesados reales
    # search_system.load_processed_documents(processed_docs)

    logger.info("💡 Para usar con tus documentos reales:")
    logger.info("   1. Ejecuta tu pipeline principal (main.py)")
    logger.info("   2. Guarda los resultados procesados")
    logger.info("   3. Carga los documentos con load_processed_documents()")
    logger.info("   4. Ejecuta search() con queries de prueba")

    # Ejemplo de uso:
    print("""
    # Ejemplo de integración:

    # 1. Después de ejecutar tu pipeline
    processed_docs = test_pipeline()  # Tu función actual

    # 2. Inicializar búsqueda
    search_system = SmartWatchSemanticSearch()
    search_system.load_processed_documents(processed_docs)

    # 3. Realizar búsquedas
    results = search_system.search("Apple Watch battery drain", top_k=5)

    # 4. Demo automática
    search_system.demonstrate_search()
    """)


if __name__ == "__main__":
    main()