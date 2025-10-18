"""
Script de prueba para el ClusterManager
"""

import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent))

from src.clustering.cluster_manager import ClusterManager
from src.storage.chroma_manager import ChromaManager
from config import *
from loguru import logger


def test_clustering_pipeline():
    """Prueba el clustering con datos reales de ChromaDB"""
    logger.info("🧪 INICIANDO PRUEBA DE CLUSTERING")
    logger.info("=" * 60)

    # 1. Conectar a ChromaDB y obtener embeddings
    logger.info("\n📡 Paso 1: Obtener embeddings de ChromaDB")
    chroma = ChromaManager()

    # Obtener todos los documentos
    all_data = chroma.collection.get(include=["embeddings", "metadatas", "documents"])

    embeddings = np.array(all_data["embeddings"])
    metadata = all_data["metadatas"]
    n_docs = len(embeddings)

    logger.info(f"✅ Obtenidos {n_docs} embeddings ({embeddings.shape})")

    # 2. Inicializar ClusterManager
    logger.info("\n🎯 Paso 2: Inicializar ClusterManager")
    cluster_manager = ClusterManager()

    # 3. Buscar K óptimo
    logger.info("\n🔍 Paso 3: Búsqueda de K óptimo")
    optimal_results = cluster_manager.find_optimal_clusters(
        embeddings, k_range=(2, 10), method="silhouette"
    )

    logger.info(f"✅ K óptimo sugerido: {optimal_results['optimal_k']}")

    # 4. Entrenar clustering
    logger.info("\n🎲 Paso 4: Entrenar modelo K-Means")
    cluster_stats = cluster_manager.fit_clusters(
        embeddings, metadata, n_clusters=optimal_results["optimal_k"]
    )

    # 5. Guardar modelo
    logger.info("\n💾 Paso 5: Guardar modelo")
    cluster_manager.save_model()

    # 6. Obtener resumen para UI
    logger.info("\n📊 Paso 6: Generar resumen para UI")
    summary = cluster_manager.get_clustering_summary()

    logger.info("\n" + "=" * 60)
    logger.info("📈 RESUMEN DE CLUSTERING:")
    logger.info(f"   Clusters: {summary['n_clusters']}")
    logger.info(f"   Documentos: {summary['n_documents']}")
    logger.info(f"   Silhouette: {summary['silhouette_score']:.3f}")

    for cluster in summary["clusters"]:
        logger.info(
            f"\n   📂 Cluster {cluster['id']}: {cluster['size']} docs ({cluster['percentage']:.1f}%)"
        )
        logger.info(f"      Marca dominante: {cluster['dominant_brand']}")
        logger.info(f"      Relevantes: {cluster['quality_relevant']}")

    # 7. Probar predicción
    logger.info("\n🔮 Paso 7: Probar predicción de cluster")
    test_embedding = embeddings[0].reshape(1, -1)
    prediction = cluster_manager.predict_cluster(test_embedding)
    logger.info(f"   Predicción: Cluster {prediction['cluster_id']}")
    logger.info(f"   Distancia: {prediction['distance_to_center']:.3f}")

    logger.info("\n✅ PRUEBA COMPLETADA EXITOSAMENTE")


if __name__ == "__main__":
    test_clustering_pipeline()