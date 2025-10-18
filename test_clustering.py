"""
Script de prueba simplificado para ClusterManager
"""

import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent))

from src.clustering.cluster_manager import ClusterManager
from src.storage.chroma_manager import ChromaManager
from loguru import logger


def test_clustering():
    """Prueba básica del clustering"""
    logger.info("🧪 PRUEBA DE CLUSTERING")
    logger.info("=" * 60)

    # 1. Obtener datos de ChromaDB
    logger.info("\n📡 Obteniendo embeddings de ChromaDB...")
    chroma = ChromaManager()
    all_data = chroma.collection.get(include=["embeddings", "metadatas"])

    embeddings = np.array(all_data["embeddings"])
    metadata = all_data["metadatas"]

    logger.info(f"✅ {len(embeddings)} embeddings obtenidos")

    # 2. Crear y entrenar cluster manager
    logger.info("\n🎯 Entrenando clustering...")
    cluster_manager = ClusterManager()

    # Entrenar (busca K óptimo automáticamente)
    results = cluster_manager.fit(embeddings, metadata)

    # 3. Guardar modelo
    cluster_manager.save()

    # 4. Obtener resumen
    logger.info("\n📊 RESUMEN:")
    summary = cluster_manager.get_summary()

    logger.info(f"   Clusters: {summary['n_clusters']}")
    logger.info(f"   Silhouette: {summary['silhouette_score']:.3f}")

    for cluster in summary["clusters"]:
        logger.info(
            f"\n   📂 Cluster {cluster['cluster_id']}: {cluster['size']} docs ({cluster['percentage']:.1f}%)"
        )
        logger.info(f"      Marca: {cluster['top_brand']}")
        logger.info(
            f"      Relevantes: {cluster['quality_distribution']['relevante']}"
        )

    # 5. Probar predicción
    logger.info("\n🔮 Probando predicción...")
    test_embedding = embeddings[0]
    predicted_cluster = cluster_manager.predict(test_embedding)
    logger.info(f"   Cluster predicho: {predicted_cluster}")

    logger.info("\n✅ PRUEBA COMPLETADA")


if __name__ == "__main__":
    test_clustering()