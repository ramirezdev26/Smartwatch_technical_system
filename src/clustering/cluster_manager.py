"""
ClusterManager - Gestión básica de clustering con K-Means
Versión simplificada enfocada en requerimientos del capstone
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from loguru import logger
import joblib
from pathlib import Path


class ClusterManager:
    """
    Gestor simplificado de clustering para embeddings
    Enfocado en funcionalidad esencial para la UI
    """

    def __init__(self, model_save_path: Optional[Path] = None):
        """
        Args:
            model_save_path: Ruta para guardar/cargar modelos
        """
        self.model_save_path = model_save_path or Path("models/cluster_model.joblib")
        self.kmeans_model = None
        self.cluster_labels = None
        self.embeddings = None
        self.metadata = None
        self.n_clusters = None

        # Crear directorio si no existe
        self.model_save_path.parent.mkdir(parents=True, exist_ok=True)

    def find_optimal_k(
        self,
        embeddings: np.ndarray,
        k_range: Tuple[int, int] = (2, 10),
    ) -> Dict[str, Any]:
        """
        Encuentra el número óptimo de clusters usando silhouette score

        Args:
            embeddings: Array de embeddings
            k_range: Rango (min_k, max_k) para evaluar

        Returns:
            Dict con k_values, silhouette_scores y optimal_k
        """
        logger.info(f"🔍 Buscando K óptimo en rango {k_range}")

        min_k, max_k = k_range
        k_values = []
        silhouette_scores = []

        for k in range(min_k, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            score = silhouette_score(embeddings, labels)

            k_values.append(k)
            silhouette_scores.append(score)

            logger.info(f"   K={k}: Silhouette={score:.3f}")

        # Seleccionar K con mejor silhouette score
        optimal_k = k_values[np.argmax(silhouette_scores)]

        logger.info(f"✅ K óptimo: {optimal_k}")

        return {
            "k_values": k_values,
            "silhouette_scores": silhouette_scores,
            "optimal_k": optimal_k,
        }

    def fit(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]],
        n_clusters: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Entrena K-Means sobre los embeddings

        Args:
            embeddings: Array de embeddings (n_samples, embedding_dim)
            metadata: Lista de metadatos por cada embedding
            n_clusters: Número de clusters (None para búsqueda automática)

        Returns:
            Dict con resultados básicos del clustering
        """
        logger.info("🎯 Entrenando K-Means clustering")

        # Búsqueda automática de K si no se especifica
        if n_clusters is None:
            optimal_results = self.find_optimal_k(embeddings)
            n_clusters = optimal_results["optimal_k"]

        # Entrenar modelo
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        self.cluster_labels = self.kmeans_model.fit_predict(embeddings)
        self.embeddings = embeddings
        self.metadata = metadata
        self.n_clusters = n_clusters

        # Calcular métricas básicas
        silhouette = silhouette_score(embeddings, self.cluster_labels)

        # Analizar distribución de clusters
        cluster_sizes = self._get_cluster_sizes()

        logger.info(f"✅ Clustering completado:")
        logger.info(f"   K={n_clusters}, Silhouette={silhouette:.3f}")
        logger.info(f"   Distribución: {cluster_sizes}")

        return {
            "n_clusters": n_clusters,
            "n_samples": len(embeddings),
            "silhouette_score": silhouette,
            "cluster_sizes": cluster_sizes,
        }

    def _get_cluster_sizes(self) -> Dict[int, int]:
        """Retorna el tamaño de cada cluster"""
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        return {int(k): int(v) for k, v in zip(unique, counts)}

    def get_cluster_info(self, cluster_id: int) -> Dict[str, Any]:
        """
        Obtiene información detallada de un cluster específico

        Args:
            cluster_id: ID del cluster

        Returns:
            Dict con información del cluster
        """
        if self.cluster_labels is None:
            raise ValueError("Modelo no entrenado")

        # Filtrar documentos del cluster
        mask = self.cluster_labels == cluster_id
        cluster_metadata = [m for m, is_in in zip(self.metadata, mask) if is_in]

        # Contar por marca
        brands = {}
        for meta in cluster_metadata:
            brand = meta.get("brand", "unknown")
            brands[brand] = brands.get(brand, 0) + 1

        # Contar por calidad
        quality_counts = {"relevante": 0, "ambiguo": 0, "irrelevante": 0}
        for meta in cluster_metadata:
            quality = meta.get("chunk_quality", "unknown")
            if quality in quality_counts:
                quality_counts[quality] += 1

        return {
            "cluster_id": cluster_id,
            "size": int(mask.sum()),
            "percentage": float(mask.sum() / len(self.cluster_labels) * 100),
            "brands": brands,
            "quality_distribution": quality_counts,
            "top_brand": max(brands.items(), key=lambda x: x[1])[0] if brands else None,
        }

    def get_all_clusters_info(self) -> List[Dict[str, Any]]:
        """
        Retorna información de todos los clusters

        Returns:
            Lista con info de cada cluster
        """
        if self.cluster_labels is None:
            raise ValueError("Modelo no entrenado")

        clusters_info = []
        for cluster_id in range(self.n_clusters):
            info = self.get_cluster_info(cluster_id)
            clusters_info.append(info)

        # Ordenar por tamaño (mayor a menor)
        clusters_info.sort(key=lambda x: x["size"], reverse=True)

        return clusters_info

    def predict(self, embedding: np.ndarray) -> int:
        """
        Predice el cluster para un nuevo embedding

        Args:
            embedding: Embedding a clasificar

        Returns:
            Cluster ID
        """
        if self.kmeans_model is None:
            raise ValueError("Modelo no entrenado")

        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        return int(self.kmeans_model.predict(embedding)[0])

    def get_cluster_documents(
        self, cluster_id: int, max_docs: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Obtiene metadatos de documentos en un cluster

        Args:
            cluster_id: ID del cluster
            max_docs: Límite de documentos a retornar

        Returns:
            Lista de metadatos
        """
        if self.cluster_labels is None:
            raise ValueError("Modelo no entrenado")

        mask = self.cluster_labels == cluster_id
        docs = [m for m, is_in in zip(self.metadata, mask) if is_in]

        if max_docs:
            docs = docs[:max_docs]

        return docs

    def save(self, filepath: Optional[Path] = None):
        """Guarda el modelo"""
        filepath = filepath or self.model_save_path

        if self.kmeans_model is None:
            raise ValueError("No hay modelo para guardar")

        model_data = {
            "kmeans_model": self.kmeans_model,
            "cluster_labels": self.cluster_labels,
            "n_clusters": self.n_clusters,
        }

        joblib.dump(model_data, filepath)
        logger.info(f"💾 Modelo guardado: {filepath}")

    def load(self, filepath: Optional[Path] = None):
        """Carga un modelo guardado"""
        filepath = filepath or self.model_save_path

        if not filepath.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {filepath}")

        model_data = joblib.load(filepath)
        self.kmeans_model = model_data["kmeans_model"]
        self.cluster_labels = model_data["cluster_labels"]
        self.n_clusters = model_data["n_clusters"]

        logger.info(f"📂 Modelo cargado: {filepath}")
        logger.info(f"   K={self.n_clusters}")

    def get_summary(self) -> Dict[str, Any]:
        """
        Resumen para la UI de Streamlit

        Returns:
            Dict con información resumida
        """
        if self.kmeans_model is None:
            return {"status": "not_trained"}

        silhouette = silhouette_score(self.embeddings, self.cluster_labels)

        return {
            "status": "trained",
            "n_clusters": self.n_clusters,
            "n_documents": len(self.cluster_labels),
            "silhouette_score": float(silhouette),
            "clusters": self.get_all_clusters_info(),
        }