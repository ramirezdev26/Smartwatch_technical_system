"""
ClusterManager - Gestión de clustering de embeddings con K-Means
Fase 3: Aprendizaje No Supervisado para visualización
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from loguru import logger
import joblib
from pathlib import Path
import time


class ClusterManager:
    """
    Gestor de clustering para embeddings de documentos técnicos
    Optimizado para ser usado por la UI de Streamlit
    """

    def __init__(self, model_save_path: Optional[Path] = None):
        """
        Inicializa el gestor de clustering

        Args:
            model_save_path: Ruta para guardar/cargar modelos entrenados
        """
        self.model_save_path = model_save_path or Path("models/cluster_model.joblib")
        self.kmeans_model = None
        self.cluster_stats = None
        self.embeddings = None
        self.metadata = None
        self.optimal_k = None

        # Crear directorio de modelos si no existe
        self.model_save_path.parent.mkdir(parents=True, exist_ok=True)

    def find_optimal_clusters(
        self,
        embeddings: np.ndarray,
        k_range: Tuple[int, int] = (2, 15),
        method: str = "elbow",
    ) -> Dict[str, Any]:
        """
        Encuentra el número óptimo de clusters usando método del codo o silhouette

        Args:
            embeddings: Array de embeddings (n_samples, embedding_dim)
            k_range: Tupla (min_k, max_k) para buscar
            method: 'elbow' o 'silhouette'

        Returns:
            Dict con resultados del análisis y k óptimo sugerido
        """
        logger.info(
            f"🔍 Buscando número óptimo de clusters (método: {method}, rango: {k_range})"
        )
        start_time = time.time()

        min_k, max_k = k_range
        results = {
            "k_values": [],
            "inertias": [],  # Para método del codo
            "silhouette_scores": [],  # Para método silhouette
            "davies_bouldin_scores": [],  # Score adicional
            "optimal_k": None,
            "method_used": method,
            "computation_time": 0,
        }

        for k in range(min_k, max_k + 1):
            logger.info(f"   Evaluando K={k}...")

            # Entrenar modelo temporal
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)

            # Calcular métricas
            results["k_values"].append(k)
            results["inertias"].append(kmeans.inertia_)

            # Silhouette score (solo si k > 1 y k < n_samples)
            if k > 1 and k < len(embeddings):
                silhouette = silhouette_score(embeddings, labels)
                davies_bouldin = davies_bouldin_score(embeddings, labels)
                results["silhouette_scores"].append(silhouette)
                results["davies_bouldin_scores"].append(davies_bouldin)
                logger.info(f"      Silhouette: {silhouette:.3f}")
            else:
                results["silhouette_scores"].append(None)
                results["davies_bouldin_scores"].append(None)

        # Determinar K óptimo según método
        if method == "elbow":
            # Método del codo: buscar "codo" en curva de inercia
            optimal_k = self._find_elbow_point(
                results["k_values"], results["inertias"]
            )
        else:  # silhouette
            # Máximo silhouette score
            valid_scores = [
                (k, score)
                for k, score in zip(results["k_values"], results["silhouette_scores"])
                if score is not None
            ]
            if valid_scores:
                optimal_k = max(valid_scores, key=lambda x: x[1])[0]
            else:
                optimal_k = min_k

        results["optimal_k"] = optimal_k
        results["computation_time"] = time.time() - start_time

        logger.info(f"✅ K óptimo encontrado: {optimal_k} ({method})")
        logger.info(f"⏱️ Tiempo: {results['computation_time']:.2f}s")

        self.optimal_k = optimal_k
        return results

    def _find_elbow_point(self, k_values: List[int], inertias: List[float]) -> int:
        """
        Encuentra el punto de codo en la curva de inercia
        Usa método de máxima curvatura
        """
        if len(k_values) < 3:
            return k_values[0]

        # Normalizar valores
        k_norm = np.array(k_values)
        inertia_norm = np.array(inertias)

        # Calcular diferencias de segunda derivada
        diffs = np.diff(inertia_norm, n=2)

        # El codo es donde hay mayor cambio (máximo de segunda derivada)
        elbow_idx = np.argmax(np.abs(diffs)) + 1  # +1 por np.diff

        return k_values[elbow_idx]

    def fit_clusters(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]],
        n_clusters: Optional[int] = None,
        auto_find_k: bool = True,
    ) -> Dict[str, Any]:
        """
        Ajusta modelo K-Means a los embeddings

        Args:
            embeddings: Array de embeddings (n_samples, embedding_dim)
            metadata: Lista de metadatos correspondientes a cada embedding
            n_clusters: Número de clusters (None para búsqueda automática)
            auto_find_k: Si buscar k óptimo automáticamente

        Returns:
            Dict con resultados del clustering
        """
        logger.info("\n🎯 INICIANDO CLUSTERING CON K-MEANS")
        logger.info("=" * 60)
        start_time = time.time()

        # Validar datos
        if len(embeddings) != len(metadata):
            raise ValueError(
                f"Mismatch: {len(embeddings)} embeddings vs {len(metadata)} metadatos"
            )

        self.embeddings = embeddings
        self.metadata = metadata

        # Determinar número de clusters
        if n_clusters is None and auto_find_k:
            logger.info("🔍 Búsqueda automática de K óptimo...")
            optimal_results = self.find_optimal_clusters(embeddings)
            n_clusters = optimal_results["optimal_k"]
        elif n_clusters is None:
            n_clusters = min(5, len(embeddings) // 10)  # Heurística simple
            logger.info(f"📊 Usando K={n_clusters} (heurística)")

        logger.info(f"🎲 Entrenando K-Means con K={n_clusters} clusters...")

        # Entrenar modelo
        self.kmeans_model = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=20,  # Más iteraciones para mejor resultado
            max_iter=300,
        )

        cluster_labels = self.kmeans_model.fit_predict(embeddings)

        # Calcular métricas de calidad
        metrics = self._compute_cluster_metrics(embeddings, cluster_labels)

        # Analizar composición de clusters
        cluster_composition = self._analyze_cluster_composition(
            cluster_labels, metadata
        )

        # Guardar estadísticas
        self.cluster_stats = {
            "n_clusters": n_clusters,
            "n_samples": len(embeddings),
            "cluster_labels": cluster_labels.tolist(),
            "cluster_centers": self.kmeans_model.cluster_centers_.tolist(),
            "metrics": metrics,
            "composition": cluster_composition,
            "training_time": time.time() - start_time,
        }

        # Log de resultados
        logger.info(f"✅ Clustering completado en {self.cluster_stats['training_time']:.2f}s")
        logger.info(f"📊 Métricas de calidad:")
        logger.info(f"   🎯 Silhouette Score: {metrics['silhouette_score']:.3f}")
        logger.info(f"   📐 Davies-Bouldin Index: {metrics['davies_bouldin_score']:.3f}")
        logger.info(f"   📏 Inercia: {metrics['inertia']:.2f}")

        logger.info(f"\n📈 Distribución de clusters:")
        for cluster_id, info in cluster_composition.items():
            logger.info(
                f"   Cluster {cluster_id}: {info['size']} docs ({info['percentage']:.1f}%)"
            )

        return self.cluster_stats

    def _compute_cluster_metrics(
        self, embeddings: np.ndarray, labels: np.ndarray
    ) -> Dict[str, float]:
        """Calcula métricas de calidad del clustering"""
        metrics = {
            "inertia": self.kmeans_model.inertia_,
            "silhouette_score": 0.0,
            "davies_bouldin_score": 0.0,
        }

        # Solo calcular si hay más de 1 cluster
        if len(np.unique(labels)) > 1:
            metrics["silhouette_score"] = silhouette_score(embeddings, labels)
            metrics["davies_bouldin_score"] = davies_bouldin_score(embeddings, labels)

        return metrics

    def _analyze_cluster_composition(
        self, labels: np.ndarray, metadata: List[Dict[str, Any]]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Analiza la composición de cada cluster (marcas, calidad, etc.)
        """
        composition = {}
        n_samples = len(labels)

        for cluster_id in np.unique(labels):
            cluster_mask = labels == cluster_id
            cluster_metadata = [m for m, mask in zip(metadata, cluster_mask) if mask]

            # Contar por marca
            brands = {}
            for meta in cluster_metadata:
                brand = meta.get("brand", "unknown")
                brands[brand] = brands.get(brand, 0) + 1

            # Contar por calidad
            quality_dist = {}
            for meta in cluster_metadata:
                quality = meta.get("chunk_quality", "unknown")
                quality_dist[quality] = quality_dist.get(quality, 0) + 1

            composition[int(cluster_id)] = {
                "size": int(np.sum(cluster_mask)),
                "percentage": float(np.sum(cluster_mask) / n_samples * 100),
                "brands": brands,
                "quality_distribution": quality_dist,
                "top_brand": max(brands.items(), key=lambda x: x[1])[0]
                if brands
                else "unknown",
            }

        return composition

    def predict_cluster(self, embedding: np.ndarray) -> Dict[str, Any]:
        """
        Predice el cluster para un nuevo embedding

        Args:
            embedding: Embedding a clasificar (debe ser 2D: [1, embedding_dim])

        Returns:
            Dict con cluster_id y distancia al centroide
        """
        if self.kmeans_model is None:
            raise ValueError("❌ Modelo no entrenado. Ejecuta fit_clusters() primero")

        # Asegurar formato correcto
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        cluster_id = int(self.kmeans_model.predict(embedding)[0])
        distance = float(
            self.kmeans_model.transform(embedding)[0, cluster_id]
        )  # Distancia al centroide

        return {
            "cluster_id": cluster_id,
            "distance_to_center": distance,
            "cluster_info": self.cluster_stats["composition"][cluster_id]
            if self.cluster_stats
            else None,
        }

    def get_cluster_documents(
        self, cluster_id: int, max_docs: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Obtiene todos los documentos de un cluster específico

        Args:
            cluster_id: ID del cluster
            max_docs: Máximo número de documentos a retornar

        Returns:
            Lista de metadatos de documentos en el cluster
        """
        if self.cluster_stats is None:
            raise ValueError("❌ No hay clustering disponible")

        labels = np.array(self.cluster_stats["cluster_labels"])
        cluster_mask = labels == cluster_id

        cluster_docs = [m for m, mask in zip(self.metadata, cluster_mask) if mask]

        if max_docs:
            cluster_docs = cluster_docs[:max_docs]

        return cluster_docs

    def save_model(self, filepath: Optional[Path] = None):
        """Guarda el modelo entrenado"""
        filepath = filepath or self.model_save_path

        if self.kmeans_model is None:
            raise ValueError("❌ No hay modelo para guardar")

        model_data = {
            "kmeans_model": self.kmeans_model,
            "cluster_stats": self.cluster_stats,
            "optimal_k": self.optimal_k,
        }

        joblib.dump(model_data, filepath)
        logger.info(f"💾 Modelo de clustering guardado en: {filepath}")

    def load_model(self, filepath: Optional[Path] = None):
        """Carga un modelo previamente entrenado"""
        filepath = filepath or self.model_save_path

        if not filepath.exists():
            raise FileNotFoundError(f"❌ Modelo no encontrado: {filepath}")

        model_data = joblib.load(filepath)
        self.kmeans_model = model_data["kmeans_model"]
        self.cluster_stats = model_data["cluster_stats"]
        self.optimal_k = model_data.get("optimal_k")

        logger.info(f"📂 Modelo de clustering cargado desde: {filepath}")
        logger.info(
            f"   📊 K={self.cluster_stats['n_clusters']}, N={self.cluster_stats['n_samples']}"
        )

    def get_clustering_summary(self) -> Dict[str, Any]:
        """
        Retorna un resumen del clustering para la UI

        Returns:
            Dict con información resumida para dashboards
        """
        if self.cluster_stats is None:
            return {"status": "not_trained", "message": "No hay modelo entrenado"}

        summary = {
            "status": "trained",
            "n_clusters": self.cluster_stats["n_clusters"],
            "n_documents": self.cluster_stats["n_samples"],
            "silhouette_score": self.cluster_stats["metrics"]["silhouette_score"],
            "davies_bouldin_score": self.cluster_stats["metrics"][
                "davies_bouldin_score"
            ],
            "clusters": [],
        }

        # Info por cluster
        for cluster_id, info in self.cluster_stats["composition"].items():
            summary["clusters"].append(
                {
                    "id": cluster_id,
                    "size": info["size"],
                    "percentage": info["percentage"],
                    "dominant_brand": info["top_brand"],
                    "quality_relevant": info["quality_distribution"].get(
                        "relevante", 0
                    ),
                }
            )

        # Ordenar por tamaño
        summary["clusters"] = sorted(
            summary["clusters"], key=lambda x: x["size"], reverse=True
        )

        return summary