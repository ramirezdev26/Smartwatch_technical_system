"""
Visualizer - Módulo de visualización de embeddings con PCA y t-SNE
Versión simplificada para requerimientos del capstone
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from loguru import logger
import joblib
from pathlib import Path


class EmbeddingVisualizer:
    """
    Gestor de visualización de embeddings usando PCA y t-SNE
    Reduce embeddings de alta dimensión a 2D/3D para visualización
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Args:
            cache_dir: Directorio para cachear transformaciones (opcional)
        """
        self.cache_dir = cache_dir or Path("models/visualization_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.pca_model = None
        self.tsne_embeddings = None
        self.pca_embeddings = None
        self.original_embeddings = None
        self.metadata = None

    def fit_pca(
        self, embeddings: np.ndarray, n_components: int = 2
    ) -> np.ndarray:
        """
        Reduce dimensionalidad con PCA

        Args:
            embeddings: Array de embeddings (n_samples, embedding_dim)
            n_components: Número de componentes (2 o 3)

        Returns:
            Embeddings reducidos (n_samples, n_components)
        """
        logger.info(f"📐 Aplicando PCA para reducir a {n_components}D")

        self.pca_model = PCA(n_components=n_components, random_state=42)
        self.pca_embeddings = self.pca_model.fit_transform(embeddings)
        self.original_embeddings = embeddings

        # Log de varianza explicada
        explained_var = self.pca_model.explained_variance_ratio_
        total_var = explained_var.sum()

        logger.info(f"✅ PCA completado:")
        logger.info(f"   Varianza explicada: {total_var:.1%}")
        for i, var in enumerate(explained_var):
            logger.info(f"   PC{i+1}: {var:.1%}")

        return self.pca_embeddings

    def fit_tsne(
        self,
        embeddings: np.ndarray,
        n_components: int = 2,
        perplexity: int = 30,
        max_iter: int = 1000,
    ) -> np.ndarray:
        """
        Reduce dimensionalidad con t-SNE

        Args:
            embeddings: Array de embeddings
            n_components: Dimensiones de salida (2 o 3)
            perplexity: Parámetro de t-SNE (5-50)
            max_iter: Iteraciones máximas

        Returns:
            Embeddings reducidos con t-SNE
        """
        logger.info(
            f"🎨 Aplicando t-SNE para reducir a {n_components}D (perplexity={perplexity})"
        )

        # t-SNE puede ser lento, avisar al usuario
        if len(embeddings) > 1000:
            logger.warning(
                f"⚠️  t-SNE con {len(embeddings)} puntos puede tardar varios minutos"
            )

        tsne_model = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            max_iter=max_iter,  # ✅ CORREGIDO: era n_iter
            random_state=42,
            verbose=0,
        )

        self.tsne_embeddings = tsne_model.fit_transform(embeddings)
        self.original_embeddings = embeddings

        logger.info(f"✅ t-SNE completado")

        return self.tsne_embeddings

    def fit_both(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]],
        n_components: int = 2,
        tsne_perplexity: int = 30,
    ) -> Dict[str, np.ndarray]:
        """
        Aplica PCA y t-SNE en secuencia

        Args:
            embeddings: Array de embeddings
            metadata: Lista de metadatos
            n_components: Dimensiones finales (2 o 3)
            tsne_perplexity: Parámetro de t-SNE

        Returns:
            Dict con 'pca' y 'tsne' embeddings
        """
        logger.info(f"🎯 Aplicando PCA y t-SNE")
        logger.info("=" * 60)

        self.metadata = metadata

        # PCA primero (rápido)
        pca_result = self.fit_pca(embeddings, n_components)

        # t-SNE después (lento)
        tsne_result = self.fit_tsne(embeddings, n_components, tsne_perplexity)

        return {"pca": pca_result, "tsne": tsne_result}

    def get_visualization_data(
        self, cluster_labels: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Prepara datos para visualización en Streamlit

        Args:
            cluster_labels: Etiquetas de cluster (opcional)

        Returns:
            Dict con datos listos para graficar
        """
        if self.pca_embeddings is None or self.tsne_embeddings is None:
            raise ValueError("Ejecuta fit_both() primero")

        # Preparar datos base
        data = {
            "pca": self.pca_embeddings.tolist(),
            "tsne": self.tsne_embeddings.tolist(),
            "metadata": self.metadata,
        }

        # Añadir clusters si están disponibles
        if cluster_labels is not None:
            data["clusters"] = cluster_labels.tolist()

        # Extraer información útil de metadata
        data["brands"] = [m.get("brand", "unknown") for m in self.metadata]
        data["quality"] = [
            m.get("chunk_quality", "unknown") for m in self.metadata
        ]
        data["doc_names"] = [
            m.get("document_name", "unknown") for m in self.metadata
        ]

        return data

    def save_cache(self, name: str = "visualization"):
        """
        Guarda las transformaciones calculadas

        Args:
            name: Nombre del archivo de cache
        """
        cache_file = self.cache_dir / f"{name}.joblib"

        cache_data = {
            "pca_model": self.pca_model,
            "pca_embeddings": self.pca_embeddings,
            "tsne_embeddings": self.tsne_embeddings,
            "metadata": self.metadata,
        }

        joblib.dump(cache_data, cache_file)
        logger.info(f"💾 Visualización cacheada: {cache_file}")

    def load_cache(self, name: str = "visualization"):
        """
        Carga transformaciones previamente calculadas

        Args:
            name: Nombre del archivo de cache
        """
        cache_file = self.cache_dir / f"{name}.joblib"

        if not cache_file.exists():
            raise FileNotFoundError(f"Cache no encontrado: {cache_file}")

        cache_data = joblib.load(cache_file)

        self.pca_model = cache_data["pca_model"]
        self.pca_embeddings = cache_data["pca_embeddings"]
        self.tsne_embeddings = cache_data["tsne_embeddings"]
        self.metadata = cache_data["metadata"]

        logger.info(f"📂 Visualización cargada desde cache")

    def get_pca_info(self) -> Dict[str, Any]:
        """
        Información sobre la transformación PCA

        Returns:
            Dict con estadísticas de PCA
        """
        if self.pca_model is None:
            return {"status": "not_fitted"}

        explained_var = self.pca_model.explained_variance_ratio_

        return {
            "status": "fitted",
            "n_components": len(explained_var),
            "explained_variance_ratio": explained_var.tolist(),
            "total_variance_explained": float(explained_var.sum()),
            "components_shape": self.pca_embeddings.shape,
        }