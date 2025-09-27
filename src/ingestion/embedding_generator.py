"""
Generador de embeddings usando all-MiniLM-L6-v1
"""
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any
from loguru import logger
import time


class EmbeddingGenerator:
    """Generador de embeddings usando all-MiniLM-L6-v1"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v1"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Carga el modelo de embeddings"""
        try:
            logger.info(f"Cargando modelo {self.model_name}...")
            start_time = time.time()

            self.model = SentenceTransformer(self.model_name)

            load_time = time.time() - start_time
            logger.info(f"Modelo cargado en {load_time:.2f} segundos")

        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            raise

    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Genera embeddings para una lista de chunks

        Args:
            chunks: Lista de chunks con texto

        Returns:
            Lista de chunks con embeddings añadidos
        """
        if not self.model:
            raise ValueError("Modelo no cargado")

        logger.info(f"Generando embeddings para {len(chunks)} chunks...")
        start_time = time.time()

        # Extraer textos
        texts = [chunk["text"] for chunk in chunks]

        # Generar embeddings en lote (más eficiente)
        embeddings = self.model.encode(
            texts,
            batch_size=32,  # Procesar en lotes de 32
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Añadir embeddings a los chunks
        enhanced_chunks = []
        for i, chunk in enumerate(chunks):
            enhanced_chunk = chunk.copy()
            enhanced_chunk["embedding"] = embeddings[i].tolist()  # Convertir a lista para JSON
            enhanced_chunk["embedding_model"] = self.model_name
            enhanced_chunk["embedding_dimension"] = len(embeddings[i])
            enhanced_chunks.append(enhanced_chunk)

        generation_time = time.time() - start_time
        logger.info(f"Embeddings generados en {generation_time:.2f} segundos")
        logger.info(f"Velocidad: {len(chunks) / generation_time:.1f} chunks/segundo")

        return enhanced_chunks

    def generate_single_embedding(self, text: str) -> np.ndarray:
        """
        Genera embedding para un texto individual

        Args:
            text: Texto de entrada

        Returns:
            Array NumPy con el embedding
        """
        if not self.model:
            raise ValueError("Modelo no cargado")

        return self.model.encode([text])[0]

    def get_model_info(self) -> Dict[str, Any]:
        """Retorna información del modelo cargado"""
        if not self.model:
            return {"status": "not_loaded"}

        return {
            "model_name": self.model_name,
            "max_seq_length": self.model.max_seq_length,
            "embedding_dimension": self.model.get_sentence_embedding_dimension(),
            "status": "loaded"
        }