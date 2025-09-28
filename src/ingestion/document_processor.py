"""
Procesador de documentos para el pipeline de ingesta
"""

import PyPDF2
import re
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from loguru import logger
import hashlib


class DocumentProcessor:
    """Procesador de documentos PDF y texto"""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def process_pdf(self, file_path: Path) -> Dict[str, Any]:
        """
        Procesa un archivo PDF y extrae el texto

        Args:
            file_path: Ruta al archivo PDF

        Returns:
            Dict con metadatos y contenido procesado
        """
        try:
            with open(file_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)

                # Extraer texto de todas las páginas
                text_content = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    text_content += page.extract_text() + "\\n"

                # Limpiar texto
                # cleaned_text = self._clean_text(text_content)

                # Crear chunks
                chunks = self._create_chunks(text_content)

                # Generar metadatos
                metadata = self._generate_metadata(
                    file_path, len(pdf_reader.pages), len(chunks)
                )

                return {
                    "metadata": metadata,
                    "chunks": chunks,
                    # "full_text": cleaned_text
                }

        except Exception as e:
            logger.error(f"Error procesando PDF {file_path}: {e}")
            return None

    def process_text_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Procesa un archivo de texto

        Args:
            file_path: Ruta al archivo de texto

        Returns:
            Dict con metadatos y contenido procesado
        """
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                text_content = file.read()

            # Limpiar texto
            # cleaned_text = self._clean_text(text_content)

            # Crear chunks
            chunks = self._create_chunks(text_content)

            # Generar metadatos
            metadata = self._generate_metadata(file_path, None, len(chunks))

            return {
                "metadata": metadata,
                "chunks": chunks,
                # "full_text": cleaned_text
            }

        except Exception as e:
            logger.error(f"Error procesando archivo de texto {file_path}: {e}")
            return None

    def _clean_text(self, text: str) -> str:
        """Limpia y normaliza el texto"""
        # Remover caracteres especiales excesivos
        # text = re.sub(r'\\s+', ' ', text)  # Múltiples espacios en blanco
        # text = re.sub(r'\\n+', '\\n', text)  # Múltiples saltos de línea

        # Remover caracteres no imprimibles
        # text = re.sub(r'[^\\x20-\\x7E\\n]', '', text)

        return text.strip()

    def _create_chunks(self, text: str) -> List[Dict[str, Any]]:
        """
        Divide el texto en chunks manejables

        Args:
            text: Texto a dividir

        Returns:
            Lista de chunks con metadatos
        """
        chunks = []
        words = text.split()

        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i : i + self.chunk_size]
            chunk_text = " ".join(chunk_words)

            # Generar ID único para el chunk
            chunk_id = hashlib.md5(chunk_text.encode()).hexdigest()[:16]

            chunk_data = {
                "id": chunk_id,
                "text": chunk_text,
                "chunk_index": len(chunks),
                "word_count": len(chunk_words),
                "start_position": i,
                "end_position": min(i + self.chunk_size, len(words)),
            }

            chunks.append(chunk_data)

        return chunks

    def _generate_metadata(
        self, file_path: Path, num_pages: int = None, num_chunks: int = 0
    ) -> Dict[str, Any]:
        """
        Genera metadatos para el documento

        Args:
            file_path: Ruta del archivo
            num_pages: Número de páginas (para PDFs)
            num_chunks: Número de chunks generados

        Returns:
            Dict con metadatos
        """
        # Detectar marca basada en la ruta del archivo
        brand = self._detect_brand(file_path)

        # Generar hash del archivo para detectar duplicados
        file_hash = self._generate_file_hash(file_path)

        metadata = {
            "file_name": file_path.name,
            "file_path": str(file_path),
            "file_extension": file_path.suffix,
            "file_size_bytes": file_path.stat().st_size,
            "file_hash": file_hash,
            "brand": brand,
            "document_type": self._classify_document_type(file_path.name),
            "num_pages": num_pages,
            "num_chunks": num_chunks,
            "processed_timestamp": pd.Timestamp.now().isoformat(),
        }

        return metadata

    def _detect_brand(self, file_path: Path) -> str:
        """Detecta la marca del smartwatch basada en la ruta"""
        path_str = str(file_path).lower()

        if "apple" in path_str or "apple_watch" in path_str:
            return "apple_watch"
        elif "garmin" in path_str:
            return "garmin"
        elif "fitbit" in path_str:
            return "fitbit"
        elif "samsung" in path_str:
            return "samsung"
        else:
            return "unknown"

    def _classify_document_type(self, filename: str) -> str:
        """Clasifica el tipo de documento basado en el nombre"""
        filename_lower = filename.lower()

        if any(word in filename_lower for word in ["manual", "guide", "user"]):
            return "user_manual"
        elif any(word in filename_lower for word in ["faq", "troubleshoot", "problem"]):
            return "troubleshooting"
        elif any(word in filename_lower for word in ["spec", "technical", "api"]):
            return "technical_specs"
        elif any(word in filename_lower for word in ["setup", "install", "config"]):
            return "setup_guide"
        else:
            return "general"

    def _generate_file_hash(self, file_path: Path) -> str:
        """Genera hash MD5 del archivo para detectar duplicados"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
