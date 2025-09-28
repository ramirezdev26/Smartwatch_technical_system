"""
Feature Extractor SIMPLIFICADO para aprender ML básico
Solo características básicas y fáciles de entender
"""
import numpy as np
from typing import Dict, List, Any
from loguru import logger


class SimpleFeatureExtractor:
    """Extractor de características MUY SIMPLE para aprendizaje de ML básico"""

    def __init__(self):
        # Solo keywords básicos y comprensibles
        self.technical_words = [
            'battery', 'charge', 'charging', 'power',
            'settings', 'problem', 'fix', 'press', 'tap',
            'watch', 'device', 'screen', 'button'
        ]

        self.irrelevant_words = [
            'page', 'copyright', 'chapter', 'section', 'contents'
        ]

    def extract_features(self, chunks: List[Dict[str, Any]]) -> np.ndarray:
        """
        Extrae SOLO 10 características básicas por chunk
        """
        logger.info(f"🔧 Extrayendo características SIMPLES de {len(chunks)} chunks...")

        features_list = []

        for i, chunk in enumerate(chunks):
            chunk_features = self._extract_simple_features(chunk)
            features_list.append(chunk_features)

            if (i + 1) % 200 == 0:
                logger.info(f"   Procesados {i + 1}/{len(chunks)} chunks")

        features_array = np.array(features_list)
        logger.info(f"✅ Características extraídas: {features_array.shape}")
        logger.info(f"📊 Solo {features_array.shape[1]} características SIMPLES por chunk")

        return features_array

    def _extract_simple_features(self, chunk: Dict[str, Any]) -> np.ndarray:
        """Extrae SOLO 10 características básicas y comprensibles"""
        text = chunk.get("text", "")
        metadata = chunk.get("metadata", {})

        if not text:
            return np.zeros(10)  # SOLO 10 características

        text_lower = text.lower()
        words = text.split()

        # === CARACTERÍSTICA 1: Número de palabras ===
        word_count = len(words)

        # === CARACTERÍSTICA 2: Número de caracteres ===
        char_count = len(text)

        # === CARACTERÍSTICA 3: ¿Es muy corto? (posible header) ===
        is_very_short = 1 if word_count < 10 else 0

        # === CARACTERÍSTICA 4: ¿Tiene números? (posible página/spec) ===
        has_numbers = 1 if any(char.isdigit() for char in text) else 0

        # === CARACTERÍSTICA 5: Cantidad de palabras técnicas ===
        technical_count = sum(1 for word in self.technical_words if word in text_lower)

        # === CARACTERÍSTICA 6: ¿Tiene palabras irrelevantes? ===
        irrelevant_count = sum(1 for word in self.irrelevant_words if word in text_lower)

        # === CARACTERÍSTICA 7: ¿Es de Apple? ===
        brand = metadata.get("brand", "").lower()
        is_apple = 1 if "apple" in brand else 0

        # === CARACTERÍSTICA 8: ¿Es de Samsung? ===
        is_samsung = 1 if "samsung" in brand else 0

        # === CARACTERÍSTICA 9: ¿Está al principio del documento? ===
        chunk_position = metadata.get("chunk_index", 0)
        is_early_chunk = 1 if chunk_position < 5 else 0

        # === CARACTERÍSTICA 10: ¿Tiene mayúsculas excesivas? ===
        uppercase_ratio = sum(1 for char in text if char.isupper()) / len(text) if text else 0
        has_many_caps = 1 if uppercase_ratio > 0.3 else 0

        # Vector final de SOLO 10 características
        features = np.array([
            word_count,           # 1. Longitud en palabras
            char_count,           # 2. Longitud en caracteres
            is_very_short,        # 3. ¿Es muy corto?
            has_numbers,          # 4. ¿Tiene números?
            technical_count,      # 5. Cantidad de palabras técnicas
            irrelevant_count,     # 6. Cantidad de palabras irrelevantes
            is_apple,             # 7. ¿Es de Apple?
            is_samsung,           # 8. ¿Es de Samsung?
            is_early_chunk,       # 9. ¿Está al principio?
            has_many_caps         # 10. ¿Tiene muchas mayúsculas?
        ])

        return features

    def get_feature_names(self) -> List[str]:
        """Nombres SIMPLES y comprensibles de las 10 características"""
        return [
            "word_count",           # Número de palabras
            "char_count",           # Número de caracteres
            "is_very_short",        # ¿Es muy corto? (1=sí, 0=no)
            "has_numbers",          # ¿Tiene números? (1=sí, 0=no)
            "technical_count",      # Cantidad de palabras técnicas
            "irrelevant_count",     # Cantidad de palabras irrelevantes
            "is_apple",             # ¿Es de Apple? (1=sí, 0=no)
            "is_samsung",           # ¿Es de Samsung? (1=sí, 0=no)
            "is_early_chunk",       # ¿Está al principio del documento? (1=sí, 0=no)
            "has_many_caps"         # ¿Tiene muchas mayúsculas? (1=sí, 0=no)
        ]

    def explain_features(self) -> Dict[str, str]:
        """Explicación en español de cada característica"""
        return {
            "word_count": "Número de palabras en el texto",
            "char_count": "Número de caracteres en el texto",
            "is_very_short": "1 si tiene menos de 10 palabras (posible header), 0 si no",
            "has_numbers": "1 si contiene números (posible página/especificación), 0 si no",
            "technical_count": "Cantidad de palabras técnicas (battery, charge, settings, etc.)",
            "irrelevant_count": "Cantidad de palabras irrelevantes (page, copyright, etc.)",
            "is_apple": "1 si es de Apple, 0 si es de otra marca",
            "is_samsung": "1 si es de Samsung, 0 si es de otra marca",
            "is_early_chunk": "1 si está en los primeros 5 chunks del documento, 0 si no",
            "has_many_caps": "1 si más del 30% son mayúsculas, 0 si no"
        }

    def analyze_features_simple(self, features: np.ndarray, labels: List[str] = None) -> None:
        """Análisis SIMPLE y comprensible de las características"""
        feature_names = self.get_feature_names()
        explanations = self.explain_features()

        logger.info("📊 ANÁLISIS SIMPLE DE CARACTERÍSTICAS:")
        logger.info("=" * 50)

        for i, name in enumerate(feature_names):
            values = features[:, i]

            logger.info(f"\n{i+1}. {name.upper()}")
            logger.info(f"   📝 {explanations[name]}")
            logger.info(f"   📊 Promedio: {np.mean(values):.2f}")
            logger.info(f"   📈 Rango: {np.min(values):.0f} - {np.max(values):.0f}")

            # Si es binario (0 o 1), mostrar porcentaje
            if set(values).issubset({0, 1}):
                percentage = np.mean(values) * 100
                logger.info(f"   ✅ {percentage:.1f}% de chunks tienen esta característica")

        # Si tenemos labels, mostrar diferencias por clase
        if labels:
            logger.info(f"\n🔍 DIFERENCIAS POR CLASE:")
            logger.info("-" * 30)

            for label in ["relevante", "irrelevante", "ambiguo"]:
                if label in labels:
                    mask = np.array(labels) == label
                    label_features = features[mask]

                    if len(label_features) > 0:
                        logger.info(f"\n📋 CHUNKS '{label.upper()}':")

                        # Solo mostrar las 3 características más diferentes
                        avg_relevant = np.mean(features[np.array(labels) == "relevante"], axis=0) if "relevante" in labels else features.mean(axis=0)
                        avg_this_label = np.mean(label_features, axis=0)
                        differences = np.abs(avg_this_label - avg_relevant)
                        top_different = np.argsort(differences)[-3:][::-1]

                        for idx in top_different:
                            diff_value = avg_this_label[idx]
                            logger.info(f"   {feature_names[idx]}: {diff_value:.2f}")


def test_simple_extractor():
    """Test del extractor simplificado"""
    logger.info("🧪 PROBANDO FEATURE EXTRACTOR SIMPLE...")

    # Ejemplos claros para entender las características
    test_chunks = [
        {
            "text": "If your Apple Watch battery drains quickly, try these solutions: Check Background App Refresh settings, reduce screen brightness, turn off unnecessary notifications.",
            "metadata": {"brand": "apple_watch", "chunk_index": 5}
        },
        {
            "text": "Page 47",  # Muy corto, irrelevante, con números
            "metadata": {"brand": "samsung", "chunk_index": 1}
        },
        {
            "text": "COPYRIGHT 2024 APPLE INC. ALL RIGHTS RESERVED.",  # Muchas mayúsculas, irrelevante
            "metadata": {"brand": "apple_watch", "chunk_index": 0}
        },
        {
            "text": "Battery life up to 18 hours with typical usage. Heart rate monitoring available 24/7.",
            "metadata": {"brand": "garmin", "chunk_index": 10}
        }
    ]

    # Extraer características
    extractor = SimpleFeatureExtractor()
    features = extractor.extract_features(test_chunks)

    # Mostrar resultados comprensibles
    feature_names = extractor.get_feature_names()
    explanations = extractor.explain_features()

    logger.info("\n📊 RESULTADOS POR CHUNK:")
    logger.info("=" * 60)

    for i, chunk in enumerate(test_chunks):
        text_preview = chunk["text"][:50] + "..." if len(chunk["text"]) > 50 else chunk["text"]

        logger.info(f"\n📝 CHUNK {i+1}: {text_preview}")
        logger.info("🔢 Características extraídas:")

        for j, (name, value) in enumerate(zip(feature_names, features[i])):
            explanation = explanations[name]
            logger.info(f"   {j+1}. {name}: {value} - {explanation}")

    return extractor, features


if __name__ == "__main__":
    test_simple_extractor()