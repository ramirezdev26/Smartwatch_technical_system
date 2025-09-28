"""
Auto-Labeler SIMPLIFICADO para aprender ML básico
Reglas simples y comprensibles para etiquetar chunks
"""
from typing import List, Dict, Any, Tuple
from loguru import logger


class SimpleAutoLabeler:
    """Auto-etiquetador SIMPLE con reglas fáciles de entender"""

    def __init__(self):
        # Palabras técnicas que indican contenido RELEVANTE
        self.good_words = [
            'battery', 'charge', 'charging', 'power',
            'settings', 'configure', 'setup',
            'problem', 'fix', 'troubleshoot', 'solve',
            'press', 'tap', 'select', 'click',
            'watch', 'device', 'screen', 'button',
            'heart rate', 'sleep', 'fitness', 'gps'
        ]

        # Palabras que indican contenido IRRELEVANTE
        self.bad_words = [
            'page', 'copyright', 'reserved', 'trademark',
            'chapter', 'section', 'contents', 'index',
            'version', 'revision', 'document'
        ]

    def auto_label_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Etiqueta chunks usando reglas SIMPLES y comprensibles
        """
        logger.info(f"🏷️ Auto-etiquetando {len(chunks)} chunks con REGLAS SIMPLES...")

        labeled_chunks = []
        stats = {"relevante": 0, "irrelevante": 0, "ambiguo": 0}

        for i, chunk in enumerate(chunks):
            # Etiquetar usando reglas simples
            label, confidence, reason = self._simple_labeling_rules(chunk)

            # Agregar etiquetas al chunk
            enhanced_chunk = chunk.copy()
            enhanced_chunk["auto_label"] = label
            enhanced_chunk["auto_confidence"] = confidence
            enhanced_chunk["auto_reason"] = reason

            labeled_chunks.append(enhanced_chunk)
            stats[label] += 1

            # Mostrar progreso
            if (i + 1) % 200 == 0:
                logger.info(f"   Etiquetados {i + 1}/{len(chunks)} chunks")

        # Mostrar estadísticas finales
        total = len(chunks)
        logger.info(f"✅ Etiquetado SIMPLE completado:")
        logger.info(f"   🟢 Relevante: {stats['relevante']} ({stats['relevante'] / total * 100:.1f}%)")
        logger.info(f"   🔴 Irrelevante: {stats['irrelevante']} ({stats['irrelevante'] / total * 100:.1f}%)")
        logger.info(f"   🟡 Ambiguo: {stats['ambiguo']} ({stats['ambiguo'] / total * 100:.1f}%)")

        return labeled_chunks

    def _simple_labeling_rules(self, chunk: Dict[str, Any]) -> Tuple[str, float, str]:
        """
        Reglas MUY SIMPLES para etiquetar (fáciles de entender y explicar)

        Returns:
            (label, confidence, reason)
        """
        text = chunk.get("text", "").strip()
        metadata = chunk.get("metadata", {})

        if not text:
            return "irrelevante", 1.0, "Texto vacío"

        text_lower = text.lower()
        word_count = len(text.split())

        # === REGLA 1: Textos MUY CORTOS son usualmente irrelevantes ===
        if word_count < 5:
            return "irrelevante", 0.9, f"Muy corto ({word_count} palabras)"

        # === REGLA 2: Si solo tiene números, probablemente es una página ===
        if text.replace(' ', '').isdigit():
            return "irrelevante", 0.95, "Solo contiene números"

        # === REGLA 3: Palabras irrelevantes obvias ===
        bad_word_count = sum(1 for word in self.bad_words if word in text_lower)
        if bad_word_count >= 2:
            return "irrelevante", 0.8, f"Contiene {bad_word_count} palabras irrelevantes"

        # === REGLA 4: Una sola palabra irrelevante fuerte ===
        strong_bad_words = ['copyright', 'page', 'chapter']
        for bad_word in strong_bad_words:
            if bad_word in text_lower:
                return "irrelevante", 0.85, f"Contiene '{bad_word}'"

        # === REGLA 5: Muchas palabras técnicas = relevante ===
        good_word_count = sum(1 for word in self.good_words if word in text_lower)
        if good_word_count >= 3:
            return "relevante", 0.8, f"Contiene {good_word_count} palabras técnicas"

        # === REGLA 6: Texto largo con algunas palabras técnicas ===
        if word_count > 30 and good_word_count >= 1:
            return "relevante", 0.7, f"Texto largo ({word_count} palabras) con contenido técnico"

        # === REGLA 7: Instrucciones típicas ===
        instruction_words = ['press', 'tap', 'go to', 'select', 'if', 'when']
        instruction_count = sum(1 for word in instruction_words if word in text_lower)
        if instruction_count >= 2:
            return "relevante", 0.75, f"Contiene {instruction_count} palabras de instrucción"

        # === REGLA 8: Posición en el documento ===
        chunk_index = metadata.get("chunk_index", 0)
        if chunk_index <= 2:  # Primeros chunks suelen ser títulos/headers
            if good_word_count == 0:
                return "ambiguo", 0.6, "Chunk al inicio sin palabras técnicas"

        # === REGLA 9: Texto mediano sin palabras malas ===
        if 10 <= word_count <= 100 and bad_word_count == 0:
            if good_word_count >= 1:
                return "relevante", 0.65, f"Texto mediano con algún contenido técnico"
            else:
                return "ambiguo", 0.5, "Texto mediano sin clasificación clara"

        # === REGLA POR DEFECTO: Ambiguo ===
        return "ambiguo", 0.4, f"No cumple reglas específicas (palabras: {word_count})"

    def explain_rules(self) -> Dict[str, str]:
        """Explica las reglas de etiquetado en español simple"""
        return {
            "Regla 1 - Textos cortos": "Si tiene menos de 5 palabras → IRRELEVANTE (probable header/página)",
            "Regla 2 - Solo números": "Si solo contiene números → IRRELEVANTE (número de página)",
            "Regla 3 - Palabras malas": "Si tiene 2+ palabras como 'copyright', 'page' → IRRELEVANTE",
            "Regla 4 - Palabras muy malas": "Si contiene 'copyright', 'page', 'chapter' → IRRELEVANTE",
            "Regla 5 - Muchas palabras técnicas": "Si tiene 3+ palabras como 'battery', 'settings' → RELEVANTE",
            "Regla 6 - Texto largo técnico": "Si tiene >30 palabras + contenido técnico → RELEVANTE",
            "Regla 7 - Instrucciones": "Si tiene 2+ palabras como 'press', 'tap', 'if' → RELEVANTE",
            "Regla 8 - Posición": "Chunks al inicio sin palabras técnicas → AMBIGUO",
            "Regla 9 - Texto mediano": "10-100 palabras con algo técnico → RELEVANTE",
            "Regla Default": "Si no cumple ninguna regla → AMBIGUO"
        }

    def analyze_labeling_results(self, labeled_chunks: List[Dict[str, Any]]) -> None:
        """Análisis simple de los resultados de etiquetado"""
        logger.info("\n📊 ANÁLISIS SIMPLE DE ETIQUETADO:")
        logger.info("=" * 50)

        # Contar por razón
        reason_counts = {}
        for chunk in labeled_chunks:
            reason = chunk["auto_reason"]
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

        # Mostrar top 10 razones más comunes
        sorted_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)

        logger.info("🔍 RAZONES DE ETIQUETADO MÁS COMUNES:")
        for i, (reason, count) in enumerate(sorted_reasons[:10], 1):
            percentage = (count / len(labeled_chunks)) * 100
            logger.info(f"   {i}. {reason}: {count} chunks ({percentage:.1f}%)")

        # Ejemplos por categoría
        logger.info(f"\n📋 EJEMPLOS POR CATEGORÍA:")
        for label in ["relevante", "irrelevante", "ambiguo"]:
            examples = [chunk for chunk in labeled_chunks if chunk["auto_label"] == label]
            if examples:
                # Tomar ejemplos de alta confianza
                high_conf_examples = sorted(examples, key=lambda x: x["auto_confidence"], reverse=True)[:3]

                logger.info(f"\n🏷️ EJEMPLOS '{label.upper()}':")
                for i, ex in enumerate(high_conf_examples, 1):
                    text_preview = ex["text"][:60] + "..." if len(ex["text"]) > 60 else ex["text"]
                    confidence = ex["auto_confidence"]
                    reason = ex["auto_reason"]

                    logger.info(f"   {i}. Confianza: {confidence:.2f} | Razón: {reason}")
                    logger.info(f"      Texto: {text_preview}")

    def get_simple_stats(self, labeled_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Estadísticas simples del etiquetado"""
        stats = {
            "total_chunks": len(labeled_chunks),
            "by_label": {},
            "high_confidence": 0,
            "avg_confidence": 0
        }

        # Por label
        for label in ["relevante", "irrelevante", "ambiguo"]:
            chunks_with_label = [c for c in labeled_chunks if c["auto_label"] == label]
            stats["by_label"][label] = {
                "count": len(chunks_with_label),
                "percentage": len(chunks_with_label) / len(labeled_chunks) * 100
            }

        # Confianza
        confidences = [chunk["auto_confidence"] for chunk in labeled_chunks]
        stats["avg_confidence"] = sum(confidences) / len(confidences)
        stats["high_confidence"] = sum(1 for c in confidences if c >= 0.8)

        return stats


def test_simple_labeler():
    """Test del auto-labeler simplificado"""
    logger.info("🧪 PROBANDO AUTO-LABELER SIMPLE...")

    # Ejemplos claros que muestren cómo funcionan las reglas
    test_chunks = [
        {
            "text": "If your Apple Watch battery drains quickly, try these solutions: Check Background App Refresh settings, reduce screen brightness.",
            "metadata": {"brand": "apple_watch", "chunk_index": 5}
        },
        {
            "text": "Page 47",
            "metadata": {"brand": "samsung", "chunk_index": 1}
        },
        {
            "text": "Copyright 2024 Apple Inc. All rights reserved.",
            "metadata": {"brand": "apple_watch", "chunk_index": 0}
        },
        {
            "text": "Battery life up to 18 hours. Heart rate monitoring available.",
            "metadata": {"brand": "garmin", "chunk_index": 10}
        },
        {
            "text": "Press the Digital Crown to wake up your device.",
            "metadata": {"brand": "apple_watch", "chunk_index": 8}
        },
        {
            "text": "See the previous section for more details.",
            "metadata": {"brand": "fitbit", "chunk_index": 15}
        }
    ]

    # Probar etiquetado
    labeler = SimpleAutoLabeler()
    labeled_chunks = labeler.auto_label_chunks(test_chunks)

    # Mostrar resultados comprensibles
    logger.info("\n📋 RESULTADOS DE ETIQUETADO:")
    logger.info("=" * 60)

    for i, chunk in enumerate(labeled_chunks, 1):
        text_preview = chunk["text"][:60] + "..." if len(chunk["text"]) > 60 else chunk["text"]
        label = chunk["auto_label"]
        confidence = chunk["auto_confidence"]
        reason = chunk["auto_reason"]

        # Emoji según el label
        emoji = "✅" if label == "relevante" else "❌" if label == "irrelevante" else "⚠️"

        logger.info(f"\n{i}. {emoji} ETIQUETA: {label.upper()} (confianza: {confidence:.2f})")
        logger.info(f"   📝 Texto: {text_preview}")
        logger.info(f"   💡 Razón: {reason}")

    # Explicar las reglas
    logger.info(f"\n📖 REGLAS DE ETIQUETADO UTILIZADAS:")
    logger.info("-" * 50)
    rules = labeler.explain_rules()
    for i, (rule_name, explanation) in enumerate(rules.items(), 1):
        logger.info(f"{i:2d}. {explanation}")

    # Análisis simple
    labeler.analyze_labeling_results(labeled_chunks)

    return labeler, labeled_chunks


if __name__ == "__main__":
    test_simple_labeler()