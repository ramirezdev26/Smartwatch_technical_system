"""
Clasificador de Calidad SIMPLIFICADO para aprender ML básico
Solo 10 características fáciles de entender + Logistic Regression básico
"""
import numpy as np
import joblib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from feature_extractor import SimpleFeatureExtractor
from auto_labeler import SimpleAutoLabeler


class SimpleQualityClassifier:
    """Clasificador MUY SIMPLE para aprender ML básico"""

    def __init__(self, model_path: Optional[Path] = None):
        self.feature_extractor = SimpleFeatureExtractor()
        self.auto_labeler = SimpleAutoLabeler()
        self.scaler = StandardScaler()

        # Modelo más simple para aprendizaje
        self.classifier = LogisticRegression(
            random_state=42,
            max_iter=500,  # Menos iteraciones
            solver='lbfgs'  # Solver simple
        )

        self.is_trained = False

        # Mapeo simple de etiquetas a números
        self.label_to_number = {"irrelevante": 0, "ambiguo": 1, "relevante": 2}
        self.number_to_label = {0: "irrelevante", 1: "ambiguo", 2: "relevante"}

        self.training_info = {}

        # Cargar modelo si existe
        if model_path and model_path.exists():
            self.load_model(model_path)

    def train(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Entrena el clasificador de forma SIMPLE y comprensible
        """
        logger.info(f"🎓 ENTRENANDO CLASIFICADOR SIMPLE con {len(chunks)} chunks...")
        logger.info("📚 Proceso de aprendizaje automático:")
        logger.info("   1️⃣ Auto-etiquetar chunks (crear datos de entrenamiento)")
        logger.info("   2️⃣ Extraer 10 características simples")
        logger.info("   3️⃣ Entrenar Logistic Regression")
        logger.info("   4️⃣ Evaluar resultados")

        # PASO 1: Auto-etiquetar (crear nuestros datos de entrenamiento)
        logger.info(f"\n1️⃣ PASO 1: Creando datos de entrenamiento automáticamente")
        logger.info("-" * 50)

        labeled_chunks = self.auto_labeler.auto_label_chunks(chunks)

        # PASO 2: Extraer características simples
        logger.info(f"\n2️⃣ PASO 2: Extrayendo 10 características por chunk")
        logger.info("-" * 50)

        features = self.feature_extractor.extract_features(labeled_chunks)

        # Preparar etiquetas como números
        labels_text = [chunk["auto_label"] for chunk in labeled_chunks]
        labels_numbers = [self.label_to_number[label] for label in labels_text]
        y = np.array(labels_numbers)

        # Mostrar distribución de datos
        logger.info(f"📊 DISTRIBUCIÓN DE DATOS DE ENTRENAMIENTO:")
        unique, counts = np.unique(labels_text, return_counts=True)
        for label, count in zip(unique, counts):
            percentage = (count / len(labels_text)) * 100
            logger.info(f"   {label}: {count} ejemplos ({percentage:.1f}%)")

        # PASO 3: Decidir estrategia según cantidad de datos
        logger.info(f"\n3️⃣ PASO 3: Decidiendo estrategia de validación")
        logger.info("-" * 50)

        # Verificar si tenemos suficientes datos para división train/test
        min_samples_per_class = min(counts)
        total_samples = len(labeled_chunks)

        if total_samples < 30 or min_samples_per_class < 2:
            # CASO: Pocos datos - usar todos para entrenamiento
            logger.info(f"⚠️ POCOS DATOS DETECTADOS:")
            logger.info(f"   📦 Total ejemplos: {total_samples}")
            logger.info(f"   📊 Clase con menos ejemplos: {min_samples_per_class}")
            logger.info(f"💡 ESTRATEGIA: Entrenar con todos los datos (sin división train/test)")
            logger.info("   📚 En ML real, necesitas más datos para validación robusta")

            X_train, X_test = features, features  # Usar todos los datos
            y_train, y_test = y, y

            logger.info(f"📚 Datos de entrenamiento: {len(X_train)} ejemplos (todos)")
            logger.info(f"🧪 Datos de prueba: {len(X_test)} ejemplos (los mismos - solo para demo)")

        else:
            # CASO: Suficientes datos - división normal
            logger.info(f"✅ SUFICIENTES DATOS para división train/test")

            X_train, X_test, y_train, y_test = train_test_split(
                features, y,
                test_size=0.2,  # 20% para prueba
                random_state=42,
                stratify=y  # Mantener proporción de clases
            )

            logger.info(f"📚 Datos de entrenamiento: {len(X_train)} ejemplos")
            logger.info(f"🧪 Datos de prueba: {len(X_test)} ejemplos")

        # PASO 4: Normalizar características
        logger.info(f"\n📏 Normalizando características (StandardScaler)...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # PASO 5: Entrenar el modelo
        logger.info(f"\n4️⃣ PASO 5: Entrenando Logistic Regression")
        logger.info("-" * 50)
        logger.info("🤖 El modelo está aprendiendo patrones en los datos...")

        self.classifier.fit(X_train_scaled, y_train)

        # PASO 6: Evaluar el modelo
        logger.info(f"\n5️⃣ PASO 6: Evaluando el modelo entrenado")
        logger.info("-" * 50)

        # Accuracy en entrenamiento y prueba
        train_accuracy = self.classifier.score(X_train_scaled, y_train)
        test_accuracy = self.classifier.score(X_test_scaled, y_test)

        logger.info(f"🎯 Accuracy en ENTRENAMIENTO: {train_accuracy:.3f} ({train_accuracy*100:.1f}%)")
        logger.info(f"🎯 Accuracy en PRUEBA: {test_accuracy:.3f} ({test_accuracy*100:.1f}%)")

        # Predicciones para análisis detallado
        y_pred = self.classifier.predict(X_test_scaled)

        # Reporte de clasificación
        test_labels_text = [self.number_to_label[num] for num in y_test]
        pred_labels_text = [self.number_to_label[num] for num in y_pred]

        report = classification_report(test_labels_text, pred_labels_text, output_dict=True)

        # Matriz de confusión
        conf_matrix = confusion_matrix(y_test, y_pred)

        # PASO 7: Análizar qué características son más importantes
        logger.info(f"\n6️⃣ PASO 7: Analizando características más importantes")
        logger.info("-" * 50)

        importance_analysis = self._analyze_feature_importance()

        # Guardar información del entrenamiento
        self.is_trained = True
        self.training_info = {
            "total_chunks": len(chunks),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "classification_report": report,
            "confusion_matrix": conf_matrix.tolist(),
            "feature_importance": importance_analysis,
            "label_distribution": {label: int(count) for label, count in zip(unique, counts)}
        }

        # Mostrar resultados finales
        logger.info(f"\n✅ ENTRENAMIENTO COMPLETADO:")
        logger.info(f"   📊 Accuracy: {test_accuracy:.3f} ({test_accuracy*100:.1f}%)")
        logger.info(f"   📈 Precisión promedio: {report['macro avg']['precision']:.3f}")
        logger.info(f"   🎯 Recall promedio: {report['macro avg']['recall']:.3f}")

        return self.training_info

    def _analyze_feature_importance(self) -> Dict[str, Any]:
        """Analiza qué características son más importantes (fácil de entender)"""
        if not self.is_trained:
            return {}

        # Para Logistic Regression, los coeficientes nos dicen la importancia
        coefficients = self.classifier.coef_  # Shape: (n_classes, n_features)

        # Tomar el promedio absoluto across todas las clases
        avg_importance = np.mean(np.abs(coefficients), axis=0)

        # Normalizar para que la suma sea 1
        if np.sum(avg_importance) > 0:
            avg_importance = avg_importance / np.sum(avg_importance)

        # Crear diccionario con nombres de características
        feature_names = self.feature_extractor.get_feature_names()
        feature_explanations = self.feature_extractor.explain_features()

        importance_list = []
        for i, (name, importance) in enumerate(zip(feature_names, avg_importance)):
            importance_list.append({
                "name": name,
                "importance": float(importance),
                "explanation": feature_explanations[name],
                "rank": 0  # Se llenará después de ordenar
            })

        # Ordenar por importancia
        importance_list.sort(key=lambda x: x["importance"], reverse=True)

        # Agregar ranking
        for i, item in enumerate(importance_list):
            item["rank"] = i + 1

        # Log características más importantes
        logger.info("🔍 TOP 5 CARACTERÍSTICAS MÁS IMPORTANTES:")
        for item in importance_list[:5]:
            logger.info(f"   {item['rank']}. {item['name']} (importancia: {item['importance']:.3f})")
            logger.info(f"      📝 {item['explanation']}")

        return {
            "top_features": importance_list[:10],  # Top 10
            "feature_count": len(feature_names)
        }

    def predict(self, chunks: List[Dict[str, Any]], explain: bool = False) -> List[Dict[str, Any]]:
        """
        Predice calidad de chunks de forma SIMPLE
        """
        if not self.is_trained:
            raise ValueError("❌ El modelo no está entrenado. Ejecuta train() primero.")

        logger.info(f"🔮 Prediciendo calidad de {len(chunks)} chunks...")

        # Extraer características
        features = self.feature_extractor.extract_features(chunks)

        # Normalizar usando el scaler entrenado
        features_scaled = self.scaler.transform(features)

        # Hacer predicciones
        predictions_numbers = self.classifier.predict(features_scaled)
        probabilities = self.classifier.predict_proba(features_scaled)

        # Convertir predicciones a texto
        enhanced_chunks = []

        for i, chunk in enumerate(chunks):
            # Prediction
            predicted_number = predictions_numbers[i]
            predicted_label = self.number_to_label[predicted_number]

            # Probabilidades por clase
            chunk_probabilities = probabilities[i]
            confidence = np.max(chunk_probabilities)

            # Agregar información al chunk
            enhanced_chunk = chunk.copy()
            enhanced_chunk["quality_label"] = predicted_label
            enhanced_chunk["quality_confidence"] = float(confidence)
            enhanced_chunk["predicted_by"] = "simple_logistic_regression"

            # Probabilidades detalladas
            enhanced_chunk["probabilities"] = {
                "irrelevante": float(chunk_probabilities[0]),
                "ambiguo": float(chunk_probabilities[1]),
                "relevante": float(chunk_probabilities[2])
            }

            enhanced_chunks.append(enhanced_chunk)

        # Mostrar estadísticas de predicciones
        pred_labels = [chunk["quality_label"] for chunk in enhanced_chunks]
        unique, counts = np.unique(pred_labels, return_counts=True)

        logger.info(f"📊 PREDICCIONES REALIZADAS:")
        for label, count in zip(unique, counts):
            percentage = (count / len(pred_labels)) * 100
            logger.info(f"   {label}: {count} chunks ({percentage:.1f}%)")

        return enhanced_chunks

    def explain_prediction(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Explica una predicción específica de forma SIMPLE"""
        if not self.is_trained:
            raise ValueError("Modelo no entrenado")

        # Extraer características del chunk
        features = self.feature_extractor.extract_features([chunk])
        features_scaled = self.scaler.transform(features)

        # Hacer predicción
        prediction = self.classifier.predict(features_scaled)[0]
        probabilities = self.classifier.predict_proba(features_scaled)[0]

        # Analizar características del chunk
        chunk_features = features[0]
        feature_names = self.feature_extractor.get_feature_names()
        feature_explanations = self.feature_extractor.explain_features()

        explanation = {
            "text_preview": chunk["text"][:100] + "..." if len(chunk["text"]) > 100 else chunk["text"],
            "predicted_label": self.number_to_label[prediction],
            "confidence": float(np.max(probabilities)),
            "probabilities": {
                "irrelevante": float(probabilities[0]),
                "ambiguo": float(probabilities[1]),
                "relevante": float(probabilities[2])
            },
            "characteristics": []
        }

        # Mostrar las características más relevantes
        for i, (name, value) in enumerate(zip(feature_names, chunk_features)):
            explanation["characteristics"].append({
                "name": name,
                "value": float(value),
                "explanation": feature_explanations[name]
            })

        return explanation

    def save_model(self, model_path: Path):
        """Guarda el modelo simple"""
        if not self.is_trained:
            logger.warning("⚠️ Intentando guardar modelo no entrenado")
            return

        model_path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "classifier": self.classifier,
            "scaler": self.scaler,
            "training_info": self.training_info,
            "is_trained": self.is_trained,
            "label_mappings": {
                "label_to_number": self.label_to_number,
                "number_to_label": self.number_to_label
            }
        }

        joblib.dump(model_data, model_path)
        logger.info(f"💾 Modelo simple guardado en: {model_path}")

    def load_model(self, model_path: Path):
        """Carga un modelo simple guardado"""
        try:
            model_data = joblib.load(model_path)

            self.classifier = model_data["classifier"]
            self.scaler = model_data["scaler"]
            self.training_info = model_data.get("training_info", {})
            self.is_trained = model_data.get("is_trained", True)

            mappings = model_data.get("label_mappings", {})
            self.label_to_number = mappings.get("label_to_number", self.label_to_number)
            self.number_to_label = mappings.get("number_to_label", self.number_to_label)

            logger.info(f"📂 Modelo simple cargado desde: {model_path}")

        except Exception as e:
            logger.error(f"❌ Error cargando modelo: {e}")
            raise

    def get_model_summary(self) -> Dict[str, Any]:
        """Resumen simple del modelo"""
        if not self.is_trained:
            return {"status": "not_trained"}

        return {
            "status": "trained",
            "model_type": "Simple Logistic Regression",
            "features_used": 10,
            "classes": ["irrelevante", "ambiguo", "relevante"],
            "test_accuracy": self.training_info.get("test_accuracy", 0),
            "total_training_samples": self.training_info.get("total_chunks", 0)
        }


def test_simple_classifier():
    """Test completo del clasificador simple"""
    logger.info("🧪 PROBANDO CLASIFICADOR SIMPLE COMPLETO...")

    # Crear datos de prueba más variados
    test_chunks = [
        # Ejemplos RELEVANTES
        {
            "text": "If your Apple Watch battery drains quickly, try these solutions: Check Background App Refresh settings, reduce screen brightness, turn off unnecessary notifications.",
            "metadata": {"brand": "apple_watch", "chunk_index": 5}
        },
        {
            "text": "To charge your Samsung Galaxy Watch, clean the charging contacts with a dry cloth and place the watch on the charging dock.",
            "metadata": {"brand": "samsung", "chunk_index": 10}
        },
        {
            "text": "Press the Digital Crown to wake up your Apple Watch. Tap Settings to configure your device preferences.",
            "metadata": {"brand": "apple_watch", "chunk_index": 8}
        },

        # Ejemplos IRRELEVANTES
        {
            "text": "Page 47",
            "metadata": {"brand": "samsung", "chunk_index": 1}
        },
        {
            "text": "Copyright 2024 Apple Inc. All rights reserved.",
            "metadata": {"brand": "apple_watch", "chunk_index": 0}
        },
        {
            "text": "Chapter 5: Advanced Features",
            "metadata": {"brand": "garmin", "chunk_index": 2}
        },

        # Ejemplos AMBIGUOS
        {
            "text": "See the previous section for battery information.",
            "metadata": {"brand": "fitbit", "chunk_index": 15}
        },
        {
            "text": "Heart rate, GPS, Sleep tracking available.",
            "metadata": {"brand": "garmin", "chunk_index": 12}
        }
    ]

    # Crear y entrenar clasificador
    classifier = SimpleQualityClassifier()

    logger.info("\n" + "="*60)
    training_results = classifier.train(test_chunks)

    # Probar predicciones en datos nuevos
    logger.info(f"\n🧪 PROBANDO PREDICCIONES EN DATOS NUEVOS:")
    logger.info("-" * 50)

    new_chunks = [
        {
            "text": "Restart your device by pressing and holding the power button for 10 seconds.",
            "metadata": {"brand": "apple_watch", "chunk_index": 20}
        },
        {
            "text": "Table of Contents",
            "metadata": {"brand": "fitbit", "chunk_index": 1}
        }
    ]

    predictions = classifier.predict(new_chunks, explain=True)

    for i, pred in enumerate(predictions, 1):
        text_preview = pred["text"][:60] + "..."
        label = pred["quality_label"]
        confidence = pred["quality_confidence"]

        emoji = "✅" if label == "relevante" else "❌" if label == "irrelevante" else "⚠️"

        logger.info(f"\n{i}. {emoji} PREDICCIÓN: {label.upper()} (confianza: {confidence:.2f})")
        logger.info(f"   📝 Texto: {text_preview}")
        logger.info(f"   🎲 Probabilidades:")
        for class_name, prob in pred["probabilities"].items():
            logger.info(f"      {class_name}: {prob:.3f}")

    # Explicar una predicción específica
    logger.info(f"\n🔍 EXPLICACIÓN DETALLADA DE UNA PREDICCIÓN:")
    logger.info("-" * 50)

    explanation = classifier.explain_prediction(new_chunks[0])

    logger.info(f"📝 Texto: {explanation['text_preview']}")
    logger.info(f"🎯 Predicción: {explanation['predicted_label']} (confianza: {explanation['confidence']:.2f})")
    logger.info(f"📊 Características del texto:")

    for char in explanation["characteristics"][:5]:  # Solo top 5
        logger.info(f"   • {char['name']}: {char['value']} - {char['explanation']}")

    # Resumen del modelo
    logger.info(f"\n📈 RESUMEN DEL MODELO:")
    logger.info("-" * 30)
    summary = classifier.get_model_summary()

    for key, value in summary.items():
        logger.info(f"   {key}: {value}")

    return classifier


if __name__ == "__main__":
    test_simple_classifier()