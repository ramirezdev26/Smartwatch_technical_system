"""
Diagnóstico SEGURO para evitar consumo excesivo de memoria
"""
import sys
import os
import psutil
from pathlib import Path
import PyPDF2
from loguru import logger
import time
import gc

# Añadir src al path
sys.path.append(str(Path(__file__).parent / "src"))

from config import *


def check_memory_usage():
    """Verifica el uso actual de memoria"""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    return memory_mb


def safe_pdf_analysis(pdf_path: Path, max_memory_mb=500, max_pages=10):
    """
    Análisis SEGURO de PDF con límites estrictos de memoria y páginas
    """
    logger.info(f"🔍 Analizando SEGURO: {pdf_path.name}")

    initial_memory = check_memory_usage()
    logger.info(f"💾 Memoria inicial: {initial_memory:.1f} MB")

    # Información básica del archivo
    file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
    logger.info(f"📁 Tamaño del archivo: {file_size_mb:.2f} MB")

    # 🚨 LÍMITE DE SEGURIDAD: Saltar archivos muy grandes
    if file_size_mb > 50:
        logger.warning(f"⚠️ ARCHIVO MUY GRANDE ({file_size_mb:.1f} MB) - SALTANDO")
        return {
            "status": "skipped_too_large",
            "file_size_mb": file_size_mb,
            "error": f"Archivo demasiado grande ({file_size_mb:.1f} MB)"
        }

    result = {
        "file_name": pdf_path.name,
        "file_size_mb": file_size_mb,
        "pages_total": 0,
        "pages_processed": 0,
        "text_length": 0,
        "success": False,
        "memory_peak_mb": initial_memory,
        "processing_time": 0,
        "status": "unknown"
    }

    start_time = time.time()

    try:
        # Abrir PDF de forma segura
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            total_pages = len(reader.pages)
            result["pages_total"] = total_pages

            logger.info(f"📄 Páginas totales: {total_pages}")

            # 🚨 LÍMITE: Máximo de páginas a procesar
            pages_to_process = min(total_pages, max_pages)
            logger.info(f"📝 Procesando solo las primeras {pages_to_process} páginas")

            extracted_text = ""

            for page_num in range(pages_to_process):
                # Verificar memoria cada 3 páginas
                if page_num % 3 == 0:
                    current_memory = check_memory_usage()
                    result["memory_peak_mb"] = max(result["memory_peak_mb"], current_memory)

                    # 🚨 LÍMITE DE MEMORIA: Parar si supera el límite
                    if current_memory > max_memory_mb:
                        logger.error(f"🚨 LÍMITE DE MEMORIA EXCEDIDO: {current_memory:.1f} MB")
                        result["status"] = "memory_limit_exceeded"
                        break

                    logger.info(f"   Página {page_num + 1}/{pages_to_process} - Memoria: {current_memory:.1f} MB")

                try:
                    page = reader.pages[page_num]
                    page_text = page.extract_text()

                    # 🚨 LÍMITE: No acumular texto infinitamente
                    if len(extracted_text) > 100000:  # 100K caracteres máximo
                        logger.warning("⚠️ Límite de texto alcanzado, parando extracción")
                        break

                    extracted_text += page_text[:5000]  # Máximo 5K caracteres por página
                    result["pages_processed"] = page_num + 1

                    # Liberar memoria explícitamente
                    del page_text

                except Exception as e:
                    logger.warning(f"Error en página {page_num + 1}: {e}")
                    continue

            # Limpiar memoria
            del reader
            gc.collect()

            result["text_length"] = len(extracted_text)
            result["processing_time"] = time.time() - start_time

            # Verificar si extrajimos algo útil
            if len(extracted_text) > 100:
                result["success"] = True
                result["status"] = "success"

                # Mostrar muestra PEQUEÑA del texto
                clean_text = ' '.join(extracted_text.split())
                sample = clean_text[:200]
                logger.info(f"📝 Texto extraído: {len(extracted_text)} caracteres")
                logger.info(f"📄 Muestra: {sample}...")

                # Verificar calidad del texto
                words = clean_text.split()
                if len(words) > 20:
                    avg_word_len = sum(len(w) for w in words[:20]) / 20
                    if 2 < avg_word_len < 12:
                        logger.info("✅ Texto parece válido")
                    else:
                        logger.warning("⚠️ Texto podría ser corrupto/escaneado")

                # 🧪 PRUEBA DE CHUNKING SIMPLE Y SEGURA
                test_simple_chunking(extracted_text[:10000])  # Solo primeros 10K caracteres

            else:
                result["status"] = "no_text_extracted"
                logger.warning(f"❌ No se extrajo texto legible")

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        logger.error(f"💥 Error procesando PDF: {e}")

    finally:
        # Limpiar memoria forzadamente
        gc.collect()
        final_memory = check_memory_usage()
        logger.info(f"💾 Memoria final: {final_memory:.1f} MB (pico: {result['memory_peak_mb']:.1f} MB)")

    return result


def test_simple_chunking(text: str):
    """Prueba de chunking SIMPLE sin cargar todo en memoria"""
    if len(text) < 100:
        return

    logger.info(f"🧪 Probando chunking en texto de {len(text)} caracteres")

    # Configuración actual
    chunk_size = CHUNK_SIZE
    overlap = CHUNK_OVERLAP

    logger.info(f"⚙️ Config actual: CHUNK_SIZE={chunk_size}, OVERLAP={overlap}")

    # Chunking simple por palabras
    words = text.split()
    total_words = len(words)

    # Simular solo los primeros chunks para evitar memoria
    chunks_simulated = 0
    start = 0

    while start < total_words and chunks_simulated < 10:  # Máximo 10 chunks para prueba
        end = min(start + chunk_size, total_words)
        chunk_words = words[start:end]

        if len(chunk_words) > 10:  # Chunk válido
            chunks_simulated += 1

        start = end - overlap
        if start >= total_words:
            break

    # Estimación total
    estimated_total_chunks = total_words // (chunk_size - overlap) if chunk_size > overlap else 1

    logger.info(f"📊 Chunking simulado:")
    logger.info(f"   - Palabras totales: {total_words}")
    logger.info(f"   - Chunks simulados: {chunks_simulated}")
    logger.info(f"   - Estimación total: {estimated_total_chunks}")

    # 🚨 DIAGNÓSTICO DEL PROBLEMA PRINCIPAL
    if estimated_total_chunks <= 1 and total_words > 500:
        logger.error(f"🚨 PROBLEMA DETECTADO: Solo {estimated_total_chunks} chunk para {total_words} palabras")
        logger.error(f"   CHUNK_SIZE={chunk_size} es DEMASIADO GRANDE")
        logger.error(f"   RECOMENDACIÓN: Cambiar CHUNK_SIZE a 150-200")

        # Mostrar lo que pasaría con configuración corregida
        corrected_chunks = total_words // 130  # ~150 palabras por chunk
        logger.info(f"💡 Con CHUNK_SIZE=150: ~{corrected_chunks} chunks")


def analyze_directory_safely():
    """Análisis seguro de todos los PDFs"""
    logger.info("🚀 DIAGNÓSTICO SEGURO DEL DIRECTORIO")
    logger.info("=" * 50)

    # Buscar archivos
    pdf_files = []
    for brand_dir in RAW_DATA_DIR.iterdir():
        if brand_dir.is_dir():
            for file_path in brand_dir.glob("*.pdf"):
                pdf_files.append(file_path)

    logger.info(f"📁 Encontrados {len(pdf_files)} archivos PDF")

    if not pdf_files:
        logger.warning("No hay archivos PDF para analizar")
        return

    # Ordenar por tamaño (más pequeños primero)
    pdf_files.sort(key=lambda p: p.stat().st_size)

    # Mostrar información de archivos
    logger.info("📋 Lista de archivos (por tamaño):")
    for i, pdf_path in enumerate(pdf_files, 1):
        size_mb = pdf_path.stat().st_size / (1024 * 1024)
        logger.info(f"   {i}. {pdf_path.parent.name}/{pdf_path.name} ({size_mb:.1f} MB)")

    # Procesar archivos de forma segura
    results = []

    for i, pdf_path in enumerate(pdf_files[:3], 1):  # Solo 3 archivos máximo
        logger.info(f"\n{'='*20} ARCHIVO {i}/3 {'='*20}")

        # Verificar memoria antes de procesar
        memory_before = check_memory_usage()
        if memory_before > 300:  # Si ya usa mucha memoria, saltar
            logger.warning(f"⚠️ Memoria alta ({memory_before:.1f} MB), saltando archivo")
            continue

        try:
            result = safe_pdf_analysis(pdf_path, max_memory_mb=400, max_pages=5)
            results.append(result)

            # Pausa entre archivos para que el sistema recupere memoria
            time.sleep(1)
            gc.collect()

        except Exception as e:
            logger.error(f"💥 Error crítico procesando {pdf_path.name}: {e}")
            break

    # Resumen final
    logger.info(f"\n{'='*50}")
    logger.info("📊 RESUMEN FINAL")

    if results:
        for result in results:
            status = result.get("status", "unknown")
            filename = result.get("file_name", "unknown")

            if status == "success":
                pages = result.get("pages_processed", 0)
                text_len = result.get("text_length", 0)
                logger.info(f"✅ {filename}: {pages} páginas, {text_len} caracteres")
            elif status == "skipped_too_large":
                logger.warning(f"⚠️ {filename}: Saltado (muy grande)")
            else:
                logger.error(f"❌ {filename}: Error - {status}")

    # Recomendaciones
    logger.info(f"\n💡 RECOMENDACIONES INMEDIATAS:")
    logger.info("1. 🔧 Cambiar en config.py: CHUNK_SIZE = 150 (no 512)")
    logger.info("2. 📉 Procesar archivos de <10MB primero")
    logger.info("3. 🧠 Verificar que los PDFs no son escaneados")
    logger.info("4. ⚡ Implementar procesamiento streaming para archivos grandes")


def main():
    """Función principal del diagnóstico seguro"""
    logger.remove()
    logger.add(sys.stdout, level="INFO")

    try:
        analyze_directory_safely()
    except KeyboardInterrupt:
        logger.info("🛑 Diagnóstico interrumpido por el usuario")
    except Exception as e:
        logger.error(f"💥 Error crítico: {e}")
    finally:
        # Limpiar memoria al final
        gc.collect()
        final_memory = check_memory_usage()
        logger.info(f"💾 Memoria final: {final_memory:.1f} MB")


if __name__ == "__main__":
    main()