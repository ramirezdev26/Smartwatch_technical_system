"""
Script de diagnóstico para identificar problemas en el pipeline
"""
import sys
from pathlib import Path
import pandas as pd
from loguru import logger
import PyPDF2
import fitz  # pymupdf
from sentence_transformers import SentenceTransformer

# Añadir src al path
sys.path.append(str(Path(__file__).parent / "src"))

from config import *


def diagnose_pdf_extraction(pdf_path: Path):
    """Diagnostica problemas de extracción de PDF"""
    logger.info(f"🔍 Diagnosticando: {pdf_path.name}")
    
    results = {
        "file_size_mb": pdf_path.stat().st_size / (1024 * 1024),
        "pypdf2_pages": 0,
        "pymupdf_pages": 0,
        "pypdf2_text_length": 0,
        "pymupdf_text_length": 0,
        "pypdf2_success": False,
        "pymupdf_success": False
    }
    
    # Probar PyPDF2
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            results["pypdf2_pages"] = len(reader.pages)
            
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            
            results["pypdf2_text_length"] = len(text)
            results["pypdf2_success"] = True
            
            # Mostrar muestra del texto
            logger.info(f"PyPDF2 - Páginas: {results['pypdf2_pages']}, Caracteres: {results['pypdf2_text_length']}")
            if text:
                sample = text[:300].replace('\n', ' ')
                logger.info(f"PyPDF2 - Muestra: {sample}...")
            
    except Exception as e:
        logger.error(f"PyPDF2 falló: {e}")
    
    # Probar PyMuPDF (más robusto)
    try:
        doc = fitz.open(pdf_path)
        results["pymupdf_pages"] = doc.page_count
        
        text = ""
        for page in doc:
            text += page.get_text()
        
        results["pymupdf_text_length"] = len(text)
        results["pymupdf_success"] = True
        
        logger.info(f"PyMuPDF - Páginas: {results['pymupdf_pages']}, Caracteres: {results['pymupdf_text_length']}")
        if text:
            sample = text[:300].replace('\n', ' ')
            logger.info(f"PyMuPDF - Muestra: {sample}...")
        
        doc.close()
        
    except Exception as e:
        logger.error(f"PyMuPDF falló: {e}")
    
    return results


def test_chunking(text: str):
    """Prueba el proceso de chunking"""
    if not text or len(text) < 100:
        logger.warning("Texto muy corto para chunking")
        return []
    
    # Simular chunking simple
    words = text.split()
    chunk_size = CHUNK_SIZE_WORDS
    overlap = CHUNK_OVERLAP_WORDS
    
    chunks = []
    start = 0
    
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text = ' '.join(chunk_words)
        
        if len(chunk_text) >= MIN_CHUNK_LENGTH:
            chunks.append({
                "text": chunk_text,
                "word_count": len(chunk_words),
                "char_count": len(chunk_text),
                "start_word": start,
                "end_word": end
            })
        
        # Mover al siguiente chunk con overlap
        start = end - overlap
        if start >= len(words):
            break
    
    logger.info(f"✅ Generados {len(chunks)} chunks")
    
    # Mostrar estadísticas de chunks
    if chunks:
        word_counts = [chunk["word_count"] for chunk in chunks]
        char_counts = [chunk["char_count"] for chunk in chunks]
        
        logger.info(f"Palabras por chunk - Min: {min(word_counts)}, Max: {max(word_counts)}, Promedio: {sum(word_counts)/len(word_counts):.1f}")
        logger.info(f"Caracteres por chunk - Min: {min(char_counts)}, Max: {max(char_counts)}, Promedio: {sum(char_counts)/len(char_counts):.1f}")
        
        # Mostrar primer chunk como ejemplo
        logger.info(f"Ejemplo de chunk: {chunks[0]['text'][:200]}...")
    
    return chunks


def test_embedding_model():
    """Prueba la carga y funcionamiento del modelo de embeddings"""
    logger.info("🧠 Probando modelo de embeddings...")
    
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        # Texto de prueba
        test_texts = [
            "My Apple Watch battery drains fast",
            "How to charge Samsung Galaxy Watch",
            "Fitbit sleep tracking not working"
        ]
        
        embeddings = model.encode(test_texts)
        
        logger.info(f"✅ Modelo cargado correctamente")
        logger.info(f"Dimensión de embeddings: {embeddings.shape[1]}")
        logger.info(f"Embeddings generados para {len(test_texts)} textos")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error con modelo de embeddings: {e}")
        return False


def main():
    """Función principal de diagnóstico"""
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    
    logger.info("🚀 DIAGNÓSTICO DEL PIPELINE DE INGESTA")
    logger.info("=" * 50)
    
    # 1. Probar modelo de embeddings
    embedding_ok = test_embedding_model()
    
    print("\n" + "="*50)
    
    # 2. Buscar y diagnosticar PDFs
    pdf_files = []
    for brand_dir in RAW_DATA_DIR.iterdir():
        if brand_dir.is_dir():
            for file_path in brand_dir.glob("*.pdf"):
                pdf_files.append(file_path)
    
    logger.info(f"📄 Encontrados {len(pdf_files)} archivos PDF")
    
    if not pdf_files:
        logger.warning("No se encontraron archivos PDF para diagnosticar")
        return
    
    # Diagnosticar cada PDF
    all_results = []
    
    for pdf_path in pdf_files[:3]:  # Limitar a 3 para la prueba
        print("\n" + "-"*30)
        
        # Diagnosticar extracción
        pdf_results = diagnose_pdf_extraction(pdf_path)
        pdf_results["file_name"] = pdf_path.name
        pdf_results["brand"] = pdf_path.parent.name
        all_results.append(pdf_results)
        
        # Probar chunking si hay texto
        best_text = ""
        if pdf_results["pymupdf_success"] and pdf_results["pymupdf_text_length"] > 0:
            # Usar PyMuPDF para chunking
            try:
                doc = fitz.open(pdf_path)
                best_text = ""
                for page in doc:
                    best_text += page.get_text()
                doc.close()
            except:
                pass
        elif pdf_results["pypdf2_success"] and pdf_results["pypdf2_text_length"] > 0:
            # Usar PyPDF2 como fallback
            try:
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    best_text = ""
                    for page in reader.pages:
                        best_text += page.extract_text()
            except:
                pass
        
        if best_text:
            logger.info(f"📝 Probando chunking para {pdf_path.name}")
            chunks = test_chunking(best_text)
            pdf_results["chunks_generated"] = len(chunks)
        else:
            logger.warning(f"❌ No se pudo extraer texto de {pdf_path.name}")
            pdf_results["chunks_generated"] = 0
    
    # Resumen final
    print("\n" + "="*50)
    logger.info("📊 RESUMEN DE DIAGNÓSTICO")
    
    df = pd.DataFrame(all_results)
    if not df.empty:
        print("\nEstadísticas por archivo:")
        print(df[["file_name", "brand", "file_size_mb", "pymupdf_pages", "pymupdf_text_length", "chunks_generated"]].to_string(index=False))
        
        # Recomendaciones
        print("\n🎯 RECOMENDACIONES:")
        
        for _, row in df.iterrows():
            if row["chunks_generated"] == 0:
                print(f"❌ {row['file_name']}: PDF problemático - posiblemente escaneado o corrupto")
            elif row["chunks_generated"] < 10:
                print(f"⚠️  {row['file_name']}: Pocos chunks ({row['chunks_generated']}) - revisar contenido")
            else:
                print(f"✅ {row['file_name']}: OK ({row['chunks_generated']} chunks)")


if __name__ == "__main__":
    main()