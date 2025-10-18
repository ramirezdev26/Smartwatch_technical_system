"""
Streamlit App - Sistema de Gestión de Conocimiento de Smartwatches
Versión básica: Visualización + Búsqueda Semántica
"""

import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Añadir src al path
sys.path.append(str(Path(__file__).parent))

from src.storage.chroma_manager import ChromaManager
from src.clustering.cluster_manager import ClusterManager
from src.visualization.visualizer import EmbeddingVisualizer
from config import *

# Configuración de página
st.set_page_config(
    page_title="Smartwatch Knowledge System",
    page_icon="⌚",
    layout="wide",
)


@st.cache_resource
def load_system():
    """Carga todos los componentes del sistema (se cachea)"""
    try:
        # ChromaDB
        chroma = ChromaManager()

        # Cargar datos
        all_data = chroma.collection.get(include=["embeddings", "metadatas"])
        embeddings = np.array(all_data["embeddings"])
        metadata = all_data["metadatas"]

        # Clustering
        cluster_manager = ClusterManager()
        try:
            cluster_manager.load()
        except FileNotFoundError:
            st.warning("⚠️ No hay modelo de clustering. Entrenando...")
            cluster_manager.fit(embeddings, metadata)
            cluster_manager.save()

        # Visualización
        visualizer = EmbeddingVisualizer()
        try:
            visualizer.load_cache()
        except FileNotFoundError:
            st.warning("⚠️ No hay visualización cacheada. Calculando...")
            visualizer.fit_both(embeddings, metadata)
            visualizer.save_cache()

        return {
            "chroma": chroma,
            "cluster_manager": cluster_manager,
            "visualizer": visualizer,
            "embeddings": embeddings,
            "metadata": metadata,
        }
    except Exception as e:
        st.error(f"❌ Error cargando sistema: {e}")
        return None


def create_plotly_scatter(coords, labels, color_by, title):
    """Crea gráfico de dispersión con Plotly"""
    fig = px.scatter(
        x=coords[:, 0],
        y=coords[:, 1],
        color=labels,
        title=title,
        labels={"x": "Dimensión 1", "y": "Dimensión 2", "color": color_by},
        opacity=0.7,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )

    fig.update_traces(marker=dict(size=5))
    fig.update_layout(
        height=500,
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return fig


def main():
    """Función principal de la app"""

    # Título
    st.title("⌚ Sistema de Gestión de Conocimiento - Smartwatches")
    st.markdown("---")

    # Cargar sistema
    with st.spinner("🔄 Cargando sistema..."):
        system = load_system()

    if system is None:
        st.stop()

    # Extraer componentes
    chroma = system["chroma"]
    cluster_manager = system["cluster_manager"]
    visualizer = system["visualizer"]
    metadata = system["metadata"]

    # Tabs principales
    tab1, tab2 = st.tabs(["🔍 Búsqueda Semántica", "📊 Visualización de Clusters"])

    # ==========================================
    # TAB 1: BÚSQUEDA SEMÁNTICA
    # ==========================================
    with tab1:
        st.header("Búsqueda Semántica")

        # Input de búsqueda
        query = st.text_input(
            "Escribe tu consulta:",
            placeholder="Ejemplo: battery life apple watch",
        )

        # Filtros
        col1, col2 = st.columns([2, 1])
        with col1:
            top_k = st.slider("Número de resultados:", 1, 20, 5)
        with col2:
            brand_filter = st.selectbox(
                "Filtrar por marca:",
                ["Todas"] + SUPPORTED_BRANDS,
            )

        # Botón de búsqueda
        if st.button("🔍 Buscar", type="primary") and query:
            with st.spinner("Buscando..."):
                # Realizar búsqueda
                brand = None if brand_filter == "Todas" else brand_filter
                results = chroma.search(
                    query=query,
                    top_k=top_k,
                    brand_filter=brand,
                    prioritize_relevant=True,
                )

                # Mostrar resultados
                if results:
                    st.success(f"✅ Encontrados {len(results)} resultados")

                    for i, result in enumerate(results, 1):
                        with st.expander(
                                f"📄 Resultado {i} - {result['metadata']['brand'].upper()} (Score: {result['similarity']:.3f})"
                        ):
                            # Metadata
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Marca", result["metadata"]["brand"])
                            with col_b:
                                st.metric(
                                    "Calidad",
                                    result["metadata"].get("chunk_quality", "N/A"),
                                )
                            with col_c:
                                st.metric(
                                    "Documento",
                                    result["metadata"]["document_name"][:20] + "...",
                                )

                            # Contenido
                            st.markdown("**Contenido:**")
                            st.info(result["text"])

                else:
                    st.warning("⚠️ No se encontraron resultados")

    # ==========================================
    # TAB 2: VISUALIZACIÓN
    # ==========================================
    with tab2:
        st.header("Exploración de Embeddings")

        # Controles
        col1, col2 = st.columns([1, 1])

        with col1:
            viz_method = st.radio(
                "Método de visualización:",
                ["t-SNE", "PCA"],
                horizontal=True,
            )

        with col2:
            color_by = st.radio(
                "Colorear por:", ["Cluster", "Marca", "Calidad"], horizontal=True
            )

        # Obtener datos de visualización
        viz_data = visualizer.get_visualization_data(cluster_manager.cluster_labels)

        # Seleccionar coordenadas
        if viz_method == "t-SNE":
            coords = np.array(viz_data["tsne"])
            title = "t-SNE - Visualización de Embeddings"
        else:
            coords = np.array(viz_data["pca"])
            pca_info = visualizer.get_pca_info()
            title = f"PCA - Visualización de Embeddings (Varianza: {pca_info['total_variance_explained']:.1%})"

        # Seleccionar colores
        if color_by == "Cluster":
            labels = [f"Cluster {c}" for c in viz_data["clusters"]]
        elif color_by == "Marca":
            labels = viz_data["brands"]
        else:  # Calidad
            labels = viz_data["quality"]

        # Crear y mostrar gráfico
        fig = create_plotly_scatter(coords, labels, color_by, title)
        st.plotly_chart(fig, use_container_width=True)

        # Información adicional
        with st.expander("ℹ️ Información del Sistema"):
            col_a, col_b, col_c = st.columns(3)

            with col_a:
                st.metric("Total Documentos", len(metadata))
            with col_b:
                st.metric("Número de Clusters", cluster_manager.n_clusters)
            with col_c:
                summary = cluster_manager.get_summary()
                # ✅ MANEJAR CASO DONDE SILHOUETTE ES NONE
                if summary['silhouette_score'] is not None:
                    st.metric("Silhouette Score", f"{summary['silhouette_score']:.3f}")
                else:
                    st.metric("Silhouette Score", "N/A")

            # Distribución de clusters
            st.markdown("**Distribución de Clusters:**")
            for cluster_info in summary["clusters"]:
                st.write(
                    f"- **Cluster {cluster_info['cluster_id']}**: {cluster_info['size']} documentos "
                    f"({cluster_info['percentage']:.1f}%) - Marca dominante: {cluster_info['top_brand']}"
                )


if __name__ == "__main__":
    main()
