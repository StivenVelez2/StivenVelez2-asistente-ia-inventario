import json
import os
import logging
from pathlib import Path
from datetime import datetime

import httpx
import pytz

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Asistente IA - Keep Control",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stSidebarNav"] {display: none;}
div[data-testid="stToolbar"] {display: none;}
div[data-testid="stDecoration"] {display: none;}
div[data-testid="stStatusWidget"] {display: none;}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
section[data-testid="stSidebar"] div.stSelectbox {display: none;}
div[class*="themeLight"], div[class*="themeDark"], div[class*="themeSystem"] {display: none !important;}
[data-testid="baseButton-header"] {display: none;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_embedding_model():
    try:
        from src.rag.embeddings import EmbeddingModel
        model = EmbeddingModel()
        logger.info("EmbeddingModel inicializado")
        return model
    except Exception as e:
        logger.error(f"Error inicializando EmbeddingModel: {e}")
        return None


@st.cache_resource
def get_vector_store(_embedding_model):
    if _embedding_model is None:
        return None
    try:
        from src.rag.vector_store import VectorStore
        chroma_path = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
        collection = os.getenv("COLLECTION_NAME", "inventario_docs")

        vs = VectorStore(
            persist_directory=chroma_path,
            collection_name=collection,
            embedding_function=_embedding_model.get_embeddings(),
        )
        stats = vs.get_collection_stats()
        logger.info(f"VectorStore inicializado con {stats['total_documents']} documentos")
        return vs
    except Exception as e:
        logger.error(f"Error inicializando VectorStore: {e}")
        return None


@st.cache_resource
def get_llm():
    try:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            logger.warning("GROQ_API_KEY no configurada")
            return None

        from langchain_groq import ChatGroq
        model = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
        llm = ChatGroq(
            model=model,
            api_key=groq_api_key,
            temperature=0.3,
        )
        test_resp = llm.invoke([{"role": "user", "content": "responde solo OK"}])
        if not hasattr(test_resp, "content") or not test_resp.content:
            raise ValueError("El LLM devolvió una respuesta vacía en la validación")
        logger.info(f"LLM inicializado con modelo: {model}")
        return llm
    except Exception as e:
        logger.error(f"Error inicializando LLM: {e}")
        return None


@st.cache_resource
def get_agent_graph(_llm, _vector_store):
    if _llm is None or _vector_store is None:
        return None
    try:
        from src.graph.agent_graph import build_agent_graph
        graph = build_agent_graph(
            llm=_llm,
            vector_store=_vector_store,
        )
        logger.info("Grafo de agentes inicializado")
        return graph
    except Exception as e:
        logger.error(f"Error inicializando grafo: {e}")
        return None


embedding_model = get_embedding_model()
vector_store = get_vector_store(embedding_model)
llm = get_llm()
graph = get_agent_graph(llm, vector_store)

chroma_ready = (
    vector_store is not None
    and hasattr(vector_store, "get_collection_stats")
    and vector_store.get_collection_stats().get("total_documents", 0) > 0
)


if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    st.title("🏪 Keep Control")
    st.markdown("**Asistente IA – Inventario y Ventas**")

    bogota_tz = pytz.timezone('America/Bogota')
    hora_colombia = datetime.now(bogota_tz).strftime('%A, %d de %B de %Y - %H:%M')
    st.caption(f"🕐 {hora_colombia}")

    st.divider()

    st.subheader("ℹ️ Información del sistema")

    llm_model = os.getenv("LLM_MODEL", "no configurado")
    embed_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    chroma_path = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")

    st.markdown(f"- **LLM:** {llm_model}")
    st.markdown(f"- **Embeddings:** {embed_model}")
    st.markdown(f"- **Vector DB:** ChromaDB")
    st.markdown(f"- **Ruta ChromaDB:** `{chroma_path}`")

    st.divider()

    if st.button("🔄 Reindexar documentos", use_container_width=True):
        try:
            from src.rag.ingestion import index_documents
            with st.spinner("Indexando documentos en ChromaDB..."):
                index_documents()
            st.success("Documentos reindexados correctamente.")
            st.cache_resource.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Error al reindexar: {e}")

    st.divider()
    modo = st.radio(
        "Modo de operación",
        options=["Chat RAG", "Generar Reporte"],
        index=0,
    )


if modo == "Chat RAG":
    st.title("💬 Asistente de Inventario")
    st.markdown("Pregunta sobre gestión de inventario, ventas, facturación, productos y más.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("📄 Fuentes consultadas"):
                    for s in msg["sources"]:
                        st.markdown(f"- `{s}`")
            if "documents" in msg and msg["documents"]:
                with st.expander("📊 Documentos recuperados y scores"):
                    for d in msg["documents"]:
                        st.markdown(f"**Fuente:** `{d['source']}` | **Score:** {d['score']:.3f}")
                        st.text(d["content"][:200] + "...")

    if prompt := st.chat_input("Escribe tu pregunta aquí..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        if graph is None or llm is None or vector_store is None:
            with st.chat_message("assistant"):
                response_text = (
                    "Los componentes del asistente no están disponibles. "
                    "Revisa la configuración (GROQ_API_KEY, ChromaDB) en la barra lateral."
                )
                st.markdown(response_text)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
            })
        else:
            with st.chat_message("assistant"):
                with st.spinner("Analizando y consultando documentos..."):
                    try:
                        additional_context = ""
                        query_lower = prompt.lower()
                        try:
                            with httpx.Client(timeout=5.0) as client:
                                if any(w in query_lower for w in ["productos", "producto", "stock"]):
                                    resp = client.get("http://localhost:8000/productos")
                                    if resp.status_code == 200:
                                        data = resp.json()
                                        additional_context += (
                                            f"\n\n[DATOS DE PRODUCTOS]:"
                                            f"\n{json.dumps(data, indent=2, ensure_ascii=False)}"
                                        )
                                if any(w in query_lower for w in ["ventas", "venta"]):
                                    resp = client.get("http://localhost:8000/ventas")
                                    if resp.status_code == 200:
                                        data = resp.json()
                                        additional_context += (
                                            f"\n\n[DATOS DE VENTAS]:"
                                            f"\n{json.dumps(data, indent=2, ensure_ascii=False)}"
                                        )
                                if any(w in query_lower for w in ["dashboard", "resumen"]):
                                    resp = client.get("http://localhost:8000/dashboard")
                                    if resp.status_code == 200:
                                        data = resp.json()
                                        additional_context += (
                                            f"\n\n[DATOS DE DASHBOARD]:"
                                            f"\n{json.dumps(data, indent=2, ensure_ascii=False)}"
                                        )
                        except httpx.HTTPError as e:
                            logger.warning(f"Error HTTP consultando backend: {e}")
                        except Exception as e:
                            logger.warning(f"Error consultando backend: {e}")

                        enriched_query = prompt + additional_context

                        from src.graph.agent_graph import run_agent_graph
                        result = run_agent_graph(
                            graph=graph,
                            query=enriched_query,
                            llm=llm,
                            vector_store=vector_store,
                        )

                        response_text = result.get("final_response", "No se pudo generar una respuesta.")
                        st.markdown(response_text)

                        sources = result.get("sources", [])
                        if sources:
                            with st.expander("📄 Fuentes consultadas"):
                                for s in sources:
                                    st.markdown(f"- `{s}`")
                        else:
                            st.caption("No se encontraron fuentes específicas en el corpus.")

                    except Exception as e:
                        logger.error(f"Error ejecutando el grafo: {e}")
                        response_text = f"Ocurrió un error al procesar tu pregunta: {str(e)}"
                        st.markdown(response_text)
                        sources = []

                    msg_entry = {
                        "role": "assistant",
                        "content": response_text,
                        "sources": sources,
                    }
                    st.session_state.messages.append(msg_entry)


elif modo == "Generar Reporte":
    st.title("📊 Generar Reporte de Ventas")
    st.markdown("Selecciona el período y genera un reporte detallado de ventas.")

    col1, col2 = st.columns([2, 1])

    with col1:
        period = st.selectbox(
            "Período del reporte",
            options=["diario", "semanal", "mensual"],
            format_func=lambda x: {"diario": "📅 Diario", "semanal": "📆 Semanal", "mensual": "📊 Mensual"}[x],
        )

    with col2:
        st.write("")
        st.write("")
        generate_btn = st.button("🚀 Generar Reporte", use_container_width=True, type="primary")

    if generate_btn:
        with st.spinner("Generando reporte..."):
            try:
                from src.skills.report_skill import generate_sales_report
                result = generate_sales_report(period)

                report_text = result["report"]
                summary = result["summary"]

                st.markdown("### Reporte generado exitosamente")

                col_metrics = st.columns(4)
                metrics_data = [
                    ("Ventas", summary.get("total_sales", 0), ""),
                    ("Facturación", f"${summary.get('total_revenue', 0):,.0f}", ""),
                    ("Ticket Prom.", f"${summary.get('average_ticket', 0):,.0f}", ""),
                    ("Stock Bajo", summary.get("low_stock_count", 0), "alertas"),
                ]
                for col, (label, value, suffix) in zip(col_metrics, metrics_data):
                    with col:
                        st.metric(label, value, suffix)

                st.text(report_text)

                st.download_button(
                    label="📥 Descargar reporte (.txt)",
                    data=report_text,
                    file_name=f"reporte_ventas_{period}_{Path(__file__).stem}.txt",
                    mime="text/plain",
                )

            except Exception as e:
                st.error(f"Error al generar el reporte: {e}")
                logger.error(f"Error en generación de reporte: {e}")
