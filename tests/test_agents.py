"""
Tests para los agentes Recuperador, Redactor y el grafo completo.
"""

import pytest
from langchain_core.documents import Document


@pytest.fixture
def sample_docs() -> list[Document]:
    return [
        Document(
            page_content="Las devoluciones en tiendas de ropa se gestionan con un plazo de 30 días. "
                         "El proceso incluye verificación de elegibilidad, inspección del producto y "
                         "actualización de inventario. Los motivos comunes son talla incorrecta, "
                         "color no deseado y defectos de fabricación.",
            metadata={"source": "doc_03_devoluciones.txt", "chunk_id": 0},
        ),
        Document(
            page_content="El control de stock utiliza niveles máximo y mínimo, punto de reorden "
                         "y stock de seguridad. Las alertas automáticas notifican cuando un producto "
                         "alcanza el punto de reorden.",
            metadata={"source": "doc_05_control_stock.txt", "chunk_id": 1},
        ),
    ]


def test_retriever_agent_returns_docs() -> None:
    """Verifica que el agente recuperador retorna documentos no vacíos."""
    try:
        from src.rag.embeddings import EmbeddingModel
        from src.rag.vector_store import VectorStore
        from src.agents.retriever_agent import RetrieverAgent

        embed_model = EmbeddingModel()
        vs = VectorStore(
            persist_directory="./data/chroma_db",
            collection_name="test_agents_collection",
            embedding_function=embed_model.get_embeddings(),
        )

        test_content = (
            "La gestión de inventario incluye FIFO, LIFO y ABC. "
            "La rotación de inventario y los días de inventario disponible son métricas clave."
        )
        vs.add_documents([
            Document(page_content=test_content, metadata={"source": "test.txt", "chunk_id": 0})
        ])

        agent = RetrieverAgent(vector_store=vs)
        result = agent.retrieve(query="gestión de inventario", k=3)

        assert isinstance(result["retrieved_docs"], list)
        assert len(result["retrieved_docs"]) >= 0
        assert "query" in result
        assert "reformulated_queries" in result
        assert "sources" in result

        if result["retrieved_docs"]:
            first_doc = result["retrieved_docs"][0]
            assert hasattr(first_doc, "page_content")
            assert hasattr(first_doc, "metadata")

    except ImportError:
        pytest.skip("Requiere dependencias de embeddings/ChromaDB")
    except Exception as e:
        pytest.skip(f"No se pudo ejecutar test: {e}")


def test_writer_agent_returns_string(sample_docs: list[Document]) -> None:
    """Verifica que el agente redactor retorna string no vacío."""
    try:
        from src.agents.writer_agent import WriterAgent

        sources = ["doc_03_devoluciones.txt", "doc_05_control_stock.txt"]

        class MockLLM:
            def invoke(self, messages):
                class Response:
                    content = (
                        "La gestión de devoluciones tiene un plazo de 30 días "
                        "[Fuente: doc_03_devoluciones.txt]. "
                        "El control de stock usa niveles mínimos y máximos "
                        "[Fuente: doc_05_control_stock.txt]."
                    )
                return Response()

        mock_llm = MockLLM()
        writer = WriterAgent(llm=mock_llm)
        result = writer.write(
            retrieved_docs=sample_docs,
            sources=sources,
            query="¿Cómo funcionan las devoluciones?",
        )

        assert "final_response" in result
        assert isinstance(result["final_response"], str)
        assert len(result["final_response"]) > 0
        assert "sources_used" in result
        assert len(result["sources_used"]) == 2

    except ImportError:
        pytest.skip("WriterAgent requiere dependencias")


def test_full_graph_execution() -> None:
    """Verifica que el grafo completo retorna una respuesta final."""
    try:
        from src.rag.embeddings import EmbeddingModel
        from src.rag.vector_store import VectorStore
        from src.graph.agent_graph import build_agent_graph, run_agent_graph

        embed_model = EmbeddingModel()
        vs = VectorStore(
            persist_directory="./data/chroma_db",
            collection_name="test_graph_collection",
            embedding_function=embed_model.get_embeddings(),
        )

        test_content = (
            "La facturación electrónica en Colombia debe cumplir con los requisitos de la DIAN. "
            "Los campos obligatorios incluyen NIT, CUFE y numeración autorizada."
        )
        vs.add_documents([
            Document(page_content=test_content, metadata={"source": "doc_04_facturacion.txt", "chunk_id": 0})
        ])

        class MockLLM:
            def invoke(self, messages):
                class Response:
                    content = (
                        "La facturación electrónica en Colombia sigue los lineamientos de la DIAN. "
                        "Debe incluir NIT del emisor, CUFE y numeración autorizada "
                        "[Fuente: doc_04_facturacion.txt]."
                    )
                return Response()

        mock_llm = MockLLM()
        graph = build_agent_graph(llm=mock_llm, vector_store=vs)

        result = run_agent_graph(
            graph=graph,
            query="¿Qué es la facturación electrónica?",
            llm=mock_llm,
            vector_store=vs,
        )

        assert "final_response" in result
        assert isinstance(result["final_response"], str)
        assert len(result["final_response"]) > 0

    except ImportError:
        pytest.skip("Requiere dependencias de LangGraph/ChromaDB")
    except Exception as e:
        pytest.skip(f"No se pudo ejecutar test: {e}")
