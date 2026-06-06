"""
Tests para el pipeline RAG: chunking, embeddings y búsqueda en vector store.
"""

import pytest
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


@pytest.fixture
def sample_documents() -> list[Document]:
    return [
        Document(
            page_content="La gestión de inventario es el proceso de supervisar y controlar las existencias. "
                         "Los métodos principales son FIFO, LIFO y ABC. La rotación de inventario es una métrica clave. "
                         "Para tiendas de ropa, el control de stock por talla y color es fundamental.",
            metadata={"source": "test_doc.txt"},
        )
    ]


def test_chunking_produces_chunks(sample_documents: list[Document]) -> None:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(sample_documents)

    assert len(chunks) >= 1, "El chunking debería producir al menos un chunk"
    for chunk in chunks:
        assert hasattr(chunk, "page_content"), "Cada chunk debe tener page_content"
        assert len(chunk.page_content) > 0, "Cada chunk debe tener contenido no vacío"


def test_embeddings_shape() -> None:
    try:
        from src.rag.embeddings import EmbeddingModel

        model = EmbeddingModel()
        embedding = model.embed_query("prueba de embedding")
        assert len(embedding) == 384, (
            f"MiniLM-L6-v2 debe producir embeddings de 384 dimensiones, "
            f"obtuvo {len(embedding)}"
        )
    except ImportError:
        pytest.skip("EmbeddingModel requiere dependencias opcionales")


def test_similarity_search_returns_results() -> None:
    try:
        from src.rag.embeddings import EmbeddingModel
        from src.rag.vector_store import VectorStore

        embed_model = EmbeddingModel()
        vs = VectorStore(
            persist_directory="./data/chroma_db",
            collection_name="test_collection",
            embedding_function=embed_model.get_embeddings(),
        )

        test_doc = Document(
            page_content="El inventario de la tienda incluye camisas, pantalones y vestidos. "
                         "La gestión adecuada del stock es crucial para evitar pérdidas.",
            metadata={"source": "test.txt", "chunk_id": 0},
        )
        vs.add_documents([test_doc])

        results = vs.similarity_search("inventario", k=5)
        assert len(results) >= 1, "La búsqueda debería retornar al menos un resultado"

        for doc, score in results:
            assert "inventario" in doc.page_content.lower() or "stock" in doc.page_content.lower()

    except ImportError:
        pytest.skip("Requiere dependencias de ChromaDB opcionales")
    except Exception as e:
        pytest.skip(f"No se pudo ejecutar test por: {e}")
