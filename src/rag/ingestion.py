import os
import logging
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.rag.embeddings import EmbeddingModel
from src.rag.vector_store import VectorStore

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

RAW_DOCS_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "inventario_docs")


def load_documents(raw_dir: str | Path) -> list:
    loader_txt = DirectoryLoader(
        str(raw_dir),
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    docs_txt = loader_txt.load()
    logger.info(f"Cargados {len(docs_txt)} documentos .txt")

    pdf_files = list(Path(raw_dir).glob("*.pdf"))
    docs_pdf = []
    for pdf_path in pdf_files:
        loader = PyPDFLoader(str(pdf_path))
        docs_pdf.extend(loader.load())
    logger.info(f"Cargados {len(docs_pdf)} documentos .pdf")

    return docs_txt + docs_pdf


def chunk_documents(documents: list, chunk_size: int = 500, overlap: int = 50) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Total chunks generados: {len(chunks)}")
    return chunks


def index_documents() -> None:
    logger.info("Iniciando pipeline de ingesta...")

    raw_dir = Path(RAW_DOCS_DIR)
    if not raw_dir.exists():
        logger.error(f"El directorio {raw_dir} no existe.")
        return

    documents = load_documents(raw_dir)
    if not documents:
        logger.warning("No se encontraron documentos para indexar.")
        return

    chunks = chunk_documents(documents)

    docs_by_source: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "desconocido")
        docs_by_source[source] = docs_by_source.get(source, 0) + 1

    for source, count in sorted(docs_by_source.items()):
        logger.info(f"  {Path(source).name}: {count} chunks")

    logger.info("Generando embeddings...")
    embedding_model = EmbeddingModel()

    logger.info("Inicializando VectorStore...")
    vector_store = VectorStore(
        persist_directory=str(CHROMA_DB_PATH),
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model.get_embeddings(),
    )

    logger.info("Agregando documentos a ChromaDB...")
    vector_store.add_documents(chunks)

    logger.info(f"Ingesta completada. {len(chunks)} chunks indexados en '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    index_documents()
