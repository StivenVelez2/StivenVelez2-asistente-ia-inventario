import os
from pathlib import Path
from typing import Any

import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from dotenv import load_dotenv

load_dotenv()


class VectorStore:
    def __init__(
        self,
        persist_directory: str | None = None,
        collection_name: str | None = None,
        embedding_function: HuggingFaceEmbeddings | None = None,
    ):
        self.persist_directory = persist_directory or os.getenv(
            "CHROMA_DB_PATH", "./data/chroma_db"
        )
        self.collection_name = collection_name or os.getenv(
            "COLLECTION_NAME", "inventario_docs"
        )
        self.embedding_function = embedding_function

        os.makedirs(self.persist_directory, exist_ok=True)

        self.client = chromadb.PersistentClient(path=self.persist_directory)

        if self.embedding_function:
            self.db = Chroma(
                client=self.client,
                collection_name=self.collection_name,
                embedding_function=self.embedding_function,
            )
        else:
            self.db = Chroma(
                client=self.client,
                collection_name=self.collection_name,
            )

        self._bm25_index: BM25Okapi | None = None
        self._all_docs: list[Document] = []

    def add_documents(self, documents: list[Document]) -> list[str]:
        for i, doc in enumerate(documents):
            source = Path(doc.metadata.get("source", "unknown")).name
            doc.metadata["source"] = source
            doc.metadata["chunk_id"] = i

        ids = self.db.add_documents(documents)
        self._bm25_index = None
        self._all_docs = []
        return ids

    def similarity_search(self, query: str, k: int = 5) -> list[tuple[Document, float]]:
        results = self.db.similarity_search_with_relevance_scores(query, k=k)
        return results

    def _ensure_bm25_index(self) -> None:
        if self._bm25_index is not None:
            return
        collection = self.client.get_or_create_collection(self.collection_name)
        all_data = collection.get(include=["documents", "metadatas"])
        self._all_docs = []
        tokenized_corpus = []
        for i, (text, meta) in enumerate(
            zip(all_data["documents"] or [], all_data["metadatas"] or [])
        ):
            doc = Document(page_content=text, metadata=meta or {})
            self._all_docs.append(doc)
            tokenized_corpus.append(self._tokenize(text))
        self._bm25_index = BM25Okapi(tokenized_corpus)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        import re
        tokens = re.findall(r"\w+", text.lower())
        return tokens

    def bm25_search(self, query: str, k: int = 5) -> list[tuple[Document, float]]:
        self._ensure_bm25_index()
        if not self._all_docs:
            return []
        tokenized_query = self._tokenize(query)
        scores = self._bm25_index.get_scores(tokenized_query)
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in indexed[:k]:
            if score > 0:
                results.append((self._all_docs[idx], float(score)))
        return results

    def hybrid_search(
        self, query: str, k: int = 5, alpha: float = 0.7
    ) -> list[tuple[Document, float]]:
        semantic_results = self.similarity_search(query, k=k * 2)
        lexical_results = self.bm25_search(query, k=k * 2)

        combined: dict[str, dict[str, Any]] = {}

        for doc, score in semantic_results:
            doc_id = doc.metadata.get("chunk_id", doc.page_content[:50])
            combined[doc_id] = {
                "doc": doc,
                "semantic_score": score,
                "lexical_score": 0.0,
                "source": doc.metadata.get("source", "unknown"),
            }

        for doc, score in lexical_results:
            doc_id = doc.metadata.get("chunk_id", doc.page_content[:50])
            if doc_id in combined:
                combined[doc_id]["lexical_score"] = score
            else:
                combined[doc_id] = {
                    "doc": doc,
                    "semantic_score": 0.0,
                    "lexical_score": score,
                    "source": doc.metadata.get("source", "unknown"),
                }

        max_sem = max(
            (v["semantic_score"] for v in combined.values()), default=1.0
        )
        max_lex = max(
            (v["lexical_score"] for v in combined.values()), default=1.0
        )

        for entry in combined.values():
            norm_sem = entry["semantic_score"] / max_sem if max_sem > 0 else 0
            norm_lex = entry["lexical_score"] / max_lex if max_lex > 0 else 0
            entry["hybrid_score"] = alpha * norm_sem + (1 - alpha) * norm_lex

        sorted_results = sorted(
            combined.values(), key=lambda x: x["hybrid_score"], reverse=True
        )

        return [
            (entry["doc"], entry["hybrid_score"])
            for entry in sorted_results[:k]
        ]

    def get_collection_stats(self) -> dict[str, int]:
        collection = self.client.get_or_create_collection(self.collection_name)
        count = collection.count()
        return {"total_documents": count}
