import logging
import re
from typing import Any

from src.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RetrieverAgent:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def reformulate_queries(self, query: str, llm: Any | None = None) -> list[str]:
        queries = [query]

        query_lower = query.lower()

        reformulations = []

        if "cómo" in query_lower or "como" in query_lower:
            reformulations.append(re.sub(r"(?i)c(o|ó)mo\s+(se\s+)?(maneja|gestiona|hace|funciona|registra|genera|aplica|realiza)\s+", "", query).strip())

        if "qué" in query_lower or "que" in query_lower:
            reformulations.append(query.replace("¿", "").replace("?", ""))

        topics = {
            "inventario": ["control de stock", "gestión de existencias", "niveles de inventario"],
            "venta": ["registro de ventas", "proceso de venta", "transacciones"],
            "devolución": ["devoluciones", "cambios", "reembolsos", "nota crédito"],
            "factura": ["facturación", "factura electrónica", "DIAN", "IVA"],
            "reporte": ["reportes", "informes", "indicadores", "KPI"],
            "usuario": ["usuarios", "roles", "permisos", "cuentas"],
            "seguridad": ["acceso", "autenticación", "contraseña"],
            "producto": ["productos", "categorías", "SKU", "búsqueda"],
            "precio": ["precios", "promociones", "descuentos"],
            "auditoría": ["auditoría", "logs", "historial"],
        }

        for keyword, alternatives in topics.items():
            if keyword in query_lower:
                for alt in alternatives:
                    alt_query = query_lower.replace(keyword, alt)
                    if alt_query != query_lower:
                        reformulations.append(alt_query)

        for alt in reformulations:
            alt_clean = alt.strip()
            if alt_clean and alt_clean not in queries:
                if len(alt_clean) > 5:
                    queries.append(alt_clean)

        return queries[:3]

    def retrieve(
        self, query: str, llm: Any | None = None, k: int = 5
    ) -> dict[str, Any]:
        logger.info(f"RetrieverAgent procesando query: {query}")

        reformulated = self.reformulate_queries(query, llm)

        all_docs_map: dict[str, tuple[Any, float]] = {}

        for q in reformulated:
            results = self.vector_store.hybrid_search(q, k=k)
            for doc, score in results:
                doc_id = doc.metadata.get("chunk_id", doc.page_content[:50])
                if doc_id not in all_docs_map or score > all_docs_map[doc_id][1]:
                    all_docs_map[doc_id] = (doc, score)

        sorted_docs = sorted(all_docs_map.values(), key=lambda x: x[1], reverse=True)
        top_docs = sorted_docs[:k]

        sources = list(set(
            doc.metadata.get("source", "desconocido") for doc, _ in top_docs
        ))

        logger.info(f"Recuperados {len(top_docs)} documentos de {len(sources)} fuentes")

        return {
            "query": query,
            "reformulated_queries": reformulated,
            "retrieved_docs": [doc for doc, _ in top_docs],
            "sources": sources,
        }
