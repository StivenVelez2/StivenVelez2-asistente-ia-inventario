import logging
from typing import Any

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "Responde usando solo el contexto. Cita las fuentes como [Fuente: nombre]. Sé claro y profesional en español."


def format_context(documents: list[tuple[Document, float]]) -> str:
    context_parts = []
    for i, (doc, score) in enumerate(documents, 1):
        source = doc.metadata.get("source", "desconocido")
        content = doc.page_content.strip()
        context_parts.append(
            f"[Documento {i}] (Fuente: {source}, Relevancia: {score:.3f})\n{content}"
        )
    return "\n\n---\n\n".join(context_parts)


def extract_sources(documents: list[tuple[Document, float]]) -> list[str]:
    sources = set()
    for doc, _score in documents:
        source = doc.metadata.get("source", "desconocido")
        sources.add(source)
    return sorted(sources)


def retrieve_and_generate(
    query: str,
    llm: Any,
    vector_store: Any,
    k: int = 5,
) -> dict[str, Any]:
    logger.info(f"Procesando query: {query}")

    hybrid_docs = vector_store.hybrid_search(query, k=k)

    if not hybrid_docs:
        return {
            "response": "No encontré información relevante en el corpus documental para responder tu pregunta.",
            "documents": [],
            "sources": [],
        }

    context = format_context(hybrid_docs)
    sources = extract_sources(hybrid_docs)

    user_prompt = f"""Contexto documental:
{context}

Pregunta del usuario: {query}

Responde la pregunta basándote exclusivamente en el contexto proporcionado arriba."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = llm.invoke(messages)
        response_text = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        logger.error(f"Error al invocar LLM: {e}")
        response_text = (
            "Ocurrió un error al generar la respuesta. Por favor, intenta de nuevo."
        )

    return {
        "response": response_text,
        "documents": [
            {
                "content": doc.page_content[:300],
                "source": doc.metadata.get("source", "desconocido"),
                "score": round(score, 3),
            }
            for doc, score in hybrid_docs
        ],
        "sources": sources,
    }
