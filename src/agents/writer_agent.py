import logging
from typing import Any

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_WRITER = "Responde usando solo el contexto. Cita las fuentes como [Fuente: nombre]. Usa español claro."


class WriterAgent:
    def __init__(self, llm: Any):
        self.llm = llm

    def write(
        self,
        retrieved_docs: list[Document],
        sources: list[str],
        query: str,
    ) -> dict[str, Any]:
        logger.info(f"WriterAgent redactando respuesta para: {query}")

        if not retrieved_docs:
            return {
                "final_response": (
                    "No encontré información en el corpus documental "
                    "para responder tu pregunta. ¿Podrías reformularla?"
                ),
                "sources_used": [],
            }

        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            source = doc.metadata.get("source", "desconocido")
            content = doc.page_content.strip()
            context_parts.append(
                f"[Documento {i}] (Fuente: {source})\n{content}"
            )
        context = "\n\n---\n\n".join(context_parts)

        user_prompt = f"""Contexto documental:
{context}

Pregunta del usuario: {query}

Basándote exclusivamente en el contexto anterior, redacta una respuesta completa y útil."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_WRITER},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = self.llm.invoke(messages)
            response_text = (
                response.content
                if hasattr(response, "content")
                else str(response)
            )
        except Exception as e:
            logger.error(f"Error al invocar LLM en WriterAgent: {e}")
            fallback = self._generate_fallback_response(retrieved_docs, query)
            response_text = fallback

        return {
            "final_response": response_text,
            "sources_used": sources,
        }

    def _generate_fallback_response(
        self, docs: list[Document], query: str
    ) -> str:
        parts = ["Basándome en la información disponible:\n"]
        seen_sources = set()
        for doc in docs:
            source = doc.metadata.get("source", "desconocido")
            if source not in seen_sources:
                content_preview = doc.page_content[:200].strip()
                parts.append(f"- {content_preview}... [Fuente: {source}]")
                seen_sources.add(source)
        parts.append(
            "\n\n*Nota: Esta respuesta se generó directamente desde los documentos "
            "sin procesamiento por LLM.*"
        )
        return "\n".join(parts)
