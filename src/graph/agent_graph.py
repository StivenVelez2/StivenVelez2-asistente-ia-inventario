import logging
from typing import Any, TypedDict

from langgraph.graph import StateGraph, END
from langchain_core.documents import Document

from src.agents.retriever_agent import RetrieverAgent
from src.agents.writer_agent import WriterAgent

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    user_query: str
    reformulated_queries: list[str]
    retrieved_docs: list[Document]
    sources: list[str]
    final_response: str
    llm: Any
    vector_store: Any


def retriever_node(state: AgentState) -> dict[str, Any]:
    logger.info("Ejecutando nodo: retriever_node")

    retriever = RetrieverAgent(vector_store=state["vector_store"])
    result = retriever.retrieve(
        query=state["user_query"],
        llm=state.get("llm"),
        k=5,
    )

    return {
        "reformulated_queries": result["reformulated_queries"],
        "retrieved_docs": result["retrieved_docs"],
        "sources": result["sources"],
    }


def writer_node(state: AgentState) -> dict[str, Any]:
    logger.info("Ejecutando nodo: writer_node")

    retrieved_docs = state.get("retrieved_docs", [])
    sources = state.get("sources", [])

    if not retrieved_docs:
        return {
            "final_response": (
                "No encontré información relevante en el corpus documental "
                "para responder tu pregunta. Por favor, intenta reformularla "
                "con términos más específicos relacionados con inventario, "
                "ventas, facturación o gestión de tiendas de ropa."
            )
        }

    writer = WriterAgent(llm=state["llm"])
    result = writer.write(
        retrieved_docs=retrieved_docs,
        sources=sources,
        query=state["user_query"],
    )

    return {
        "final_response": result["final_response"],
    }


def should_continue(state: AgentState) -> str:
    docs = state.get("retrieved_docs", [])
    if docs:
        return "writer"
    return "skip"


def build_agent_graph(llm: Any, vector_store: Any) -> Any:
    workflow = StateGraph(AgentState)

    workflow.add_node("retriever", retriever_node)
    workflow.add_node("writer", writer_node)

    workflow.set_entry_point("retriever")

    workflow.add_conditional_edges(
        "retriever",
        should_continue,
        {
            "writer": "writer",
            "skip": END,
        },
    )

    workflow.add_edge("writer", END)

    graph = workflow.compile()

    logger.info("Grafo de agentes construido y compilado")
    return graph


def run_agent_graph(
    graph: Any,
    query: str,
    llm: Any,
    vector_store: Any,
) -> dict[str, Any]:
    initial_state: AgentState = {
        "user_query": query,
        "reformulated_queries": [],
        "retrieved_docs": [],
        "sources": [],
        "final_response": "",
        "llm": llm,
        "vector_store": vector_store,
    }

    result = graph.invoke(initial_state)

    logger.info(f"Grafo ejecutado. Respuesta generada: {len(result.get('final_response', ''))} chars")
    return result
