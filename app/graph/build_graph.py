from langgraph.graph import StateGraph, END

from app.graph.state import RAGState
from app.graph.nodes import (
    guardrail_node,
    route_after_guardrail,
    refuse_node,
    make_retrieve_node,
    make_generate_node,
)


def build_rag_graph(collection, settings):
    graph = StateGraph(RAGState)

    graph.add_node("guardrail", guardrail_node)
    graph.add_node("refuse", refuse_node)
    graph.add_node("retrieve", make_retrieve_node(collection, settings))
    graph.add_node("generate", make_generate_node(settings))

    graph.set_entry_point("guardrail")

    graph.add_conditional_edges(
        "guardrail",
        route_after_guardrail,
        {
            "retrieve": "retrieve",
            "refuse": "refuse",
        },
    )

    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    graph.add_edge("refuse", END)

    return graph.compile()