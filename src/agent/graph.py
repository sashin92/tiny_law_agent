from typing import Any, Dict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END

from src.agent.state import GraphState
from src.agent.configuration import Configuration
from src.agent.graph_component import check_question, retrieval_node, call_rag_model


async def is_ragable(state: GraphState, config: RunnableConfig) -> Dict[str, Any]:
    if state.answer == "<answer>yes</answer>":
        return "rag"
    return "normal"


graph = (
    StateGraph(GraphState, config_schema=Configuration)
    .add_node(check_question)
    .add_node(retrieval_node)
    .add_node(call_rag_model)
    .add_edge(START, "check_question")
    .add_conditional_edges(
    "check_question",
    is_ragable,
    {
        "rag": "retrieval_node",
        "normal": END,
    }
    )
    .add_edge("retrieval_node", "call_rag_model")
    .add_edge("call_rag_model", END)
    .compile(name="Law Search Graph")
)

