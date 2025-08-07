from typing import Any, Dict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END, MessagesState

from src.agent.state import GraphState
from src.agent.configuration import Configuration
from src.agent.graph_component import check_question, retrieval_node, call_rag_model, start_node, end_node


async def is_ragable(state: GraphState, config: RunnableConfig) -> str:
    if state["answer"] == "<answer>yes</answer>":
        return "rag"
    return "normal"



graph = (
    StateGraph(GraphState, input_schema=MessagesState)
    .add_node(start_node)
    .add_node(check_question)
    .add_node(retrieval_node)
    .add_node(call_rag_model)
    .add_node(end_node)
    .add_edge(START, "start_node")
    .add_edge("start_node", "check_question")
    .add_conditional_edges(
    "check_question",
    is_ragable,
    {
        "rag": "retrieval_node",
        "normal": "end_node",
    }
    )
    .add_edge("retrieval_node", "call_rag_model")
    .add_edge("call_rag_model", "end_node")
    .add_edge("end_node", END)
    .compile(
        name="Law Search Graph"
    )
)

