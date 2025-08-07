from dataclasses import dataclass
from langgraph.graph import MessagesState


class GraphState(MessagesState):
    question: str
    context: str = ""
    answer: str = ""
    
