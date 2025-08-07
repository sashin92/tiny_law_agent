from dataclasses import dataclass
from langgraph.graph import MessagesState

@dataclass
class GraphState:
    question: str
    context: str = ""
    answer: str = ""
    
