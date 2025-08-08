from typing import Any, Dict

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import MessagesState

from src.agent.state import GraphState
from src.util.qdrant_handler import QdrantHandler
from src.agent.prompt import call_rag_system_prompt, check_question_system_prompt
from langchain_core.messages import AIMessage

async def end_node(state: GraphState, config: RunnableConfig) -> Dict[str, Any]:
    state["messages"] += [AIMessage(content=state["answer"])]
    return state

async def call_rag_model(state: GraphState, config: RunnableConfig) -> Dict[str, Any]:
    question = state["messages"][-1].content
    
    model = ChatOpenAI(
            temperature=0,
            model="gpt-4.1-mini"
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                call_rag_system_prompt
            ),
            (
                "human", "# Question: {question}"
            )
        ]
    )
    chain = prompt | model | StrOutputParser()
    response = await chain.ainvoke(
        {"question": question, "context": state["context"]}
    )
    state["answer"]=response
    return state

async def retrieval_node(state: GraphState, config: RunnableConfig) -> Dict[str, Any]:
    # TODO: Question으로 검색 후 결과값 context에 저장, state 리턴
    qdrant = QdrantHandler()
    
    results = await qdrant.search_text("law_collection", state["question"][0]["text"], limit=20)
    context_parts = []
    for i, result in enumerate(results, 1):
        source = result['payload'].get('source', 'N/A')
        page = result['payload'].get('page', 'N/A')
        content = result['payload']['text']
        formatted_str = f"<document><content>{content}</content><metadata>, page: {page + 1}, source: {source}</metadata></document>"
        context_parts.append(formatted_str)
    
    state["context"] = "\n\n".join(context_parts)
    return state

async def check_question(state: GraphState, config: RunnableConfig) -> Dict[str, Any]:
    question = state["messages"][-1].content
    state["question"] = question
    
    model = ChatOpenAI(
        temperature=0,
        model="gpt-4.1-mini"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                check_question_system_prompt
            ),
            (
                "human", "# Question: {question}"
            )
        ]
    )
    chain = prompt | model | StrOutputParser()
    response = await chain.ainvoke(
        {"question": question}
    )
    state["answer"]=response
    return state