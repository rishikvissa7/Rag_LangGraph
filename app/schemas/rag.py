# app/schemas/rag.py
# State schema for LangGraph

from typing import TypedDict, List

class RAGState(TypedDict):
    question: str
    sub_questions: List[str]
    companies: List[str]
    tools: List[str]
    answers: List[str]
    final_answer: str