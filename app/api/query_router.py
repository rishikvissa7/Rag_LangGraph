from fastapi import APIRouter, Query
from app.agents.langgraph_flow import run_rag_agent

router = APIRouter()

@router.get("/ask")
async def ask(question: str = Query(...)):
    answer = run_rag_agent(question)
    return {"question": question, "answer": answer}
