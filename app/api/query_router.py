from fastapi import APIRouter, Query
from app.agents.langgraph_flow import run_rag_agent

router = APIRouter()

@router.get("/ask")
async def ask(question: str = Query(...), user_id: int = 0):
    answer = run_rag_agent(question, user_id)
    return {"question": question, "answer": answer}
