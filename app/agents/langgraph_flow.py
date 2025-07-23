# app/agents/langgraph_flow.py
# LangGraph workflow execution

from app.services.postgres_storage import get_history, save_turn
from app.schemas.rag import RAGState
from app.workflow import workflow
from app.workflow.workflow import get_graph
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_context(history: list) -> str:
    """Convert history into a conversation string."""
    return "\n".join(f"{turn['role'].capitalize()}: {turn['content']}" for turn in history)

def run_rag_agent(question: str, user_id: str = "default_user", session_id: str = "default_session") -> str:
    state = {
        "question": question,
        "sub_questions": [],
        "companies": [],
        "tools": [],
        "answers": [],
        "final_answer": "",
        "session_id": session_id
    }

    app = workflow.compile()
    result = app.invoke(state, config={"configurable": {"user_id": user_id}})
    return result.get("final_answer", "Sorry, I couldn't find the answer.")
