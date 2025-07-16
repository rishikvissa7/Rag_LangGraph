from langgraph.graph import StateGraph
from app.services.litellm_llm import get_litellm
from app.services.db import save_query_history, save_checkpoint, get_similar_history, get_recent_history
from app.tools.rag_tools import (
    google_rag_tool, apple_rag_tool, amazon_rag_tool, microsoft_rag_tool, meta_rag_tool,
)
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType

llm = get_litellm()
tools = [
    Tool.from_function(google_rag_tool, name="google_rag_tool", description="Search Google collection"),
    Tool.from_function(apple_rag_tool, name="apple_rag_tool", description="Search Apple collection"),
    Tool.from_function(amazon_rag_tool, name="amazon_rag_tool", description="Search Amazon collection"),
    Tool.from_function(microsoft_rag_tool, name="microsoft_rag_tool", description="Search Microsoft collection"),
    Tool.from_function(meta_rag_tool, name="meta_rag_tool", description="Search Meta collection"),
]
rag_agent = initialize_agent(tools=tools, llm=llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)

def agent_node(state: dict):
    question = state["question"]
    user_id = state.get("user_id", 0)

    save_checkpoint("question_received", {"question": question}, user_id)

    response = rag_agent.run(question)

    save_query_history(question, response, user_id)
    save_checkpoint("generated_answer", {"answer": response}, user_id)

    return {"question": question, "response": response}

def end_node(state: dict):
    return {"final_answer": state["response"]}

def get_rag_graph():
    builder = StateGraph(dict)
    builder.add_node("main_agent", agent_node)
    builder.set_entry_point("main_agent")
    builder.add_node("end", end_node)
    builder.add_edge("main_agent", "end")
    return builder.compile()

def run_rag_agent(question: str, user_id: int = 0):
    similar = get_similar_history(question, user_id)
    if similar:
        return f"(From memory)\n{similar}"

    history = get_recent_history(user_id)
    history_prompt = "\n".join([f"Q: {q}\nA: {a}" for q, a in history])
    full_question = f"{history_prompt}\n\nQ: {question}" if history else question

    graph = get_rag_graph()
    result = graph.invoke({"question": full_question, "user_id": user_id})
    return result["final_answer"]
