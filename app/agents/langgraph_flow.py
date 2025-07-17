from langgraph.graph import StateGraph
from app.services.litellm_llm import get_litellm
from app.services.db import save_query_history, save_checkpoint
from langchain.agents import initialize_agent, Tool, AgentType
from app.tools.rag_tool import rag_search_tool
from app.tools.web_tool import web_search_tool
from app.tools.stock_tool import stock_trend_tool
from app.tools.wiki_tool import wiki_tool

COMPANIES = ["google", "apple", "amazon", "microsoft", "meta"]

def detect_company_from_question(question: str) -> str:
    lowered = question.lower()
    for company in COMPANIES:
        if company in lowered:
            return company
    return "google"  # fallback if nothing matches

def run_company_agent(company: str, state: dict):
    question = state["question"]
    user_id = state.get("user_id", 0)

    tools = [
        Tool.from_function(
            lambda q: rag_search_tool(company, q),
            name="CompanyRAG",
            description="Search Qdrant vector collection for company-specific info"
        ),
        Tool.from_function(
            web_search_tool,
            name="WebSearch",
            description="Search the web for recent and real-time information"
        ),
        Tool.from_function(
            stock_trend_tool,
            name="StockInfo",
            description="Fetch current stock data for the company"
        ),
        Tool.from_function(
            wiki_tool,
            name="Wikipedia",
            description="Get a summary of any topic from Wikipedia"
        ),
    ]

    llm = get_litellm()
    agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS)

    save_checkpoint("question_received", {"question": question}, user_id)
    answer = agent.run(question)
    save_query_history(question, answer, user_id)
    save_checkpoint("generated_answer", {"answer": answer}, user_id)

    return {"question": question, "response": answer}

def end_node(state: dict):
    return {"final_answer": state["response"]}

def get_rag_graph():
    builder = StateGraph(dict)

    for company in COMPANIES:
        builder.add_node(f"{company}_node", lambda s, c=company: run_company_agent(c, s))
    
    builder.add_node("end", end_node)

    for company in COMPANIES:
        builder.add_edge(f"{company}_node", "end")

    builder.set_entry_point("google_node")  # dummy entry point (overridden later)
    return builder.compile()

def run_rag_agent(question: str, user_id: int = 0):
    detected_company = detect_company_from_question(question)
    graph = get_rag_graph()
    entry_node = f"{detected_company}_node"
    result = graph.invoke({"question": question, "user_id": user_id}, config={"entry_point": entry_node})
    return result["final_answer"]
