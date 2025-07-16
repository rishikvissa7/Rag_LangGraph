# app/services/langgraph_builder.py

from langgraph.graph import StateGraph, END
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType

from app.tools.rag_tools import (
    google_rag_tool,
    apple_rag_tool,
    amazon_rag_tool,
    microsoft_rag_tool,
    meta_rag_tool,
)

from app.services.litellm_llm import get_litellm

# Initialize the LLM
llm = get_litellm()

# Tools for the agent
tools = [
    Tool.from_function(google_rag_tool, name="google_rag_tool", description="Search Google collection"),
    Tool.from_function(apple_rag_tool, name="apple_rag_tool", description="Search Apple collection"),
    Tool.from_function(amazon_rag_tool, name="amazon_rag_tool", description="Search Amazon collection"),
    Tool.from_function(microsoft_rag_tool, name="microsoft_rag_tool", description="Search Microsoft collection"),
    Tool.from_function(meta_rag_tool, name="meta_rag_tool", description="Search Meta collection"),
]

# Main RAG Agent
rag_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
)

# LangGraph node: agent
def agent_node(state: dict):
    question = state["question"]
    response = rag_agent.run(question)
    return {"question": question, "response": response}

# LangGraph node: end
def end_node(state: dict):
    return {"final_answer": state["response"]}

# Define LangGraph graph
def get_rag_graph():
    builder = StateGraph(dict)  # ✅ Specify dict as state type
    builder.add_node("main_agent", agent_node)
    builder.set_entry_point("main_agent")
    builder.add_node("end", end_node)
    builder.add_edge("main_agent", "end")
    return builder.compile()

# Run the RAG pipeline
def run_rag_agent(question: str):
    graph = get_rag_graph()
    result = graph.invoke({"question": question})
    return result["final_answer"]
