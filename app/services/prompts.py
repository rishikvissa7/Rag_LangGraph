# app/services/prompts.py

from langchain_core.prompts import ChatPromptTemplate

tool_selector_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are an assistant that decides the best tool to answer questions about companies. "
     "You have access to tools like:\n"
     "- stock_price_tool: For stock prices.\n"
     "- qdrant_search_tool: For company facts (e.g., founder, CEO).\n"
     "- web_search_tool: For recent news or if others fail.\n"
     "Return the best tool for the job."),
    ("user", "{question}")
])
