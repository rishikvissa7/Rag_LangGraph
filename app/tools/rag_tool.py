# File: app/services/tools/rag_tool.py
from langchain.tools import tool
from app.services.qdrant_search import search_collection

@tool
def rag_search_tool(collection: str, query: str):
    """Search in the specified company's collection using vector similarity"""
    return search_collection(collection, query)