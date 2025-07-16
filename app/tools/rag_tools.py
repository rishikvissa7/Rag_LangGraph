from langchain.tools import tool
from app.services.qdrant_search import search_collection

@tool
def google_rag_tool(query: str):
    """Search Google collection"""
    return search_collection("google", query)

@tool
def apple_rag_tool(query: str):
    """Search Apple collection"""
    return search_collection("apple", query)

@tool
def amazon_rag_tool(query: str):
    """Search Amazon collection"""
    return search_collection("amazon", query)

@tool
def microsoft_rag_tool(query: str):
    """Search Microsoft collection"""
    return search_collection("microsoft", query)

@tool
def meta_rag_tool(query: str):
    """Search Meta collection"""
    return search_collection("meta", query)
