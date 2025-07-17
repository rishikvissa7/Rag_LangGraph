# File: app/services/tools/wiki_tool.py
from langchain.tools import tool
import wikipedia

@tool
def wiki_tool(topic: str):
    """Fetch summary from Wikipedia"""
    try:
        return wikipedia.summary(topic, sentences=3)
    except:
        return "No summary found on Wikipedia."