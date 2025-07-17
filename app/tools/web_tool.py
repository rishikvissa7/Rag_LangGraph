import os
import requests
from langchain.tools import tool

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
SERPER_API_URL = "https://google.serper.dev/search"

@tool
def web_search_tool(query: str) -> str:
    """Search the web for the most recent and relevant information using Serper API"""
    if not SERPER_API_KEY:
        return "Serper API key not found. Please set SERPER_API_KEY environment variable."

    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }

    data = {
        "q": query
    }

    try:
        response = requests.post(SERPER_API_URL, headers=headers, json=data)
        response.raise_for_status()
        results = response.json()

        # Format top 3 search results
        answer = ""
        for idx, result in enumerate(results.get("organic", [])[:3], 1):
            title = result.get("title")
            link = result.get("link")
            snippet = result.get("snippet")
            answer += f"{idx}. {title}\n{snippet}\n{link}\n\n"

        return answer.strip() or "No relevant results found."
    
    except Exception as e:
        return f"Web search failed: {e}"
