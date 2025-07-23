# app/tools/tools.py
# LangChain tools for handling stock, Qdrant, and web searches

from langchain_core.tools import tool
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import yfinance as yf
import requests
import os
from dotenv import load_dotenv
import logging
from typing import Dict
import time
from bs4 import BeautifulSoup

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize SentenceTransformer
embedder = SentenceTransformer("all-MiniLM-L6-v2")
logger.info("SentenceTransformer initialized successfully")


@tool
def stock_price_tool(company: str) -> str:
    """Get the live stock price of a company using Yahoo Finance web scraping."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }

        # Step 1: Search for the company on Yahoo Finance
        search_url = f"https://finance.yahoo.com/lookup?s={company}"
        search_resp = requests.get(search_url, headers=headers)
        soup = BeautifulSoup(search_resp.text, "html.parser")

        # Find the first stock ticker link
        link = soup.select_one("table tbody tr td a")
        if not link:
            raise Exception("Couldn't find a ticker for the company.")

        ticker = link.text.strip()

        # Step 2: Go to the ticker's stock page
        stock_url = f"https://finance.yahoo.com/quote/{ticker}"
        stock_resp = requests.get(stock_url, headers=headers)
        stock_soup = BeautifulSoup(stock_resp.text, "html.parser")

        # Extract the current price
        price_span = stock_soup.select_one('fin-streamer[data-field="regularMarketPrice"]')
        if not price_span:
            raise Exception("Couldn't find the stock price on Yahoo Finance page.")

        price = price_span.text.strip()
        return f"The current stock price of {company} ({ticker}) is ${price}."

    except Exception as e:
        raise Exception(f"Yahoo Finance scraping error: {e}")
@tool
def qdrant_search_tool(question: str, company: str) -> str:
    """Search Qdrant company_info collection for company-related information."""
    try:
        # Initialize Qdrant client
        qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        query_vector = embedder.encode([question])[0].tolist()
        search_result = qdrant_client.search(
            collection_name="company_info",
            query_vector=query_vector,
            query_filter={
                "must": [
                    {"key": "company", "match": {"value": company}}
                ]
            },
            limit=1
        )
        if search_result:
            answer = search_result[0].payload.get("info", "")
            logger.info(f"Found answer in Qdrant for company {company}: {answer[:50]}...")
            return answer
        logger.info(f"No results found in Qdrant for company {company}")
        return ""
    except Exception as e:
        logger.error(f"Qdrant search failed: {str(e)}")
        raise Exception(f"Qdrant search failed: {str(e)}")

@tool
def web_search_tool(question: str, company: str = "") -> str:
    """Search the web using Serper API."""
    try:
        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": os.getenv("SERPER_API_KEY"),
            "Content-Type": "application/json"
        }
        query = f"{company} {question}" if company else question
        payload = {"q": query}
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        results = response.json().get("organic", [])
        if results:
            for result in results:
                snippet = result.get("snippet", "")
                if snippet:
                    logger.info(f"Found web answer for {query}: {snippet[:50]}...")
                    return snippet
        logger.info(f"No web results found for {query}")
        return "No relevant information found on the web."
    except Exception as e:
        logger.error(f"Web search failed: {str(e)}")
        raise Exception(f"Web search failed: {str(e)}")

# List of available tools (add new tools here)
TOOLS = [stock_price_tool, qdrant_search_tool, web_search_tool]