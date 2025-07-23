# app/services/llm_config.py

import os
from dotenv import load_dotenv
from langchain_litellm import ChatLiteLLM
from app.tools.tools import TOOLS

load_dotenv()

llm = ChatLiteLLM(
    model="gemini/gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY")
)

llm_with_tools = llm.bind_tools(TOOLS)
