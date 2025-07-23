# app/services/company_extractor.py

from langchain_litellm import ChatLiteLLM
import os
import json
from dotenv import load_dotenv

load_dotenv()

llm = ChatLiteLLM(
    model="gemini/gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY")
)

def extract_questions_and_companies(query: str):
    system_prompt = """
You are a helpful assistant. Break down the user's multi-part question into individual sub-questions, and identify the company for each.
Respond in JSON format like this:
[
  {"question": "sub-question 1", "company": "Company1"},
  {"question": "sub-question 2", "company": "Company2"}
]
If no company is present, set company to "".
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]

    try:
        response = llm.invoke(messages)
        return json.loads(response.content)
    except Exception as e:
        raise ValueError(f"Gemini parsing failed: {str(e)}")
