import json
import re
import logging
from fastapi import APIRouter, HTTPException
from app.tools.tools import qdrant_search_tool, web_search_tool, stock_price_tool
from app.services.llm_config import llm
from app.services.postgres_storage import save_turn

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/ask")
async def ask_question(question: str, user_id: str = "default_user", session_id: str = "default_session"):
    try:
        logger.info(f"Received question: {question}")

        # 1. Ask LLM to split compound question
        split_prompt = f"""
Split the following question into multiple clear sub-questions. Return as a JSON list of strings.

Question: "{question}"
Example output: ["Who is the CEO of Tesla?", "What is the stock price of Amazon?"]
Only return the JSON array.
"""
        split_response = llm.invoke(split_prompt)
        raw_response = split_response.content.strip()
        logger.info(f"[Split Response] {raw_response}")

        clean_json_str = re.sub(r"^```(?:json)?\n|\n```$", "", raw_response.strip(), flags=re.IGNORECASE)
        sub_questions = json.loads(clean_json_str)

        final_answers = []

        for sub_q in sub_questions:
            logger.info(f"\n🔍 Processing sub-question: {sub_q}")
            await save_turn(user_id=user_id, role="user", content=sub_q, session_id=session_id)

            # Extract company using LLM
            company_prompt = f"Extract the company name from: '{sub_q}'. Return only the name or empty string."
            company_resp = llm.invoke(company_prompt)
            company = company_resp.content.strip()
            logger.info(f"[Company Extraction] -> '{company}'")

            # Detect intent
            intent_prompt = f"""Classify the following question:
- If about stock price, return 'stock'.
- If about leadership/services/policies, return 'rag'.
- If about general info/news, return 'web'.

Question: '{sub_q}'"""
            intent_resp = llm.invoke(intent_prompt)
            intent = intent_resp.content.strip().lower()

            logger.info(f"[Routing] Sub-question: '{sub_q}' | Company: '{company}' | Intent: '{intent}'")

            # Route based on detected intent
            if intent == "stock":
                if not company:
                    answer = "No company found for stock query."
                else:
                    result = stock_price_tool.invoke({"company": company})
                    if "error" in result:
                        answer = f"❌ {result['error']}"
                    else:
                        answer = f"The stock price of {company} ({result['ticker']}) is {result['price']} {result['currency']}."

            elif intent == "rag":
                answer = qdrant_search_tool.invoke({
                    "question": sub_q,
                    "company": company,
                    "user_id": user_id,
                    "session_id": session_id
                })
                if not answer or not answer.strip():
                    logger.info("[RAG fallback] No RAG result, using web.")
                    answer = web_search_tool.invoke({
                        "question": sub_q,
                        "company": company,
                        "user_id": user_id,
                        "session_id": session_id
                    })

            else:  # default to web
                answer = web_search_tool.invoke({"question": sub_q, "company": company})

            await save_turn(user_id=user_id, role="assistant", content=answer, session_id=session_id)
            final_answers.append(f"Q: {sub_q}\nA: {answer}")

        return {
            "question": question,
            "answer": "\n\n".join(final_answers)
        }

    except Exception as e:
        logger.exception("❌ Failed to process multi-question input")
        raise HTTPException(status_code=500, detail="Failed to process your question.")
