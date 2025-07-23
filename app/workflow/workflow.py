# app/workflow/workflow.py
# LangGraph workflow for multi-agent system with conversation state persistence

from langgraph.graph import StateGraph
from app.services.postgres_storage import get_session_history, save_session_checkpoint
from app.tools.tools import qdrant_search_tool, stock_price_tool, web_search_tool
from app.schemas.rag import RAGState
from litellm import completion
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def company_selector_node(state: RAGState) -> RAGState:
    session_id = state.get("session_id")
    prev_state = get_session_history(session_id) if session_id else {}

    # Build prompt input with previous final answer if available (for context)
    if prev_state and prev_state.get("final_answer"):
        user_question_with_context = (
            f"Previous answer: {prev_state['final_answer']}\nNew question: {state['question']}"
        )
    else:
        user_question_with_context = state["question"]

    system_prompt = (
        "You are an intelligent assistant that breaks down a multi-question user input. "
        "Your job is to extract a list of {question, company} pairs in JSON format.\n\n"
        "For example:\n"
        "Input: 'Tell me the stock price of Amazon and latest news about Qualcomm.'\n"
        "Output: [\n"
        "  {\"question\": \"What is the stock price of Amazon?\", \"company\": \"Amazon\"},\n"
        "  {\"question\": \"What are the latest news about Qualcomm?\", \"company\": \"Qualcomm\"}\n"
        "]\n\n"
        "Only respond with a valid JSON list. Do not include extra text or explanation."
    )

    try:
        response = completion(
            model="gemini/gemini-2.5-flash",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_question_with_context},
            ],
        )
        raw = response.choices[0].message.content.strip()
        logger.info(f"Gemini company split response: {raw}")

        parsed = json.loads(raw)
        sub_questions = [q["question"] for q in parsed]
        companies = [q["company"] for q in parsed]

        # If no company extracted and previous state has a last company, reuse it
        if not any(companies) and prev_state.get("companies"):
            companies = prev_state["companies"][-1:]  # use last company
            # Adjust sub_questions by appending company to them if missing
            sub_questions = [f"{sq} about {companies[0]}" if companies[0] not in sq else sq for sq in sub_questions]

        state["sub_questions"] = sub_questions
        state["companies"] = companies

        # Save progress after this node
        if session_id:
            save_session_checkpoint(session_id, state)

        return state

    except Exception as e:
        logger.error(f"[CompanySelector] Error: {str(e)}")
        raise ValueError("Gemini failed to extract sub-questions and companies.")


def tool_selector_node(state: RAGState) -> RAGState:
    session_id = state.get("session_id")

    state["tools"] = []
    state["answers"] = []

    from app.db.db import SessionLocal
    from app.models.models import Company

    db = SessionLocal()

    try:
        for sub_question, company in zip(state["sub_questions"], state["companies"]):
            tool = "web_search"  # default fallback
            try:
                system_prompt = (
                    "You are an assistant that selects the best tool to answer a question about a company. "
                    f"Question: {sub_question} "
                    f"Company: {company} "
                    "Available tools: "
                    "- qdrant_search: For general company information (e.g., CEO, founder). "
                    "- stock_details: For stock price information. "
                    "- web_search: For recent news or updates. "
                    "Return only the tool name as a string."
                )
                response = completion(
                    model="gemini/gemini-2.5-flash",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Question: {sub_question}, Company: {company}"},
                    ],
                )
                tool = response.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"Tool selection failed for {sub_question}: {str(e)}")

            state["tools"].append(tool)

            try:
                if tool == "qdrant_search":
                    answer = qdrant_search_tool.invoke({"question": sub_question, "company": company})
                elif tool == "stock_details":
                    # Use stock_price_tool with company name directly
                    answer = stock_price_tool.invoke({"company": company})
                elif tool == "web_search":
                    answer = web_search_tool.invoke({"question": f"{company} {sub_question}" if company else sub_question, "company": company})
                else:
                    answer = "Tool not recognized."
            except Exception as e:
                logger.error(f"Tool execution failed for {tool}: {str(e)}")
                answer = f"Error executing {tool}: {str(e)}"

            state["answers"].append(answer)

            # Save checkpoint after each tool call
            if session_id:
                save_session_checkpoint(session_id, state)

    finally:
        db.close()

    return state


def end_node(state: RAGState) -> RAGState:
    session_id = state.get("session_id")
    state["final_answer"] = " | ".join(state["answers"])
    if session_id:
        save_session_checkpoint(session_id, state)
    return state


def get_graph():
    builder = StateGraph(state_schema=RAGState)

    builder.add_node("company_selector", company_selector_node)
    builder.add_node("tool_selector", tool_selector_node)
    builder.add_node("end", end_node)

    builder.set_entry_point("company_selector")
    builder.add_edge("company_selector", "tool_selector")
    builder.add_edge("tool_selector", "end")

    return builder.compile()