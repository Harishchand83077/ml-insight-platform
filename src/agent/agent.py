"""
Churn-analytics agent: ChatGroq (openai/gpt-oss-120b) with four tools
bound to it, via LangChain's create_agent - the current LangGraph-based
tool-calling agent. (create_tool_calling_agent + AgentExecutor were
removed in LangChain 1.0; create_agent is the replacement.)

Model note: llama-3.3-70b-versatile is no longer available on Groq for
this API key (Groq's lineup has moved on) - openai/gpt-oss-120b is the
largest general-purpose chat model currently available on this key.

Prerequisites:
- FastAPI serving app running on localhost:8000
  (uvicorn src.serving.api:app --reload --port 8000) - needed for
  predict_churn_tool, which calls it over HTTP.
- postgres-ml container running - needed for get_churn_rate_by_column and
  get_customer_count.
- python src/agent/build_knowledge_base.py already run at least once -
  needed for query_project_docs_tool, which reads from data/chroma_db/.
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_groq import ChatGroq

# Absolute import when loaded as part of the src package (e.g. by
# src/serving/api.py via `uvicorn src.serving.api:app`), same-directory
# import when this file's own directory is run/imported directly (e.g.
# `python src/agent/test_agent.py`, which does `from agent import agent`).
try:
    from src.agent.tools import (
        get_churn_rate_by_column,
        get_customer_count,
        predict_churn_tool,
        query_project_docs_tool,
    )
except ImportError:
    from tools import (
        get_churn_rate_by_column,
        get_customer_count,
        predict_churn_tool,
        query_project_docs_tool,
    )

load_dotenv()

SYSTEM_PROMPT = (
    "You are a customer analytics assistant. You have four tools:\n"
    "- predict_churn_tool: looks up a specific customer's churn risk. Use "
    "this when the user asks about one customer by ID.\n"
    "- get_churn_rate_by_column: returns the churn rate broken down by a "
    "customer_features column (e.g. contract, internet_service, "
    "payment_method). Use this for aggregate questions like 'what "
    "percentage of X customers churn?' or 'which contract type churns "
    "most?'.\n"
    "- get_customer_count: counts customers matching simple equality "
    "filters on customer_features. Use this for 'how many customers "
    "have/are X?' questions.\n"
    "- query_project_docs_tool: retrieves context from this project's own "
    "documentation (glossary and design-decision log). Use this for "
    "questions about project methodology, metric definitions (e.g. why "
    "PR-AUC over accuracy), or design decisions (e.g. the data leakage "
    "issue, why both models were kept) - NOT for customer-specific "
    "queries, which the other three tools handle. This tool only returns "
    "raw context chunks; you must synthesize the actual answer yourself "
    "from what it returns, grounded in that context rather than general "
    "knowledge.\n"
    "Always use the right tool to get real data rather than guessing or "
    "estimating a number yourself. Summarize results in plain language."
)

model = ChatGroq(model="openai/gpt-oss-120b", api_key=os.environ["GROQ_API_KEY"])

agent = create_agent(
    model,
    tools=[predict_churn_tool, get_churn_rate_by_column, get_customer_count, query_project_docs_tool],
    system_prompt=SYSTEM_PROMPT,
)
