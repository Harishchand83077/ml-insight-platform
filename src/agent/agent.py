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
        explain_churn_tool,
        recommend_retention_tool,
        simulate_churn_tool,
        predict_churn_tool,
        query_project_docs_tool,
    )
except ImportError:
    from tools import (
        get_churn_rate_by_column,
        get_customer_count,
        explain_churn_tool,
        recommend_retention_tool,
        simulate_churn_tool,
        predict_churn_tool,
        query_project_docs_tool,
    )

load_dotenv()

SYSTEM_PROMPT = (
    "You are a customer analytics assistant. You have seven tools:\n"
    "- predict_churn_tool: looks up a specific customer's churn risk. Use "
    "this when the user asks about one customer by ID and wants a number.\n"
    "- explain_churn_tool: lists the features that moved a specific "
    "customer's predicted churn risk up or down, as statistical "
    "contributions from the trained model. Use this when the user asks WHY "
    "a customer is at risk, or what is driving their predicted risk, not "
    "just what it is. Call predict_churn_tool too if the user also wants "
    "the probability.\n"
    "- recommend_retention_tool: suggests one retention action for a specific "
    "at-risk customer, chosen by a fixed rule from that customer's largest "
    "risk factor. Use it when the user asks what should be done about a "
    "customer's risk. Say plainly that this is a simple rule-based "
    "suggestion, not a validated or optimized policy, and that it does not "
    "show the action would reduce churn unless the tool reports a projected "
    "change from the model. If the priority is low, say no action is needed "
    "rather than inventing one.\n"
    "- simulate_churn_tool: answers what-if questions about one customer by "
    "re-scoring them with some account fields changed, e.g. {\"contract\": "
    "\"Two year\"}. Use this for hypothetical questions ('what if they had "
    "a 2-year contract?'), not for the current risk. Report both the "
    "before and after probabilities and the direction the risk moved. "
    "State that this is the trained model's output with the inputs changed, "
    "not a causal forecast of what would happen to the customer. If the "
    "tool returns an error about an invalid field or value, relay that "
    "error to the user and do not guess a number.\n"
    "For explain_churn_tool results, follow these rules "
    "strictly: (1) state each contribution as a model output, e.g. 'a "
    "month-to-month contract increased the predicted risk by 0.86 "
    "log-odds'; (2) never say why a feature matters - no reasons, "
    "motivations, emotions, frustration, satisfaction, expectations, "
    "'essential', 'easy out', or 'resolved promptly'; (3) do not add facts "
    "about customer behavior or population churn rates that the tool output "
    "does not contain, unless you retrieved them with query_project_docs_tool "
    "and cite them.\n"
    "- get_churn_rate_by_column: returns the churn rate broken down by a "
    "customer_features column (e.g. contract, internet_service, "
    "payment_method). Use this for aggregate questions like 'what "
    "percentage of X customers churn?' or 'which contract type churns "
    "most?'.\n"
    "- get_customer_count: counts customers matching simple equality "
    "filters on customer_features. Use this for 'how many customers "
    "have/are X?' questions.\n"
    "- query_project_docs_tool: retrieves context from this project's own "
    "documentation - the glossary, the design-decision log, and Vantrix's "
    "(the fictional telecom company this project models) policy docs: "
    "refund/cancellation policy, contract terms and early termination "
    "fees, support SLA, fiber pricing/promotions, autopay/payment "
    "policy, retention playbooks, the churn model's model card, and the "
    "customer_features data dictionary. Use this for questions about "
    "project methodology, metric definitions (e.g. why PR-AUC over "
    "accuracy), design decisions (e.g. the data leakage issue, why both "
    "models were kept), or Vantrix policy questions (e.g. early "
    "termination fees, refund windows, SLA credits) - NOT for "
    "customer-specific queries, which the other three tools handle. This "
    "tool only returns raw context chunks, each labeled with a citation "
    "like \"[source: contract_terms.md > 2-Year Contract]\"; you must "
    "synthesize the actual answer yourself from what it returns, "
    "grounded in that context rather than general knowledge.\n"
    "When you answer using query_project_docs_tool's results, cite the "
    "source(s) you used in that same \"[source: file.md > Section]\" "
    "format so the user can see where the answer came from. If the tool "
    "returns \"No relevant context found in the project docs\" or "
    "nothing applicable to the question, say so explicitly to the user "
    "(e.g. \"I couldn't find anything in the project docs about that\") "
    "rather than answering generically from general knowledge.\n"
    "Always use the right tool to get real data rather than guessing or "
    "estimating a number yourself. Summarize results in plain language."
)

_agent = None


def get_agent():
    """Builds the LangGraph agent on first use and caches it. Built lazily so
    importing this module doesn't require GROQ_API_KEY to be set."""
    global _agent
    if _agent is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env before using the chat agent.")
        # Bounded so one slow LLM call can't hold a request for minutes. The
        # SDK default is 60 s read timeout with 2 retries; 30 s with 1 retry
        # keeps a worst-case turn well under a minute before /chat returns 503.
        model = ChatGroq(model="openai/gpt-oss-120b", api_key=api_key, timeout=30, max_retries=1)
        _agent = create_agent(
            model,
            tools=[predict_churn_tool, explain_churn_tool, simulate_churn_tool, recommend_retention_tool, get_churn_rate_by_column, get_customer_count, query_project_docs_tool],
            system_prompt=SYSTEM_PROMPT,
        )
    return _agent
