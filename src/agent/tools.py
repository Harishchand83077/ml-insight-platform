"""
Tools the churn-analytics agent can call.

Prerequisite: the FastAPI serving app must already be running
(uvicorn src.serving.api:app --reload --port 8000) - predict_churn_tool
calls it over HTTP rather than importing the model directly, so the agent
talks to the same serving path a real client would. get_churn_rate_by_column
and get_customer_count talk to Postgres directly and only need the
postgres-ml container running.

get_churn_rate_by_column and get_customer_count are the "for now" stand-in
for a natural-language query tool: instead of letting the agent construct
raw SQL (a SQL injection risk we're deliberately not taking on yet), it
picks one of these two fixed, parameterized query functions and supplies
structured arguments. Column *names* can't be parameterized by the DB
driver the way values can, so they're checked against ALLOWED_COLUMNS - a
static allowlist mirroring the real customer_features schema - before
ever being interpolated into a query string. Filter *values* always go
through psycopg2 as query parameters, never string-interpolated.

query_project_docs_tool retrieves from the local Chroma vector store built
by build_knowledge_base.py (run that script first, and re-run it if
docs/glossary.md or docs/decisions.md change) - it only retrieves raw
context chunks, it does not answer the question itself; the agent's LLM
is responsible for synthesizing an answer from what comes back.
"""

import requests
from langchain_chroma import Chroma
from langchain_core.tools import tool

# Absolute import when loaded as part of the src package; src.common isn't
# a sibling of this file, so the fallback explicitly puts the project root
# on sys.path first - needed when this file's own directory is
# run/imported directly.
try:
    from src.common.db import connect_local
    from src.common.embedding_model import get_embedder
except ImportError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.common.db import connect_local
    from src.common.embedding_model import get_embedder

PREDICT_URL = "http://localhost:8000/predict"

CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "project_docs"
_vectorstore = None

# Mirrors customer_features' real columns (excluding id/customer_id) -
# update this if that table's schema changes.
ALLOWED_COLUMNS = {
    "tenure",
    "monthly_charges",
    "contract",
    "payment_method",
    "internet_service",
    "senior_citizen",
    "partner",
    "dependents",
    "num_addons_active",
    "total_logins_90d",
    "recent_30d_vs_older_60d_ratio",
    "avg_session_duration_recent_30d",
    "num_tickets",
    "pct_unresolved",
    "avg_resolution_time_hours",
    "avg_usage_count",
}


@tool
def predict_churn_tool(customer_id: str) -> str:
    """Look up a customer's churn risk by calling the churn-prediction API.

    Args:
        customer_id: The customer's ID, e.g. "7590-VHVEG".
    """
    try:
        response = requests.post(PREDICT_URL, json={"customer_id": customer_id}, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        return f"Error calling prediction API for customer_id '{customer_id}': {e}"

    data = response.json()
    return (
        f"customer_id: {data['customer_id']}\n"
        f"churn_probability: {data['churn_probability']}\n"
        f"prediction: {data['prediction']}\n"
        f"cache_hit: {data['cache_hit']}"
    )


@tool
def get_churn_rate_by_column(column_name: str) -> str:
    """Get the churn rate broken down by a customer_features column.

    Args:
        column_name: One of: tenure, monthly_charges, contract, payment_method,
            internet_service, senior_citizen, partner, dependents,
            num_addons_active, total_logins_90d, recent_30d_vs_older_60d_ratio,
            avg_session_duration_recent_30d, num_tickets, pct_unresolved,
            avg_resolution_time_hours, avg_usage_count.
    """
    if column_name not in ALLOWED_COLUMNS:
        return f"Error: '{column_name}' is not an allowed column. Allowed columns: {sorted(ALLOWED_COLUMNS)}"

    # column_name is safe to interpolate here only because it was just
    # checked against the static ALLOWED_COLUMNS allowlist above.
    query = f"""
        SELECT cf.{column_name} AS group_value,
               COUNT(*) AS customer_count,
               ROUND(AVG(CASE WHEN c.churn = 'Yes' THEN 1.0 ELSE 0 END) * 100, 2) AS churn_rate_pct
        FROM customer_features cf
        JOIN customers c ON cf.customer_id = c.customer_id
        GROUP BY cf.{column_name}
        ORDER BY cf.{column_name}
    """

    conn = connect_local()
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
    finally:
        conn.close()

    lines = [f"Churn rate by {column_name}:"]
    for group_value, customer_count, churn_rate_pct in rows:
        lines.append(f"  {group_value}: {churn_rate_pct}% churn ({customer_count} customers)")
    return "\n".join(lines)


@tool
def get_customer_count(filters: dict) -> str:
    """Count customers matching simple equality filters on customer_features.

    Args:
        filters: A dict of column_name -> value to filter on, e.g.
            {"contract": "Month-to-month", "senior_citizen": 1}. Pass an
            empty dict to get the total customer count. Allowed columns:
            tenure, monthly_charges, contract, payment_method,
            internet_service, senior_citizen, partner, dependents,
            num_addons_active, total_logins_90d, recent_30d_vs_older_60d_ratio,
            avg_session_duration_recent_30d, num_tickets, pct_unresolved,
            avg_resolution_time_hours, avg_usage_count.
    """
    bad_keys = [k for k in filters if k not in ALLOWED_COLUMNS]
    if bad_keys:
        return f"Error: filter columns {bad_keys} are not allowed. Allowed columns: {sorted(ALLOWED_COLUMNS)}"

    if filters:
        # column names are safe to interpolate here only because every key
        # was just checked against the static ALLOWED_COLUMNS allowlist;
        # values are always passed as query parameters, never interpolated.
        where_clause = " AND ".join(f"{col} = %s" for col in filters)
        query = f"SELECT COUNT(*) FROM customer_features WHERE {where_clause}"
        params = tuple(filters.values())
    else:
        query = "SELECT COUNT(*) FROM customer_features"
        params = ()

    conn = connect_local()
    try:
        with conn.cursor() as cur:
            cur.execute(query, params)
            count = cur.fetchone()[0]
    finally:
        conn.close()

    filter_desc = filters if filters else "no filters (all customers)"
    return f"Customer count for {filter_desc}: {count}"


def _get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=get_embedder(),
            persist_directory=CHROMA_DIR,
        )
    return _vectorstore


@tool
def query_project_docs_tool(question: str) -> str:
    """Retrieve relevant context about this project's methodology, metric
    definitions, or design decisions (e.g. why PR-AUC over accuracy, what
    num_addons_active means, the data leakage issue and fix). Returns raw
    context chunks for you to synthesize an answer from - it does not
    answer the question itself.

    Args:
        question: A natural-language question about the project.
    """
    vectorstore = _get_vectorstore()
    results = vectorstore.similarity_search(question, k=3)

    if not results:
        return "No relevant context found in the project docs."

    chunks = []
    for i, doc in enumerate(results, start=1):
        source = doc.metadata.get("source", "unknown")
        chunks.append(f"[Context {i} - source: {source}]\n{doc.page_content}")
    return "\n\n".join(chunks)
