"""
Tools the churn-analytics agent can call.

predict_churn_tool calls src.serving.prediction.predict_churn() directly,
in-process - not an HTTP request to this server's own /predict endpoint.
An earlier version did call over HTTP, which meant either exempting
/predict from end-user JWT auth for this one caller or minting the agent
process its own internal credential just to talk to itself; both are
more moving parts than a function call needs, and the HTTP round trip
was also quietly loading a second, independent copy of the model into
memory if it had ever gone through a path that didn't share
src.common.production_model's single loaded pipeline. get_churn_rate_by_column
and get_customer_count talk to Postgres directly and only need Supabase
(or a local Postgres) reachable.

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
docs/glossary.md, docs/decisions.md, or anything under
docs/knowledge_base/ changes) - it only retrieves raw context chunks,
labeled with a "source.md > Section" citation built from each chunk's
metadata, it does not answer the question itself; the agent's LLM is
responsible for synthesizing an answer from what comes back, citing
those labels per SYSTEM_PROMPT's instructions.

A BM25+vector hybrid retriever (src/agent/hybrid_retriever.py, weighted
Reciprocal Rank Fusion over the same chunks) was built and measured
against this plain vector search via tests/eval/rag_eval.py and is NOT
wired in here - see docs/decisions.md's "Hybrid search evaluated, not
adopted" entry for the numbers. In short: at its default 0.5/0.5 weight
it was strictly worse (no exact-term gain, worse paraphrase retrieval,
worse false-confidence on unanswerable questions); at 0.7 BM25/0.3
vector it did clearly fix exact-term hit@1 (0.67->1.0) but at the cost
of paraphrase hit@1 dropping 0.875->0.625 and EVERY unanswerable test
question scoring above the suspicious-confidence threshold (vs. 1/3 for
vector-only) - a regression against this tool's own "say so if nothing
relevant" contract. hybrid_retriever.py is kept, tested, and documented
in case a future corpus or question mix makes that tradeoff worth
revisiting, but plain vector search is what's actually live.
"""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.tools import tool

# Absolute import when loaded as part of the src package; src.common isn't
# a sibling of this file, so the fallback explicitly puts the project root
# on sys.path first - needed when this file's own directory is
# run/imported directly.
try:
    from src.common.db import get_shared_pool, run_read
    from src.common.embedding_model import get_embedder
    from src.serving.explain import explain_prediction
    from src.serving.prediction import predict_churn, recommend_retention_action, simulate_prediction
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.common.db import get_shared_pool, run_read
    from src.common.embedding_model import get_embedder
    from src.serving.explain import explain_prediction
    from src.serving.prediction import predict_churn, recommend_retention_action, simulate_prediction

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


SQL_STATEMENT_TIMEOUT = "5s"


def _run_bounded_read(query, params=()):
    """Runs a read-only tool query on the shared pool, with a statement timeout.
    SET LOCAL is used because it applies only to this transaction: the pool
    rolls the transaction back on return, so the timeout doesn't leak into the
    next query that reuses the pooled connection."""

    def _query(conn):
        with conn.cursor() as cur:
            cur.execute(f"SET LOCAL statement_timeout = '{SQL_STATEMENT_TIMEOUT}'")
            cur.execute(query, params or None)
            return cur.fetchall()

    return run_read(get_shared_pool(), _query)


@tool
def predict_churn_tool(customer_id: str) -> str:
    """Look up a customer's churn risk by running the trained model.

    Args:
        customer_id: The customer's ID, e.g. "7590-VHVEG".
    """
    try:
        result = predict_churn(customer_id)
    except Exception as e:
        # A tool that raises breaks the whole agent turn; returning a
        # string instead lets the LLM relay a clear failure to the user
        # (e.g. a transient Postgres/Redis error from get_customer_features -
        # RuntimeError from get_pipeline() shouldn't actually be reachable
        # here, since /chat already refuses to invoke the agent at all
        # until model_state["ready"] is True, which is only set after
        # set_pipeline() has run).
        return f"Error looking up customer_id '{customer_id}': {e}"

    if result is None:
        return f"Error: customer_id '{customer_id}' not found"

    return (
        f"customer_id: {result['customer_id']}\n"
        f"churn_probability: {result['churn_probability']}\n"
        f"prediction: {result['prediction']}\n"
        f"cache_hit: {result['cache_hit']}"
    )


@tool
def explain_churn_tool(customer_id: str) -> str:
    """List the features that moved a specific customer's predicted churn risk
    up or down the most, as statistical contributions from the trained model.
    These describe the model's output, not causes of customer behavior. Use
    this when asked why a customer is at risk, not just what their risk is.

    Args:
        customer_id: The customer's ID, e.g. "7590-VHVEG".
    """
    try:
        result = explain_prediction(customer_id)
    except Exception as e:
        return f"Error explaining customer_id '{customer_id}': {e}"

    if result is None:
        return f"Error: customer_id '{customer_id}' not found"

    lines = [
        f"customer_id: {result['customer_id']}",
        f"churn_probability: {result['churn_probability']}",
        "Contributions are log-odds units from the model: a positive value increased the predicted",
        "churn risk, a negative value decreased it. Baseline plus all contributions equals the model's score.",
        f"Baseline log-odds (average customer): {result['base_log_odds']}",
        "Top drivers, largest first:",
    ]
    for item in result["top_features"]:
        lines.append(
            f"- {item['feature']}={item['value']}: {item['log_odds']:+.4f} ({item['direction']})"
        )
    lines.append(
        "Report only these numbers and their direction. Do not give reasons, motivations, "
        "or emotional interpretations for any feature."
    )
    return "\n".join(lines)


@tool
def simulate_churn_tool(customer_id: str, overrides: dict) -> str:
    """Answer a what-if question about one customer: re-score their churn risk
    with some account fields changed, and report the before and after
    probabilities. Use this for hypothetical questions such as "what if they
    had a 2-year contract instead?". Changeable fields: contract ("Month-to-month",
    "One year", "Two year"), payment_method, internet_service ("DSL", "Fiber optic",
    "No"), partner and dependents ("Yes"/"No"), senior_citizen (0 or 1), tenure
    (0-72 months), monthly_charges (18.25-118.75), num_addons_active (0-6).

    Args:
        customer_id: The customer's ID, e.g. "2691-NZETQ".
        overrides: Field-to-new-value pairs, e.g. {"contract": "Two year"}.
    """
    try:
        result = simulate_prediction(customer_id, overrides)
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error simulating customer_id '{customer_id}': {e}"

    if result is None:
        return f"Error: customer_id '{customer_id}' not found"

    changes = ", ".join(f"{field}={value}" for field, value in result["overrides"].items())
    lines = [
        f"customer_id: {result['customer_id']}",
        f"what-if: {changes}",
        f"original_probability: {result['original_probability']}",
        f"modified_probability: {result['modified_probability']}",
        f"delta: {result['delta']:+.4f} (churn risk {result['direction']})",
        "This is the trained model's output with those inputs changed, not a causal forecast.",
    ]
    return "\n".join(lines)


@tool
def recommend_retention_tool(customer_id: str) -> str:
    """Suggest a retention action for one at-risk customer. This is a simple
    rule-based mapping from the customer's single largest risk-increasing
    factor to a fixed action, not a learned or optimized policy, and it has
    not been validated against outcomes. Use this when asked what should be
    done about a specific customer's churn risk.

    Args:
        customer_id: The customer's ID, e.g. "2691-NZETQ".
    """
    try:
        result = recommend_retention_action(customer_id)
    except Exception as e:
        return f"Error recommending for customer_id '{customer_id}': {e}"

    if result is None:
        return f"Error: customer_id '{customer_id}' not found"

    lines = [
        f"customer_id: {result['customer_id']}",
        f"churn_probability: {result['churn_probability']}",
        f"priority: {result['priority']}",
    ]
    factor = result["triggering_factor"]
    if factor is not None:
        lines.append(
            f"triggering_factor: {factor['feature']}={factor['value']} "
            f"(contributed {factor['log_odds']:+.4f} log-odds toward churn)"
        )
    if result["action"] is not None:
        lines.append(f"suggested_action: {result['action']}")
    if result["projected"] is not None:
        proj = result["projected"]
        changes = ", ".join(f"{k}={v}" for k, v in proj["overrides"].items())
        lines.append(
            f"projected_if_applied ({changes}): probability {proj['original_probability']} -> "
            f"{proj['modified_probability']} (delta {proj['delta']:+.4f}, model output, not a forecast)"
        )
    if result["note"] is not None:
        lines.append(f"note: {result['note']}")
    lines.append(
        "This is a rule-based suggestion from the top risk factor, not a validated or optimized policy."
    )
    return "\n".join(lines)


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

    try:
        rows = _run_bounded_read(query)
    except Exception as e:
        return f"Error computing the churn rate by {column_name}: {e}"

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

    try:
        count = _run_bounded_read(query, params)[0][0]
    except Exception as e:
        return f"Error counting customers: {e}"

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
    definitions, design decisions (e.g. why PR-AUC over accuracy, what
    num_addons_active means, the data leakage issue and fix), or about
    Vantrix's (the fictional telecom company this project models)
    policies - refunds/cancellation, contract terms and early termination
    fees, support SLA, fiber pricing/promotions, autopay/payment
    handling, retention playbooks, the churn model's model card, or the
    customer_features data dictionary. Returns raw context chunks for you
    to synthesize an answer from, each labeled with its source file and
    section - it does not answer the question itself.

    Args:
        question: A natural-language question about the project or about
            Vantrix's policies.
    """
    vectorstore = _get_vectorstore()
    results = vectorstore.similarity_search(question, k=3)

    if not results:
        return "No relevant context found in the project docs."

    chunks = []
    for doc in results:
        source = Path(doc.metadata.get("source", "unknown")).name
        section = doc.metadata.get("section")
        label = f"{source} > {section}" if section else source
        chunks.append(f"[source: {label}]\n{doc.page_content}")
    return "\n\n".join(chunks)
