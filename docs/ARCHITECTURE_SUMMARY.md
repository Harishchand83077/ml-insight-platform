# Architecture summary

## What it does

A churn-prediction service for a synthetic telecom customer base. An XGBoost
model scores each customer's churn risk and explains the score. A LangGraph
agent answers natural-language questions by calling seven tools, including
retrieval over the project's own policy documents with citations. A React
frontend talks to a JWT-authenticated FastAPI backend, deployed on Render with
Supabase Postgres, Upstash Redis, and Groq for the LLM. All data is synthetic.

## The seven agent tools

The agent has seven tools, not six. Each runs in-process.

1. **predict_churn_tool**: one customer's churn probability.
2. **explain_churn_tool**: the top five feature contributions to that score, in log-odds, from XGBoost's native `pred_contribs`.
3. **simulate_churn_tool**: re-scores a customer with validated account-field changes, such as a one-year contract.
4. **recommend_retention_tool**: one of six hand-written actions, chosen by the customer's largest risk factor. It's a rule table, not a learned policy.
5. **get_churn_rate_by_column**: churn rate by an allowlisted column. The agent never writes SQL.
6. **get_customer_count**: count of customers matching allowlisted filters.
7. **query_project_docs_tool**: vector search over the glossary, decision log, and policy documents, returning labeled chunks for citation.

## Hardest problems

**Out-of-memory on the 512 MB free tier.** The idle footprint was 749 MB at a
900 MB test, and the container was OOM-killed during startup at the real limit.
Diagnosis came from reproducing the limit locally with `docker run
--memory=512m --cpus=0.1`. Removing mlflow from the serving path (loading the
model with skops), and moving embedding calls off the event loop, brought the
worst-case test to 489.8 MiB, about 22 MB of headroom with no growth across
repeated calls. A later explainability feature added about 64 MB, and switching
to XGBoost's native contributions removed that cost.

**The same import-time bug in five places.** The Postgres pools in three
modules, the JWT secret check, and the Groq client each ran at module import.
Any test or CI job that imported the app then needed a live database and every
secret set. That broke CI. The fix was the same each time: move the work into a
getter that runs on first use and caches the result. A strict test patches
`psycopg2.connect` and the pool constructor to fail, and confirms the app
still imports with the environment empty. The CI simulation runs the unit suite
from a copy of the repo with no `.env` and an empty environment, and passes 42
of 42.

**Hybrid search rejected by measurement.** BM25 plus vector search with
reciprocal rank fusion was built and tested on a 17-question retrieval eval.
At 0.7 BM25 / 0.3 vector, exact-term hit@1 rose from 0.67 to 1.00, but
paraphrase hit@1 fell from 0.875 to 0.625. All three unanswerable questions
also scored above the suspicious-confidence threshold, against one of three for
vector search alone. Vector search stays live, and the hybrid module stays in the
repo, documented and unused.

## Key numbers

| Area | Number |
|---|---|
| Churn model (XGBoost) | F1 0.69, PR-AUC 0.78 (LogReg: 0.66, 0.77) |
| Memory, worst case | 489.8 MiB of a 512 MB limit |
| Memory, before fix | 749 MB idle at 900 MB; OOM at 512 MB |
| RAG, vector only | exact-term hit@1 0.67, paraphrase hit@1 0.875, hit@3 1.00 |
| Unit tests | 42, all external services mocked |
| Docker build check | the RAG index must contain at least 50 chunks; the build produced 128 |

## Known limits

Chat history is held in process memory, so it's lost on restart. The stack
runs as a single instance on the free tier. Ingestion, RabbitMQ, and Celery
retraining run locally only. Explanations and what-if results describe the
model's output, not causal effects. The retention rules are hand-written and
haven't been validated against outcomes.
