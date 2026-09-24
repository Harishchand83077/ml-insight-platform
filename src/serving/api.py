"""
Churn-prediction API: GET /health, POST /predict, POST /chat,
DELETE /chat/{session_id}, and GET /metrics (Prometheus). The XGBoost
model (the full preprocessing + classifier pipeline, from its MLflow run
artifact) is loaded once at startup, not per-request. The LangChain agent
(src/agent/agent.py) is imported once at module load for the same reason.

/metrics gets the standard HTTP metrics (request count, latency
histograms, status codes per endpoint) for free from
prometheus-fastapi-instrumentator. Beyond that, three app-specific
metrics actually show what's interesting about this service's behavior:
feature_cache_requests_total (feature_cache.py), semantic_cache_hits_total
(semantic_cache.py), and agent_response_seconds (below) - raw HTTP
latency on /chat wouldn't distinguish a semantic-cache-hit response
(near-instant) from a real agent call (LLM + tool round trips).
"""

import logging
from contextlib import asynccontextmanager

import groq
import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from src.agent.agent import agent as churn_agent
from src.agent.semantic_cache import check_semantic_cache, store_in_semantic_cache
from src.models.train_baseline import prepare_model_input
from src.serving.feature_cache import get_customer_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("api")

model_state = {}


PRODUCTION_MODEL_DIR = "models/production_model"


def load_production_model():
    """Loads the XGBoost pipeline directly from models/production_model/ -
    a plain local MLflow model directory produced by
    scripts/export_model_for_deployment.py - instead of querying the
    MLflow tracking store (sqlite:///mlruns.db) at startup. That store is
    dev-time experiment history and isn't shipped with the deployed image;
    to ship a new model, re-run the export script and commit the updated
    models/production_model/."""
    pipeline = mlflow.sklearn.load_model(PRODUCTION_MODEL_DIR)
    logger.info("Loaded xgboost model from %s", PRODUCTION_MODEL_DIR)
    return pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_state["pipeline"] = load_production_model()
    yield
    model_state.clear()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ml-insight-platform.vercel.app",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)  # GET /metrics

AGENT_RESPONSE_TIME = Histogram(
    "agent_response_seconds",
    "Time spent in churn_agent.invoke() - the LLM + tool-call round trip for a real "
    "(non-cached) /chat response, separate from raw HTTP request latency",
)

# Groq's SDK maps HTTP 429 to groq.RateLimitError, but under load we've also
# observed 413 ("Request too large ... tokens per minute (TPM)") for the same
# underlying cause - Groq's own quota code for that response is still
# "rate_limit_exceeded". 413 isn't one of groq._exceptions' explicitly mapped
# status codes, so it comes back as the generic groq.APIStatusError rather
# than RateLimitError; catching APIStatusError and checking status_code
# catches both.
GROQ_RATE_LIMIT_STATUS_CODES = {429, 413}

# Session_id -> full LangChain message list for that conversation. In-memory
# only: this is deliberately simple for now - it won't survive a server
# restart (--reload will wipe it on every code change too) and isn't safe
# across multiple server instances/workers, since each process has its own
# dict. A real deployment would back this with Redis or a proper session
# store instead.
chat_sessions: dict[str, list] = {}


class PredictRequest(BaseModel):
    customer_id: str


class ChatRequest(BaseModel):
    session_id: str
    message: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    features, cache_hit = get_customer_features(req.customer_id)
    if features is None:
        raise HTTPException(status_code=404, detail=f"customer_id '{req.customer_id}' not found")

    X = prepare_model_input(features)
    churn_probability = float(model_state["pipeline"].predict_proba(X)[0, 1])
    prediction = "Yes" if churn_probability >= 0.5 else "No"

    return {
        "customer_id": req.customer_id,
        "churn_probability": round(churn_probability, 4),
        "prediction": prediction,
        "cache_hit": cache_hit,
    }


@app.post("/chat")
def chat(req: ChatRequest):
    previous_messages = chat_sessions.get(req.session_id, [])
    is_first_message = not previous_messages

    # Semantic cache only applies to the first message of a session - the
    # same question mid-conversation can mean something different given
    # prior context, so it's only safe to short-circuit on a fresh session.
    if is_first_message:
        cached = check_semantic_cache(req.message)
        if cached is not None:
            chat_sessions[req.session_id] = [
                {"role": "user", "content": req.message},
                {"role": "assistant", "content": cached["response"]},
            ]
            return {
                "session_id": req.session_id,
                "response": cached["response"],
                "tool_calls": cached["tool_calls"],
            }

    input_messages = previous_messages + [{"role": "user", "content": req.message}]

    try:
        with AGENT_RESPONSE_TIME.time():
            result = churn_agent.invoke({"messages": input_messages})
    except groq.APIStatusError as e:
        if e.status_code in GROQ_RATE_LIMIT_STATUS_CODES:
            logger.warning("Groq rate-limited this request (status %s): %s", e.status_code, e.message)
            return JSONResponse(
                status_code=503,
                content={"error": "Upstream LLM provider rate-limited, please retry shortly"},
            )
        raise

    chat_sessions[req.session_id] = result["messages"]

    # only the messages generated by this turn (not prior turns already
    # reported to the caller before), so tool_calls reflects just this call
    new_messages = result["messages"][len(input_messages):]

    tool_calls = [
        {"tool": call["name"], "args": call["args"]}
        for msg in new_messages
        for call in (getattr(msg, "tool_calls", None) or [])
    ]

    response_text = new_messages[-1].content if new_messages else ""

    if is_first_message:
        store_in_semantic_cache(req.message, response_text, tool_calls)

    return {
        "session_id": req.session_id,
        "response": response_text,
        "tool_calls": tool_calls,
    }


@app.delete("/chat/{session_id}")
def clear_chat(session_id: str):
    existed = chat_sessions.pop(session_id, None) is not None
    return {"session_id": session_id, "cleared": existed}
