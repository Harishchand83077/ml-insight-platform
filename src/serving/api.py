"""
Churn-prediction API: GET /health, POST /auth/signup, POST /auth/login,
POST /predict, POST /chat, DELETE /chat/{session_id}, and GET /metrics
(Prometheus). The XGBoost model (the full preprocessing + classifier
pipeline, from its MLflow run artifact) is loaded once at startup, not
per-request. The LangChain agent (src/agent/agent.py) is imported once
at module load for the same reason.

/predict and /chat require a valid JWT (Authorization: Bearer <token>,
obtained from /auth/signup or /auth/login) - see src/serving/auth.py for
password hashing, token issuance/verification, and the get_current_user
dependency. Every call to /predict and /chat is recorded in the
audit_logs table (auth.log_audit), tied to the authenticated user.

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
import torch
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_huggingface import HuggingFaceEmbeddings
from prometheus_client import Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, constr

from src.agent.agent import agent as churn_agent
from src.agent.semantic_cache import check_semantic_cache, store_in_semantic_cache
from src.common.embedding_model import EMBEDDING_MODEL_NAME, set_embedder
from src.models.train_baseline import prepare_model_input
from src.serving.auth import (
    EmailAlreadyExistsError,
    create_access_token,
    create_user,
    get_current_user,
    get_user_by_email,
    hash_password,
    log_audit,
    verify_password,
)
from src.serving.feature_cache import get_customer_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("api")

# Limiting torch's intra-op thread pool matters even on CPU-only torch (see
# Dockerfile): unconstrained, it defaults to one thread per core, and on a
# memory/CPU-constrained free-tier instance that contention adds to the
# same pressure that's been crashing the instance, not just raw model
# memory. Set before any model (xgboost's torch-free, but the embedding
# model is not) is loaded.
torch.set_num_threads(1)

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


def load_embedder():
    """Loads the one shared sentence-transformers embedding model instance
    - see src/common/embedding_model.py for why this must happen exactly
    once, here, rather than lazily in the RAG tool or semantic cache."""
    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    logger.info("Loaded embedding model %s", EMBEDDING_MODEL_NAME)
    return embedder


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_state["pipeline"] = load_production_model()
    model_state["embedder"] = load_embedder()
    set_embedder(model_state["embedder"])
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

# (user_id, session_id) -> full LangChain message list for that
# conversation - keyed by user as well as session_id so one authenticated
# user can never read or continue another user's session, even if they
# guessed or reused the same session_id. In-memory only: this is
# deliberately simple for now - it won't survive a server restart
# (--reload will wipe it on every code change too) and isn't safe across
# multiple server instances/workers, since each process has its own dict.
# A real deployment would back this with Redis or a proper session store
# instead.
chat_sessions: dict[tuple[str, str], list] = {}


class SignupRequest(BaseModel):
    email: str
    password: constr(min_length=8, max_length=72)  # bcrypt's own input limit is 72 bytes


class LoginRequest(BaseModel):
    email: str
    password: str


class PredictRequest(BaseModel):
    customer_id: str


class ChatRequest(BaseModel):
    session_id: str
    message: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/auth/signup", status_code=201)
def signup(req: SignupRequest):
    if get_user_by_email(req.email) is not None:
        raise HTTPException(status_code=409, detail="email already registered")

    try:
        user = create_user(req.email, hash_password(req.password))
    except EmailAlreadyExistsError:
        # the pre-check above already covers the common case; this catches
        # the race where two signups for the same email land concurrently
        raise HTTPException(status_code=409, detail="email already registered") from None

    token = create_access_token(user["id"], user["email"])
    return {"access_token": token, "token_type": "bearer"}


@app.post("/auth/login")
def login(req: LoginRequest):
    user = get_user_by_email(req.email)
    # Same generic error whether the email doesn't exist or the password is
    # wrong - not revealing which, so a login attempt can't be used to
    # enumerate registered emails.
    if user is None or not verify_password(req.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="invalid email or password")

    token = create_access_token(user["id"], user["email"])
    return {"access_token": token, "token_type": "bearer"}


@app.post("/predict")
def predict(req: PredictRequest, current_user: dict = Depends(get_current_user)):
    log_audit(current_user["id"], "/predict", req.customer_id)

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
def chat(req: ChatRequest, current_user: dict = Depends(get_current_user)):
    log_audit(current_user["id"], "/chat", req.message)

    session_key = (current_user["id"], req.session_id)
    previous_messages = chat_sessions.get(session_key, [])
    is_first_message = not previous_messages

    # Semantic cache only applies to the first message of a session - the
    # same question mid-conversation can mean something different given
    # prior context, so it's only safe to short-circuit on a fresh session.
    if is_first_message:
        cached = check_semantic_cache(req.message)
        if cached is not None:
            chat_sessions[session_key] = [
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

    chat_sessions[session_key] = result["messages"]

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
def clear_chat(session_id: str, current_user: dict = Depends(get_current_user)):
    # Protected (not explicitly requested, but required for correctness):
    # chat_sessions is keyed by (user_id, session_id) now, so this needs
    # current_user to build the same key - and as a side effect, one user
    # can no longer clear another user's session by guessing its id.
    existed = chat_sessions.pop((current_user["id"], session_id), None) is not None
    return {"session_id": session_id, "cleared": existed}
