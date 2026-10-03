"""
Churn-prediction API: GET /health, POST /auth/signup, POST /auth/login,
POST /predict, POST /chat, DELETE /chat/{session_id}, POST /feedback, and
GET /metrics (Prometheus). The XGBoost model (the full preprocessing +
classifier pipeline, exported by scripts/export_model_for_deployment.py
to models/production_model/model.skops - see load_production_model) is
loaded once at startup, not per-request, via skops.io.load() directly -
not mlflow, which isn't a serve-time dependency at all (kept in
requirements.txt for training, dropped from requirements-docker.txt).
The LangChain agent (src/agent/agent.py) is built lazily on first use via
get_agent(), so importing this module doesn't need GROQ_API_KEY set.

/predict and /chat require a valid JWT (Authorization: Bearer <token>,
obtained from /auth/signup or /auth/login) - see src/serving/auth.py for
password hashing, token issuance/verification, and the get_current_user
dependency. Every call to /predict and /chat is recorded in the
audit_logs table (auth.log_audit), tied to the authenticated user.

Model loading (XGBoost + the embedding model, together often 30-60s on a
cold instance - see load_embedder's docstring) happens in a background
asyncio task kicked off from lifespan, not awaited during startup - so
the app finishes startup and starts accepting connections immediately.
/health always returns 200 the moment the process is up, regardless of
model readiness; /predict and /chat check model_state["ready"] and
return 503 until loading finishes, rather than hanging or crashing
against a model_state that isn't populated yet.

/metrics gets the standard HTTP metrics (request count, latency
histograms, status codes per endpoint) for free from
prometheus-fastapi-instrumentator. Beyond that, three app-specific
metrics actually show what's interesting about this service's behavior:
feature_cache_requests_total (feature_cache.py), semantic_cache_hits_total
(semantic_cache.py), and agent_response_seconds (below) - raw HTTP
latency on /chat wouldn't distinguish a semantic-cache-hit response
(near-instant) from a real agent call (LLM + tool round trips).

/chat is the one `async def` handler (every other route is plain `def`,
which FastAPI dispatches to a worker thread automatically - see each
handler's own comments for why that's fine for them). /chat needs to be
async so its embedding calls (via semantic_cache.py) can genuinely be
awaited off the event loop via asyncio.to_thread, matching the pattern
_load_models uses at startup - see that async-ness's own comment, right
above where it's declared, for what that then requires of every other
blocking call made directly in this handler.
"""

import asyncio
import hmac
import logging
import os
from contextlib import asynccontextmanager
from typing import Literal

import groq
import skops.io
import torch
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_huggingface import HuggingFaceEmbeddings
from prometheus_client import Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, constr

from src.agent.agent import get_agent
from src.agent.semantic_cache import check_semantic_cache, store_in_semantic_cache
from src.common.embedding_model import EMBEDDING_MODEL_NAME, set_embedder
from src.common.production_model import set_pipeline
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
from src.serving.feedback import store_feedback
from src.serving.prediction import predict_churn

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
_startup_task = None  # holds a strong reference - see lifespan()


PRODUCTION_MODEL_PATH = "models/production_model/model.skops"
# Must match scripts/export_model_for_deployment.py's save_model call (see
# models/production_model/MLmodel's own skops_trusted_types field, which
# records the same list) - the pipeline's XGBoost step won't deserialize
# without skops being told these specific types are safe to unpickle.
PRODUCTION_MODEL_TRUSTED_TYPES = ["xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"]


def load_production_model():
    """Loads the XGBoost pipeline directly from
    models/production_model/model.skops via skops.io.load() - the same
    file mlflow.sklearn.load_model() used to read (mlflow's sklearn/skops
    flavor is a thin wrapper around this exact call), but without needing
    mlflow itself installed just to load one file at serve time. Produced
    by scripts/export_model_for_deployment.py; to ship a new model, re-run
    that script and commit the updated models/production_model/."""
    pipeline = skops.io.load(PRODUCTION_MODEL_PATH, trusted=PRODUCTION_MODEL_TRUSTED_TYPES)
    logger.info("Loaded xgboost model from %s", PRODUCTION_MODEL_PATH)
    return pipeline


def load_embedder():
    """Loads the one shared sentence-transformers embedding model instance
    - see src/common/embedding_model.py for why this must happen exactly
    once, here, rather than lazily in the RAG tool or semantic cache."""
    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    logger.info("Loaded embedding model %s", EMBEDDING_MODEL_NAME)
    return embedder


async def _load_models():
    """Runs as a background task, not awaited by lifespan - both loaders
    are blocking calls (skops.io.load does file I/O;
    HuggingFaceEmbeddings does a network round trip to the HF Hub even
    when the model is already cached locally - see load_embedder's
    docstring), so each runs in a worker thread via asyncio.to_thread
    rather than directly on the event loop. Running them directly here
    would block the loop for the exact duration this whole refactor is
    meant to avoid blocking - /health would go unresponsive right along
    with everything else."""
    try:
        model_state["pipeline"] = await asyncio.to_thread(load_production_model)
        set_pipeline(model_state["pipeline"])
        model_state["embedder"] = await asyncio.to_thread(load_embedder)
        set_embedder(model_state["embedder"])
        model_state["ready"] = True
        logger.info("Model loading complete - now serving /predict and /chat")
    except Exception:
        logger.exception("Model loading failed - /predict and /chat will keep returning 503")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _startup_task
    model_state["ready"] = False
    # Not awaited: startup returns immediately (so the app can start
    # accepting connections - in particular /health - right away) while
    # this keeps running in the background. Kept in a module-level
    # variable so it isn't garbage-collected mid-flight - asyncio only
    # holds a weak reference to a task returned by create_task.
    _startup_task = asyncio.create_task(_load_models())
    yield
    _startup_task.cancel()
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


class FeedbackRequest(BaseModel):
    session_id: str
    message_content: str
    rating: Literal["up", "down"]


@app.get("/health")
def health():
    # Always 200 the moment the process is up - a liveness check, not a
    # readiness check. models_ready in the body lets a caller distinguish
    # "alive but still loading" from "alive and serving" without /predict
    # or /chat needing to be hit (and bounced with 503) just to find out.
    return {"status": "ok", "models_ready": model_state.get("ready", False)}


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
    if not model_state.get("ready", False):
        raise HTTPException(status_code=503, detail="Models still loading, please retry shortly")

    log_audit(current_user["id"], "/predict", req.customer_id)

    result = predict_churn(req.customer_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"customer_id '{req.customer_id}' not found")

    return result


# Test-only switch to skip the semantic cache. Off unless CACHE_BYPASS_TOKEN is
# set on the server; a request must send the same value in X-Cache-Bypass.
# Ordinary users can't reach it without the secret, so it can't be used to
# burn Groq quota. A wrong token is a 403, not a silent fallback, so a
# misconfigured test fails loudly.
CACHE_BYPASS_TOKEN = os.environ.get("CACHE_BYPASS_TOKEN", "")


def _cache_bypass_requested(header_value):
    if header_value is None:
        return False
    if not CACHE_BYPASS_TOKEN or not hmac.compare_digest(header_value, CACHE_BYPASS_TOKEN):
        raise HTTPException(status_code=403, detail="Invalid cache-bypass token")
    return True


@app.post("/chat")
async def chat(
    req: ChatRequest,
    current_user: dict = Depends(get_current_user),
    x_cache_bypass: str | None = Header(default=None, alias="X-Cache-Bypass"),
):
    if not model_state.get("ready", False):
        raise HTTPException(status_code=503, detail="Models still loading, please retry shortly")

    # This handler is async so the embedding calls inside check_semantic_cache/
    # store_in_semantic_cache (both async, awaiting asyncio.to_thread
    # internally - see semantic_cache.py) can actually be awaited rather than
    # running directly on the event loop. That makes every OTHER blocking
    # call made directly in this function - not FastAPI dependencies like
    # get_current_user, which FastAPI already threadpools automatically -
    # something that now needs the same treatment, or it'd be worse off than
    # before: log_audit (a psycopg2 call) and churn_agent.invoke() (an LLM
    # call plus, when the agent uses query_project_docs_tool, the RAG
    # retrieval step) are both wrapped in asyncio.to_thread below for exactly
    # this reason. query_project_docs_tool itself can't be made async - its
    # @tool decorator requires sync invocation when bound to this agent's
    # sync-only tool-calling path (verified directly: an async tool raises
    # "StructuredTool does not support sync invocation" here) - so
    # offloading the whole invoke() call is what actually keeps its
    # retrieval step off the loop, not a wrap inside the tool itself.
    await asyncio.to_thread(log_audit, current_user["id"], "/chat", req.message)

    session_key = (current_user["id"], req.session_id)
    previous_messages = chat_sessions.get(session_key, [])
    is_first_message = not previous_messages

    # Semantic cache only applies to the first message of a session - the
    # same question mid-conversation can mean something different given
    # prior context, so it's only safe to short-circuit on a fresh session.
    use_semantic_cache = is_first_message and not _cache_bypass_requested(x_cache_bypass)
    if use_semantic_cache:
        cached = await check_semantic_cache(req.message)
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
            churn_agent = await asyncio.to_thread(get_agent)
            result = await asyncio.to_thread(churn_agent.invoke, {"messages": input_messages})
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

    if use_semantic_cache:
        await store_in_semantic_cache(req.message, response_text, tool_calls)

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


@app.post("/feedback", status_code=201)
def feedback(req: FeedbackRequest, current_user: dict = Depends(get_current_user)):
    store_feedback(current_user["id"], req.session_id, req.message_content, req.rating)
    return {"status": "recorded"}
