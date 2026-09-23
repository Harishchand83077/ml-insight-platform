"""
Semantic cache for agent responses: embeds incoming questions with the
same embedding model used for RAG (BAAI/bge-small-en-v1.5 - must match
src/agent/tools.py and build_knowledge_base.py), and if a new question is
highly similar (cosine similarity >= 0.90) to a previously cached
question, returns that cached response directly instead of calling the
agent (and its LLM/tool calls) again.

Only meaningful for single-turn-equivalent questions: the same question
can mean something different mid-conversation depending on prior context,
so the caller (src/serving/api.py) is responsible for only consulting
this cache when there's no prior conversation history for the session -
i.e. only on the first message of a session - and only storing a new
entry when that first message was itself a genuine cache miss.

Entries are stored in Redis as a capped list (max 200) under key
"semantic_cache", each a JSON object: {question, embedding, response,
tool_calls}. Newest entries are pushed to the head; LTRIM evicts the
oldest once the cap is reached.
"""

import json
import logging

import numpy as np
import redis
from langchain_huggingface import HuggingFaceEmbeddings
from prometheus_client import Counter

logger = logging.getLogger("semantic_cache")

REDIS_HOST = "localhost"
REDIS_PORT = 6379
CACHE_KEY = "semantic_cache"
MAX_ENTRIES = 200
SIMILARITY_THRESHOLD = 0.90
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # must match src/agent/tools.py

SEMANTIC_CACHE_HIT_COUNTER = Counter(
    "semantic_cache_hits_total",
    "Agent /chat requests answered from the semantic cache instead of a real agent call",
)

_redis_client = None
_embeddings = None


def _get_redis_client():
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    return _redis_client


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def _cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def check_semantic_cache(question):
    """Returns {"response": str, "tool_calls": list} on a hit above the
    similarity threshold, else None."""
    raw_entries = _get_redis_client().lrange(CACHE_KEY, 0, -1)
    if not raw_entries:
        return None

    query_embedding = _get_embeddings().embed_query(question)

    best_score = -1.0
    best_entry = None
    for raw in raw_entries:
        entry = json.loads(raw)
        score = _cosine_similarity(query_embedding, entry["embedding"])
        if score > best_score:
            best_score = score
            best_entry = entry

    if best_entry is not None and best_score >= SIMILARITY_THRESHOLD:
        logger.info(
            "SEMANTIC CACHE HIT (similarity=%.4f) matched question: %r",
            best_score,
            best_entry["question"],
        )
        SEMANTIC_CACHE_HIT_COUNTER.inc()
        return {"response": best_entry["response"], "tool_calls": best_entry["tool_calls"]}

    return None


def store_in_semantic_cache(question, response, tool_calls):
    r = _get_redis_client()
    embedding = _get_embeddings().embed_query(question)
    entry = json.dumps(
        {"question": question, "embedding": embedding, "response": response, "tool_calls": tool_calls}
    )
    r.lpush(CACHE_KEY, entry)
    r.ltrim(CACHE_KEY, 0, MAX_ENTRIES - 1)  # keep newest MAX_ENTRIES, evict the rest
