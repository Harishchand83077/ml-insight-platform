"""
Semantic cache for agent responses: embeds incoming questions with the
same shared embedding model instance used for RAG (get_embedder() from
src.common.embedding_model - see that module for why it's a single
instance rather than one per caller), and if a new question is highly
similar (cosine similarity >= 0.90) to a previously cached question,
returns that cached response directly instead of calling the agent (and
its LLM/tool calls) again.

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
from prometheus_client import Counter

# Absolute import when loaded as part of the src package; src.common isn't
# a sibling of this file, so the fallback explicitly puts the project root
# on sys.path first - needed when this file's own directory is
# run/imported directly. get_redis_client aliased to _get_redis_client (not
# get_redis_client) to match this module's existing private-helper naming
# and so tests/unit/test_semantic_cache.py's patch.object(semantic_cache,
# "_get_redis_client", ...) keeps working unchanged.
try:
    from src.common.embedding_model import get_embedder
    from src.common.redis_client import get_redis_client as _get_redis_client
except ImportError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.common.embedding_model import get_embedder
    from src.common.redis_client import get_redis_client as _get_redis_client

logger = logging.getLogger("semantic_cache")

CACHE_KEY = "semantic_cache"
MAX_ENTRIES = 200
SIMILARITY_THRESHOLD = 0.90

SEMANTIC_CACHE_HIT_COUNTER = Counter(
    "semantic_cache_hits_total",
    "Agent /chat requests answered from the semantic cache instead of a real agent call",
)
SEMANTIC_CACHE_MISS_COUNTER = Counter(
    "semantic_cache_misses_total",
    "First-message /chat requests that did not match anything in the semantic cache "
    "(empty cache, or best match below the similarity threshold)",
)

def _cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def check_semantic_cache(question):
    """Returns {"response": str, "tool_calls": list} on a hit above the
    similarity threshold, else None."""
    raw_entries = _get_redis_client().lrange(CACHE_KEY, 0, -1)
    if not raw_entries:
        SEMANTIC_CACHE_MISS_COUNTER.inc()
        return None

    query_embedding = get_embedder().embed_query(question)

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

    SEMANTIC_CACHE_MISS_COUNTER.inc()
    return None


def store_in_semantic_cache(question, response, tool_calls):
    r = _get_redis_client()
    embedding = get_embedder().embed_query(question)
    entry = json.dumps(
        {"question": question, "embedding": embedding, "response": response, "tool_calls": tool_calls}
    )
    r.lpush(CACHE_KEY, entry)
    r.ltrim(CACHE_KEY, 0, MAX_ENTRIES - 1)  # keep newest MAX_ENTRIES, evict the rest
