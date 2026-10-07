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
tool_calls, created_at}. Newest entries are pushed to the head; LTRIM evicts
the oldest once the cap is reached.

Two exclusions, both enforced here so every caller gets them:

- Questions containing a customer ID (\\d{4}-[A-Za-z]{5}) are never read
  from or written to the cache. Their answers depend on per-customer live
  data, and the embedding can't reliably tell customer IDs or contract
  terms apart: "What if customer 2691-NZETQ switched to a 2-year contract?"
  and the same question with "1-year" scored 0.975 against each other, above
  the threshold. The feature cache already makes repeat customer lookups
  cheap, so skipping this cache costs little.
- Only answers whose tool calls were all query_project_docs_tool (or with
  no tool calls) are written. Answers that used SQL, prediction, what-if,
  explain, or retention tools are never stored, because their parameters
  (customer, contract term, metric name) are what the embedding can't tell
  apart. Reads are unchanged, so entries written before this rule remain
  readable until they expire.
- Entries older than TTL_SECONDS (24 hours) are ignored on read. A
  knowledge-base rebuild or a change in customer data then can't keep
  serving answers from before it for longer than a day. Entries without a
  created_at field, written before this check existed, count as expired.
  Expired entries stay in the list until LTRIM evicts them; they are never
  returned.

Both public functions are async and run their embed_query() call via
asyncio.to_thread - that call is CPU-bound (a local sentence-transformers
forward pass), and api.py's /chat handler is async, so without this it
would run directly on the event loop. The Redis calls are left
synchronous (plain redis-py, not aioredis) since they're comparatively
fast network round trips, not the CPU-bound part this exists to offload.
"""

import asyncio
import json
import logging
import re
import time

import numpy as np
import redis
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
TTL_SECONDS = 24 * 60 * 60
CUSTOMER_ID_PATTERN = re.compile(r"\d{4}-[A-Za-z]{5}")
DOCS_TOOL_NAME = "query_project_docs_tool"

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


def contains_customer_id(question):
    """True if the question names a specific customer (e.g. 7590-VHVEG).
    Such questions bypass the semantic cache entirely - see the module docstring."""
    return CUSTOMER_ID_PATTERN.search(question) is not None


def _is_fresh(entry, now):
    return now - entry.get("created_at", 0) < TTL_SECONDS


async def check_semantic_cache(question):
    """Returns {"response": str, "tool_calls": list} on a hit above the
    similarity threshold, else None. Always None, with no Redis or embedding
    work, for questions that contain a customer ID."""
    if contains_customer_id(question):
        return None

    # Redis is best-effort: an error or timeout counts as a miss, so /chat
    # still reaches the agent when Redis is down.
    try:
        raw_entries = _get_redis_client().lrange(CACHE_KEY, 0, -1)
    except (redis.RedisError, TimeoutError) as e:
        logger.warning("Semantic cache read failed, treating as a miss: %s", e)
        SEMANTIC_CACHE_MISS_COUNTER.inc()
        return None
    entries = [json.loads(raw) for raw in raw_entries]
    now = time.time()
    entries = [entry for entry in entries if _is_fresh(entry, now)]
    if not entries:
        SEMANTIC_CACHE_MISS_COUNTER.inc()
        return None

    query_embedding = await asyncio.to_thread(get_embedder().embed_query, question)

    best_score = -1.0
    best_entry = None
    for entry in entries:
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


def is_cacheable_answer(tool_calls):
    """Only answers grounded purely in the project docs (or with no tool calls
    at all) are cacheable. Data-dependent answers (SQL, prediction, what-if,
    explain, retention) are never stored: the embedding can't distinguish
    their parameters, so a near-identical question with different numbers
    would get the wrong stored answer. Measured: "Why did you choose PR-AUC
    over accuracy?" vs "...ROC-AUC over accuracy?" scored 0.907, above the
    threshold, and the closest legitimate rewording scored 0.910, so no
    threshold separates them."""
    return all(call.get("tool") == DOCS_TOOL_NAME for call in tool_calls)


async def store_in_semantic_cache(question, response, tool_calls):
    if contains_customer_id(question):
        return
    if not is_cacheable_answer(tool_calls):
        return
    r = _get_redis_client()
    embedding = await asyncio.to_thread(get_embedder().embed_query, question)
    entry = json.dumps(
        {
            "question": question,
            "embedding": embedding,
            "response": response,
            "tool_calls": tool_calls,
            "created_at": time.time(),
        }
    )
    try:
        r.lpush(CACHE_KEY, entry)
        r.ltrim(CACHE_KEY, 0, MAX_ENTRIES - 1)  # keep newest MAX_ENTRIES, evict the rest
    except (redis.RedisError, TimeoutError) as e:
        logger.warning("Semantic cache write failed, skipping the store: %s", e)
