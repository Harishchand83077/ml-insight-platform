"""
Cache-aside lookup of a customer's row in customer_features: check Redis
first (key "features:{customer_id}"), and on a miss fall back to Postgres
(via a small pooled connection, not one connection per call), cache the
result in Redis with a 300s TTL, then return it.
"""

import json
import logging

import psycopg2.extras
import redis
from prometheus_client import Counter

# Absolute import when loaded as part of the src package (e.g. by
# src/serving/api.py); src.common isn't a sibling of this file, so the
# fallback explicitly puts the project root on sys.path first - needed
# when this file's own directory is run/imported directly (e.g.
# `python src/serving/test_cache.py`, which does `from feature_cache
# import ...` and never puts the project root on sys.path itself).
try:
    from src.common.db import get_shared_pool, run_read
    from src.common.redis_client import get_redis_client
except ImportError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.common.db import get_shared_pool, run_read
    from src.common.redis_client import get_redis_client

CACHE_TTL_SECONDS = 300
CACHE_KEY_PREFIX = "features"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("feature_cache")


def _get_pool():
    # This used to be its own pool, sized up from 5 to 20 under load testing.
    # It's now the one pool shared with auth, feedback, and the agent's SQL
    # tools - see src/common/db.py for why (Supabase pooler limits, the
    # bounded-wait semaphore) and the current sizing.
    return get_shared_pool()


FEATURE_CACHE_COUNTER = Counter(
    "feature_cache_requests_total",
    "get_customer_features lookups, by whether they hit Redis or fell through to Postgres",
    ["result"],  # "hit" or "miss"
)


def get_customer_features(customer_id):
    """Returns (features_dict_or_None, cache_hit)."""
    r = get_redis_client()
    cache_key = f"{CACHE_KEY_PREFIX}:{customer_id}"

    # Redis is best-effort: an error or timeout here is treated as a miss, so
    # a Redis outage slows /predict down to a Postgres lookup instead of
    # failing it.
    try:
        cached = r.get(cache_key)
        redis_get_failed = False
    except (redis.RedisError, TimeoutError) as e:
        logger.warning("Redis GET failed for %s, falling back to Postgres: %s", customer_id, e)
        cached = None
        redis_get_failed = True

    if cached is not None:
        logger.info("Cache HIT for %s", customer_id)
        FEATURE_CACHE_COUNTER.labels(result="hit").inc()
        return json.loads(cached), True

    logger.info("Cache MISS for %s", customer_id)
    FEATURE_CACHE_COUNTER.labels(result="miss").inc()

    def _query(conn):
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM customer_features WHERE customer_id = %s", (customer_id,))
            return cur.fetchone()

    # Read-only lookup, so run_read may retry it once on a fresh connection.
    row = run_read(_get_pool(), _query)

    if row is None:
        return None, False

    row = dict(row)
    for key, value in row.items():
        if hasattr(value, "__float__") and not isinstance(value, (int, float, bool)):
            row[key] = float(value)  # e.g. Decimal from NUMERIC columns

    # If the GET just failed, Redis is unreachable right now. Skip the SETEX so
    # this request doesn't spend a second timeout on a write that will fail too.
    if redis_get_failed:
        return row, False
    try:
        r.setex(cache_key, CACHE_TTL_SECONDS, json.dumps(row))
    except (redis.RedisError, TimeoutError) as e:
        logger.warning("Redis SETEX failed for %s, returning the row uncached: %s", customer_id, e)
    return row, False
