"""
Cache-aside lookup of a customer's row in customer_features: check Redis
first (key "features:{customer_id}"), and on a miss fall back to Postgres
(via a small pooled connection, not one connection per call), cache the
result in Redis with a 300s TTL, then return it.
"""

import json
import logging

import psycopg2.extras
import psycopg2.pool
from prometheus_client import Counter

# Absolute import when loaded as part of the src package (e.g. by
# src/serving/api.py); src.common isn't a sibling of this file, so the
# fallback explicitly puts the project root on sys.path first - needed
# when this file's own directory is run/imported directly (e.g.
# `python src/serving/test_cache.py`, which does `from feature_cache
# import ...` and never puts the project root on sys.path itself).
try:
    from src.common.db import get_database_url
    from src.common.redis_client import get_redis_client
except ImportError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.common.db import get_database_url
    from src.common.redis_client import get_redis_client

CACHE_TTL_SECONDS = 300
CACHE_KEY_PREFIX = "features"
POOL_MIN_CONN = 1
# Bumped from 5 -> 20 after load testing showed /predict's p95/p99 latency
# degrading badly under concurrent load (requests queuing for a pooled
# connection). A local Postgres instance handles 20 connections easily, so
# this is a cheap fix - but it's not unlimited scaling: it just raises the
# concurrency level where the same queuing problem reappears, rather than
# removing it. A sustained load higher than this would need the same
# investigation again (or a properly sized pool per expected traffic,
# read replicas, etc.).
POOL_MAX_CONN = 20

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("feature_cache")

_pg_pool = psycopg2.pool.SimpleConnectionPool(POOL_MIN_CONN, POOL_MAX_CONN, get_database_url())

FEATURE_CACHE_COUNTER = Counter(
    "feature_cache_requests_total",
    "get_customer_features lookups, by whether they hit Redis or fell through to Postgres",
    ["result"],  # "hit" or "miss"
)


def get_customer_features(customer_id):
    """Returns (features_dict_or_None, cache_hit)."""
    r = get_redis_client()
    cache_key = f"{CACHE_KEY_PREFIX}:{customer_id}"

    cached = r.get(cache_key)
    if cached is not None:
        logger.info("Cache HIT for %s", customer_id)
        FEATURE_CACHE_COUNTER.labels(result="hit").inc()
        return json.loads(cached), True

    logger.info("Cache MISS for %s", customer_id)
    FEATURE_CACHE_COUNTER.labels(result="miss").inc()
    conn = _pg_pool.getconn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM customer_features WHERE customer_id = %s", (customer_id,))
            row = cur.fetchone()
    finally:
        _pg_pool.putconn(conn)

    if row is None:
        return None, False

    row = dict(row)
    for key, value in row.items():
        if hasattr(value, "__float__") and not isinstance(value, (int, float, bool)):
            row[key] = float(value)  # e.g. Decimal from NUMERIC columns

    r.setex(cache_key, CACHE_TTL_SECONDS, json.dumps(row))
    return row, False
