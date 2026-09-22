"""
Cache-aside lookup of a customer's row in customer_features: check Redis
first (key "features:{customer_id}"), and on a miss fall back to Postgres
(via a small pooled connection, not one connection per call), cache the
result in Redis with a 300s TTL, then return it.
"""

import json
import logging
import os

import psycopg2.extras
import psycopg2.pool
import redis

PG_DSN = {
    "host": os.environ.get("PG_HOST", "localhost"),
    "port": os.environ.get("PG_PORT", "5432"),
    "dbname": os.environ.get("PG_DATABASE", "ml_insight"),
    "user": os.environ.get("PG_USER", "postgres"),
    "password": os.environ.get("PG_PASSWORD", "devpassword"),
}

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
CACHE_TTL_SECONDS = 300
CACHE_KEY_PREFIX = "features"
POOL_MIN_CONN = 1
POOL_MAX_CONN = 5

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("feature_cache")

_redis_client = None
_pg_pool = psycopg2.pool.SimpleConnectionPool(POOL_MIN_CONN, POOL_MAX_CONN, **PG_DSN)


def get_redis_client():
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    return _redis_client


def get_customer_features(customer_id):
    """Returns (features_dict_or_None, cache_hit)."""
    r = get_redis_client()
    cache_key = f"{CACHE_KEY_PREFIX}:{customer_id}"

    cached = r.get(cache_key)
    if cached is not None:
        logger.info("Cache HIT for %s", customer_id)
        return json.loads(cached), True

    logger.info("Cache MISS for %s", customer_id)
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
