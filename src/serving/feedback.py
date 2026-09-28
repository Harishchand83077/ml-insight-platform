"""
Stores /chat response feedback (thumbs up/down on an assistant message -
see api.py's POST /feedback and frontend/src/components/Message.jsx).

Its own small connection pool, same pattern as feature_cache.py/auth.py
but sized down (1-5, not 1-20): feedback writes are a low-frequency,
low-concurrency path compared to those, so there's no reason to reserve
as many connections against Supabase's pooler for this.
"""

import psycopg2.pool

try:
    from src.common.db import get_database_url
except ImportError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.common.db import get_database_url

POOL_MIN_CONN = 1
POOL_MAX_CONN = 5
_pg_pool = psycopg2.pool.SimpleConnectionPool(POOL_MIN_CONN, POOL_MAX_CONN, get_database_url())

MESSAGE_CONTENT_MAX_LEN = 1000


def store_feedback(user_id: str, session_id: str, message_content: str, rating: str) -> None:
    truncated = (message_content or "")[:MESSAGE_CONTENT_MAX_LEN]
    conn = _pg_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO feedback (user_id, session_id, message_content, rating) "
                "VALUES (%s, %s, %s, %s)",
                (user_id, session_id, truncated, rating),
            )
        conn.commit()
    finally:
        _pg_pool.putconn(conn)
