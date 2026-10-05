"""
Stores /chat response feedback (thumbs up/down on an assistant message -
see api.py's POST /feedback and frontend/src/components/Message.jsx).

Its own small connection pool, same pattern as feature_cache.py/auth.py
but sized down (1-5, not 1-20): feedback writes are a low-frequency,
low-concurrency path compared to those, so there's no reason to reserve
as many connections against Supabase's pooler for this.
"""

try:
    from src.common.db import checkout_for_write, make_pool
except ImportError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.common.db import checkout_for_write, make_pool

POOL_MIN_CONN = 1
POOL_MAX_CONN = 5
_pg_pool = None


def _get_pool():
    global _pg_pool
    if _pg_pool is None:
        _pg_pool = make_pool(POOL_MIN_CONN, POOL_MAX_CONN)
    return _pg_pool

MESSAGE_CONTENT_MAX_LEN = 1000


def store_feedback(user_id: str, session_id: str, message_content: str, rating: str) -> None:
    # Write: pinged on checkout, never retried after the INSERT is sent (see src/common/db.py).
    truncated = (message_content or "")[:MESSAGE_CONTENT_MAX_LEN]
    with checkout_for_write(_get_pool()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO feedback (user_id, session_id, message_content, rating) "
                "VALUES (%s, %s, %s, %s)",
                (user_id, session_id, truncated, rating),
            )
        conn.commit()
