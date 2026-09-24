"""
Shared Redis connection helper, mirroring src/common/db.py's pattern:
reads REDIS_URL if set (defaults to local redis://localhost:6379
otherwise, for backward-compatible local dev), and calls load_dotenv()
itself so no caller needs to remember to. This is what makes pointing at
a different Redis (local -> Upstash, or anywhere else) just an env var
change: set REDIS_URL and nothing in the calling code changes.

Upstash (and most hosted Redis) requires TLS. redis-py's Redis.from_url()
auto-detects this from the URL scheme - verified directly against
redis-py 8.1.0's redis.connection.parse_url source, not assumed: a
rediss:// URL (note the double s) sets connection_class=SSLConnection
automatically, the same "https vs http" convention Upstash's own
dashboard connection string already follows. No separate ssl=True flag
is needed as long as the URL scheme is correct - confirmed with a live
PING/SET/GET against the actual Upstash instance this project uses. If a
copied URL is missing the extra 's' (plain redis:// pointed at a host
that actually requires TLS), the connection will fail outright rather
than silently working unencrypted; check Upstash's dashboard for the
exact connection string if that happens.

Redis.from_url() creates and manages its own connection pool internally
(the standard redis-py pattern) - no separate ConnectionPool object is
needed on top of it.
"""

import os

import redis
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")

_redis_client = None


def get_redis_client():
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    return _redis_client
