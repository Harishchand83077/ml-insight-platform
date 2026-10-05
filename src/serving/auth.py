"""
JWT-based auth for the serving API: password hashing, token issuance/
verification, user lookup, and the get_current_user dependency that
protects /predict and /chat in src/serving/api.py. Also owns audit
logging (log_audit) - a separate concern in principle, but every caller
of it is already an authenticated endpoint going through this module's
DB pool, so it lives here rather than standing up a second pool for one
INSERT statement.

Password hashing: passlib's CryptContext(schemes=["bcrypt"]) - note
passlib 1.7.4 (its last release) probes bcrypt.__about__.__version__ to
detect the backend version, an attribute bcrypt itself removed in 4.1;
left unpinned, that raises inside passlib before a single password is
ever hashed. requirements.txt/requirements-docker.txt pin bcrypt==4.0.1
(the last version passlib's probe works against) specifically to avoid
this - don't upgrade bcrypt here without re-verifying passlib still works.

JWT: PyJWT, HS256, signed with JWT_SECRET_KEY (see .env.example - this
repo's copy was generated once for local/demo use; rotate it for any
real deployment, since anyone holding it can mint valid tokens for any
user id).

Tokens carry {"sub": user_id, "email": ..., "iat", "exp"} and are
verified on every protected request - get_current_user decodes the
token AND re-fetches the user row from Postgres (rather than trusting
the token's email claim alone), so a user deleted after a token was
issued is correctly rejected rather than treated as still valid until
expiry.
"""

import logging
import os
import time
import uuid

import jwt
import psycopg2
import psycopg2.errors
from dotenv import load_dotenv
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext

try:
    from src.common.db import checkout_for_write, make_pool, run_read
except ImportError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.common.db import checkout_for_write, make_pool, run_read

load_dotenv()

logger = logging.getLogger("auth")

JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_SECONDS = 60 * 60 * 24  # 24h

def _get_jwt_secret() -> str:
    secret = os.environ.get("JWT_SECRET_KEY")
    if not secret:
        raise RuntimeError(
            "JWT_SECRET_KEY is not set. Add it to your .env (see .env.example) - "
            "generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
        )
    return secret


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

POOL_MIN_CONN = 1
POOL_MAX_CONN = 20
_pg_pool = None


def _get_pool():
    global _pg_pool
    if _pg_pool is None:
        _pg_pool = make_pool(POOL_MIN_CONN, POOL_MAX_CONN)
    return _pg_pool

# HTTPBearer's default auto_error=True raises 403 (not 401) when the
# Authorization header is missing entirely - auto_error=False here, with
# get_current_user raising 401 itself, so "no token" and "bad token" both
# come back 401.
_bearer_scheme = HTTPBearer(auto_error=False)


class EmailAlreadyExistsError(Exception):
    pass


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, hashed_password: str) -> bool:
    return pwd_context.verify(password, hashed_password)


def create_access_token(user_id: str, email: str) -> str:
    now = int(time.time())
    payload = {"sub": user_id, "email": email, "iat": now, "exp": now + ACCESS_TOKEN_EXPIRE_SECONDS}
    return jwt.encode(payload, _get_jwt_secret(), algorithm=JWT_ALGORITHM)


def create_user(email: str, hashed_password: str) -> dict:
    # Write: checkout_for_write, not run_read - see the policy in src/common/db.py.
    user_id = str(uuid.uuid4())
    with checkout_for_write(_get_pool()) as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    "INSERT INTO users (id, email, hashed_password) VALUES (%s, %s, %s) "
                    "RETURNING id, email, created_at",
                    (user_id, email, hashed_password),
                )
            except psycopg2.errors.UniqueViolation:
                conn.rollback()
                raise EmailAlreadyExistsError(email) from None
            row = cur.fetchone()
        conn.commit()
    return {"id": str(row[0]), "email": row[1], "created_at": row[2]}


def get_user_by_email(email: str) -> dict | None:
    def _query(conn):
        with conn.cursor() as cur:
            cur.execute("SELECT id, email, hashed_password FROM users WHERE email = %s", (email,))
            return cur.fetchone()

    row = run_read(_get_pool(), _query)
    if row is None:
        return None
    return {"id": str(row[0]), "email": row[1], "hashed_password": row[2]}


def get_user_by_id(user_id: str) -> dict | None:
    def _query(conn):
        with conn.cursor() as cur:
            cur.execute("SELECT id, email FROM users WHERE id = %s", (user_id,))
            return cur.fetchone()

    row = run_read(_get_pool(), _query)
    if row is None:
        return None
    return {"id": str(row[0]), "email": row[1]}


def get_current_user(credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme)) -> dict:
    if credentials is None:
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        payload = jwt.decode(credentials.credentials, _get_jwt_secret(), algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token") from None

    user = get_user_by_id(payload.get("sub", ""))
    if user is None:
        raise HTTPException(status_code=401, detail="User no longer exists")
    return user


REQUEST_SUMMARY_MAX_LEN = 500


def log_audit(user_id: str, endpoint: str, request_summary: str) -> None:
    """Best-effort: a logging failure shouldn't take down the request it's
    logging, so this only warns on error rather than raising."""
    truncated = (request_summary or "")[:REQUEST_SUMMARY_MAX_LEN]
    try:
        with checkout_for_write(_get_pool()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO audit_logs (user_id, endpoint, request_summary) VALUES (%s, %s, %s)",
                    (user_id, endpoint, truncated),
                )
            conn.commit()
    except Exception:
        logger.warning("Failed to write audit log for user=%s endpoint=%s", user_id, endpoint, exc_info=True)
