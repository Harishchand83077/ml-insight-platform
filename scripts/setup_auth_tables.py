"""
One-time setup: creates the `users` and `audit_logs` tables used by JWT
auth (src/serving/auth.py) and by the /auth/signup, /auth/login, /predict
and /chat endpoints in src/serving/api.py.

Connects via src.common.db:connect_local() - DATABASE_URL/SUPABASE_DB_URL
if set (this project's deployment target, e.g. Supabase - see .env.example),
else the local PG_HOST/PG_PORT/... vars. Same connection helper every
other serving-path module already uses, so this creates the tables
wherever the app itself will actually look for them.

`id` is generated in application code (uuid.uuid4(), see
src/serving/auth.py:create_user) rather than a Postgres-side DEFAULT, so
this doesn't depend on the pgcrypto/uuid-ossp extension being enabled.

Run: python scripts/setup_auth_tables.py
"""

import sys
from pathlib import Path

# scripts/ lives outside the src package, so add the project root to
# sys.path - running this file directly only puts scripts/ itself on the
# path, and `from src...` would otherwise fail.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.db import connect_local  # noqa: E402

CREATE_TABLE_SQL = {
    "users": """
        CREATE TABLE IF NOT EXISTS users (
            id UUID PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """,
    "audit_logs": """
        CREATE TABLE IF NOT EXISTS audit_logs (
            id BIGSERIAL PRIMARY KEY,
            user_id UUID NOT NULL REFERENCES users(id),
            endpoint TEXT NOT NULL,
            request_summary TEXT,
            timestamp TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """,
}


def main():
    conn = connect_local()
    try:
        with conn.cursor() as cur:
            for table, sql in CREATE_TABLE_SQL.items():
                cur.execute(sql)
                print(f"Ensured table exists: {table}")
        conn.commit()
    finally:
        conn.close()
    print("Auth tables ready.")


if __name__ == "__main__":
    main()
