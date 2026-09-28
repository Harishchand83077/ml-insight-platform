"""
One-time setup: creates the `feedback` table used by POST /feedback in
src/serving/api.py (thumbs up/down on an assistant chat response, from
frontend/src/components/Message.jsx).

Connects via src.common.db:connect_local() - same connection target every
other serving-path setup script uses (see scripts/setup_auth_tables.py).
References users(id), so run scripts/setup_auth_tables.py first if that
table doesn't exist yet.

Run: python scripts/setup_feedback_table.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.db import connect_local  # noqa: E402

CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS feedback (
        id BIGSERIAL PRIMARY KEY,
        user_id UUID NOT NULL REFERENCES users(id),
        session_id TEXT NOT NULL,
        message_content TEXT,
        rating TEXT NOT NULL CHECK (rating IN ('up', 'down')),
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
"""


def main():
    conn = connect_local()
    try:
        with conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)
        conn.commit()
    finally:
        conn.close()
    print("Ensured table exists: feedback")


if __name__ == "__main__":
    main()
