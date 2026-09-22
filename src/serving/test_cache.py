"""
Picks 5 random customer_ids from customer_features and calls
get_customer_features on each twice, timing both calls to show the
DB-miss vs cache-hit latency difference.
"""

import os
import time

import psycopg2

from feature_cache import get_customer_features

PG_DSN = {
    "host": os.environ.get("PG_HOST", "localhost"),
    "port": os.environ.get("PG_PORT", "5432"),
    "dbname": os.environ.get("PG_DATABASE", "ml_insight"),
    "user": os.environ.get("PG_USER", "postgres"),
    "password": os.environ.get("PG_PASSWORD", "devpassword"),
}


def pick_random_customer_ids(n=5):
    conn = psycopg2.connect(**PG_DSN)
    with conn.cursor() as cur:
        cur.execute("SELECT customer_id FROM customer_features ORDER BY random() LIMIT %s", (n,))
        ids = [row[0] for row in cur.fetchall()]
    conn.close()
    return ids


def timed_call(customer_id):
    start = time.perf_counter()
    _, cache_hit = get_customer_features(customer_id)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return cache_hit, elapsed_ms


def main():
    customer_ids = pick_random_customer_ids(5)
    print(f"Testing customer_ids: {customer_ids}\n")

    for customer_id in customer_ids:
        first_hit, first_ms = timed_call(customer_id)
        second_hit, second_ms = timed_call(customer_id)
        print(
            f"{customer_id}: first call (hit={first_hit}) = {first_ms:.2f} ms, "
            f"second call (hit={second_hit}) = {second_ms:.2f} ms, "
            f"speedup = {first_ms / second_ms:.1f}x\n"
        )


if __name__ == "__main__":
    main()
