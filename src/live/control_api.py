"""
Stage 1 control API: starts/stops the live EventGenerator and reports its
status. Bind to 127.0.0.1:8001 only - this is a local development tool, not
meant to be reachable from anywhere but this machine, and has nothing to do
with the public Supabase/Upstash-backed serving app (src/serving/api.py).

Run with:
    uvicorn src.live.control_api:app --host 127.0.0.1 --port 8001
"""

import os

import psycopg2
import requests
from fastapi import FastAPI
from pydantic import BaseModel

from src.live.generator import QUEUES, EventGenerator
from src.live.migrate_event_tables import migrate_event_tables
from src.live.safety import local_pg_dsn

app = FastAPI(title="Stage 1 live-data control API")
_generator = EventGenerator()

RABBITMQ_MGMT_PORT = os.environ.get("RABBITMQ_MGMT_PORT", "15672")
RABBITMQ_MGMT_USER = os.environ.get("RABBITMQ_MGMT_USER", "guest")
RABBITMQ_MGMT_PASSWORD = os.environ.get("RABBITMQ_MGMT_PASSWORD", "guest")
# %2F is the default vhost "/", URL-encoded, as the management API requires.
RABBITMQ_MGMT_URL = f"http://localhost:{RABBITMQ_MGMT_PORT}/api/queues/%2F/" + "{queue}"


class StartRequest(BaseModel):
    rate: float = 5.0
    drift: bool = False


@app.post("/generator/start")
def start_generator(req: StartRequest):
    conn = psycopg2.connect(**local_pg_dsn())
    try:
        migrate_event_tables(conn)
    finally:
        conn.close()

    started = _generator.start(events_per_second=req.rate, drift=req.drift)
    return {"started": started, **_generator.status()}


@app.post("/generator/stop")
def stop_generator():
    stopped = _generator.stop()
    return {"stopped": stopped, **_generator.status()}


def _row_counts():
    conn = psycopg2.connect(**local_pg_dsn())
    try:
        with conn.cursor() as cur:
            counts = {}
            for table in QUEUES:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                counts[table] = cur.fetchone()[0]
            return counts
    finally:
        conn.close()


def _queue_depths():
    depths = {}
    for queue_name in QUEUES:
        try:
            resp = requests.get(
                RABBITMQ_MGMT_URL.format(queue=queue_name),
                auth=(RABBITMQ_MGMT_USER, RABBITMQ_MGMT_PASSWORD),
                timeout=2,
            )
            resp.raise_for_status()
            depths[queue_name] = resp.json().get("messages")
        except requests.RequestException as exc:
            depths[queue_name] = f"unavailable: {exc}"
    return depths


@app.get("/generator/status")
def generator_status():
    status = _generator.status()
    status["rows_landed"] = _row_counts()
    status["queue_depth"] = _queue_depths()
    return status
