"""
Stage 1/2 control API: starts/stops the live EventGenerator, reports its
status (including the Stage 2 simulated clock), and drives the Stage 2
pipeline (reset / rebuild-features / drift). Bind to 127.0.0.1:8001 only -
this is a local development tool, not meant to be reachable from anywhere
but this machine, and has nothing to do with the public Supabase/Upstash-
backed serving app (src/serving/api.py).

Run with:
    uvicorn src.live.control_api:app --host 127.0.0.1 --port 8001
"""

import os
import tempfile
from pathlib import Path

import psycopg2
import requests
import skops.io
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from src.live import drift as drift_module
from src.live import gate as gate_module
from src.live import pipeline
from src.live import promote as promote_module
from src.live import serving_compat
from src.live.generator import QUEUES, EventGenerator
from src.live.migrate_event_tables import migrate_event_tables
from src.live.promote import resolve_target_dir
from src.live.safety import local_pg_dsn
from src.live.serving_compat import SKOPS_TRUSTED_TYPES
from src.live.simclock import SimClock

app = FastAPI(title="Stage 1/2/3 live-data control API")
_clock = SimClock()
_generator = EventGenerator(clock=_clock)
_candidate_state = {}  # set by POST /retrain: pipeline, holdout, evaluation, decision, compat_result


@app.exception_handler(drift_module.NoReferenceError)
def _no_reference_handler(request, exc):
    return JSONResponse(status_code=409, content={"error": str(exc)})


@app.exception_handler(promote_module.PromotionRefused)
def _promotion_refused_handler(request, exc):
    return JSONResponse(status_code=409, content={"error": str(exc)})


PANEL_HTML_PATH = Path(__file__).parent / "panel.html"


@app.get("/", response_class=HTMLResponse)
def control_panel():
    """The control panel: polls /generator/status and shows a warning
    banner when detect_queue_warnings() flags a growing, unconsumed
    queue."""
    return PANEL_HTML_PATH.read_text(encoding="utf-8")

RABBITMQ_MGMT_PORT = os.environ.get("RABBITMQ_MGMT_PORT", "15672")
RABBITMQ_MGMT_USER = os.environ.get("RABBITMQ_MGMT_USER", "guest")
RABBITMQ_MGMT_PASSWORD = os.environ.get("RABBITMQ_MGMT_PASSWORD", "guest")
# %2F is the default vhost "/", URL-encoded, as the management API requires.
RABBITMQ_MGMT_URL = f"http://localhost:{RABBITMQ_MGMT_PORT}/api/queues/%2F/" + "{queue}"


class StartRequest(BaseModel):
    rate: float = 5.0
    drift: bool = False
    drift_fraction: float = 0.3


@app.post("/generator/start")
def start_generator(req: StartRequest):
    conn = psycopg2.connect(**local_pg_dsn())
    try:
        migrate_event_tables(conn)
    finally:
        conn.close()

    started = _generator.start(events_per_second=req.rate, drift=req.drift, drift_fraction=req.drift_fraction)
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


def _queue_depths_and_consumers():
    """{queue_name: {"messages": int|None, "consumers": int|None}} - both
    come from the same RabbitMQ management API call, so a single request
    per queue covers depth and consumer count together."""
    info = {}
    for queue_name in QUEUES:
        try:
            resp = requests.get(
                RABBITMQ_MGMT_URL.format(queue=queue_name),
                auth=(RABBITMQ_MGMT_USER, RABBITMQ_MGMT_PASSWORD),
                timeout=2,
            )
            resp.raise_for_status()
            data = resp.json()
            info[queue_name] = {"messages": data.get("messages"), "consumers": data.get("consumers")}
        except requests.RequestException as exc:
            info[queue_name] = {"messages": None, "consumers": None, "error": str(exc)}
    return info


def detect_queue_warnings(queue_info, previous_depths):
    """queue_info: this call's {queue_name: {"messages", "consumers"}}.
    previous_depths: {queue_name: int|None} remembered from the previous
    /generator/status call (or {} on the very first call - nothing to
    compare against yet, so no warnings then).

    Returns (warnings, updated_previous_depths): a list of human-readable
    warning strings, and the depths to remember for the next call.

    A queue is flagged only when its depth actually GREW since the last
    reading while it has 0 consumers - not just "depth is nonzero", which
    would also flag a queue that's draining normally after a burst, or one
    with a harmless leftover backlog that isn't getting worse. Growing
    depth with zero consumers is specifically "nothing is reading this
    queue and it's piling up" - the event_consumer.py-not-running case this
    exists to catch."""
    warnings = []
    updated = {}
    for queue_name, info in queue_info.items():
        messages = info.get("messages")
        consumers = info.get("consumers")
        previous = previous_depths.get(queue_name)
        if messages is not None and consumers == 0 and previous is not None and messages > previous:
            warnings.append(
                f"{queue_name}: queue depth growing ({previous} -> {messages}) with 0 "
                "consumers - is event_consumer.py running?"
            )
        updated[queue_name] = messages
    return warnings, updated


_last_queue_depths = {}


@app.get("/generator/status")
def generator_status():
    global _last_queue_depths
    status = _generator.status()
    status["rows_landed"] = _row_counts()

    queue_info = _queue_depths_and_consumers()
    status["queue_depth"] = {q: info["messages"] for q, info in queue_info.items()}
    status["queue_consumers"] = {q: info["consumers"] for q, info in queue_info.items()}

    warnings, _last_queue_depths = detect_queue_warnings(queue_info, _last_queue_depths)
    status["warnings"] = warnings
    return status


@app.post("/pipeline/reset")
def pipeline_reset():
    conn = psycopg2.connect(**local_pg_dsn())
    try:
        return pipeline.reset_pipeline(conn, _generator, _clock)
    finally:
        conn.close()


@app.post("/pipeline/rebuild-features")
def pipeline_rebuild_features():
    conn = psycopg2.connect(**local_pg_dsn())
    try:
        return pipeline.rebuild_features_pipeline(conn, _clock)
    finally:
        conn.close()


@app.get("/pipeline/drift")
def pipeline_drift():
    return drift_module.get_drift_report()


# --- Stage 3b: the promotion gate --------------------------------------


def _load_current_production_pipeline():
    """The model /retrain compares the candidate against: whatever is at
    MODEL_DIR (default models/production_model) right now, loaded through
    the same skops.io.load() call + trusted-types list src.serving.api
    uses at startup - imported from serving_compat, not re-declared."""
    model_dir = resolve_target_dir()
    return skops.io.load(f"{model_dir}/model.skops", trusted=SKOPS_TRUSTED_TYPES)


def _sample_customer_ids(n=5):
    conn = psycopg2.connect(**local_pg_dsn())
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT customer_id FROM customer_features ORDER BY customer_id LIMIT %s", (n,))
            return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()


class RetrainRequest(BaseModel):
    label_table: str = gate_module.DEFAULT_LABEL_TABLE
    strength: float | None = None  # required when label_table is the Scenario B table


@app.post("/retrain")
def retrain(req: RetrainRequest):
    """Freezes a holdout, trains a candidate with production's own
    hyperparameters, and runs the statistical gate against whatever model
    is currently at MODEL_DIR. Does not touch the serving-compat check or
    promote anything - that's POST /promote, and only after this result
    shows promoted=True."""
    holdout = gate_module.freeze_holdout(label_table=req.label_table, strength=req.strength)
    candidate = gate_module.train_candidate(holdout)
    production = _load_current_production_pipeline()

    evaluation = gate_module.evaluate(candidate, production, holdout)
    decision = gate_module.decide(
        evaluation["pr_auc_diff_point"],
        evaluation["bootstrap"]["ci_lower"],
        evaluation["bootstrap"]["ci_upper"],
        holdout_metadata=holdout["metadata"],
    )

    _candidate_state.clear()
    _candidate_state.update(pipeline=candidate, holdout=holdout, evaluation=evaluation, decision=decision, compat_result=None)

    return {"evaluation": evaluation, "decision": decision}


@app.get("/candidate/compare")
def candidate_compare():
    if "decision" not in _candidate_state:
        raise HTTPException(status_code=404, detail="no candidate trained yet - call POST /retrain first")
    return {
        "evaluation": _candidate_state["evaluation"],
        "decision": _candidate_state["decision"],
        "compat_result": _candidate_state.get("compat_result"),
    }


@app.post("/promote")
def promote_candidate():
    if "decision" not in _candidate_state:
        raise HTTPException(status_code=404, detail="no candidate trained yet - call POST /retrain first")

    decision = _candidate_state["decision"]
    if not decision.get("promoted"):
        return JSONResponse(status_code=409, content={"error": "gate rejected this candidate", "decision": decision})

    export_dir = tempfile.mkdtemp(prefix="stage3b_candidate_")
    sample_ids = _sample_customer_ids(5)
    compat_result = serving_compat.check_serving_compat(_candidate_state["pipeline"], sample_ids, export_dir)
    _candidate_state["compat_result"] = compat_result

    if not compat_result["passed"]:
        return JSONResponse(
            status_code=409, content={"error": "serving-compat check failed", "compat_result": compat_result}
        )

    version_info = promote_module.promote(
        export_dir,
        decision,
        compat_result,
        metrics={
            "production": _candidate_state["evaluation"]["production"],
            "candidate": _candidate_state["evaluation"]["candidate"],
        },
    )
    return version_info
