"""
GET /pipeline/drift: Evidently drift of the rebuilt customer_features
against the training reference (X_train), per column.

Reuses src/models/train_baseline.py's column lists and load_dataset() query
and src/models/monitor_drift.py's is_drifted() threshold convention directly
- not a second copy of either.

The reference (X_train) is captured once, by cache_reference() - called by
POST /pipeline/reset right after it rebuilds customer_features to the
pristine seed-42 baseline. It is NOT re-queried from whatever
customer_features holds at drift-check time: once live events (and
build_features.py's windowing) have changed customer_features, re-deriving
"the reference" from that same, now-different table would compare drifted
data against itself. cache_reference() always re-derives it the same way
train_baseline.py does (load_dataset() + the identical train_test_split
call, same random_state=42), so it matches what the deployed model was
actually trained on, as long as reset last ran against the real baseline.
"""

import pandas as pd
import psycopg2
from evidently import Report
from evidently.presets import DataDriftPreset
from sklearn.model_selection import train_test_split

from src.live.safety import local_pg_dsn
from src.models.monitor_drift import is_drifted
from src.models.train_baseline import BINARY_YES_NO_COLS, CATEGORICAL_COLS, NUMERIC_COLS, load_dataset

FEATURE_COLUMNS = CATEGORICAL_COLS + BINARY_YES_NO_COLS + NUMERIC_COLS

_reference = None  # cached X_train DataFrame; set by cache_reference()


class NoReferenceError(RuntimeError):
    """GET /pipeline/drift was called before any POST /pipeline/reset in
    this process, so there's nothing to compare against yet."""


def _load_current(conn):
    df = pd.read_sql(
        """
        SELECT cf.*, c.churn
        FROM customer_features cf
        JOIN customers c ON cf.customer_id = c.customer_id
        """,
        conn,
    )
    df["partner"] = (df["partner"] == "Yes").astype(int)
    df["dependents"] = (df["dependents"] == "Yes").astype(int)
    return df[FEATURE_COLUMNS]


def cache_reference():
    """Re-derives X_train from whatever customer_features holds right now,
    the same way train_baseline.py does, and caches it as the fixed
    reference for subsequent get_drift_report() calls."""
    global _reference
    X, y = load_dataset()
    X_train, _, _, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    _reference = X_train
    return _reference


def get_drift_report():
    if _reference is None:
        raise NoReferenceError(
            "no drift reference cached yet - call POST /pipeline/reset first"
        )

    conn = psycopg2.connect(**local_pg_dsn())
    try:
        current = _load_current(conn)
    finally:
        conn.close()

    report = Report([DataDriftPreset()])
    snapshot = report.run(current, _reference)
    result = snapshot.dict()

    columns = []
    drifted_count = 0
    share = 0.0
    for m in result["metrics"]:
        cfg = m.get("config", {})
        if cfg.get("type") == "evidently:metric_v2:ValueDrift":
            column, method, threshold, value = cfg["column"], cfg["method"], cfg["threshold"], m["value"]
            drifted = is_drifted(method, value, threshold)
            columns.append(
                {
                    "column": column,
                    "method": method,
                    # numpy scalars (bool_/float64) aren't JSON-serializable
                    # through FastAPI's jsonable_encoder - cast to native
                    # Python types.
                    "distance": float(value),
                    "threshold": float(threshold),
                    "drifted": bool(drifted),
                }
            )
            if drifted:
                drifted_count += 1
        elif cfg.get("type") == "evidently:metric_v2:DriftedColumnsCount":
            share = float(m["value"]["share"])

    return {
        "columns": columns,
        "drifted_columns": drifted_count,
        "total_columns": len(FEATURE_COLUMNS),
        "share": share,
        "dataset_drift": bool(share >= 0.5),
    }
