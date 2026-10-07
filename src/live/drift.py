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

EXCLUDED_COLUMNS: recent_30d_vs_older_60d_ratio and
avg_session_duration_recent_30d are left out of the drift view entirely -
see the reasons below, which are the result of a direct measurement, not
a guess. Both come from build_login_features()'s "recent 30 days" window
in src/pipelines/build_features.py, which is correct code doing what a
rolling calendar window should do - the problem is a scale mismatch, not a
bug: isolated (zero live traffic, same historical data), the reference's
own feature values (zero_share 0.33%, well-distributed) go to 80.6% zero by
as_of = END_DATE + 30 simulated days, and to a literal constant 0 (100%
zero, 1 unique value for all 7043 customers) by +60 days - purely from the
simulated clock advancing past the fixed 90-day historical window, before
any live traffic or drift mode is involved. Feeding the live generator
enough traffic to keep the window populated at the reference's density
would need roughly 7043 customers x ~5 logins/30d (the historical rate) =
~35,000 login events per 30 simulated days - at the default clock speed
(1 sim day/real second) that's ~1,167 events/sec, versus the generator's
measured achievable throughput of ~30 events/sec (confirmed live; see the
Stage 2 report). No window-formula change closes a ~40x gap like that
without either slowing the clock enough to defeat its own purpose, or
artificially thinning the reference to match the live system's limited
scale (which would stop it being the real training distribution). That's
why this is excluded rather than "fixed": the window computation itself
isn't wrong.
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

EXCLUDED_COLUMNS = {
    "recent_30d_vs_older_60d_ratio": (
        "Calendar-windowed to as_of, not to the data's own timeline. Measured "
        "in isolation (zero live traffic): zero_share goes from 0.33% at the "
        "reference's as_of=END_DATE to 80.6% at +30 simulated days and a "
        "literal constant 0 (every one of 7043 customers) at +60 days, before "
        "any live traffic or drift is involved. With live traffic, sustaining "
        "the reference's density would need ~1,167 login events/sec at the "
        "default clock speed, ~40x the generator's measured ~30/sec. A scale "
        "mismatch, not a window-formula bug - see src/live/drift.py's "
        "module docstring for the full measurement."
    ),
    "avg_session_duration_recent_30d": (
        "Same recent-30-day window in build_login_features() as "
        "recent_30d_vs_older_60d_ratio, and the same cause - see that "
        "column's reason."
    ),
}

# Columns actually fed to Evidently: FEATURE_COLUMNS minus the excluded ones.
DRIFT_VIEW_COLUMNS = [c for c in FEATURE_COLUMNS if c not in EXCLUDED_COLUMNS]

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
    snapshot = report.run(current[DRIFT_VIEW_COLUMNS], _reference[DRIFT_VIEW_COLUMNS])
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
        "total_columns": len(DRIFT_VIEW_COLUMNS),
        "share": share,
        "dataset_drift": bool(share >= 0.5),
        "excluded_columns": [
            {"column": column, "reason": reason} for column, reason in EXCLUDED_COLUMNS.items()
        ],
    }
