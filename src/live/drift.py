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

# These aren't excluded (they're real, informative features) but they ARE
# unwindowed all-time counts (build_features.py's build_login_features/
# build_ticket_features never filter these by as_of at all - see each
# one's own comment there) - they mechanically grow as more events land,
# live or historical, drifted or not. Measured: at drift=False, these are
# exactly the columns that cross the drift threshold first (see the
# "control at equal volume" measurement in this module's own report),
# purely from accumulated volume. Labeled, not excluded: a real shift here
# is still worth seeing, just not mistaken for a drift-toggle effect.
VOLUME_SENSITIVE_COLUMNS = {"total_logins_90d", "num_tickets"}

# --- Event-level drift ("stream_drift"): compares the LAST N live events'
# own raw metric against a reference sample from the historical events,
# per event type - not a customer-level, all-time aggregate. This responds
# directly to the drift multipliers (DRIFT_RESOLUTION_TIME_SCALE,
# DRIFT_USAGE_COUNT_SCALE in src/live/generator.py) because it looks only
# at recent events' own values, with no dilution from thousands of
# pre-existing historical rows the way avg_resolution_time_hours/
# avg_usage_count (customer-level, all-time averages) are diluted.
# session_duration_seconds is tracked for logins too, for symmetry and as
# a useful negative control: drift mode does NOT scale login session
# duration (only selection frequency, via DRIFT_LOGIN_FREQUENCY_SCALE), so
# this one is expected to stay flat even when drift is clearly on - that's
# a feature of the check (specificity), not a gap in it.
STREAM_EVENT_METRICS = {
    "login_events": "session_duration_seconds",
    "support_tickets": "resolution_time_hours",
    "feature_usage_logs": "usage_count",
}
# No fixed STREAM_DRIFT_THRESHOLD constant: Evidently picks the test (and
# its threshold) per sample - see get_stream_drift_report()'s own comment
# on is_drifted() for why that matters.
#
# N=100, not 200: at the achievable ~30 events/sec (70/15/15 split across
# event types), tickets/usage events arrive at ~4.5/sec, so n=200 needs
# ~45s just to fill the window - too close to the 60-90s detection target
# to leave margin. n=100 fills in ~22s, and is still large enough to avoid
# the small-sample noise get_stream_drift_report() guards against (caught
# live at n_live=20-33: noisy enough to cross threshold by sampling
# variance alone, with drift OFF) - verified at n=100 with a live
# drift=False control before trusting it for the timing measurement.
STREAM_DRIFT_SAMPLE_N = 100

# Debounce: a p-value-based test (Evidently's own choice for some of these
# comparisons - see get_stream_drift_report()'s is_drifted() comment) has
# an inherent ~5% false-positive rate per check at the conventional 0.05
# threshold, by construction. That's fine for ONE check, but /pipeline/
# drift gets polled repeatedly (the panel, or anyone watching for the
# toggle) - caught live: feature_usage_logs flagged "drifted" with drift
# OFF on a single poll (p=0.00012, confirmed genuinely significant, just
# a false positive - this will happen on ~1 in 20 checks of a true-null
# metric by design, and compounds fast across repeated polls and 3
# metrics). CONSECUTIVE_CONFIRMATIONS_REQUIRED means "drifted" in the
# response only turns true after that many consecutive raw reads agree,
# per table - a real shift (once the window is saturated with drifted
# data) stays flagged on every subsequent read, so this costs detection
# speed only on the scale of one extra poll interval, while cutting
# compounding false positives sharply. raw_drifted is still reported
# alongside, for anyone who wants the undebounced read.
CONSECUTIVE_CONFIRMATIONS_REQUIRED = 2
_stream_drift_history = {}  # table -> list of recent raw "drifted" bools


def reset_stream_drift_history():
    """Called by cache_reference() (i.e. on every POST /pipeline/reset) -
    a reset starts a fresh event-table baseline, so a debounce streak
    carried over from before it would misrepresent "confirmed" against
    data that no longer exists in the same form."""
    _stream_drift_history.clear()


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
    reference for subsequent get_drift_report() calls. Also clears the
    stream_drift debounce history - see reset_stream_drift_history()."""
    global _reference
    X, y = load_dataset()
    X_train, _, _, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    _reference = X_train
    reset_stream_drift_history()
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
                    "volume_sensitive": column in VOLUME_SENSITIVE_COLUMNS,
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
        "stream_drift": get_stream_drift_report(),
    }


def _load_event_sample(conn, table, column, n, live):
    """live=True: the N most recently inserted live events (event_id IS
    NOT NULL), newest first by the serial id - "the last N events" in
    insertion order. live=False: a deterministic reference sample, the N
    most recent HISTORICAL events (event_id IS NULL) - "recent history,
    right before live traffic started", not a random sample, so repeated
    calls compare against the same fixed reference."""
    predicate = "event_id IS NOT NULL" if live else "event_id IS NULL"
    query = (
        f"SELECT {column} AS value FROM {table} "
        f"WHERE {predicate} AND {column} IS NOT NULL "
        f"ORDER BY id DESC LIMIT %(n)s"
    )
    return pd.read_sql(query, conn, params={"n": n})


def get_stream_drift_report(n=STREAM_DRIFT_SAMPLE_N):
    """Event-level drift: the last `n` live events' own metric value vs a
    same-sized historical reference sample, per event type - see this
    module's docstring/STREAM_EVENT_METRICS for why this is the view that
    actually responds to the drift toggle rather than to elapsed volume.

    Requires a FULL window of live events (n_live >= n) before reporting a
    real distance - caught live: with only a handful of live events (e.g.
    33), the Wasserstein distance against a 200-row reference is itself
    noisy enough to cross 0.1 by sampling variance alone, firing
    "drifted" on login_events within a few seconds even with drift OFF.
    That's sample-size noise, not a signal. Below a full window, this
    reports "not enough events yet" the same way zero live events does."""
    conn = psycopg2.connect(**local_pg_dsn())
    try:
        report = {}
        for table, column in STREAM_EVENT_METRICS.items():
            historical = _load_event_sample(conn, table, column, n, live=False)
            live_sample = _load_event_sample(conn, table, column, n, live=True)

            if len(live_sample) < n or len(historical) < n:
                report[table] = {
                    "metric": column,
                    "n_live": len(live_sample),
                    "n_reference": len(historical),
                    "distance": None,
                    "threshold": None,  # not known yet - Evidently picks the test (and its threshold) per sample
                    "raw_drifted": False,
                    "drifted": False,
                    "note": f"not enough events yet to compare (need a full window of {n})",
                }
                continue

            snapshot = Report([DataDriftPreset()]).run(live_sample, historical)
            result = snapshot.dict()
            distance, method, threshold = None, None, None
            for m in result["metrics"]:
                cfg = m.get("config", {})
                if cfg.get("type") == "evidently:metric_v2:ValueDrift" and cfg.get("column") == "value":
                    distance, method, threshold = float(m["value"]), cfg["method"], float(cfg["threshold"])

            # Evidently auto-picks the test per column/sample size - a
            # Wasserstein-style distance (drift when ABOVE threshold) for
            # some, a p-value test like K-S (drift when BELOW threshold)
            # for others. Caught live: at n=100 for login_events, Evidently
            # picked "K-S p_value" (p=0.47); a flat "distance > threshold"
            # check called that "drifted" when a p-value that high means
            # the opposite. is_drifted() (same helper the feature-level
            # view uses) gets the direction right either way - and
            # threshold comes from Evidently's own config for this metric,
            # not a hardcoded value that may not even apply to the test it
            # chose.
            raw_drifted = bool(distance is not None and is_drifted(method, distance, threshold))

            # Debounce: only report "drifted" once the last
            # CONSECUTIVE_CONFIRMATIONS_REQUIRED raw reads all agree - see
            # the module-level comment on CONSECUTIVE_CONFIRMATIONS_REQUIRED
            # for why a single raw read isn't trustworthy enough on its own.
            history = _stream_drift_history.setdefault(table, [])
            history.append(raw_drifted)
            del history[: -CONSECUTIVE_CONFIRMATIONS_REQUIRED]
            confirmed_drifted = len(history) >= CONSECUTIVE_CONFIRMATIONS_REQUIRED and all(history)

            report[table] = {
                "metric": column,
                "n_live": len(live_sample),
                "n_reference": len(historical),
                "method": method,
                "distance": distance,
                "threshold": threshold,
                "raw_drifted": raw_drifted,
                "drifted": confirmed_drifted,
            }
        return report
    finally:
        conn.close()
