"""
Stage 2 pipeline orchestration for POST /pipeline/reset and
POST /pipeline/rebuild-features: ties together the event-table reset
(src/live/reset.py), the simulated clock (src/live/simclock.py), the
generator's own state, build_features.py's existing atomic
read+build+write (src/pipelines/build_features.run_build - reused, not
forked), and the drift reference cache (src/live/drift.py).
"""

from src.live import drift, reset
from src.pipelines.build_features import run_build

CHECKSUM_TABLES = [*reset.EVENT_TABLES, "customer_features"]


def _all_checksums(conn):
    checksums = {}
    for table in reset.EVENT_TABLES:
        checksums[table] = reset.table_checksum(conn, table)
    checksums["customer_features"] = reset.table_checksum(conn, "customer_features", order_by="customer_id")
    return checksums


def reset_pipeline(conn, generator, clock):
    """Stops the generator (idempotent), restores the event tables to the
    seed-42 baseline, resets the sim clock and the generator's own counters/
    drifted-customer state, rebuilds customer_features at that baseline
    (as_of = exactly the clock's reset point, e.g. END_DATE), and re-caches
    the drift reference from that pristine rebuild. Returns the post-reset
    checksums (event tables + customer_features) and sim time, so two
    resets in a row can be compared directly.

    as_of uses clock.reset()'s return value, not a later clock.now() call:
    reset_event_tables()'s bulk reload takes a real-measurable amount of
    time, and now() would have advanced by that much (times speed) in sim
    time - a few sim-minutes is enough to flip a handful of historical
    events across a day boundary relative to END_DATE, making
    customer_features subtly different between two resets even though the
    event tables themselves are identical. Pinning as_of to the reset point
    itself removes that source of drift entirely."""
    generator.stop()
    reset.reset_event_tables(conn)
    reset_sim_time = clock.reset()
    generator.reset_state()

    features = run_build(conn, as_of=reset_sim_time)
    drift.cache_reference()

    return {
        "sim_time": reset_sim_time.isoformat(sep=" "),
        "checksums": _all_checksums(conn),
        "customer_features_rows": len(features),
    }


def rebuild_features_pipeline(conn, clock):
    """Atomic rebuild with as_of = simulated now."""
    sim_now = clock.now()  # one read: run_build and the reported sim_time must agree exactly
    features = run_build(conn, as_of=sim_now)
    return {
        "sim_time": sim_now.isoformat(sep=" "),
        "customer_features_rows": len(features),
        "columns": list(features.columns),
    }
