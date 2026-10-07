"""
Unit tests for src/live/pipeline.py's orchestration (POST /pipeline/reset,
POST /pipeline/rebuild-features). No real database: reset_event_tables,
run_build, and table_checksum are faked, since what's under test here is
the ORDER and the glue, not the SQL - "two resets give identical checksums"
against the real database is proven live, not here.
"""

from unittest.mock import MagicMock, patch

import pandas as pd

from src.live import pipeline

FEATURES_DF = pd.DataFrame({"customer_id": ["a", "b"], "tenure": [1, 2]})
SIM_NOW = pd.Timestamp("2024-08-15 12:00:00")
SIM_NOW_STR = SIM_NOW.isoformat(sep=" ")


def _reset_mocks():
    return (
        patch.object(pipeline.reset, "reset_event_tables"),
        patch.object(pipeline.reset, "table_checksum", return_value="checksum-x"),
        patch.object(pipeline, "run_build", return_value=FEATURES_DF),
        patch.object(pipeline.drift, "cache_reference"),
    )


class TestResetPipeline:
    def test_stops_resets_rebuilds_and_recaches_the_drift_reference(self):
        conn, generator, clock = MagicMock(), MagicMock(), MagicMock()
        clock.reset.return_value = SIM_NOW

        p1, p2, p3, p4 = _reset_mocks()
        with p1 as reset_event_tables, p2 as table_checksum, p3 as run_build, p4 as cache_reference:
            result = pipeline.reset_pipeline(conn, generator, clock)

        generator.stop.assert_called_once()
        reset_event_tables.assert_called_once_with(conn)
        clock.reset.assert_called_once()
        generator.reset_state.assert_called_once()
        run_build.assert_called_once_with(conn, as_of=SIM_NOW)
        cache_reference.assert_called_once()
        assert table_checksum.call_count == 4  # 3 event tables + customer_features
        assert result["customer_features_rows"] == 2
        assert result["sim_time"] == SIM_NOW_STR
        assert result["checksums"].keys() == {"login_events", "support_tickets", "feature_usage_logs", "customer_features"}

    def test_as_of_uses_reset_s_return_value_not_a_later_now_call(self):
        # Pinning as_of to reset()'s return value (not a later now() call)
        # is what makes two resets reproducible even if reset_event_tables()
        # takes a different amount of real time each run - see the
        # docstring in src/live/pipeline.py and src/live/simclock.py.
        conn, generator, clock = MagicMock(), MagicMock(), MagicMock()
        clock.reset.return_value = SIM_NOW
        clock.now.return_value = SIM_NOW + pd.Timedelta(days=999)  # must NOT be used

        p1, p2, p3, p4 = _reset_mocks()
        with p1, p2, p3 as run_build, p4:
            result = pipeline.reset_pipeline(conn, generator, clock)

        run_build.assert_called_once_with(conn, as_of=SIM_NOW)
        assert result["sim_time"] == SIM_NOW_STR
        clock.now.assert_not_called()

    def test_calls_happen_in_the_documented_order(self):
        conn, generator, clock = MagicMock(), MagicMock(), MagicMock()
        clock.reset.return_value = SIM_NOW

        manager = MagicMock()
        p1, p2, p3, p4 = _reset_mocks()
        with p1 as reset_event_tables, p2, p3 as run_build, p4 as cache_reference:
            manager.attach_mock(generator.stop, "stop")
            manager.attach_mock(reset_event_tables, "reset_event_tables")
            manager.attach_mock(clock.reset, "clock_reset")
            manager.attach_mock(generator.reset_state, "reset_state")
            manager.attach_mock(run_build, "run_build")
            manager.attach_mock(cache_reference, "cache_reference")
            pipeline.reset_pipeline(conn, generator, clock)

        names = [c[0] for c in manager.mock_calls]
        assert names == [
            "stop",
            "reset_event_tables",
            "clock_reset",
            "reset_state",
            "run_build",
            "cache_reference",
        ]

    def test_two_resets_in_a_row_return_identical_checksums(self):
        conn, generator, clock = MagicMock(), MagicMock(), MagicMock()
        clock.reset.return_value = SIM_NOW

        p1, p2, p3, p4 = _reset_mocks()
        with p1, p2, p3, p4:
            first = pipeline.reset_pipeline(conn, generator, clock)
            second = pipeline.reset_pipeline(conn, generator, clock)

        assert first["checksums"] == second["checksums"]


class TestRebuildFeaturesPipeline:
    def test_calls_run_build_with_sim_now_and_reports_shape(self):
        conn, clock = MagicMock(), MagicMock()
        clock.now.return_value = SIM_NOW

        with patch.object(pipeline, "run_build", return_value=FEATURES_DF) as run_build:
            result = pipeline.rebuild_features_pipeline(conn, clock)

        run_build.assert_called_once_with(conn, as_of=SIM_NOW)
        assert result == {
            "sim_time": SIM_NOW_STR,
            "customer_features_rows": 2,
            "columns": ["customer_id", "tenure"],
        }
