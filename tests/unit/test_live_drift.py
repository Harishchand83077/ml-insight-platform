"""
Unit tests for src/live/drift.py's get_drift_report() response shape, and
that it refuses to run before any reference has been cached (i.e. before
POST /pipeline/reset has run once in this process). Evidently's report and
the database read are both faked - what's under test is the shape this
project's code produces from Evidently's output, not Evidently itself.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.live import drift


def _fake_snapshot_dict(column_results):
    """column_results: list of (column, method, threshold, value)."""
    metrics = []
    drifted_count = 0
    for column, method, threshold, value in column_results:
        metrics.append(
            {
                "config": {
                    "type": "evidently:metric_v2:ValueDrift",
                    "column": column,
                    "method": method,
                    "threshold": threshold,
                },
                "value": value,
            }
        )
        if drift.is_drifted(method, value, threshold):
            drifted_count += 1
    share = drifted_count / len(column_results) if column_results else 0.0
    metrics.append(
        {
            "config": {"type": "evidently:metric_v2:DriftedColumnsCount"},
            "value": {"count": drifted_count, "share": share},
        }
    )
    return {"metrics": metrics}


@pytest.fixture(autouse=True)
def _clear_reference():
    drift._reference = None
    drift._stream_drift_history.clear()
    yield
    drift._reference = None
    drift._stream_drift_history.clear()


class TestNoReferenceCachedYet:
    def test_raises_before_pipeline_reset_has_ever_run(self):
        with pytest.raises(drift.NoReferenceError):
            drift.get_drift_report()


FAKE_STREAM_DRIFT = {
    "login_events": {"metric": "session_duration_seconds", "n_live": 200, "n_reference": 200, "distance": 0.01, "threshold": 0.1, "drifted": False},
    "support_tickets": {"metric": "resolution_time_hours", "n_live": 200, "n_reference": 200, "distance": 0.3, "threshold": 0.1, "drifted": True},
    "feature_usage_logs": {"metric": "usage_count", "n_live": 200, "n_reference": 200, "distance": 0.25, "threshold": 0.1, "drifted": True},
}


class TestResponseShape:
    def test_shape_once_a_reference_is_cached(self):
        view_columns = ["tenure", "monthly_charges", "avg_usage_count"]
        drift._reference = pd.DataFrame({c: [1, 2, 3] for c in view_columns})
        fake_snapshot = MagicMock()
        fake_snapshot.dict.return_value = _fake_snapshot_dict(
            [
                ("tenure", "wasserstein", 0.1, 0.05),  # distance < threshold: not drifted
                ("monthly_charges", "wasserstein", 0.1, 0.30),  # distance > threshold: drifted
                ("avg_usage_count", "wasserstein", 0.1, 0.02),  # distance < threshold: not drifted
            ]
        )

        with patch.object(drift, "DRIFT_VIEW_COLUMNS", view_columns), \
             patch.object(drift, "_load_current", return_value=pd.DataFrame({c: [1] for c in view_columns})), \
             patch.object(drift, "psycopg2") as fake_psycopg2, \
             patch.object(drift, "Report") as fake_report_cls, \
             patch.object(drift, "get_stream_drift_report", return_value=FAKE_STREAM_DRIFT):
            fake_psycopg2.connect.return_value = MagicMock()
            fake_report_cls.return_value.run.return_value = fake_snapshot

            result = drift.get_drift_report()

        assert set(result) == {
            "columns",
            "drifted_columns",
            "total_columns",
            "share",
            "dataset_drift",
            "excluded_columns",
            "stream_drift",
        }
        assert result["drifted_columns"] == 1
        assert result["total_columns"] == 3
        assert result["share"] == pytest.approx(1 / 3)
        assert result["dataset_drift"] is False  # share < 0.5
        assert len(result["columns"]) == 3
        for col in result["columns"]:
            assert set(col) == {"column", "method", "distance", "threshold", "drifted", "volume_sensitive"}
        assert result["stream_drift"] == FAKE_STREAM_DRIFT

        drifted_names = {c["column"] for c in result["columns"] if c["drifted"]}
        assert drifted_names == {"monthly_charges"}

    def test_dataset_drift_true_when_share_at_or_above_half(self):
        view_columns = ["tenure", "monthly_charges"]
        drift._reference = pd.DataFrame({c: [1, 2, 3] for c in view_columns})
        fake_snapshot = MagicMock()
        fake_snapshot.dict.return_value = _fake_snapshot_dict(
            [
                ("tenure", "wasserstein", 0.1, 0.30),  # drifted
                ("monthly_charges", "wasserstein", 0.1, 0.05),  # not drifted
            ]
        )

        with patch.object(drift, "DRIFT_VIEW_COLUMNS", view_columns), \
             patch.object(drift, "_load_current", return_value=pd.DataFrame({c: [1] for c in view_columns})), \
             patch.object(drift, "psycopg2") as fake_psycopg2, \
             patch.object(drift, "Report") as fake_report_cls, \
             patch.object(drift, "get_stream_drift_report", return_value=FAKE_STREAM_DRIFT):
            fake_psycopg2.connect.return_value = MagicMock()
            fake_report_cls.return_value.run.return_value = fake_snapshot

            result = drift.get_drift_report()

        assert result["share"] == 0.5
        assert result["dataset_drift"] is True  # share >= 0.5


class TestVolumeSensitiveLabel:
    def test_count_type_columns_are_labeled_volume_sensitive(self):
        assert drift.VOLUME_SENSITIVE_COLUMNS == {"total_logins_90d", "num_tickets"}

    def test_label_appears_on_the_right_columns_only(self):
        view_columns = ["total_logins_90d", "num_tickets", "tenure"]
        drift._reference = pd.DataFrame({c: [1, 2, 3] for c in view_columns})
        fake_snapshot = MagicMock()
        fake_snapshot.dict.return_value = _fake_snapshot_dict(
            [
                ("total_logins_90d", "wasserstein", 0.1, 0.02),
                ("num_tickets", "wasserstein", 0.1, 0.02),
                ("tenure", "wasserstein", 0.1, 0.02),
            ]
        )

        with patch.object(drift, "DRIFT_VIEW_COLUMNS", view_columns), \
             patch.object(drift, "_load_current", return_value=pd.DataFrame({c: [1] for c in view_columns})), \
             patch.object(drift, "psycopg2") as fake_psycopg2, \
             patch.object(drift, "Report") as fake_report_cls, \
             patch.object(drift, "get_stream_drift_report", return_value=FAKE_STREAM_DRIFT):
            fake_psycopg2.connect.return_value = MagicMock()
            fake_report_cls.return_value.run.return_value = fake_snapshot

            result = drift.get_drift_report()

        flagged = {c["column"] for c in result["columns"] if c["volume_sensitive"]}
        assert flagged == {"total_logins_90d", "num_tickets"}


class TestStreamDriftReport:
    def _fake_df(self, values):
        return pd.DataFrame({"value": values})

    def test_a_high_p_value_method_is_not_misread_as_drifted(self):
        # Caught live: Evidently chose "K-S p_value" for a real
        # login_events comparison and returned p=0.47 (not significant -
        # no drift) - a flat "distance > threshold" check called that
        # "drifted" anyway, since 0.47 > 0.1. is_drifted() must get the
        # direction right for a p-value method (drift means LOW p, not
        # high).
        fake_snapshot = MagicMock()
        fake_snapshot.dict.return_value = {
            "metrics": [
                {
                    "config": {
                        "type": "evidently:metric_v2:ValueDrift",
                        "column": "value",
                        "method": "K-S p_value",
                        "threshold": 0.1,
                    },
                    "value": 0.469506448503778,
                }
            ]
        }

        def fake_load(conn, table, column, n, live):
            return self._fake_df([1.0] * n)

        with patch.object(drift, "psycopg2") as fake_psycopg2, \
             patch.object(drift, "_load_event_sample", side_effect=fake_load), \
             patch.object(drift, "Report") as fake_report_cls:
            fake_psycopg2.connect.return_value = MagicMock()
            fake_report_cls.return_value.run.return_value = fake_snapshot
            report = drift.get_stream_drift_report(n=100)

        for table in drift.STREAM_EVENT_METRICS:
            assert report[table]["method"] == "K-S p_value"
            assert report[table]["distance"] == pytest.approx(0.469506448503778)
            assert report[table]["drifted"] is False  # p=0.47 is not significant

    def test_reports_each_event_types_metric_and_drifted_flag(self):
        def fake_load(conn, table, column, n, live):
            # historical centered at 10, live centered at 10 (no shift) for
            # login/usage, live shifted for tickets - exercised below
            return self._fake_df([10.0] * n)

        with patch.object(drift, "psycopg2") as fake_psycopg2, \
             patch.object(drift, "_load_event_sample", side_effect=fake_load):
            fake_psycopg2.connect.return_value = MagicMock()
            report = drift.get_stream_drift_report(n=50)

        assert set(report) == set(drift.STREAM_EVENT_METRICS)
        for table, column in drift.STREAM_EVENT_METRICS.items():
            assert report[table]["metric"] == column
            assert set(report[table]) == {
                "metric", "n_live", "n_reference", "method", "distance", "threshold", "raw_drifted", "drifted",
            }

    def test_no_live_events_yet_is_reported_not_crashed(self):
        def fake_load(conn, table, column, n, live):
            return self._fake_df([]) if live else self._fake_df([10.0] * n)

        with patch.object(drift, "psycopg2") as fake_psycopg2, \
             patch.object(drift, "_load_event_sample", side_effect=fake_load):
            fake_psycopg2.connect.return_value = MagicMock()
            report = drift.get_stream_drift_report(n=50)

        for table in drift.STREAM_EVENT_METRICS:
            assert report[table]["drifted"] is False
            assert report[table]["n_live"] == 0
            assert "note" in report[table]

    def test_a_partial_window_is_not_trusted_even_if_noisy(self):
        # Caught live: with only a few live events (well under a full
        # window), the Wasserstein distance against the reference is
        # itself noisy enough to cross threshold by sampling variance
        # alone - a partial window must report "not enough events", not a
        # real (and possibly spuriously "drifted") distance.
        rng = np.random.default_rng(0)

        def fake_load(conn, table, column, n, live):
            if live:
                return self._fake_df(rng.normal(loc=50, scale=5, size=20).tolist())  # far short of n=200
            return self._fake_df(rng.normal(loc=10, scale=1, size=n).tolist())  # very different distribution

        with patch.object(drift, "psycopg2") as fake_psycopg2, \
             patch.object(drift, "_load_event_sample", side_effect=fake_load):
            fake_psycopg2.connect.return_value = MagicMock()
            report = drift.get_stream_drift_report(n=200)

        for table in drift.STREAM_EVENT_METRICS:
            assert report[table]["n_live"] == 20
            assert report[table]["distance"] is None
            assert report[table]["drifted"] is False
            assert "note" in report[table]

    def test_a_full_window_is_trusted(self):
        rng = np.random.default_rng(0)

        def fake_load(conn, table, column, n, live):
            loc = 50 if live else 10  # genuinely different distributions
            return self._fake_df(rng.normal(loc=loc, scale=1, size=n).tolist())

        with patch.object(drift, "psycopg2") as fake_psycopg2, \
             patch.object(drift, "_load_event_sample", side_effect=fake_load):
            fake_psycopg2.connect.return_value = MagicMock()
            report = drift.get_stream_drift_report(n=200)

        for table in drift.STREAM_EVENT_METRICS:
            assert report[table]["n_live"] == 200
            assert report[table]["distance"] is not None
            assert "note" not in report[table]


class TestDebounce:
    """Caught live: a single poll's raw p-value check has a real ~5%
    false-positive rate, which compounds fast under repeated polling
    across 3 metrics - CONSECUTIVE_CONFIRMATIONS_REQUIRED exists to absorb
    that. These tests drive is_drifted()'s return value directly across
    successive get_stream_drift_report() calls, rather than depending on
    Evidently's actual statistics."""

    def _patches(self, raw_drifted_holder, n=50):
        def fake_load(conn, table, column, size, live):
            return pd.DataFrame({"value": [1.0] * size})

        fake_snapshot = MagicMock()
        fake_snapshot.dict.return_value = {
            "metrics": [
                {
                    "config": {
                        "type": "evidently:metric_v2:ValueDrift",
                        "column": "value",
                        "method": "wasserstein",
                        "threshold": 0.1,
                    },
                    "value": 0.5,
                }
            ]
        }
        return (
            patch.object(drift, "psycopg2"),
            patch.object(drift, "_load_event_sample", side_effect=fake_load),
            patch.object(drift, "Report", return_value=MagicMock(run=lambda *a, **k: fake_snapshot)),
            patch.object(drift, "is_drifted", side_effect=lambda *a, **k: raw_drifted_holder["value"]),
        )

    def test_a_single_raw_positive_is_not_yet_confirmed(self):
        raw = {"value": True}
        p1, p2, p3, p4 = self._patches(raw)
        with p1 as fake_psycopg2, p2, p3, p4:
            fake_psycopg2.connect.return_value = MagicMock()
            result = drift.get_stream_drift_report(n=50)

        for table in drift.STREAM_EVENT_METRICS:
            assert result[table]["raw_drifted"] is True
            assert result[table]["drifted"] is False  # only 1 of 2 required confirmations

    def test_two_consecutive_raw_positives_confirm_drifted(self):
        raw = {"value": True}
        p1, p2, p3, p4 = self._patches(raw)
        with p1 as fake_psycopg2, p2, p3, p4:
            fake_psycopg2.connect.return_value = MagicMock()
            drift.get_stream_drift_report(n=50)
            result = drift.get_stream_drift_report(n=50)

        for table in drift.STREAM_EVENT_METRICS:
            assert result[table]["drifted"] is True

    def test_a_disagreement_prevents_confirmation(self):
        raw = {"value": True}
        p1, p2, p3, p4 = self._patches(raw)
        with p1 as fake_psycopg2, p2, p3, p4:
            fake_psycopg2.connect.return_value = MagicMock()
            drift.get_stream_drift_report(n=50)  # True
            raw["value"] = False
            result = drift.get_stream_drift_report(n=50)  # False - breaks the streak

        for table in drift.STREAM_EVENT_METRICS:
            assert result[table]["raw_drifted"] is False
            assert result[table]["drifted"] is False

    def test_reset_stream_drift_history_clears_state(self):
        raw = {"value": True}
        p1, p2, p3, p4 = self._patches(raw)
        with p1 as fake_psycopg2, p2, p3, p4:
            fake_psycopg2.connect.return_value = MagicMock()
            drift.get_stream_drift_report(n=50)
            drift.get_stream_drift_report(n=50)  # now confirmed

            drift.reset_stream_drift_history()
            result = drift.get_stream_drift_report(n=50)  # back to 1 of 2

        for table in drift.STREAM_EVENT_METRICS:
            assert result[table]["drifted"] is False

    def test_cache_reference_resets_the_debounce_history_too(self):
        drift._stream_drift_history["login_events"] = [True, True]
        with patch.object(drift, "load_dataset", return_value=(pd.DataFrame({"a": [1]}), pd.Series([0]))), \
             patch("src.live.drift.train_test_split", return_value=(pd.DataFrame({"a": [1]}), None, None, None)):
            drift.cache_reference()

        assert drift._stream_drift_history == {}


class TestExcludedColumns:
    def test_recent_window_columns_are_excluded_from_the_drift_view(self):
        assert "recent_30d_vs_older_60d_ratio" not in drift.DRIFT_VIEW_COLUMNS
        assert "avg_session_duration_recent_30d" not in drift.DRIFT_VIEW_COLUMNS
        # every other feature column is still in the view
        assert len(drift.DRIFT_VIEW_COLUMNS) == len(drift.FEATURE_COLUMNS) - 2

    def test_response_documents_the_exclusion(self):
        view_columns = ["tenure"]
        drift._reference = pd.DataFrame({"tenure": [1, 2, 3]})
        fake_snapshot = MagicMock()
        fake_snapshot.dict.return_value = _fake_snapshot_dict([("tenure", "wasserstein", 0.1, 0.02)])

        with patch.object(drift, "DRIFT_VIEW_COLUMNS", view_columns), \
             patch.object(drift, "_load_current", return_value=pd.DataFrame({"tenure": [1]})), \
             patch.object(drift, "psycopg2") as fake_psycopg2, \
             patch.object(drift, "Report") as fake_report_cls, \
             patch.object(drift, "get_stream_drift_report", return_value=FAKE_STREAM_DRIFT):
            fake_psycopg2.connect.return_value = MagicMock()
            fake_report_cls.return_value.run.return_value = fake_snapshot

            result = drift.get_drift_report()

        excluded_names = {e["column"] for e in result["excluded_columns"]}
        assert excluded_names == {"recent_30d_vs_older_60d_ratio", "avg_session_duration_recent_30d"}
        for entry in result["excluded_columns"]:
            assert entry["reason"]  # non-empty, documented reason
        # the excluded columns never appear in the per-column results
        assert {c["column"] for c in result["columns"]}.isdisjoint(excluded_names)
