"""
Unit tests for the Stage 1 live EventGenerator (src/live/generator.py).
psycopg2 and pika are faked throughout - the global conftest.py guard now
also forbids a real pika.BlockingConnection, matching the guard already in
place for psycopg2.connect and redis.Redis.from_url, so a test here that
forgot to mock RabbitMQ would fail loudly instead of hanging.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.live import generator as gen
from src.live.safety import NonLocalHostError


def _customer(customer_id, churn="No", tech_support="Yes", **addons):
    row = {
        "customerID": customer_id,
        "Churn": churn,
        "TechSupport": tech_support,
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
    }
    row.update(addons)
    return row


SAMPLE_CUSTOMERS = [
    _customer("c-churn-1", churn="Yes", StreamingTV="Yes", StreamingMovies="Yes"),
    _customer("c-churn-2", churn="Yes", OnlineBackup="Yes"),
    _customer("c-stay-1", churn="No", StreamingTV="Yes"),
    _customer("c-stay-2", churn="No", OnlineSecurity="Yes", DeviceProtection="Yes"),
]


def _generator(drift=False, seed=1, events_per_second=10):
    g = gen.EventGenerator(
        events_per_second=events_per_second, drift=drift, rng=np.random.default_rng(seed)
    )
    g._customers = [dict(c) for c in SAMPLE_CUSTOMERS]
    g._drifted_ids = g._pick_drifted_customers(g._customers)
    return g


def _patched_start(g):
    """Context manager patching everything start() touches except the
    generator itself, so start()/stop() can be exercised without a real DB
    or broker."""
    return (
        patch.object(gen.EventGenerator, "_load_customers", return_value=[dict(SAMPLE_CUSTOMERS[0])]),
        patch.object(gen, "local_pg_dsn", return_value={}),
        patch.object(gen, "local_rabbitmq_host", return_value="localhost"),
        patch.object(gen, "pika"),
    )


class TestLifecycle:
    def test_start_is_idempotent(self):
        g = _generator()
        p1, p2, p3, p4 = _patched_start(g)
        with p1, p2, p3, p4 as fake_pika:
            fake_pika.BlockingConnection.return_value.channel.return_value = MagicMock()
            first = g.start()
            second = g.start()  # already running: no-op
            g.stop()

        assert first is True
        assert second is False

    def test_stop_before_start_is_a_no_op(self):
        g = _generator()
        assert g.stop() is False
        assert g.is_running() is False

    def test_stop_closes_the_rabbitmq_connection_cleanly(self):
        g = _generator()
        p1, p2, p3, p4 = _patched_start(g)
        with p1, p2, p3, p4 as fake_pika:
            fake_connection = MagicMock()
            fake_pika.BlockingConnection.return_value = fake_connection
            g.start()
            stopped = g.stop()

        assert stopped is True
        assert g.is_running() is False
        fake_connection.close.assert_called_once()

    def test_refuses_to_start_against_a_non_local_database(self, monkeypatch):
        monkeypatch.setenv("PG_HOST", "db.abcdefgh.supabase.co")
        g = _generator()

        with pytest.raises(NonLocalHostError):
            g.start()

        assert g.is_running() is False

    def test_status_reports_running_published_and_drift(self):
        g = _generator(drift=True)
        p1, p2, p3, p4 = _patched_start(g)
        with p1, p2, p3, p4 as fake_pika:
            fake_pika.BlockingConnection.return_value.channel.return_value = MagicMock()
            g.start()
            status = g.status()
            g.stop()

        assert status["running"] is True
        assert status["drift"] is True
        assert "events_published" in status


class TestCustomerSelection:
    def test_login_selection_weighted_against_drifted_customers(self):
        g = gen.EventGenerator(events_per_second=10, drift=True, rng=np.random.default_rng(13))
        g._customers = [_customer("c-drift", churn="No"), _customer("c-normal", churn="No")]
        g._drifted_ids = {"c-drift"}

        picks = [g._pick_customer(weighted_for_login=True)["customerID"] for _ in range(4000)]
        drift_share = picks.count("c-drift") / len(picks)

        # weights [0.5, 1.0] normalize to [1/3, 2/3]
        assert 0.27 <= drift_share <= 0.40

    def test_non_login_selection_is_uniform_regardless_of_drift(self):
        g = gen.EventGenerator(events_per_second=10, drift=True, rng=np.random.default_rng(13))
        g._customers = [_customer("c-drift", churn="No"), _customer("c-normal", churn="No")]
        g._drifted_ids = {"c-drift"}

        picks = [g._pick_customer()["customerID"] for _ in range(4000)]
        drift_share = picks.count("c-drift") / len(picks)

        assert 0.45 <= drift_share <= 0.55


class TestEventShapes:
    def test_login_event_has_the_expected_shape(self):
        g = _generator()
        queue_name, payload = g._login_event()

        assert queue_name == "login_events"
        assert payload["customer_id"] in {c["customerID"] for c in SAMPLE_CUSTOMERS}
        assert payload["session_duration_seconds"] >= 5
        assert payload["device"] in gen.DEVICES
        assert payload["event_id"]

    def test_support_ticket_event_has_the_expected_shape(self):
        g = _generator()
        queue_name, payload = g._support_ticket_event()

        assert queue_name == "support_tickets"
        assert payload["category"] in gen.TICKET_CATEGORIES
        assert isinstance(payload["resolved"], bool)
        if not payload["resolved"]:
            assert payload["resolution_time_hours"] is None

    def test_feature_usage_event_only_picks_a_service_the_customer_actually_has(self):
        g = _generator()
        g._customers = [_customer("c-only-streaming", tech_support="No", StreamingTV="Yes")]

        for _ in range(20):
            queue_name, payload = g._feature_usage_event()
            assert queue_name == "feature_usage_logs"
            assert payload["feature_name"] == "StreamingTV"

    def test_feature_usage_event_is_skipped_for_a_customer_with_no_addons(self):
        g = _generator()
        g._customers = [_customer("c-no-addons", tech_support="No")]

        assert g._feature_usage_event() is None


class TestDriftShiftsTheDistributions:
    """Statistical checks with a seeded RNG: drift mode measurably shifts
    the generated distributions, not just a single sample."""

    N = 4000

    def _resolution_times(self, drift):
        g = gen.EventGenerator(events_per_second=10, drift=drift, rng=np.random.default_rng(7))
        g._customers = [_customer("c-drift", churn="Yes")]
        g._drifted_ids = {"c-drift"} if drift else set()
        times = []
        for _ in range(self.N):
            _, payload = g._support_ticket_event()
            if payload["resolved"]:
                times.append(payload["resolution_time_hours"])
        return times

    def _usage_counts(self, drift):
        g = gen.EventGenerator(events_per_second=10, drift=drift, rng=np.random.default_rng(11))
        g._customers = [_customer("c-drift", churn="Yes", OnlineBackup="Yes")]
        g._drifted_ids = {"c-drift"} if drift else set()
        return [g._feature_usage_event()[1]["usage_count"] for _ in range(self.N)]

    def test_resolution_time_is_about_1_5x_under_drift(self):
        baseline = np.mean(self._resolution_times(drift=False))
        drifted = np.mean(self._resolution_times(drift=True))

        assert 1.45 <= drifted / baseline <= 1.55

    def test_usage_count_is_about_0_6x_under_drift(self):
        baseline = np.mean(self._usage_counts(drift=False))
        drifted = np.mean(self._usage_counts(drift=True))

        assert 0.45 <= drifted / baseline <= 0.75
