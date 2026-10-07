"""
Unit tests for src/live/safety.py: every Stage 1 live-data component must
refuse to start unless the database (and RabbitMQ) host is localhost or
127.0.0.1 - this must never reach Supabase or Upstash.
"""

import pytest

from src.live import safety


class TestAssertLocalHost:
    @pytest.mark.parametrize("host", ["localhost", "127.0.0.1"])
    def test_allows_local_hosts(self, host):
        assert safety.assert_local_host(host, "database") == host

    @pytest.mark.parametrize(
        "host",
        [
            "db.abcdefgh.supabase.co",
            "some-project.upstash.io",
            "203.0.113.5",
            "example.com",
            "",
        ],
    )
    def test_refuses_non_local_hosts(self, host):
        with pytest.raises(safety.NonLocalHostError):
            safety.assert_local_host(host, "database")


class TestLocalPgDsn:
    def test_defaults_to_localhost_and_the_usual_pg_vars(self, monkeypatch):
        monkeypatch.delenv("PG_HOST", raising=False)
        monkeypatch.delenv("PG_PORT", raising=False)
        monkeypatch.delenv("PG_DATABASE", raising=False)

        dsn = safety.local_pg_dsn()

        assert dsn["host"] == "localhost"
        assert dsn["port"] == "5432"
        assert dsn["dbname"] == "ml_insight"

    def test_refuses_a_non_local_pg_host(self, monkeypatch):
        monkeypatch.setenv("PG_HOST", "db.abcdefgh.supabase.co")
        with pytest.raises(safety.NonLocalHostError):
            safety.local_pg_dsn()

    def test_127_0_0_1_is_accepted(self, monkeypatch):
        monkeypatch.setenv("PG_HOST", "127.0.0.1")
        assert safety.local_pg_dsn()["host"] == "127.0.0.1"


class TestLocalRabbitmqHost:
    def test_defaults_to_localhost(self, monkeypatch):
        monkeypatch.delenv("RABBITMQ_HOST", raising=False)
        assert safety.local_rabbitmq_host() == "localhost"

    def test_refuses_a_non_local_rabbitmq_host(self, monkeypatch):
        monkeypatch.setenv("RABBITMQ_HOST", "some-remote-broker.example.com")
        with pytest.raises(safety.NonLocalHostError):
            safety.local_rabbitmq_host()
