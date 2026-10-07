"""
Unit tests for the column-allowlist defense in src/agent/tools.py's
DB-backed tools (get_churn_rate_by_column, get_customer_count). No
database connection is made or needed: for a rejected (disallowed)
column, run_read (src/common/db.py) is patched to raise
AssertionError, so the test fails loudly if validation ever falls
through to a real DB call.

Both get_shared_pool and run_read are patched together, not run_read
alone: tools.py's _run_bounded_read calls get_shared_pool() to build the
pool argument before it ever calls run_read(pool, query) - and
get_shared_pool() (via make_pool/psycopg2.pool.SimpleConnectionPool)
opens a real connection immediately on first use, regardless of what
run_read does with the result. Patching run_read alone leaves that real
connection attempt in place; tests/unit/conftest.py's guard exists to
catch exactly this if it's ever missed again.
"""

from contextlib import contextmanager
from unittest.mock import patch

import pytest

from src.agent.tools import ALLOWED_COLUMNS, get_churn_rate_by_column, get_customer_count

INJECTION_ATTEMPT = "DROP TABLE customers; --"


@contextmanager
def _never_connect():
    with patch("src.agent.tools.get_shared_pool", side_effect=AssertionError("must not touch the DB")), \
         patch("src.agent.tools.run_read", side_effect=AssertionError("must not touch the DB")):
        yield


@contextmanager
def _connect_reaches_here():
    with patch("src.agent.tools.get_shared_pool", return_value="fake-pool-not-a-real-connection"), \
         patch("src.agent.tools.run_read", side_effect=RuntimeError("reached DB call")):
        yield


class TestGetChurnRateByColumnAllowlist:
    def test_rejects_injection_attempt(self):
        with _never_connect():
            result = get_churn_rate_by_column.invoke({"column_name": INJECTION_ATTEMPT})

        assert "not an allowed column" in result
        assert INJECTION_ATTEMPT in result

    def test_rejects_arbitrary_unknown_column(self):
        with _never_connect():
            result = get_churn_rate_by_column.invoke({"column_name": "not_a_real_column"})

        assert "not an allowed column" in result

    @pytest.mark.parametrize("column", sorted(ALLOWED_COLUMNS))
    def test_every_allowlisted_column_passes_validation(self, column):
        # a real DB call happens next for a valid column - proven here by
        # the patched run_read raising instead of silently no-opping
        with _connect_reaches_here():
            result = get_churn_rate_by_column.invoke({"column_name": column})
        # validation passed and the query was reached; the DB error comes back as a tool message
        assert "reached DB call" in result


class TestGetCustomerCountAllowlist:
    def test_rejects_injection_attempt_in_filter_key(self):
        with _never_connect():
            result = get_customer_count.invoke({"filters": {INJECTION_ATTEMPT: "x"}})

        assert "not allowed" in result
        assert INJECTION_ATTEMPT in result

    def test_rejects_one_bad_column_even_when_others_are_valid(self):
        with _never_connect():
            result = get_customer_count.invoke(
                {"filters": {"contract": "Month-to-month", INJECTION_ATTEMPT: "x"}}
            )

        assert "not allowed" in result

    def test_valid_single_filter_passes_validation(self):
        with _connect_reaches_here():
            result = get_customer_count.invoke({"filters": {"contract": "Month-to-month"}})
        assert "reached DB call" in result

    def test_empty_filters_means_no_filter_and_still_passes_validation(self):
        with _connect_reaches_here():
            result = get_customer_count.invoke({"filters": {}})
        assert "reached DB call" in result
