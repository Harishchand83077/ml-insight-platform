"""
Unit tests for the column-allowlist defense in src/agent/tools.py's
DB-backed tools (get_churn_rate_by_column, get_customer_count). No
database connection is made or needed: for a rejected (disallowed)
column, run_read (src/common/db.py) is patched to raise
AssertionError, so the test fails loudly if validation ever falls
through to a real DB call.
"""

from unittest.mock import patch

import pytest

from src.agent.tools import ALLOWED_COLUMNS, get_churn_rate_by_column, get_customer_count

INJECTION_ATTEMPT = "DROP TABLE customers; --"


def _never_connect():
    return patch("src.agent.tools.run_read", side_effect=AssertionError("must not touch the DB"))


def _connect_reaches_here():
    return patch("src.agent.tools.run_read", side_effect=RuntimeError("reached DB call"))


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
