"""
Unit tests for the pure (no-I/O) logic in build_features.py and
load_static_data.py, using small in-memory fixtures - no Postgres.
"""

import pandas as pd

from src.pipelines.build_features import EVENT_DERIVED_COLUMNS, build_customer_features, build_feature_table
from src.pipelines.load_static_data import clean_total_charges


def make_customers_df():
    base = {
        "customer_id": None,
        "tenure": 12,
        "monthly_charges": 50.0,
        "contract": "Month-to-month",
        "payment_method": "Electronic check",
        "internet_service": "Fiber optic",
        "senior_citizen": 0,
        "partner": "No",
        "dependents": "No",
        "online_security": "No",
        "online_backup": "No",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "No",
        "streaming_movies": "No",
    }

    active_addons = dict(base, customer_id="cust-with-addons")
    active_addons.update(online_security="Yes", online_backup="Yes", tech_support="Yes")

    no_addons = dict(base, customer_id="cust-no-events")  # also used as the "no events" customer

    no_events_partner = dict(base, customer_id="cust-with-events")

    return pd.DataFrame([active_addons, no_addons, no_events_partner])


class TestNumAddonsActive:
    def test_counts_yes_across_six_addon_columns(self):
        customers = make_customers_df()
        result = build_customer_features(customers)

        counts = result.set_index("customer_id")["num_addons_active"]
        assert counts["cust-with-addons"] == 3
        assert counts["cust-no-events"] == 0
        assert counts["cust-with-events"] == 0

    def test_addon_columns_dropped_from_output(self):
        customers = make_customers_df()
        result = build_customer_features(customers)

        for col in ["online_security", "online_backup", "device_protection", "tech_support", "streaming_tv", "streaming_movies"]:
            assert col not in result.columns


class TestZeroFillForNoEvents:
    def test_customer_with_no_rows_in_any_event_table_gets_zero_not_null(self):
        customers = make_customers_df()

        # only "cust-with-events" has any events; the other two have none
        login_events = pd.DataFrame(
            {
                "customer_id": ["cust-with-events"],
                "timestamp": [pd.Timestamp("2024-06-15")],
                "session_duration_seconds": [400],
            }
        )
        support_tickets = pd.DataFrame(
            {
                "customer_id": ["cust-with-events"],
                "timestamp": [pd.Timestamp("2024-06-10")],
                "resolved": [True],
                "resolution_time_hours": [5.0],
            }
        )
        feature_usage_logs = pd.DataFrame(
            {
                "customer_id": ["cust-with-events"],
                "timestamp": [pd.Timestamp("2024-06-12")],
                "usage_count": [8],
            }
        )

        features = build_feature_table(customers, login_events, support_tickets, feature_usage_logs)
        features = features.set_index("customer_id")

        for customer_id in ["cust-no-events", "cust-with-addons"]:
            row = features.loc[customer_id, EVENT_DERIVED_COLUMNS]
            assert not row.isna().any(), f"{customer_id} has NaN event-derived features, expected 0-fill"
            assert (row == 0).all(), f"{customer_id} expected all-zero event features, got {row.to_dict()}"

    def test_customer_with_events_gets_nonzero_features(self):
        customers = make_customers_df()
        login_events = pd.DataFrame(
            {
                "customer_id": ["cust-with-events"] * 3,
                "timestamp": [pd.Timestamp("2024-06-15")] * 3,
                "session_duration_seconds": [400, 500, 600],
            }
        )
        empty_tickets = pd.DataFrame(columns=["customer_id", "timestamp", "resolved", "resolution_time_hours"])
        empty_usage = pd.DataFrame(columns=["customer_id", "timestamp", "usage_count"])

        features = build_feature_table(customers, login_events, empty_tickets, empty_usage)
        features = features.set_index("customer_id")

        assert features.loc["cust-with-events", "total_logins_90d"] == 3
        assert features.loc["cust-with-events", "avg_session_duration_recent_30d"] == 500
        # customers with no tickets/usage still get 0 for those specifically
        assert features.loc["cust-with-events", "num_tickets"] == 0
        assert features.loc["cust-with-events", "avg_usage_count"] == 0


class TestCleanTotalCharges:
    def test_blank_and_whitespace_become_zero(self):
        df = pd.DataFrame({"TotalCharges": ["", "  ", "29.85", "1889.5"]})
        result = clean_total_charges(df)

        assert result["TotalCharges"].tolist() == [0.0, 0.0, 29.85, 1889.5]

    def test_result_is_numeric_dtype(self):
        df = pd.DataFrame({"TotalCharges": ["", "100.5"]})
        result = clean_total_charges(df)

        assert pd.api.types.is_numeric_dtype(result["TotalCharges"])

    def test_does_not_mutate_input_in_place(self):
        df = pd.DataFrame({"TotalCharges": [""]})
        clean_total_charges(df)

        # original column untouched (still the raw string) - clean_total_charges
        # returns a new frame rather than mutating the caller's copy
        assert df["TotalCharges"].tolist() == [""]
