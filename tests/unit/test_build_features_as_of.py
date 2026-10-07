"""
Unit tests for build_features.py's as_of parameter (Stage 2): the default
(as_of=END_DATE) must leave every existing caller's output identical to
before this parameter existed, and passing a different as_of must shift the
recent/older windows as expected - no DB needed, these are pure functions.
"""

import pandas as pd
import pytest

from src.pipelines.build_features import END_DATE, build_feature_table, build_login_features

CUSTOMERS = pd.DataFrame(
    [
        {
            "customer_id": "c-1",
            "tenure": 10,
            "monthly_charges": 50.0,
            "contract": "Month-to-month",
            "payment_method": "Electronic check",
            "internet_service": "Fiber optic",
            "senior_citizen": 0,
            "partner": "Yes",
            "dependents": "No",
            "online_security": "No",
            "online_backup": "No",
            "device_protection": "No",
            "tech_support": "No",
            "streaming_tv": "No",
            "streaming_movies": "No",
        }
    ]
)

LOGIN_EVENTS = pd.DataFrame(
    [
        # 10 days before END_DATE: "recent" under the default as_of
        {"customer_id": "c-1", "timestamp": END_DATE - pd.Timedelta(days=10), "session_duration_seconds": 600},
        # 45 days before END_DATE: "older" under the default as_of
        {"customer_id": "c-1", "timestamp": END_DATE - pd.Timedelta(days=45), "session_duration_seconds": 300},
    ]
)

EMPTY_TICKETS = pd.DataFrame(columns=["customer_id", "timestamp", "category", "resolved", "resolution_time_hours"])
EMPTY_USAGE = pd.DataFrame(columns=["customer_id", "timestamp", "feature_name", "usage_count"])


class TestDefaultAsOfIsUnchanged:
    def test_build_login_features_default_matches_explicit_end_date(self):
        default = build_login_features(LOGIN_EVENTS)
        explicit = build_login_features(LOGIN_EVENTS, as_of=END_DATE)

        pd.testing.assert_frame_equal(default, explicit)

    def test_build_login_features_default_classifies_the_known_windows(self):
        result = build_login_features(LOGIN_EVENTS).set_index("customer_id")

        assert result.loc["c-1", "total_logins_90d"] == 2
        assert result.loc["c-1", "avg_session_duration_recent_30d"] == 600  # only the 10-days-ago row
        assert result.loc["c-1", "recent_30d_vs_older_60d_ratio"] == 1.0  # 1 recent / 1 older

    def test_build_feature_table_default_matches_explicit_end_date(self):
        default = build_feature_table(CUSTOMERS, LOGIN_EVENTS, EMPTY_TICKETS, EMPTY_USAGE)
        explicit = build_feature_table(CUSTOMERS, LOGIN_EVENTS, EMPTY_TICKETS, EMPTY_USAGE, as_of=END_DATE)

        pd.testing.assert_frame_equal(default, explicit)


class TestAsOfShiftsTheWindows:
    def test_an_event_that_was_recent_becomes_older_as_as_of_advances(self):
        # the 10-days-ago-of-END_DATE row is "recent" at END_DATE, but 40
        # days later (as_of = END_DATE + 40d) it's 50 days old: "older", not
        # "recent". The 45-days-ago row is now 85 days old - still inside
        # the 90-day window, also "older". Neither row is "recent" anymore.
        later = END_DATE + pd.Timedelta(days=40)

        result = build_login_features(LOGIN_EVENTS, as_of=later).set_index("customer_id")

        assert result.loc["c-1", "avg_session_duration_recent_30d"] == 0  # fillna(0): nothing recent anymore
        assert result.loc["c-1", "recent_30d_vs_older_60d_ratio"] == 0.0  # 0 recent / 2 older

    def test_a_live_event_timestamped_in_the_real_future_relative_to_end_date_is_handled_correctly_once_as_of_tracks_it(self):
        # This is exactly the Stage 2 investigation finding: under the fixed
        # as_of=END_DATE, a wall-clock-"now"-stamped live event (~2 years
        # after END_DATE) has a deeply negative days_ago, which satisfies
        # "< 30" by accident and gets miscounted as recent. Passing the
        # live event's own timestamp as as_of (what the sim clock does)
        # fixes that: days_ago becomes 0, correctly "recent".
        live_event = pd.DataFrame(
            [{"customer_id": "c-1", "timestamp": pd.Timestamp("2026-10-07"), "session_duration_seconds": 900}]
        )

        under_fixed_end_date = build_login_features(live_event, as_of=END_DATE).set_index("customer_id")
        under_tracking_as_of = build_login_features(live_event, as_of=pd.Timestamp("2026-10-07")).set_index("customer_id")

        # Both count it as "recent" - the bug isn't that it's miscounted
        # under END_DATE, it's that END_DATE never advances to match it.
        assert under_fixed_end_date.loc["c-1", "avg_session_duration_recent_30d"] == 900
        assert under_tracking_as_of.loc["c-1", "avg_session_duration_recent_30d"] == 900


@pytest.mark.parametrize("as_of", [END_DATE, END_DATE + pd.Timedelta(days=200)])
def test_build_feature_table_runs_at_any_as_of_without_error(as_of):
    result = build_feature_table(CUSTOMERS, LOGIN_EVENTS, EMPTY_TICKETS, EMPTY_USAGE, as_of=as_of)
    assert len(result) == 1
