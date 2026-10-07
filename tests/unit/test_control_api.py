"""
Unit tests for src/live/control_api.py's queue-depth-growing-with-no-
consumers warning (detect_queue_warnings): /generator/status and the
control panel should surface this when nothing is reading a queue and it's
piling up - the "event_consumer.py isn't running" case.
"""

from src.live.control_api import detect_queue_warnings


class TestDetectQueueWarnings:
    def test_no_warning_on_the_first_call_ever(self):
        # nothing to compare against yet - growth can't be judged from a
        # single reading.
        queue_info = {"login_events": {"messages": 500, "consumers": 0}}

        warnings, updated = detect_queue_warnings(queue_info, previous_depths={})

        assert warnings == []
        assert updated == {"login_events": 500}

    def test_warns_when_depth_grew_with_zero_consumers(self):
        queue_info = {"login_events": {"messages": 550, "consumers": 0}}

        warnings, updated = detect_queue_warnings(queue_info, previous_depths={"login_events": 500})

        assert len(warnings) == 1
        assert "login_events" in warnings[0]
        assert "500" in warnings[0] and "550" in warnings[0]
        assert "0 consumers" in warnings[0]
        assert updated == {"login_events": 550}

    def test_no_warning_when_a_consumer_is_present_even_if_depth_grew(self):
        queue_info = {"login_events": {"messages": 550, "consumers": 1}}

        warnings, _ = detect_queue_warnings(queue_info, previous_depths={"login_events": 500})

        assert warnings == []

    def test_no_warning_when_depth_is_flat_with_zero_consumers(self):
        queue_info = {"login_events": {"messages": 500, "consumers": 0}}

        warnings, _ = detect_queue_warnings(queue_info, previous_depths={"login_events": 500})

        assert warnings == []

    def test_no_warning_when_depth_is_shrinking_with_zero_consumers(self):
        # a leftover backlog that's draining on its own isn't the problem
        # this warning exists to catch.
        queue_info = {"login_events": {"messages": 400, "consumers": 0}}

        warnings, _ = detect_queue_warnings(queue_info, previous_depths={"login_events": 500})

        assert warnings == []

    def test_no_crash_and_no_warning_when_rabbitmq_is_unreachable(self):
        queue_info = {"login_events": {"messages": None, "consumers": None, "error": "timeout"}}

        warnings, updated = detect_queue_warnings(queue_info, previous_depths={"login_events": 500})

        assert warnings == []
        assert updated == {"login_events": None}

    def test_only_the_qualifying_queue_is_flagged_among_several(self):
        queue_info = {
            "login_events": {"messages": 550, "consumers": 0},  # growing, no consumer: warn
            "support_tickets": {"messages": 120, "consumers": 1},  # has a consumer: no warn
            "feature_usage_logs": {"messages": 80, "consumers": 0},  # flat: no warn
        }
        previous = {"login_events": 500, "support_tickets": 100, "feature_usage_logs": 80}

        warnings, updated = detect_queue_warnings(queue_info, previous)

        assert len(warnings) == 1
        assert "login_events" in warnings[0]
        assert updated == {"login_events": 550, "support_tickets": 120, "feature_usage_logs": 80}
