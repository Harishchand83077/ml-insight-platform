"""
Unit tests for src/live/simclock.py: the clock advances at the configured
speed using a fake time source (no real sleeping), reset() restarts it from
a new point, and set_speed() changes pace without jumping time.
"""

import pandas as pd

from src.live.simclock import SimClock
from src.pipelines.build_features import END_DATE


class FakeRealTime:
    """A controllable stand-in for time.monotonic(): advance() moves it
    forward by a given number of real seconds, with no real sleeping."""

    def __init__(self, start=0.0):
        self._t = start

    def advance(self, seconds):
        self._t += seconds

    def __call__(self):
        return self._t


class TestAdvancesAtConfiguredSpeed:
    def test_default_speed_is_one_sim_day_per_real_second(self):
        real_time = FakeRealTime()
        start = pd.Timestamp("2024-06-30")
        clock = SimClock(start=start, speed=1.0, real_time_fn=real_time)

        real_time.advance(5)

        assert clock.now() == start + pd.Timedelta(days=5)

    def test_half_speed_is_half_a_sim_day_per_real_second(self):
        real_time = FakeRealTime()
        start = pd.Timestamp("2024-06-30")
        clock = SimClock(start=start, speed=0.5, real_time_fn=real_time)

        real_time.advance(10)

        assert clock.now() == start + pd.Timedelta(days=5)

    def test_zero_elapsed_real_time_gives_exactly_the_start_time(self):
        real_time = FakeRealTime()
        start = pd.Timestamp("2024-06-30")
        clock = SimClock(start=start, real_time_fn=real_time)

        assert clock.now() == start

    def test_defaults_to_build_features_end_date(self):
        real_time = FakeRealTime()
        clock = SimClock(real_time_fn=real_time)

        assert clock.now() == END_DATE


class TestReset:
    def test_reset_restarts_from_end_date_by_default(self):
        real_time = FakeRealTime()
        clock = SimClock(start=pd.Timestamp("2025-01-01"), real_time_fn=real_time)
        real_time.advance(100)

        clock.reset()

        assert clock.now() == END_DATE

    def test_reset_to_an_explicit_start(self):
        real_time = FakeRealTime()
        clock = SimClock(real_time_fn=real_time)
        new_start = pd.Timestamp("2025-01-01")

        clock.reset(start=new_start)

        assert clock.now() == new_start

    def test_reset_returns_the_exact_start_value(self):
        # A caller needing "as_of = the reset point" (POST /pipeline/reset's
        # rebuild) must use this return value, not a later now() call: any
        # real time spent between reset() and now() would otherwise leak
        # into "as_of" by speed sim-days per real second.
        real_time = FakeRealTime()
        clock = SimClock(speed=1.0, real_time_fn=real_time)

        returned = clock.reset()
        assert returned == END_DATE

        real_time.advance(5)  # time spent doing other work after reset()
        assert clock.now() == END_DATE + pd.Timedelta(days=5)  # now() moved on...
        assert returned == END_DATE  # ...but the earlier return value didn't

    def test_time_advances_again_after_a_reset(self):
        real_time = FakeRealTime()
        clock = SimClock(speed=1.0, real_time_fn=real_time)
        real_time.advance(50)
        clock.reset()

        real_time.advance(5)

        assert clock.now() == END_DATE + pd.Timedelta(days=5)


class TestSetSpeed:
    def test_changing_speed_does_not_jump_time(self):
        real_time = FakeRealTime()
        clock = SimClock(speed=1.0, real_time_fn=real_time)
        real_time.advance(5)
        before = clock.now()

        clock.set_speed(2.0)

        assert clock.now() == before  # no jump at the instant of the change

    def test_new_speed_takes_effect_after_the_change(self):
        real_time = FakeRealTime()
        clock = SimClock(speed=1.0, real_time_fn=real_time)
        real_time.advance(5)
        clock.set_speed(2.0)
        before = clock.now()

        real_time.advance(3)

        assert clock.now() == before + pd.Timedelta(days=6)  # 3 real seconds x 2.0 sim-days/s


class TestStatus:
    def test_status_reports_sim_time_and_speed(self):
        real_time = FakeRealTime()
        clock = SimClock(speed=3.0, real_time_fn=real_time)

        status = clock.status()

        assert status == {
            "sim_time": clock.now().isoformat(sep=" "),
            "speed_sim_days_per_real_second": 3.0,
        }
