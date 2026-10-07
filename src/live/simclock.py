"""
Stage 2 simulated clock: maps real elapsed time to a simulated timestamp, at
a configurable speed. Sim time starts at END_DATE (src/pipelines/
build_features.py's own anchor - "the end of the historical data", the same
reference date src/data_gen/generate_events.py generated events up to) and
advances by `speed` simulated days per real second, default 1.0 (1 real
second = 1 simulated day).

The generator (src/live/generator.py) stamps every event with
clock.now() instead of wall-clock "now", so simulated events land right
after the historical data instead of ~2 years after it (see the Stage 2
investigation: build_features.py's "recent 30d" window is anchored to a
fixed as_of, not to real now() - a wall-clock-stamped live event's days_ago
comes out deeply negative, which is *less* than the 30-day recent cutoff,
so it was being counted as "recent" by accident, not excluded as "too old"
the way it looks like it should be).

Thread-safe: now()/reset()/set_speed() all take the same lock, since the
generator thread calls now() for every event while a control_api request
can reset() or change speed concurrently.
"""

import threading
import time

import pandas as pd

from src.pipelines.build_features import END_DATE


class SimClock:
    def __init__(self, start=None, speed=1.0, real_time_fn=time.monotonic):
        self._real_time_fn = real_time_fn
        self._lock = threading.Lock()
        self._start_sim_time = start if start is not None else END_DATE
        self.speed = speed  # simulated days per real second
        self._real_start = self._real_time_fn()

    def now(self):
        with self._lock:
            elapsed_real_seconds = self._real_time_fn() - self._real_start
            elapsed_sim_days = elapsed_real_seconds * self.speed
            return self._start_sim_time + pd.Timedelta(days=elapsed_sim_days)

    def reset(self, start=None):
        """Back to start (default END_DATE) and restarts the real-time
        reference point. Returns the exact start value: a caller that needs
        "as_of = the reset point" (POST /pipeline/reset's rebuild) should
        use this return value, not a subsequent now() - any real time spent
        between reset() and now() (e.g. reset_event_tables()'s bulk reload)
        would otherwise leak a few sim-minutes into "as_of", which is enough
        to flip a handful of historical events across a day-boundary
        relative to END_DATE and make the rebuilt customer_features subtly
        non-reproducible across repeated resets - caught live: two resets
        taken a different number of real seconds apart gave different
        customer_features checksums even though the event tables' checksums
        matched exactly."""
        with self._lock:
            self._start_sim_time = start if start is not None else END_DATE
            self._real_start = self._real_time_fn()
            return self._start_sim_time

    def set_speed(self, speed):
        """Changes speed without a time jump: the current sim time becomes
        the new reference point, so a mid-flight speed change doesn't warp
        time between the old and new speed's reckoning."""
        with self._lock:
            elapsed_real_seconds = self._real_time_fn() - self._real_start
            elapsed_sim_days = elapsed_real_seconds * self.speed
            self._start_sim_time = self._start_sim_time + pd.Timedelta(days=elapsed_sim_days)
            self._real_start = self._real_time_fn()
            self.speed = speed

    def status(self):
        now = self.now()
        with self._lock:
            speed = self.speed
        return {"sim_time": now.isoformat(sep=" "), "speed_sim_days_per_real_second": speed}
