"""
Stage 1 live synthetic event stream: a background thread that picks random
existing customers from the LOCAL customers table and publishes login,
support-ticket, and feature-usage events to RabbitMQ in real time,
timestamped with the Stage 2 simulated clock's current time (src/live/
simclock.py) - not wall-clock now(). Events used to be wall-clock-stamped;
see the Stage 2 investigation for why that was a problem (a live event
stamped "now" lands ~2 years after the historical data, which
build_features.py's fixed as_of anchor was silently treating as "recent").

The churn-conditioned shape of each event - device mix, session-duration
scaling by engagement trend, ticket base rate, resolve probability,
resolution time, feature-usage rate - reuses generate_events.py's constants
and pure per-event helper functions directly (see the import below). This is
the same model the batch CSVs are generated from, not a second copy of it:
if generate_events.py's numbers change, this stream picks the change up
automatically.

Drift is a live-only concept layered on top (there's no "drift" in the
static batch model, which has no notion of real time passing): while
drift=True, a configurable fraction of customers has shifted engagement for
as long as drift stays on - half the login frequency, 0.6x usage counts,
1.5x resolution times. See DRIFT_* below.

Safety: start() refuses to run unless both the Postgres and RabbitMQ hosts
are local (src.live.safety) - this must never reach Supabase or Upstash.
"""

import json
import threading
import time
import uuid

import numpy as np
import pika
import psycopg2
import psycopg2.extras

from src.data_gen.generate_events import (
    ADDON_SERVICES,
    CHURNED_TREND_PROBS,
    DEVICE_PROBS,
    DEVICES,
    NON_CHURNED_TREND_PROBS,
    TICKET_CATEGORIES,
    TREND_CATEGORIES,
    feature_usage_lambda,
    ticket_resolution_mean,
    ticket_resolution_time,
    ticket_resolve_prob,
    trend_duration_low,
)
from src.live.safety import local_pg_dsn, local_rabbitmq_host
from src.live.simclock import SimClock

QUEUES = ["login_events", "support_tickets", "feature_usage_logs"]

# Logins dominate the batch data too (10-30 per customer over 90 days, vs
# 0-5 tickets and a handful of usage events per active add-on), so the live
# stream mirrors that skew rather than splitting event types evenly.
EVENT_TYPE_WEIGHTS = {"login": 0.7, "support_ticket": 0.15, "feature_usage": 0.15}

# Drift: the fraction of customers whose engagement is shifted, and by how
# much, while drift mode is on. These numbers are the live generator's own
# spec, not generate_events.py's - there is no "drift" in the static model.
DRIFT_FRACTION = 0.3
DRIFT_LOGIN_FREQUENCY_SCALE = 0.5
DRIFT_USAGE_COUNT_SCALE = 0.6
DRIFT_RESOLUTION_TIME_SCALE = 1.5

CUSTOMER_SELECT_SQL = """
    SELECT
        customer_id AS "customerID",
        churn AS "Churn",
        tech_support AS "TechSupport",
        online_security AS "OnlineSecurity",
        online_backup AS "OnlineBackup",
        device_protection AS "DeviceProtection",
        streaming_tv AS "StreamingTV",
        streaming_movies AS "StreamingMovies"
    FROM customers
"""


class EventGenerator:
    """events_per_second: target publish rate. drift: whether a configurable
    fraction of customers has shifted engagement right now. rng: an optional
    seeded numpy Generator, for reproducible tests - defaults to an
    unseeded one for real use. clock: the SimClock events are stamped with
    (defaults to a fresh one starting at END_DATE) - control_api.py shares
    one clock across the generator and /generator/status."""

    def __init__(self, events_per_second=5.0, drift=False, drift_fraction=DRIFT_FRACTION, rng=None, clock=None):
        self.events_per_second = events_per_second
        self.drift = drift
        self.drift_fraction = drift_fraction
        self._rng = rng if rng is not None else np.random.default_rng()
        self.clock = clock if clock is not None else SimClock()
        self._customers = None
        self._drifted_ids = set()
        self._thread = None
        self._stop_event = threading.Event()
        self._lifecycle_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self.published = 0
        self.published_by_queue = {q: 0 for q in QUEUES}

    # --- lifecycle ----------------------------------------------------

    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self, events_per_second=None, drift=None, drift_fraction=None):
        """Idempotent: a no-op (returns False) if already running. Params
        are only applied on the transition from stopped to running - call
        stop() first to change the rate, drift, or drift_fraction of a
        running generator."""
        with self._lifecycle_lock:
            if self.is_running():
                return False
            if events_per_second is not None:
                self.events_per_second = events_per_second
            if drift is not None:
                self.drift = drift
            if drift_fraction is not None:
                self.drift_fraction = drift_fraction

            local_pg_dsn()  # raises NonLocalHostError before anything else if misconfigured
            local_rabbitmq_host()

            self._customers = self._load_customers()
            self._drifted_ids = self._pick_drifted_customers(self._customers)
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            return True

    def stop(self):
        """Idempotent: a no-op (returns False) if already stopped. Blocks
        until the publish loop's current iteration (if any) finishes and
        the RabbitMQ connection is closed - there's no in-flight publish
        left running when this returns."""
        with self._lifecycle_lock:
            if not self.is_running():
                return False
            self._stop_event.set()
        self._thread.join()
        with self._lifecycle_lock:
            self._thread = None
        return True

    def status(self):
        with self._stats_lock:
            status = {
                "running": self.is_running(),
                "events_published": self.published,
                "events_published_by_queue": dict(self.published_by_queue),
                "drift": self.drift,
                "drift_fraction": self.drift_fraction,
            }
        status.update(self.clock.status())
        return status

    def reset_state(self):
        """Clears published counters, the drifted-customer set, and the
        cached customer pool, so the next start() reloads customers fresh
        and begins counting from zero. Does not touch the clock - callers
        (POST /pipeline/reset) reset that separately, since it has its own
        start point. Must be called while stopped."""
        if self.is_running():
            raise RuntimeError("reset_state() requires the generator to be stopped first")
        with self._stats_lock:
            self.published = 0
            self.published_by_queue = {q: 0 for q in QUEUES}
        self._customers = None
        self._drifted_ids = set()

    # --- customer pool --------------------------------------------------

    def _load_customers(self):
        conn = psycopg2.connect(**local_pg_dsn())
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(CUSTOMER_SELECT_SQL)
                rows = cur.fetchall()
        finally:
            conn.close()
        if not rows:
            raise RuntimeError(
                "the local customers table is empty - run load_static_data.py first"
            )
        return [dict(r) for r in rows]

    def _pick_drifted_customers(self, customers):
        if not self.drift:
            return set()
        n = max(1, int(len(customers) * self.drift_fraction))
        chosen = self._rng.choice(len(customers), size=n, replace=False)
        return {customers[i]["customerID"] for i in chosen}

    def _is_drifted(self, customer):
        return self.drift and customer["customerID"] in self._drifted_ids

    def _pick_customer(self, weighted_for_login=False):
        if not weighted_for_login or not self._drifted_ids:
            return self._customers[self._rng.integers(len(self._customers))]
        weights = np.array(
            [
                DRIFT_LOGIN_FREQUENCY_SCALE if c["customerID"] in self._drifted_ids else 1.0
                for c in self._customers
            ]
        )
        idx = self._rng.choice(len(self._customers), p=weights / weights.sum())
        return self._customers[idx]

    # --- event generation (reuses generate_events.py's churn-conditioned model) --

    def _login_event(self):
        # Drift's "login frequency x0.5" is expressed here as a selection
        # weight: a drifted customer is half as likely to be the one a given
        # login tick is generated for, rather than scaling a per-event field.
        customer = self._pick_customer(weighted_for_login=True)
        churned = customer["Churn"] == "Yes"
        trend_probs = CHURNED_TREND_PROBS if churned else NON_CHURNED_TREND_PROBS
        trend = self._rng.choice(TREND_CATEGORIES, p=trend_probs)
        duration_low = trend_duration_low(self._rng, trend)
        duration = max(5, int(self._rng.lognormal(mean=6.2, sigma=0.5) * duration_low))
        return "login_events", {
            "event_id": str(uuid.uuid4()),
            "customer_id": customer["customerID"],
            "timestamp": self.clock.now().isoformat(sep=" "),
            "session_duration_seconds": duration,
            "device": str(self._rng.choice(DEVICES, p=DEVICE_PROBS)),
        }

    def _support_ticket_event(self):
        customer = self._pick_customer()
        churned = customer["Churn"] == "Yes"
        resolve_prob = ticket_resolve_prob(self._rng, churned)
        resolved = bool(self._rng.random() < resolve_prob)
        resolution_time = None
        if resolved:
            resolution_mean = ticket_resolution_mean(self._rng, churned)
            resolution_time = ticket_resolution_time(self._rng, resolution_mean)
            if self._is_drifted(customer):
                resolution_time = round(resolution_time * DRIFT_RESOLUTION_TIME_SCALE, 2)
        return "support_tickets", {
            "event_id": str(uuid.uuid4()),
            "customer_id": customer["customerID"],
            "timestamp": self.clock.now().isoformat(sep=" "),
            "category": str(self._rng.choice(TICKET_CATEGORIES)),
            "resolved": resolved,
            "resolution_time_hours": resolution_time,
        }

    def _feature_usage_event(self):
        customer = self._pick_customer()
        churned = customer["Churn"] == "Yes"
        active_services = [s for s in ADDON_SERVICES if customer.get(s) == "Yes"]
        if not active_services:
            return None  # this customer has no add-ons; nothing to generate
        service = active_services[self._rng.integers(len(active_services))]
        lam = feature_usage_lambda(self._rng, churned)
        usage_count = int(self._rng.poisson(lam)) + 1
        if self._is_drifted(customer):
            usage_count = max(1, int(round(usage_count * DRIFT_USAGE_COUNT_SCALE)))
        return "feature_usage_logs", {
            "event_id": str(uuid.uuid4()),
            "customer_id": customer["customerID"],
            "timestamp": self.clock.now().isoformat(sep=" "),
            "feature_name": service,
            "usage_count": usage_count,
        }

    def _generate_one(self):
        kinds = list(EVENT_TYPE_WEIGHTS)
        weights = list(EVENT_TYPE_WEIGHTS.values())
        kind = self._rng.choice(kinds, p=weights)
        if kind == "login":
            return self._login_event()
        if kind == "support_ticket":
            return self._support_ticket_event()
        return self._feature_usage_event()

    # --- publish loop ---------------------------------------------------

    def _run(self):
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=local_rabbitmq_host())
        )
        try:
            channel = connection.channel()
            channel.confirm_delivery()  # publisher confirms: basic_publish blocks for the ack
            for queue_name in QUEUES:
                channel.queue_declare(queue=queue_name, durable=True)

            interval = 1.0 / self.events_per_second if self.events_per_second > 0 else 0.0
            next_tick = time.monotonic()
            while not self._stop_event.is_set():
                result = self._generate_one()
                if result is not None:
                    queue_name, payload = result
                    channel.basic_publish(
                        exchange="",
                        routing_key=queue_name,
                        body=json.dumps(payload),
                        properties=pika.BasicProperties(delivery_mode=2),  # persistent
                    )
                    with self._stats_lock:
                        self.published += 1
                        self.published_by_queue[queue_name] += 1

                next_tick += interval
                remaining = next_tick - time.monotonic()
                if remaining > 0:
                    # wakes immediately on stop() instead of sleeping blindly,
                    # so stop() doesn't have to wait out a full idle tick
                    self._stop_event.wait(remaining)
        finally:
            connection.close()
