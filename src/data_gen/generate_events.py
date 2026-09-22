"""
Generates synthetic customer behavioral events for the Telco churn dataset
(data/raw/telco_churn.csv.csv), correlated with each customer's real Churn
label, Contract type, and service columns so downstream models see
realistic, IMPERFECT signal rather than a clean split. Output is written to
data/synthetic/ as three CSVs and NOT published to RabbitMQ (that comes
after the data is validated).

Correlation logic per event type:

1. login_events (10-30 events per customer, spread over the last 90 days)
   - Each customer is randomly assigned a 30-day engagement trend -
     "declining", "flat", or "increasing" - drawn from a churn-conditional
     distribution: churned customers draw from [0.45, 0.35, 0.20]
     (declining/flat/increasing), non-churned from [0.30, 0.40, 0.30]. So
     most churned customers still trend down, but a meaningful minority
     (55%) are flat or even increasing, and 30% of non-churned customers
     trend down too - real overlap, not a clean split.
   - The trend sets both the recent-30-day login-probability curve and the
     recent-session-duration scaling, each with its own per-customer random
     draw within the trend's range (e.g. a "declining" customer's session
     durations shrink to somewhere between 35% and 65% of normal, not a
     fixed amount), so no two customers in the same trend look identical.

2. support_tickets (0-5 tickets per customer, over the last 90 days)
   - Ticket count is drawn from a Poisson distribution whose mean is higher
     for customers with TechSupport == "No" (they have no support add-on,
     so more issues end up as tickets instead of being self-served),
     clipped to [0, 5].
   - Each customer gets their own resolve probability, drawn from a normal
     distribution centered lower for churned customers (0.60) than
     non-churned (0.82) but with enough spread (sigma 0.16) that plenty of
     churned customers land in the high-resolve range and vice versa -
     some churned customers had a fine support experience, some
     non-churned customers had a bad one.
   - Each customer also gets their own mean resolution time (log-normal,
     centered higher for churned customers), and each individual ticket's
     resolution_time_hours is then drawn around that per-customer mean -
     two layers of noise instead of one fixed value per group. Unresolved
     tickets have no resolution time (NaN).

3. feature_usage_logs (per active add-on service per customer)
   - For each of OnlineSecurity, OnlineBackup, DeviceProtection,
     TechSupport, StreamingTV, StreamingMovies where the customer's value
     is "Yes", a handful of usage events are generated over the last 90
     days.
   - Each customer gets their own usage_count rate, drawn from a
     log-normal distribution centered lower for churned customers (3.0)
     than non-churned (5.0) with sigma 0.6 - wide enough that the two
     groups' usage_count distributions overlap substantially rather than
     occupying clearly separated ranges. Event-count ranges per service
     also overlap (3-10 for churned, 4-14 for non-churned) rather than
     being disjoint.

All randomness goes through a single numpy Generator seeded with SEED, so
re-running this script produces byte-identical CSVs.
"""

import numpy as np
import pandas as pd

SEED = 42
WINDOW_DAYS = 90
RECENT_DAYS = 30
END_DATE = pd.Timestamp("2024-06-30")

RAW_CSV = "data/raw/telco_churn.csv.csv"
OUT_DIR = "data/synthetic"

DEVICES = ["mobile", "desktop", "tablet"]
DEVICE_PROBS = [0.55, 0.35, 0.10]

TICKET_CATEGORIES = [
    "Billing",
    "Technical",
    "Account",
    "Service Outage",
    "General Inquiry",
]

ADDON_SERVICES = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

TREND_CATEGORIES = ["declining", "flat", "increasing"]
CHURNED_TREND_PROBS = [0.45, 0.35, 0.20]
NON_CHURNED_TREND_PROBS = [0.30, 0.40, 0.30]


def random_timestamps(rng, day_weights, n):
    """Pick n days-ago values from day_weights (index = days ago, 0..WINDOW_DAYS-1),
    then jitter each with a random time-of-day, returning a list of Timestamps."""
    days_ago = rng.choice(len(day_weights), size=n, p=day_weights / day_weights.sum())
    seconds_of_day = rng.integers(0, 24 * 3600, size=n)
    return [
        END_DATE - pd.Timedelta(days=int(d)) + pd.Timedelta(seconds=int(s))
        for d, s in zip(days_ago, seconds_of_day)
    ]


def trend_end_scale(rng, trend):
    """Recent-30d login-probability multiplier at day 0 (ramps from 1.0 at day 29)."""
    if trend == "declining":
        return rng.uniform(0.35, 0.65)
    if trend == "flat":
        return rng.uniform(0.85, 1.15)
    return rng.uniform(1.15, 1.45)  # increasing


def trend_duration_low(rng, trend):
    """Recent-30d session-duration multiplier at day 0 (ramps from 1.0 at day 29)."""
    if trend == "declining":
        return rng.uniform(0.35, 0.65)
    if trend == "flat":
        return rng.uniform(0.85, 1.05)
    return rng.uniform(1.05, 1.3)  # increasing


def generate_login_events(rng, customers):
    rows = []
    days_ago = np.arange(WINDOW_DAYS)  # 0 = today, WINDOW_DAYS-1 = oldest
    is_recent = days_ago < RECENT_DAYS

    for cust in customers:
        n_events = rng.integers(10, 31)
        churned = cust["Churn"] == "Yes"
        trend_probs = CHURNED_TREND_PROBS if churned else NON_CHURNED_TREND_PROBS
        trend = rng.choice(TREND_CATEGORIES, p=trend_probs)

        weights = np.ones(WINDOW_DAYS)
        weights[is_recent] = np.linspace(1.0, trend_end_scale(rng, trend), RECENT_DAYS)

        timestamps = random_timestamps(rng, weights, n_events)
        base_duration = rng.lognormal(mean=6.2, sigma=0.5, size=n_events)  # ~ hundreds of secs
        duration_low = trend_duration_low(rng, trend)

        for ts, dur in zip(timestamps, base_duration):
            days_before = (END_DATE - ts).days
            if days_before < RECENT_DAYS:
                dur *= duration_low + (1 - duration_low) * (days_before / RECENT_DAYS)

            rows.append(
                {
                    "customer_id": cust["customerID"],
                    "timestamp": ts,
                    "session_duration_seconds": max(5, int(dur)),
                    "device": rng.choice(DEVICES, p=DEVICE_PROBS),
                }
            )

    return pd.DataFrame(rows).sort_values(["customer_id", "timestamp"]).reset_index(drop=True)


def generate_support_tickets(rng, customers):
    rows = []
    uniform_weights = np.ones(WINDOW_DAYS)

    for cust in customers:
        churned = cust["Churn"] == "Yes"
        base_lambda = 1.2
        if cust["TechSupport"] == "No":
            base_lambda += 1.3

        n_tickets = int(np.clip(rng.poisson(base_lambda), 0, 5))
        if n_tickets == 0:
            continue

        timestamps = random_timestamps(rng, uniform_weights, n_tickets)

        resolve_prob = float(np.clip(rng.normal(0.60 if churned else 0.82, 0.16), 0.05, 0.98))
        resolved_flags = rng.random(n_tickets) < resolve_prob

        # per-customer mean resolution time, then per-ticket noise around it
        resolution_mean = float(rng.lognormal(mean=np.log(18.0 if churned else 8.0), sigma=0.7))
        categories = rng.choice(TICKET_CATEGORIES, size=n_tickets)

        for ts, resolved, category in zip(timestamps, resolved_flags, categories):
            resolution_time = (
                round(float(rng.lognormal(mean=np.log(resolution_mean), sigma=0.5)), 2)
                if resolved
                else np.nan
            )
            rows.append(
                {
                    "customer_id": cust["customerID"],
                    "timestamp": ts,
                    "category": category,
                    "resolved": bool(resolved),
                    "resolution_time_hours": resolution_time,
                }
            )

    return pd.DataFrame(rows).sort_values(["customer_id", "timestamp"]).reset_index(drop=True)


def generate_feature_usage_logs(rng, customers):
    rows = []
    uniform_weights = np.ones(WINDOW_DAYS)

    for cust in customers:
        churned = cust["Churn"] == "Yes"
        n_events_range = (3, 10) if churned else (4, 14)
        # per-customer usage rate, wide enough to overlap the other group's range
        usage_lambda = float(rng.lognormal(mean=np.log(3.0 if churned else 5.0), sigma=0.6))

        for service in ADDON_SERVICES:
            if cust[service] != "Yes":
                continue

            n_events = rng.integers(*n_events_range)
            timestamps = random_timestamps(rng, uniform_weights, n_events)
            usage_counts = rng.poisson(usage_lambda, size=n_events) + 1

            for ts, count in zip(timestamps, usage_counts):
                rows.append(
                    {
                        "customer_id": cust["customerID"],
                        "timestamp": ts,
                        "feature_name": service,
                        "usage_count": int(count),
                    }
                )

    return pd.DataFrame(rows).sort_values(["customer_id", "timestamp"]).reset_index(drop=True)


def main():
    rng = np.random.default_rng(SEED)

    df = pd.read_csv(RAW_CSV)
    customers = df.to_dict("records")

    login_events = generate_login_events(rng, customers)
    support_tickets = generate_support_tickets(rng, customers)
    feature_usage_logs = generate_feature_usage_logs(rng, customers)

    import os

    os.makedirs(OUT_DIR, exist_ok=True)
    login_events.to_csv(f"{OUT_DIR}/login_events.csv", index=False)
    support_tickets.to_csv(f"{OUT_DIR}/support_tickets.csv", index=False)
    feature_usage_logs.to_csv(f"{OUT_DIR}/feature_usage_logs.csv", index=False)

    print(f"login_events: {len(login_events)} rows -> {OUT_DIR}/login_events.csv")
    print(f"support_tickets: {len(support_tickets)} rows -> {OUT_DIR}/support_tickets.csv")
    print(f"feature_usage_logs: {len(feature_usage_logs)} rows -> {OUT_DIR}/feature_usage_logs.csv")


if __name__ == "__main__":
    main()
