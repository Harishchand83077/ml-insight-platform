"""
Generates synthetic customer behavioral events for the Telco churn dataset
(data/raw/telco_churn.csv.csv), correlated with each customer's real Churn
label, Contract type, and service columns so downstream models see realistic
signal. Output is written to data/synthetic/ as three CSVs and NOT published
to RabbitMQ (that comes after the data is validated).

Correlation logic per event type:

1. login_events (10-30 events per customer, spread over the last 90 days)
   - Every customer gets a baseline daily login probability for the older
     half of the window (days 31-90 ago).
   - Churned customers ("Churn" == "Yes") get that probability multiplied
     down sharply for the most recent 30 days, and their session_duration
     for those recent events is scaled down too - modeling a customer who
     is disengaging right before they leave. The result is a declining
     login-frequency AND declining session-length trend near day 0.
   - Non-churned customers keep a flat-to-slightly-increasing probability
     and session length in the recent 30 days, modeling steady or growing
     engagement.

2. support_tickets (0-5 tickets per customer, over the last 90 days)
   - Ticket count is drawn from a Poisson distribution whose mean is higher
     for customers with TechSupport == "No" (they have no support add-on,
     so more issues end up as tickets instead of being self-served),
     clipped to [0, 5].
   - resolved is a Bernoulli draw whose success probability is lower for
     churned customers, modeling unresolved issues as a churn driver.
   - resolution_time_hours (only set when resolved) is drawn from a
     log-normal distribution with a higher mean for churned customers,
     modeling slower support as another churn driver. Unresolved tickets
     have no resolution time (NaN).

3. feature_usage_logs (per active add-on service per customer)
   - For each of OnlineSecurity, OnlineBackup, DeviceProtection,
     TechSupport, StreamingTV, StreamingMovies where the customer's value
     is "Yes", a handful of usage events are generated over the last 90
     days.
   - Churned customers get fewer events per service (sparser usage) and a
     lower per-event usage_count (drawn from a Poisson with a smaller
     mean) than non-churned customers, modeling declining/low engagement
     with the services they are paying for right before they leave.

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


def random_timestamps(rng, day_weights, n):
    """Pick n days-ago values from day_weights (index = days ago, 0..WINDOW_DAYS-1),
    then jitter each with a random time-of-day, returning a list of Timestamps."""
    days_ago = rng.choice(len(day_weights), size=n, p=day_weights / day_weights.sum())
    seconds_of_day = rng.integers(0, 24 * 3600, size=n)
    return [
        END_DATE - pd.Timedelta(days=int(d)) + pd.Timedelta(seconds=int(s))
        for d, s in zip(days_ago, seconds_of_day)
    ]


def generate_login_events(rng, customers):
    rows = []
    days_ago = np.arange(WINDOW_DAYS)  # 0 = today, WINDOW_DAYS-1 = oldest
    is_recent = days_ago < RECENT_DAYS

    for cust in customers:
        n_events = rng.integers(10, 31)
        churned = cust["Churn"] == "Yes"

        weights = np.ones(WINDOW_DAYS)
        if churned:
            # sharp drop-off in login probability as we approach day 0
            recent_scale = np.linspace(1.0, 0.15, RECENT_DAYS)
            weights[is_recent] = recent_scale
        else:
            # flat-to-slightly-increasing probability near day 0
            recent_scale = np.linspace(1.0, 1.4, RECENT_DAYS)
            weights[is_recent] = recent_scale

        timestamps = random_timestamps(rng, weights, n_events)
        base_duration = rng.lognormal(mean=6.2, sigma=0.5, size=n_events)  # ~ hundreds of secs

        for ts, dur in zip(timestamps, base_duration):
            days_before = (END_DATE - ts).days
            if churned and days_before < RECENT_DAYS:
                dur *= 0.3 + 0.6 * (days_before / RECENT_DAYS)  # shorter the more recent

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
        resolve_prob = 0.55 if churned else 0.88
        resolved_flags = rng.random(n_tickets) < resolve_prob
        resolution_mean = 24.0 if churned else 6.0
        categories = rng.choice(TICKET_CATEGORIES, size=n_tickets)

        for ts, resolved, category in zip(timestamps, resolved_flags, categories):
            resolution_time = (
                round(float(rng.lognormal(mean=np.log(resolution_mean), sigma=0.6)), 2)
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
        n_events_range = (2, 8) if churned else (5, 15)
        usage_lambda = 2.0 if churned else 6.0

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
