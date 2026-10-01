---
title: "Data Dictionary: customer_features"
doc_type: data_dictionary
version: "2.0"
last_updated: "2026-07-05"
---

# Data Dictionary: customer_features

Column-by-column reference for the `customer_features` Postgres table,
the feature source for both model training (`src/models/train_baseline.py`)
and serving-time inference (`src/serving/prediction.py` via
`src/models/inference.py`). Joined with `customers.churn` for the
training label. This describes a synthetic dataset built for this
project, not real customer records.

## Identifier

- **customer_id** (text, primary key): Unique customer identifier,
  format `NNNN-XXXXX` (e.g. `7590-VHVEG`) — four digits, a hyphen, five
  uppercase letters. Used as the lookup key for `predict_churn_tool` and
  the `/predict` endpoint, and as the Redis cache key suffix
  (`features:{customer_id}`) in the feature cache.

## Categorical Columns

- **contract** (text): One of `Month-to-month`, `One year`, `Two year`.
  Maps to the contract terms and ETF rules in Contract Terms
  (POL-CTR-009). The single strongest churn predictor found in EDA —
  month-to-month customers churn at 42.7% vs. 2.8% for two-year
  contracts.
- **payment_method** (text): One of `Electronic check`, `Mailed check`,
  `Bank transfer (automatic)`, `Credit card (automatic)`. The two
  `(automatic)` values correspond to autopay enrollment under
  Autopay and Payment Method Policy (POL-PAY-018).
- **internet_service** (text): One of `DSL`, `Fiber optic`, `No`. See
  Fiber-Optic Pricing and Promotions (POL-FIB-022) for current fiber
  pricing; fiber customers show a higher churn rate (42%) than DSL (19%)
  despite being the premium tier, attributed to competitive-market
  overlap rather than a quality issue.

## Binary Columns (Yes/No in Postgres, 0/1 after `prepare_model_input`)

- **partner** (text in DB: `Yes`/`No`; converted to int 0/1 for the
  model): Whether the customer has a partner on the account.
- **dependents** (text in DB: `Yes`/`No`; converted to int 0/1 for the
  model): Whether the customer has dependents.

## Numeric Columns

- **tenure** (integer, months): Months since the customer's account was
  activated. Strongest behavioral signal after contract type — customers
  in their first year churn at ~47%, dropping to ~9.5% after 4+ years.
- **monthly_charges** (numeric, USD): Current monthly bill amount,
  reflects plan tier and any active discounts/promotions (see
  Fiber-Optic Pricing and Promotions, POL-FIB-022).
- **senior_citizen** (integer, 0/1): 1 if the customer is 65+, else 0.
  Relevant to the senior discount in POL-FIB-022.
- **num_addons_active** (integer, 0-6): Count of active add-on services
  out of six possible (Online Security, Online Backup, Device
  Protection, Tech Support, Streaming TV, Streaming Movies). Engineered
  during feature engineering to collapse six redundant categorical
  columns into one denser numeric signal — see the glossary entry for
  the full rationale. Higher values correlate with lower churn.
- **total_logins_90d** (integer): Count of account logins (web or app)
  in the trailing 90 days. Part of the synthetic behavioral-event layer.
- **recent_30d_vs_older_60d_ratio** (numeric): Ratio of activity in the
  most recent 30 days vs. the preceding 60 days. A value below 1.0
  indicates declining engagement; used as an early-warning signal
  distinct from raw login counts.
- **avg_session_duration_recent_30d** (numeric, minutes): Average session
  length over the last 30 days.
- **num_tickets** (integer): Count of support tickets opened in the
  observation window. Primary driver for routing into the Service
  Quality retention playbook (POL-RET-102).
- **pct_unresolved** (numeric, 0-1): Fraction of that customer's support
  tickets that closed unresolved rather than resolved. High values
  combined with high `num_tickets` are a strong churn-risk combination.
- **avg_resolution_time_hours** (numeric): Average hours to resolve a
  ticket for this customer, compared against the Support SLA's
  resolution targets (POL-SLA-007).
- **avg_usage_count** (numeric): Average count of core-service usage
  events (calls, data sessions, etc.) per period — a general engagement
  proxy alongside the login-based features.

## Excluded Column: TotalCharges

The original source data's `TotalCharges` column is **not** part of
`customer_features`. It was dropped during feature engineering because
it is highly collinear with `tenure` (r=0.83) and `monthly_charges`
(r=0.65), being roughly their product — keeping it alongside both
independent signals would add redundant, collinear input without new
information. See the glossary's "Tenure" entry for the related EDA
discussion.

## Caching

`customer_features` rows are cached in Redis under cache-aside semantics
(`features:{customer_id}`, 300-second TTL) — see the glossary entry
"Cache-aside pattern" for the full mechanism and measured latency
numbers.
