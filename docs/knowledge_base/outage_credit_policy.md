---
title: Service Outage Credit Policy
doc_type: policy
version: "1.9"
last_updated: "2026-04-22"
---

# Service Outage Credit Policy

> **Synthetic document.** Vantrix Communications is a fictional telecom
> company invented for this project. All policy IDs, dollar amounts, and
> procedures below are made up for retrieval-testing purposes and do not
> describe any real company's terms.

Policy ID: **POL-OUT-003**. Defines how service credits for outages are
calculated and issued, extending the uptime guarantee in the Support SLA
(POL-SLA-007).

## Credit Calculation

For any outage beyond the 99.5% monthly uptime allowance (roughly 3
hours 39 minutes/month for residential, 43 minutes for business-tier),
the credit is:

```
credit = (plan's daily rate) x (ceil(downtime_hours_beyond_allowance / 24))
```

In plain terms: one full day of service value is credited for every 24
hours (or fraction thereof) of downtime beyond the allowance. Example: a
Fiber 500 1-year customer ($64.99/month ≈ $2.17/day) who experiences 30
hours of outage beyond the allowance is credited for 2 days
(ceil(30/24) = 2), or **$4.34**.

## Automatic vs. Manual Credits

Outages detected and confirmed by Vantrix's own network operations
monitoring are credited **automatically** within 5 business days of the
outage ending — no customer action required. A customer who experiences
downtime that was not automatically detected (e.g., a localized issue
network monitoring didn't flag) can request a manual credit review by
providing the outage window; manual reviews are decided using network
operations logs as the source of truth, not customer self-reports alone,
though a customer's reported timestamps are used to narrow the log
search.

## Planned Maintenance Exception

Outages during a properly-announced maintenance window (72+ hours
advance notice via email and account dashboard, per POL-SLA-007) do not
count toward credit-eligible downtime. If planned maintenance runs
**longer** than its announced window, the overage portion does count,
and is calculated the same way as an unplanned outage.

## Interaction with Early Termination Fees

Three or more credit-eligible outage events in a rolling 90-day window
entitle the customer to a full early termination fee waiver if they
choose to cancel during or shortly after that window (cross-referenced
in Contract Terms, POL-CTR-009, and the Service Quality retention
playbook, POL-RET-102). This is tracked automatically on the account —
agents do not need to manually count breach events, the account flag
`sla_breach_3x_90d` is set by the billing system once the third
qualifying event posts.

## Multi-Service Accounts

For a bundled account (e.g., fiber internet + home phone), the outage
credit is calculated per affected service, not against the full bundle
rate — an internet-only outage credits only the internet portion of the
bill, even if the phone service was technically unaffected and still
usable during the outage window.

## Disputing a Credit Amount

A customer who believes a credit was calculated incorrectly can request
a recalculation review within **30 days** of the credit posting, citing
the specific outage window. Reviews beyond 30 days are not guaranteed
but may still be considered at a retention specialist's discretion,
particularly if the account is otherwise in good standing.
