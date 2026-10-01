---
title: Support Service Level Agreement
doc_type: policy
version: "2.1"
last_updated: "2026-04-10"
---

# Support Service Level Agreement

> **Synthetic document.** Vantrix Communications is a fictional telecom
> company invented for this project. All policy IDs, dollar amounts, and
> procedures below are made up for retrieval-testing purposes and do not
> describe any real company's terms.

Policy ID: **POL-SLA-007**. This SLA defines Vantrix's commitments on
network uptime, response times, and the service credits owed when those
commitments are missed.

## Uptime Guarantee

Vantrix guarantees **99.5% monthly uptime** for residential fiber and
cable service, measured at the network edge (not inclusive of
customer-side equipment failures or in-home wiring issues). 99.5%
monthly uptime permits roughly **3 hours 39 minutes** of downtime per
30-day month before the guarantee is breached. Business-tier accounts
carry a stricter **99.9% uptime guarantee** (about 43 minutes/month).

## Response Time Targets by Channel

- **Live chat**: first response within **2 minutes**, 24/7.
- **Phone support**: answered within **5 minutes** average hold time
  during business hours (7am-11pm local), **15 minutes** overnight.
- **Email/ticket support**: first response within **4 business hours**,
  resolution target **24 hours** for non-technical issues.
- **Field technician dispatch**: next-available appointment within
  **48 hours** for an outage-class issue, **5 business days** for a
  scheduled install or non-urgent service call.

## Severity Levels

Support tickets are triaged into four severity levels:
- **SEV-1 (total outage)**: no connectivity at all. Target: technician
  dispatched or remote fix within 4 hours, 24/7.
- **SEV-2 (degraded service)**: connectivity present but significantly
  impaired (packet loss >5%, speeds under 50% of plan). Target: 24-hour
  resolution.
- **SEV-3 (intermittent issue)**: occasional drops, not continuously
  reproducible. Target: 3 business days.
- **SEV-4 (billing/account, non-technical)**: handled by the standard
  email/ticket targets above.

## Service Credits for SLA Breach

If monthly uptime falls below the 99.5% guarantee, the customer is owed
an automatic service credit calculated as **1 day of service value per
hour of downtime beyond the 3h39m allowance**, applied to the next
billing statement without the customer needing to request it. A customer
who believes a credit was missed can request a manual review by citing
the outage window; Vantrix's network operations logs are the source of
truth for downtime duration, not customer-reported estimates. See the
Service Outage Credit Policy (POL-OUT-003) for the full credit
calculation and the interaction with contract ETF waivers.

## What Counts Against the SLA

Counted: core network outages, regional fiber cuts, Vantrix equipment
failures (ONT, head-end hardware), and planned maintenance that exceeds
its announced window. Not counted: customer-side router/modem failures
not supplied by Vantrix, in-home wiring issues, Wi-Fi coverage
complaints where the wired connection is healthy, and outages during a
properly-announced maintenance window (Vantrix commits to announcing
planned maintenance at least 72 hours in advance via email and account
dashboard).

## Escalation Path

A customer whose SEV-1 or SEV-2 ticket is not resolved within target can
request escalation to a Tier 2 specialist at any time — there is no
minimum wait before escalation is available. Three or more missed SLA
targets in a rolling 90-day window entitles the customer to both a
service credit for each breach and a full early termination fee waiver
if they choose to cancel (cross-referenced in Contract Terms,
POL-CTR-009).
