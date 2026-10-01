---
title: "Retention Playbook: Service Quality Complaints"
doc_type: playbook
version: "1.4"
last_updated: "2026-06-18"
---

# Retention Playbook: Service Quality Complaints

> **Synthetic document.** Vantrix Communications is a fictional telecom
> company invented for this project. All policy IDs, dollar amounts, and
> procedures below are made up for retrieval-testing purposes and do not
> describe any real company's terms.

Playbook ID: **POL-RET-102**. Use this playbook when a customer's
cancellation reason is service reliability — repeated outages, slow
speeds, or unresolved support tickets — rather than price. Churn-model
flags where `num_tickets` and `pct_unresolved` are top contributors, or
`avg_resolution_time_hours` is well above the SLA target, should route
here.

## Who This Applies To

Accounts with 2 or more open or recently-closed SEV-1/SEV-2 tickets
(see Support SLA, POL-SLA-007) in the last 90 days, or any account that
has breached the SLA uptime guarantee at least once in that window.
This segment responds poorly to discount offers — the complaint is about
reliability, not cost, and leading with a credit without addressing the
underlying issue reads as dismissive.

## Required First Step: Resolve, Don't Discount

Before any retention offer, confirm whether the underlying technical
issue is actually resolved. If there is an open ticket, the retention
agent must either resolve it on the call (where possible — e.g., a
remote line reset) or get a firm, specific commitment for resolution
(technician appointment date/time) before discussing retention at all.
Offering a discount on a still-broken service is explicitly discouraged
in agent training, since it treats the symptom (the customer is upset)
rather than the cause.

## Offer Ladder

1. **SLA service credit**, if not already applied automatically — check
   the account before assuming it posted (see POL-SLA-007 and the
   Service Outage Credit Policy, POL-OUT-003).
2. **Priority technician dispatch**: for accounts in this playbook, the
   standard 48-hour dispatch window can be compressed to **same-day or
   next-day**, flagged in the dispatch system as `RETENTION_PRIORITY`.
3. **Free equipment upgrade**: if the issue traces to aging customer
   equipment (older router/ONT hardware), a free upgrade is offered
   regardless of contract term or tenure — equipment age is a legitimate
   Vantrix-side cause and is not treated as a discretionary perk.
4. **ETF waiver escalation**: if three or more SLA breaches are
   confirmed in the rolling 90-day window, the customer already
   qualifies for a full ETF waiver per Contract Terms (POL-CTR-009) —
   agents should proactively mention this rather than waiting for the
   customer to ask, even if the agent's goal is still retention; forcing
   a customer to discover their own contractual rights reads as bad
   faith and has driven negative reviews when it's happened.

## Escalation for Repeat Flags

An account that triggers this playbook twice in 6 months, regardless of
whether retention succeeded both times, is automatically escalated to a
**Network Operations review** — repeated individual-account reliability
issues can indicate a local infrastructure problem (e.g., a degraded
node serving several addresses) that retention conversations alone won't
fix. Reason code: `QUALITY_REPEAT_FLAG`.
