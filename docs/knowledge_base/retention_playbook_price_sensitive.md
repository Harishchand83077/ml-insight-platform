---
title: "Retention Playbook: Price-Sensitive Customers"
doc_type: playbook
version: "1.6"
last_updated: "2026-06-18"
---

# Retention Playbook: Price-Sensitive Customers

> **Synthetic document.** Vantrix Communications is a fictional telecom
> company invented for this project. All policy IDs, dollar amounts, and
> procedures below are made up for retrieval-testing purposes and do not
> describe any real company's terms.

Playbook ID: **POL-RET-101**. Use this playbook when a customer's stated
reason for canceling, or the churn model's top driver for a flagged
account, is cost — "too expensive," "found a cheaper plan elsewhere," or
a high `monthly_charges` value combined with low `num_addons_active`.

## Who This Applies To

Primarily month-to-month customers (no term discount already applied)
and customers on a plan tier above what their usage patterns justify —
e.g., a Fiber 1 Gig customer whose `avg_usage_count` and
`total_logins_90d` suggest light actual usage. Churn-model flags with
`monthly_charges` among the top 3 SHAP contributors toward churn risk
are a strong signal to route the account through this playbook rather
than the generic save flow.

## Offer Ladder (use in order, stop at first acceptance)

1. **Term-contract conversion**: if the customer is on month-to-month,
   offer the equivalent 1-year or 2-year rate from Fiber Pricing and
   Promotions (POL-FIB-022) — this alone is often a $15-30/month
   reduction with no feature loss.
2. **Autopay discount**: if not already enrolled, the $5-7/month autopay
   discount (POL-PAY-018) is a no-cost retention lever — always confirm
   this is active before escalating to a discretionary offer.
3. **Tier right-sizing**: if usage data supports it, proactively suggest
   stepping down one speed tier (e.g., Fiber 1 Gig → Fiber 500) framed
   as "most households don't need the top tier" rather than a downgrade
   — retains the customer at a lower price point instead of losing them
   entirely.
4. **Discretionary retention credit**: agents may apply up to a
   **$15/month credit for 6 months** without manager approval; credits
   beyond that require Tier 2 retention-specialist sign-off. This is the
   last offer in the ladder, used only if 1-3 don't resolve the
   objection.

## What Not to Do

Do not lead with the discretionary credit (step 4) — it trains customers
to threaten cancellation for a discount and costs more than the
contract-conversion or autopay levers, which solve the same problem at
lower cost to Vantrix. Do not stack more than one discretionary credit
on the same account within a rolling 12-month period without manager
override.

## Logging Outcomes

Every retention conversation under this playbook is logged with the
offer(s) presented and the outcome (`accepted`, `declined_canceled`,
`declined_stayed_anyway`) using reason code `PRICE_SENSITIVE`. This
feeds back into the churn-rate-by-segment reporting
(`get_churn_rate_by_column`) so the retention team can see which offers
actually move the needle for this segment over time, rather than relying
on anecdote.
