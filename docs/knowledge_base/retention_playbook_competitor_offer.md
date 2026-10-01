---
title: "Retention Playbook: Competitor Offer / Win-Back"
doc_type: playbook
version: "1.2"
last_updated: "2026-05-30"
---

# Retention Playbook: Competitor Offer / Win-Back

> **Synthetic document.** Vantrix Communications is a fictional telecom
> company invented for this project. All policy IDs, dollar amounts, and
> procedures below are made up for retrieval-testing purposes and do not
> describe any real company's terms.

Playbook ID: **POL-RET-103**. Use this playbook when a customer states
they have a specific competing offer in hand (not a vague "I found
something cheaper" — an actual named competitor and price/term) and are
calling to cancel or to ask Vantrix to match it.

## Who This Applies To

Any customer who names a specific competitor and quotes a specific
price. This playbook explicitly does not apply to vague price
complaints without a competing offer in hand — those route to the
Price-Sensitive Customers playbook (POL-RET-101) instead, which uses a
different, lower-cost offer ladder.

## Verification Before Matching

Agents should ask for the competing offer's term length and any
promotional conditions (new-customer-only pricing, contract length,
equipment fees) before matching — many competitor "deals" are
new-customer-only introductory rates that revert to a much higher price
after 12 months, which is worth pointing out factually rather than
dismissively.

## Match Authority

- Agents may match a competing offer **up to 15% below** Vantrix's
  current list price for the equivalent tier without manager approval,
  for a **12-month locked term**.
- Matches beyond 15% below list price require Tier 2 retention-specialist
  approval, capped at **25% below list** in any case — Vantrix does not
  authorize matching below that floor regardless of the competing offer,
  since service below that margin is not sustainable to deliver at
  current support-SLA standards.
- A matched rate always requires a 12-month (or longer) contract — Vantrix
  does not offer win-back matching on month-to-month terms, since the
  whole point of the match is securing a committed term in exchange for
  the discount.

## Win-Back for Already-Canceled Accounts

A customer who already canceled and is calling back within **90 days**
is eligible for the **Win-Back FIBER200OFF+** promotion: the standard
$200 new-customer credit (POL-FIB-022) plus a waived installation fee,
specifically to lower the friction of coming back versus the cost of
being treated as a brand-new signup. After 90 days, a returning customer
is processed as a standard new signup with standard new-customer
promotions only (no additional win-back bonus).

## What Not to Match

Vantrix does not match on service characteristics it cannot actually
deliver — e.g., a competitor's "unlimited data with no throttling" claim
on a technology where Vantrix's own network has a documented fair-use
threshold. Agents should be transparent that Vantrix can match price but
not misrepresent service characteristics; offering a price match while
implying a technical capability Vantrix doesn't have is explicitly
prohibited and has triggered compliance review in the past.

## Logging

All competitor win-back conversations are logged with the named
competitor, quoted price, and outcome under reason code
`COMPETITOR_MATCH`, separate from `PRICE_SENSITIVE`, so Vantrix can track
competitive pressure by region distinctly from general price
sensitivity.
