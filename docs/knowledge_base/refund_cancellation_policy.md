---
title: Refund and Cancellation Policy
doc_type: policy
version: "3.2"
last_updated: "2026-06-01"
---

# Refund and Cancellation Policy

> **Synthetic document.** Vantrix Communications is a fictional telecom
> company invented for this project. All policy IDs, dollar amounts, and
> procedures below are made up for retrieval-testing purposes and do not
> describe any real company's terms.

Policy ID: **POL-REF-014**. This document governs how Vantrix handles
cancellations initiated by the customer and the conditions under which a
refund is issued.

## Cancellation Window

A customer may cancel any new service within **14 calendar days** of
activation for a full refund of the first month's charges, no questions
asked — this is the "buyer's remorse" window referenced internally as
the **14-Day Satisfaction Guarantee**. Equipment (routers, ONT boxes)
must be returned within 10 days of cancellation using the prepaid return
label emailed at signup, or a **$150 unreturned-equipment fee** is
charged to the account on file. After the 14-day window closes, standard
cancellation terms (see Contract Terms, POL-CTR-009) apply instead,
including any early termination fee for customers under a 1-year or
2-year contract.

## Prorated Refunds Mid-Cycle

Vantrix bills in advance on a monthly cycle. If a customer cancels
mid-cycle outside the 14-day window, Vantrix does **not** refund the
unused portion of the current month — service continues through the end
of the paid period and then terminates. The one exception is a
Vantrix-caused service failure (see Service Outage Credit Policy,
POL-OUT-003): if the account has an active, unresolved outage credit at
the time of cancellation, that credit is refunded in cash rather than
forfeited.

## Refund Method and Timeline

Refunds are issued to the original payment method on file. Credit/debit
card refunds post within **5-7 business days**; ACH/bank-transfer
refunds take **7-10 business days**. Refunds are never issued as account
credit unless the customer explicitly requests store credit instead of a
cash refund, in which case a **10% bonus** is added (e.g., a $50 refund
becomes $55 in account credit) as an incentive to stay in the Vantrix
billing ecosystem.

## Disputed Charges

A customer disputing a charge should contact support within **60 days**
of the statement date. Vantrix support can reverse billing errors
directly; charges older than 60 days must go through the customer's card
issuer. Fraudulent-charge disputes (unauthorized account access) have no
time limit and are escalated to the Trust & Safety team the same day.

## Non-Refundable Items

The following are explicitly non-refundable under POL-REF-014:
- Installation and activation fees ($49.99 standard install), once a
  technician has been dispatched, even if the customer cancels before
  the appointment is completed.
- Premium channel add-ons (sports packages, movie bundles) once the
  billing cycle in which they were activated has started.
- Early termination fees themselves (see POL-CTR-009) — these are a
  contract-exit charge, not a service charge, and are not refundable
  even under the 14-day window if the customer is canceling a
  multi-year contract specifically to switch providers.

## Retention Offer Before Cancellation

Before processing any cancellation request, support agents are required
to check the account for retention-offer eligibility (see the retention
playbooks, POL-RET-101 through POL-RET-103) and present at least one
applicable offer. A customer who declines the offer and proceeds with
cancellation has that decision logged with reason code `CANCEL_AFTER_OFFER`
for churn-model monitoring purposes — this signal is distinct from a
customer who cancels without support ever being contacted.
