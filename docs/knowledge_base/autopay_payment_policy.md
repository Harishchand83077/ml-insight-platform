---
title: Autopay and Payment Method Policy
doc_type: policy
version: "2.4"
last_updated: "2026-03-05"
---

# Autopay and Payment Method Policy

> **Synthetic document.** Vantrix Communications is a fictional telecom
> company invented for this project. All policy IDs, dollar amounts, and
> procedures below are made up for retrieval-testing purposes and do not
> describe any real company's terms.

Policy ID: **POL-PAY-018**. Governs accepted payment methods, autopay
enrollment, the autopay discount, and failed-payment handling.

## Accepted Payment Methods

Vantrix accepts credit card, debit card, and direct bank transfer (ACH).
Vantrix does **not** accept cryptocurrency, money orders, or cash
payments at this time — a customer asking about these should be told
they are not currently supported, not given a workaround. Electronic
check (one-time ACH, not enrolled autopay) is available for manual
monthly payment but does not qualify for the autopay discount below.

## Autopay Discount

Enrolling in autopay (automatic monthly charge to a card or bank
account on file) earns a **$5/month discount**, applied automatically
starting the second billing cycle after enrollment. Autopay via bank
transfer (ACH) earns an additional **$2/month** on top of the base
$5 discount (total $7/month) because ACH processing costs Vantrix less
than card-network interchange fees. Discount is removed immediately
(next cycle) if autopay is disabled or a payment fails.

## Failed Payment Handling

A failed autopay charge triggers this sequence:
1. **Day 0**: payment fails, customer notified by email and SMS.
2. **Day 3**: automatic retry.
3. **Day 7**: second automatic retry; if this fails, a **$10 late fee**
   (POL-PAY-018 §4) is applied and the account is flagged `payment_risk`.
4. **Day 15**: service is suspended (not canceled) if payment still
   has not cleared — the customer loses connectivity but the account and
   all settings are preserved.
5. **Day 30**: if unresolved, the account is canceled and sent to
   collections per standard terms; this is also when the early
   termination fee (if under a term contract) is applied, since
   non-payment cancellation is treated the same as customer-initiated
   cancellation for ETF purposes.

A customer can avoid the Day 15 suspension at any point by making a
manual payment, even partial, which pauses the sequence and restarts the
retry clock from that payment.

## Payment Method Changes

Customers can update their payment method at any time via the account
portal with no fee. Changing payment method does **not** affect autopay
enrollment status — if autopay was on, it stays on with the new method
unless explicitly disabled. A card that expires is detected
automatically 30 days before expiration (based on the card's stored
expiry date) and triggers a reminder notice, rather than waiting for a
decline to surface the issue.

## Billing Cycle and Due Dates

Bills are generated on a fixed day of the month assigned at signup
(the "anniversary date") and are due **21 days** from the statement
date — longer than the single-digit due windows some competitors use,
intentionally, to reduce late-payment volume. Autopay charges run on the
due date itself, not the statement date, giving autopay customers the
full 21-day float before the charge hits.

## Disputing a Charge vs. Payment Method Policy

Payment-method and autopay issues are handled under this document;
disputing the legitimacy of a charge itself (e.g., billed for a service
never received) falls under the Refund and Cancellation Policy's
Disputed Charges section (POL-REF-014) instead, which has a separate
60-day dispute window.
