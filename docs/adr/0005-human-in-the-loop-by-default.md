# ADR-0005: Human review is the default route, automation is earned

Status: accepted. Date: 2026-09-13.

## Context
Privilege calls are near-unappeasable if wrong (clawback, waiver, sanctions);
hot-doc detection carries case-outcome weight. No confidence number a model
emits justifies auto-closing a review decision without a human signature.

## Decision
Every answer carries `needs_human_review`. Routing is forced when: the
verifier rejects any sentence, confidence is below 0.85, the answer is a
refusal, or evidence fell under the relevance floor. Accepted answers may
still be sampled for review. Attorney decisions (accept/edit/reject with
notes + reviewer) are recorded as first-class audit events, making the queue
the training signal for future automation.

## Consequences
- Throughput is lower than an ungated auto-classifier by design; the eval
  reports needs_human_review_rate so the trade is visible, not hidden.
- The queue + decisions form the labeled dataset for a future
  privilege-classifier, whose rollout can then be gated on measured precision.
