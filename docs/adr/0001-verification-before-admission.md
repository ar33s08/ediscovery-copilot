# ADR-0001: The LLM proposes, an independent verifier disposes

Status: accepted. Date: 2026-09-13.

## Context
LLM providers self-report confidence and cite passages that may not support the
prose (citation hallucination). In a review workflow an ungrounded answer that
looks right is worse than no answer: it biases a human reviewer who pays for it.

## Decision
The provider's output is never admitted on its own word. After synthesis,
`agents.verify_answer` re-checks every answer sentence against the *text of the
chunks the answer itself cited*: a sentence is grounded only if it appears
verbatim (normalized) in a cited chunk or clears an 80% token-overlap floor.
Any ungrounded sentence docks confidence below the admission threshold and the
item is force-routed to the human queue. The verifier is deterministic, has no
network dependency, and runs on the same footing in tests and production.

## Consequences
- Hallucinations are measured (`evalkit.hallucination_rate`) and gated in CI.
- Latency grows by one local pass (negligible vs provider round-trip).
- A correct-but-paraphrasing model can be over-routed to humans: an intentional,
  tunable bias (see MIN_SUPPORT_RATIO). Recall/precision belong to the reviewer.
