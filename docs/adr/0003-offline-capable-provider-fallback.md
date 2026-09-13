# ADR-0003: Providers behind one Protocol, offline extractive fallback first-class

Status: accepted. Date: 2026-09-13.

## Context
The product must be demoable, testable, and runnable in air-gapped review
environments where no model API exists or keys are not permitted. Tests must
never hit the network: CI needs determinism and no secrets.

## Decision
A provider is any object with `synthesize(query, evidence) -> RawAnswer`.
Two implementations ship: `OpenAICompatibleProvider` (chat completions via
httpx against any OpenAI-style base URL) and `ExtractiveFallbackProvider`, a
deterministic composer that quotes the highest-ranked evidence sentences with
inline chunk citations. With no `EDISCOVERY_API_KEY` in the environment the
fallback is selected automatically — the same pipeline, verifier, queue, and
audit trail work end-to-end offline.

## Consequences
- CI runs the whole product (not a mock) with zero credentials.
- The fallback's answers are literal quotes, so the verifier sees 100% grounded
  answers and the eval still exercises the full admission path.
- Providers that raise `LLMError` surface as refusals (routed to humans) rather
  than crashes: degraded availability must never look like a wrong answer.
