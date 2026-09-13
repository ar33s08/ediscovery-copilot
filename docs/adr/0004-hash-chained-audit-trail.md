# ADR-0004: Tamper-evident audit trail (prev-hash linked JSONL)

Status: accepted. Date: 2026-09-13.

## Context
An AI-assisted review is only defensible if every suggestion and every human
decision is reconstructable later — including proof that the record was not
silently edited. A plain log file cannot evidence its own integrity.

## Decision
Every audit entry embeds the SHA-256 of `canonical_json(payload) | prev_hash |
seq`, chaining each record to its predecessor from a genesis digest.
`AuditTrail.verify_chain()` returns `(valid, first_bad_seq)` and is exercised
in tests against tampered payloads **and** deleted entries. Writes are
append-only; verification is cheap and can run at export time or on read.

## Consequences
- Detects editing and deletion of history (both cases have dedicated tests).
- It is tamper-*evident*, not tamper-*proof*: a full-registry attacker with write
  access can rebuild a chain. Production posture therefore appends to
  append-only storage (S3 object-lock / Postgres with row-level security) so an
  attacker cannot rewrite the file — the chain detects, storage prevents.
- Canonical serialization (sort_keys, tight separators) is part of the hash
  contract: readers must reproduce it exactly or they will see false breaks.
