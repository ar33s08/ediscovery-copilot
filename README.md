# eDiscovery Copilot

**Citation-grounded, audit-grade document review for e-discovery — an AI copilot that
answers review questions only when it can prove the answer, and refuses otherwise.**

[![CI](https://github.com/ar33s08/ediscovery-copilot/actions/workflows/CI/badge.svg)](https://github.com/ar33s08/ediscovery-copilot/actions)
[![Python 3.11+](https://shields.io/badge/Python-3.11%2B-blue.svg)]()
[![License MIT](https://shields.io/badge/License-MIT-green.svg)]()

## The problem this solves

Reviewing produced documents (email, contracts, memos, transcripts) is the expensive
core of litigation support. Generic LLM chatbots are unusable in that workflow because
they hallucinate, cite nothing, and leave no record an attorney can defend. A real
review tool must answer three questions a chatbot cannot:

1. **Where did that come from?** — every sentence traceable to a Bates-stamped chunk.
2. **What happens when the evidence isn't there?** — refuse and route to a human.
3. **Who decided what, and when?** — a tamper-evident log of every AI suggestion and
   every attorney decision.

This project implements all three as a small, fully testable service.

## How it works

```
question
   v
[Retriever]  BM25 (Okapi, tuned k1/b) + TF-IDF cosine, fused with Reciprocal Rank Fusion
   v          deterministic: same corpus + question => same ranking (CI-testable)
[Provider]   OpenAI-compatible chat API *or* a deterministic extractive fallback so
             the whole pipeline runs offline (tests, CI, air-gapped demos)
   v
[Verifier]   THE TRUST BOUNDARY: every sentence of the answer must be a verbatim quote
             of its own cited chunk or exceed a token-overlap floor; if any claim is
             ungrounded, confidence drops and the item is force-routed to the queue
   v
[Queue]      human-in-the-loop: attorney accepts/edits/rejects -> recorded as a decision
   v
[Audit]      hash-chained JSONL: each entry embeds the digest of the previous one, so
             editing or deleting any past record is detectable (verify_chain())
```

Design choices worth stealing (see `docs/adr/`):

- **The LLM proposes, the verifier disposes.** No answer reaches the queue without
  passing an independent grounding check that never trusts the model's self-report.
- **Deterministic-by-default retrieval.** Pure-Python BM25 + numpy TF-IDF with RRF
  fusion; rankings are reproducible and unit-testable without any network.
- **Offline first.** With no API key configured, an extractive provider keeps the
  entire product honest: answers are literal quotes and the eval still passes.
- **Refusal is a feature.** A relevance floor drops weak evidence; the system prefers
  `INSUFFICIENT_EVIDENCE` + human review over a plausible guess.
- **Auditability as a data structure.** The audit trail is a hash chain, not just a log.

## Quickstart

```bash
git clone https://github.com/ar33s08/ediscovery-copilot.git
cd ediscovery-copilot
uv sync --extra dev          # or: pip install -e ".[dev]"

uv run ediscovery "Are the deal terms confidential under the NDA?"
```

Example output (against the shipped synthetic corpus):

```text
The parties executed a mutual NDA on March 1 2025 [PROD-0003:0000-0003] The deal
terms disclosed under the NDA remain confidential for five years [PROD-0003:0000-0003]
------------------------------------------------------------
confidence=1.0 needs_human_review=False

[PROD-0003:0000-0003] (PROD-0003) The parties executed a mutual NDA on March 1 2025.
```

Run the service:

```bash
uv run uvicorn ediscovery_copilot.server:APP
# POST /questions | GET /questions | POST /decisions | GET /audit | GET /health
```

Use the API:

```bash
curl -X POST http://127.0.0.1:8000/questions \
  -H "Content-Type: application/json" \
  -d '{"query": "Which document shows potential fraud or accounting misconduct?"}'
```

## Evaluation (measures hallucination rate on a gold set)

```bash
uv run python -m ediscovery_copilot.evalkit
```

```json
{
  "total": 10,
  "retrieval_hit_rate": 0.9,
  "answer_rate": 0.778,
  "grounded_answer_rate": 1.0,
  "hallucination_rate": 0.0,
  "refusal_correctness": 1.0,
  "needs_human_review_rate": 0.3
}
```

- **hallucination_rate = 0.0** — zero answerable items were answered without passing
  the independent verifier.
- **refusal_correctness = 1.0** — the unanswerable gold item was refused, not guessed.
- **answer_rate vs needs_human_review_rate** — the deliberate trade: 78% of answerable
  questions auto-answered with full grounding; the rest goes to humans with the evidence
  attached. The thresholds are in `ediscovery_copilot/agents.py`.

## Quality gates

```bash
uv run make gate    # ruff lint + format-check + pytest (38 tests) + eval thresholds
```

CI (github Actions) runs the same steps on every push: lint, format, `pytest --cov`,
and the eval must clear its thresholds or the build is red.

## Project layout

```text
ediscovery_copilot/
  models.py      documents, sentence-addressable chunks (stable char offsets), questions
  corpus.py      JSON ingest, dedup by content hash
  retrieval.py   BM25 + TF-IDF + RRF fusion, stemming, stopwords
  llm.py         provider protocol; OpenAI-compatible + offline extractive fallback
  agents.py      retrieve -> synthesize -> VERIFY -> route; decision recording
  audit.py       hash-chained tamper-evident JSONL trail
  server.py      FastAPI service (questions, review queue, decisions, audit)
  cli.py         ediscovery command
  evalkit.py     gold-set harness: recall, answer rate, hallucination rate, refusals
data/            synthetic litigation corpus (8 docs) + 10-item gold set
docs/adr/        architecture decision records (why, not just what)
tests/           38 tests: unit + behaviour + API + tamper-detection + eval gate
scripts/gate.py  the CI gate in one file
```

## Interview-level questions this repo is built to answer

- *How do you prevent RAG hallucination?* → `agents.verify_answer`: independent
  sentence-level grounding check + refusal floor; measured, not promised.
- *Why not just embeddings?* → hybrid BM25/TF-IDF/RRF beats single-branch on keyword
  legal terms; rankings deterministic and unit-tested.
- *Human-in-the-loop, concretely?* → confidence + verification drive a queue;
  accept/edit/reject decisions are first-class records.
- *How would you audit an AI system?* → prev-hash linked trail; tamper tests prove
  detection of both payload edits and entry deletion.
- *What breaks in production?* → ADR-003 covers provider fallback; ADR-004 covers
  why chunk addresses must be content-stable for Bates-style citation.

## Roadmap

1. Embedding branch behind the same Provider protocol (drop-in for the TF-IDF side).
2. Postgres + pgvector store with a migration layer (swap the in-memory index).
3. Streaming responses over the fused top-k while verification stays synchronous.
4. Privilege-log export (producer/title/type columns) from the decision queue.
5. Batch review runs with per-custodian recall reports for review-planner sign-off.

## Disclaimer

All corpus content is synthetic and generated for demonstration. The project does not
connect to any real litigation platform and includes no real party data.
