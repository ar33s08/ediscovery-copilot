# ADR-0002: Hybrid BM25 + TF-IDF with RRF, not a single embedding branch

Status: accepted. Date: 2026-09-13.

## Context
Legal review language contains exact identifiers (custodian names, dates,
Bates ranges, defined terms like "litigation hold"). Pure ANN/embedding stacks
excel at semantics but blur exact lexical hits, and they make rankings
model-version-dependent — hard to pin in tests and in front of a review planner.

## Decision
Score every query through two complementary branches — Okapi BM25 (k1=1.5,
b=0.75) over an inverted index and TF-IDF cosine over a numpy matrix — and fuse
with Reciprocal Rank Fusion (k=60). Everything is in-repo and deterministic:
identical corpus + identical query yields byte-identical rankings, which the
test suite asserts (`test_fused_search_deterministic`). The gold-set eval
measures retrieval hit-rate and fails the build below 0.80.

## Consequences
- Keyword-critical queries (ADR/gold g4 "preservation obligations") need the
  lexical branch; paraphrase queries lean on the shared vocabulary vector side.
- The TF-IDF branch is a drop-in seam for a real embedding provider later: same
  interface, one branch swapped, eval thresholds unchanged as the acceptance test.
- On corpus sizes where the TF-IDF matrix is large, this design moves to a
  vector store behind the same interface (roadmap 2) rather than being rewritten.
