"""Provider abstraction for answer synthesis.

Two providers ship in the box:

* OpenAICompatibleProvider -> real model calls over httpx against any
  OpenAI-style /chat/completions endpoint (configurable base URL).
* ExtractiveFallbackProvider -> deterministic, no-network, extractive answer
  composer. Used when no API key is configured so tests, demos, and CI run
  anywhere and produce citation-verifiable answers.

Both return a RawAnswer: proposed text plus the chunk ids it claims to be
grounded in. The verifier in agents.py decides whether the claim holds.
"""

from __future__ import annotations

from typing import Any, Protocol

from ediscovery_copilot.retrieval import Scored, tokenize


class LLMError(RuntimeError):
    """Raised when a provider call fails hard."""


class Provider(Protocol):
    name: str

    def synthesize(self, query: str, evidence: list[Scored]) -> RawAnswer: ...


class RawAnswer:
    def __init__(self, text: str, support_ids: list[str], provider: str) -> None:
        self.text = text
        self.support_ids = list(support_ids)
        self.provider = provider

    def asdict(self) -> dict[str, Any]:
        return {"text": self.text, "support_ids": self.support_ids, "provider": self.provider}


_PROMPT = (
    "You are a litigation-support review assistant. Answer the query using ONLY the "
    "numbered evidence passages. Cite passages inline as [chunk_id]. If the evidence "
    "is insufficient, answer exactly INSUFFICIENT_EVIDENCE. Never invent facts.\n\n"
    "QUERY: {query}\n\nEVIDENCE:\n{passages}\n\n"
)


class OpenAICompatibleProvider:
    name = "openai-compatible"

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def synthesize(self, query: str, evidence: list[Scored]) -> RawAnswer:
        import httpx

        passages = "\n".join(f"[{s.chunk.chunk_id}] {s.chunk.text}" for s in evidence)
        body = {
            "model": self.model,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": "Follow citation discipline strictly."},
                {"role": "user", "content": _PROMPT.format(query=query, passages=passages)},
            ],
        }
        try:
            resp = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=body,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:  # network/config issues degrade to fallback upstream
            raise LLMError(str(exc)) from exc
        claimed = [s.chunk.chunk_id for s in evidence if s.chunk.chunk_id in text]
        return RawAnswer(text, claimed or [s.chunk.chunk_id for s in evidence[:2]], self.name)


class ExtractiveFallbackProvider:
    """Deterministic extractive composer: selects the highest-scoring sentences."""

    name = "extractive-fallback"

    def __init__(
        self, max_sentences: int = 3, min_score: float = 1e-9, min_sentence_overlap: int = 2
    ) -> None:
        self.max_sentences = max_sentences
        self.min_score = min_score
        self.min_sentence_overlap = min_sentence_overlap

    def synthesize(self, query: str, evidence: list[Scored]) -> RawAnswer:
        """Compose an answer from verbatim sentences of retrieved chunks.

        Sentence-level evidence floor (refusal is a feature): a sentence may be
        quoted only if it shares >= min_sentence_overlap stemmed content tokens
        with the question. A single incidental shared word (e.g. "employees"
        bridging a dividend query to a litigation-hold notice) is NOT topical
        evidence -- quoting it would be a false-confidence failure, worse than
        silence. When no retrieved sentence clears the floor the provider
        REFUSES (INSUFFICIENT_EVIDENCE) and the item routes to a human. This is
        the offline/deterministic precision bias; paraphrase answering is the
        LLM provider's documented job (see ADR-0003).
        """
        usable = [s for s in evidence if s.score >= self.min_score]
        if not usable:
            return RawAnswer("INSUFFICIENT_EVIDENCE", [], self.name)
        query_terms = set(tokenize(query))
        lines: list[str] = []
        used_ids: list[str] = []
        seen: set[str] = set()
        for scored in usable:
            for sentence in scored.chunk.text.split(". "):
                clean = sentence.strip().rstrip(".")
                key = clean.casefold()
                if len(clean) < 25 or key in seen:
                    continue
                if query_terms:
                    shared = len(query_terms & set(tokenize(clean)))
                    if shared < self.min_sentence_overlap:
                        continue
                seen.add(key)
                lines.append(f"{clean} [{scored.chunk.chunk_id}]")
                used_ids.append(scored.chunk.chunk_id)
                if len(lines) >= self.max_sentences:
                    break
            if len(lines) >= self.max_sentences:
                break
        if not lines:
            return RawAnswer("INSUFFICIENT_EVIDENCE", [], self.name)
        return RawAnswer(" ".join(lines), list(dict.fromkeys(used_ids)), self.name)


def provider_from_env(env: dict[str, str]) -> Provider:
    """Pick the strongest provider the environment supports."""
    key = env.get("EDISCOVERY_API_KEY") or env.get("OPENAI_API_KEY")
    base = env.get("EDISCOVERY_BASE_URL") or env.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = env.get("EDISCOVERY_MODEL", "gpt-4o-mini")
    if key:
        return OpenAICompatibleProvider(base, key, model)
    return ExtractiveFallbackProvider()
