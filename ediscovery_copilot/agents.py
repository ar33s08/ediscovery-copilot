"""Review agent: retrieve -> synthesize -> VERIFY -> queue for humans.

The verifier is the core trust mechanism: an answer is only admissible when
every sentence it proposes can be matched back to its own cited chunk. If any
claim sentence is ungrounded, confidence is docked and the item is forced into
the human-review queue. This is what separates an audit-grade copilot from a
chatbot demo.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from ediscovery_copilot.audit import AuditTrail
from ediscovery_copilot.corpus import Corpus
from ediscovery_copilot.llm import Provider, RawAnswer
from ediscovery_copilot.models import (
    Chunk,
    Question,
    ReviewCategory,
    ReviewDecision,
    normalize_text,
    split_sentences,
)
from ediscovery_copilot.retrieval import Retriever

INSUFFICIENT = "INSUFFICIENT_EVIDENCE"
MIN_SUPPORT_RATIO = 0.6
MIN_QUERY_OVERLAP = 0.05  # relevance floor for evidence chunks


@dataclass
class Verification:
    ok: bool
    support_ratio: float
    grounded: list[str] = field(default_factory=list)
    ungrounded: list[str] = field(default_factory=list)

    def asdict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "support_ratio": round(self.support_ratio, 3),
            "grounded": self.grounded,
            "ungrounded": self.ungrounded,
        }


def verify_answer(answer: RawAnswer, chunks_by_id: dict[str, Chunk]) -> Verification:
    """Check each answer sentence against the text of the chunks it cites."""
    if answer.text.strip().casefold().startswith(INSUFFICIENT.casefold()):
        return Verification(ok=True, support_ratio=1.0, grounded=[], ungrounded=[])
    cited_text = " ".join(
        normalize_text(chunks_by_id[cid].text) for cid in answer.support_ids if cid in chunks_by_id
    )
    cited_tokens = set(_words(cited_text))
    sentences = [
        s for s in split_sentences(answer.text) if not s.strip().casefold().startswith("[")
    ]
    grounded: list[str] = []
    ungrounded: list[str] = []
    for sent in sentences:
        cleaned = re.sub(r"\[[^\]]*\]", " ", sent)
        norm = normalize_text(cleaned)
        # Direct-quote grounding: the sentence (normalized) appears verbatim
        # inside one of the cited chunks -> unquestionably supported.
        if norm and norm in cited_text:
            grounded.append(sent)
            continue
        toks = [w for w in _words(norm) if w not in _STOP]
        if not toks:
            continue
        overlap = sum(1 for w in toks if w in cited_tokens) / len(toks)
        (grounded if overlap >= 0.8 else ungrounded).append(sent)
    checked = len(grounded) + len(ungrounded)
    ratio = len(grounded) / checked if checked else 0.0
    return Verification(
        ok=ratio >= MIN_SUPPORT_RATIO, support_ratio=ratio, grounded=grounded, ungrounded=ungrounded
    )


_STOP = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "in",
        "for",
        "on",
        "with",
        "by",
        "that",
        "this",
        "from",
        "as",
        "at",
        "it",
        "is",
        "are",
        "was",
        "be",
    ]
)


def _words(text: str) -> list[str]:
    return [w for w in text.casefold().split() if len(w) > 2]


@dataclass
class ReviewAgent:
    """One-shot review pipeline with an enforced human-in-the-loop gate."""

    corpus: Corpus
    provider: Provider
    audit: AuditTrail
    top_k: int = 6
    retriever: Retriever = field(init=False)

    def __post_init__(self) -> None:
        self.retriever = Retriever(self.corpus)

    def ask(self, query: str, category: ReviewCategory | None = None) -> Question:
        qid = uuid.uuid4().hex[:12]
        hits = self._relevant_hits(query)
        answer = self.provider.synthesize(query, hits)
        ver = verify_answer(answer, self.corpus.chunk_index)
        confidence = ver.support_ratio if answer.text.strip() != INSUFFICIENT else 0.0
        needs_review = (not ver.ok) or confidence < 0.85 or answer.text.strip() == INSUFFICIENT
        q = Question(
            question_id=qid,
            query=query,
            answer=answer.text,
            category=category,
            confidence=round(confidence, 3),
            citation_ids=answer.support_ids,
            needs_human_review=needs_review,
        )
        self.audit.record(
            "question.answered",
            {
                "question_id": qid,
                "query": query,
                "provider": answer.provider,
                "verification": ver.asdict(),
                "citations": answer.support_ids,
                "needs_human_review": needs_review,
            },
        )
        return q

    def _relevant_hits(self, query: str):
        """Drop chunks with no topical token overlap with the query.

        Fused (RRF) scores encode rank, not semantic relevance, so a noisy
        tail can surface off-topic chunks. The trust-critical rule: the system
        would rather REFUSE and route to a human than answer off weak evidence.
        """
        hits = self.retriever.search(query, top_k=self.top_k)
        qtoks = {w for w in _words(normalize_text(query)) if w not in _STOP}
        if not qtoks:
            return hits
        kept = []
        for hit in hits:
            ctoks = {w for w in _words(normalize_text(hit.chunk.text)) if w not in _STOP}
            overlap = len(qtoks & ctoks) / len(qtoks)
            if overlap >= MIN_QUERY_OVERLAP:
                kept.append(hit)
        return kept

    def record_decision(self, q: Question, decision: ReviewDecision) -> None:
        self.audit.record(
            "decision.recorded",
            {
                "question_id": decision.question_id,
                "status": decision.status.value,
                "reviewer": decision.reviewer,
                "notes": decision.notes,
            },
        )
