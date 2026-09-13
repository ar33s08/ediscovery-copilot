"""Core domain models shared across the copilot."""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?\])])\s+(?=[\"(\[]?[A-Z0-9])")


def normalize_text(text: str) -> str:
    """Normalize text for dedup and matching: casefold, collapse whitespace."""
    return " ".join(text.casefold().split())


def content_hash(text: str) -> str:
    """Stable sha256 hex digest of normalized content."""
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text.strip())]
    return [p for p in parts if p]


class DocumentType(str, Enum):
    EMAIL = "email"
    MEMO = "memo"
    CONTRACT = "contract"
    POLICY = "policy"
    TRANSCRIPT = "transcript"
    OTHER = "other"


class ReviewCategory(str, Enum):
    PRIVILEGE = "privilege"
    HOT_DOC = "hot-doc"
    RESPONSIVENESS = "responsiveness"
    IRRELEVANT = "irrelevant"


class ReviewStatus(str, Enum):
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    EDITED = "edited"
    REJECTED = "rejected"


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Document(BaseModel):
    """A produced document in the corpus."""

    doc_id: str
    title: str
    author: str | None = None
    recipients: list[str] = Field(default_factory=list)
    created_at: datetime | None = None
    doc_type: DocumentType = DocumentType.OTHER
    custodian: str | None = None
    native_metadata: dict[str, Any] = Field(default_factory=dict)
    text: str

    def sentences(self) -> list[str]:
        return split_sentences(self.text)


class Chunk(BaseModel):
    """Addressable sentence window used for citation and retrieval."""

    chunk_id: str
    doc_id: str
    text: str
    sent_start: int
    sent_end: int
    char_start: int
    char_end: int
    page: int | None = None
    content_hash: str
    sentence_count: int

    @classmethod
    def build(cls, doc: Document, *, max_sentences: int = 6) -> list[Chunk]:
        """Deterministic sentence-window chunker.

        Chunk ids and char offsets are stable for a given (document, params),
        which is what makes citations auditable across re-ingest.
        """
        sents = doc.sentences()
        spans: list[tuple[int, int]] = []
        cursor = 0
        for s in sents:
            start = doc.text.find(s, cursor)
            if start == -1:
                start = cursor
            end = start + len(s)
            spans.append((start, end))
            cursor = end
        chunks: list[Chunk] = []
        i = 0
        while i < len(sents):
            j = min(i + max_sentences, len(sents))
            window = sents[i:j]
            text = " ".join(window)
            c_start = spans[i][0]
            c_end = spans[j - 1][1]
            chunks.append(
                cls(
                    chunk_id=f"{doc.doc_id}:{i:04d}-{j:04d}",
                    doc_id=doc.doc_id,
                    text=text,
                    sent_start=i,
                    sent_end=j,
                    char_start=c_start,
                    char_end=c_end,
                    content_hash=content_hash(text),
                    sentence_count=j - i,
                )
            )
            i = j
        return chunks


class Question(BaseModel):
    """A review question answered against the corpus with cited evidence."""

    question_id: str
    query: str
    answer: str
    category: ReviewCategory | None = None
    confidence: float = 0.0
    citation_ids: list[str] = Field(default_factory=list)
    needs_human_review: bool = True
    created_at: datetime = Field(default_factory=_utcnow)


class ReviewDecision(BaseModel):
    """Attorney-recorded outcome for a proposed answer."""

    question_id: str
    status: ReviewStatus
    notes: str | None = None
    reviewer: str | None = None
    at: datetime = Field(default_factory=_utcnow)
