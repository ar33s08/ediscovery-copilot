"""Hybrid retriever: BM25 lexical scoring + TF-IDF cosine + RRF fusion.

Zero external model dependencies so the pipeline is fully offline-testable in
CI. Scoring is deterministic: identical corpus + query => identical ranking.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

from ediscovery_copilot.corpus import Corpus
from ediscovery_copilot.models import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "he",
        "her",
        "his",
        "i",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "which",
        "who",
        "will",
        "with",
        "you",
        "your",
        "not",
    ]
)


_SUFFIXES = (
    "izations",
    "ization",
    "ities",
    "ility",
    "ations",
    "ation",
    "ements",
    "ement",
    "ingly",
    "edly",
    "ings",
    "ing",
    "ied",
    "ies",
    "ers",
    "est",
    "ed",
    "es",
    "s",
    "ment",
    "ness",
    "less",
    "ful",
    "ous",
    "ive",
    "ity",
    "ly",
    "sion",
    "e",
    "tion",
)


def stem(word: str) -> str:
    """Aggressive suffix stripper sufficient for legal-prose recall.

    Not a full Porter implementation, but deterministic and dependency-free;
    collapses preserv/preservation/preserves and account/accounting.
    """
    for suf in _SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 4:
            return word[: len(word) - len(suf)]
    return word


def tokenize(text: str) -> list[str]:
    """Lowercase stemmed tokens with a small stopword filter."""
    return [
        stem(t) for t in _TOKEN_RE.findall(text.casefold()) if t not in _STOPWORDS and len(t) > 2
    ]


@dataclass
class Scored:
    chunk: Chunk
    score: float
    source: str  # "bm25" | "tfidf" | "fused"


@dataclass
class Retriever:
    """Builds lexical + vector indices over a corpus and answers top-k queries."""

    corpus: Corpus
    k1: float = 1.5
    b: float = 0.75
    chunks: list[Chunk] = field(init=False)
    _df: dict[str, int] = field(init=False, default_factory=dict)
    _postings: dict[str, dict[int, int]] = field(init=False, default_factory=dict)
    _avgdl: float = field(init=False, default=0.0)
    _tfidf: np.ndarray | None = field(init=False, default=None)
    _idf: np.ndarray | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.chunks = list(self.corpus.chunks)
        self._build()

    # ------------------------------------------------------------------ build
    def _build(self) -> None:
        self._postings = {}
        doc_len = []
        for idx, chunk in enumerate(self.chunks):
            toks = tokenize(chunk.text)
            doc_len.append(max(len(toks), 1))
            tf: dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            for term, freq in tf.items():
                self._postings.setdefault(term, {})[idx] = freq
        self._df = {term: len(pl) for term, pl in self._postings.items()}
        n = max(len(self.chunks), 1)
        self._avgdl = sum(doc_len) / n
        vocab = sorted(self._df)
        self._vocab = {term: vi for vi, term in enumerate(vocab)}
        self._idf = np.array(
            [math.log(1 + (n - self._df[t] + 0.5) / (self._df[t] + 0.5)) for t in vocab]
        )
        mat = np.zeros((n, max(len(vocab), 1)), dtype=np.float32)
        for idx, chunk in enumerate(self.chunks):
            for t, freq in tokenize_counts(chunk.text).items():
                vi = self._vocab.get(t)
                if vi is not None:
                    mat[idx, vi] = (1 + math.log(freq)) * self._idf[vi]
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self._tfidf = mat / norms

    # ------------------------------------------------------------------- bm25
    def bm25(self, query: str, top_k: int = 10) -> list[Scored]:
        toks = tokenize(query)
        n = max(len(self.chunks), 1)
        scores = np.zeros(n, dtype=np.float64)
        for term in toks:
            plist = self._postings.get(term)
            if not plist:
                continue
            df = self._df[term]
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for idx, freq in plist.items():
                dl = max(self.chunks[idx].sentence_count, 1)
                num = freq * (self.k1 + 1)
                den = freq + self.k1 * (1 - self.b + self.b * dl / max(self._avgdl, 1))
                scores[idx] += idf * num / den
        order = np.argsort(-scores, kind="stable")
        out = []
        for idx in order:
            if scores[idx] <= 0:
                break
            out.append(Scored(self.chunks[int(idx)], float(scores[idx]), "bm25"))
            if len(out) >= top_k:
                break
        return out

    # ------------------------------------------------------------------ tfidf
    def tfidf(self, query: str, top_k: int = 10) -> list[Scored]:
        if self._tfidf is None:
            return []
        qv = np.zeros(self._tfidf.shape[1], dtype=np.float32)
        for t, freq in tokenize_counts(query).items():
            vi = self._vocab.get(t)
            if vi is not None:
                qv[vi] = (1 + math.log(freq)) * self._idf[vi]
        qn = np.linalg.norm(qv)
        if qn == 0:
            return []
        sims = self._tfidf @ (qv / qn)
        order = np.argsort(-sims, kind="stable")
        out = []
        for idx in order:
            if sims[idx] <= 1e-9:
                break
            out.append(Scored(self.chunks[int(idx)], float(sims[idx]), "tfidf"))
            if len(out) >= top_k:
                break
        return out

    # ------------------------------------------------------------------- fuse
    def search(self, query: str, top_k: int = 6, rrf_k: int = 60) -> list[Scored]:
        """Reciprocal Rank Fusion over the two retrievers (deterministic)."""
        fused: dict[str, tuple[float, Chunk]] = {}
        for branch in (self.bm25(query, top_k=top_k * 3), self.tfidf(query, top_k=top_k * 3)):
            for rank, sc in enumerate(branch, start=1):
                prev = fused.get(sc.chunk.chunk_id)
                add = 1.0 / (rrf_k + rank)
                fused[sc.chunk.chunk_id] = ((prev[0] if prev else 0.0) + add, sc.chunk)
        ranked = sorted(fused.items(), key=lambda kv: (-kv[1][0], kv[0]))
        return [Scored(ch, score, "fused") for _, (score, ch) in ranked[:top_k]]


def tokenize_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for t in tokenize(text):
        counts[t] = counts.get(t, 0) + 1
    return counts
