"""Shared fixtures for the test suite."""

from __future__ import annotations

import os
import tempfile

import pytest

from ediscovery_copilot.agents import ReviewAgent
from ediscovery_copilot.audit import AuditTrail
from ediscovery_copilot.corpus import Corpus
from ediscovery_copilot.llm import ExtractiveFallbackProvider
from ediscovery_copilot.models import Document

_ROOT = os.path.dirname(os.path.dirname(__file__))
CORPUS_PATH = os.path.join(_ROOT, "data", "corpus.json")
GOLD_PATH = os.path.join(_ROOT, "data", "gold.json")


@pytest.fixture()
def corpus() -> Corpus:
    return Corpus.from_json(CORPUS_PATH)


@pytest.fixture()
def agent(corpus: Corpus) -> ReviewAgent:
    tmp = os.path.join(tempfile.mkdtemp(), "audit.jsonl")
    return ReviewAgent(corpus=corpus, provider=ExtractiveFallbackProvider(), audit=AuditTrail(tmp))


@pytest.fixture()
def tiny_corpus() -> Corpus:
    docs = [
        Document(
            doc_id="t1",
            title="NDA",
            text="The parties signed a confidentiality agreement. The terms stay secret for five years.",
        )
    ]
    return Corpus(docs)
