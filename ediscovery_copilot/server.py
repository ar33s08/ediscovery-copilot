"""FastAPI service exposing the review pipeline + human review queue.

Endpoints
  GET  /health                 liveness
  POST /questions              submit a review question, get grounded answer
  GET  /questions              list the human-review queue
  POST /decisions              record attorney accept/edit/reject decision
  GET  /audit                  dump the audit trail + chain validity

Run locally with: uvicorn ediscovery_copilot.server:APP
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from pydantic import BaseModel

from ediscovery_copilot.agents import ReviewAgent
from ediscovery_copilot.audit import AuditTrail
from ediscovery_copilot.corpus import Corpus
from ediscovery_copilot.llm import provider_from_env
from ediscovery_copilot.models import Question, ReviewCategory, ReviewDecision, ReviewStatus

APP = FastAPI(title="eDiscovery Copilot", version="0.1.0")

_DEFAULT_CORPUS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "corpus.json")

_state: dict[str, ReviewAgent] = {}


def get_agent() -> ReviewAgent:
    if "agent" not in _state:
        corpus_path = os.environ.get("EDISCOVERY_CORPUS", _DEFAULT_CORPUS)
        audit_path = os.environ.get("EDISCOVERY_AUDIT_PATH", "var/audit.jsonl")
        provider = provider_from_env(dict(os.environ))
        _state["agent"] = ReviewAgent(
            corpus=Corpus.from_json(corpus_path),
            provider=provider,
            audit=AuditTrail(audit_path),
        )
    return _state["agent"]


class QuestionIn(BaseModel):
    query: str
    category: ReviewCategory | None = None


class DecisionIn(BaseModel):
    question_id: str
    status: ReviewStatus
    notes: str | None = None
    reviewer: str | None = None


@APP.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "provider": get_agent().provider.name}


@APP.post("/questions", response_model=Question)
def answer_question(payload: QuestionIn) -> Question:
    return get_agent().ask(payload.query, category=payload.category)


@APP.get("/questions")
def review_queue() -> list[dict]:
    agent = get_agent()
    return [
        {
            "question_id": e["payload"]["question_id"],
            "query": e["payload"]["query"],
            "verification": e["payload"]["verification"],
            "citation_ids": e["payload"]["citations"],
        }
        for e in agent.audit.entries()
        if e["event"] == "question.answered" and e["payload"].get("needs_human_review")
    ]


@APP.post("/decisions")
def record_decision(payload: DecisionIn) -> dict[str, bool]:
    agent = get_agent()
    decision = ReviewDecision(
        question_id=payload.question_id,
        status=payload.status,
        notes=payload.notes,
        reviewer=payload.reviewer,
    )
    agent.record_decision(None, decision)
    return {"recorded": True}


@APP.get("/audit")
def audit_view() -> dict:
    agent = get_agent()
    valid, _bad = agent.audit.verify_chain()
    return {"valid": valid, "first_bad_seq": _bad, "entries": agent.audit.entries()}
