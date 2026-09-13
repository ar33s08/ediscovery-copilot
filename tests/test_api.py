import os
import tempfile

import pytest
from fastapi.testclient import TestClient

os.environ["EDISCOVERY_CORPUS"] = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "corpus.json"
)
os.environ["EDISCOVERY_AUDIT_PATH"] = os.path.join(tempfile.mkdtemp(), "api-audit.jsonl")

from ediscovery_copilot import server as srv


@pytest.fixture()
def client():
    srv._state.clear()
    return TestClient(srv.APP)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_answer_question_returns_citations(client):
    resp = client.post(
        "/questions", json={"query": "Are the deal terms confidential under the NDA?"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["citation_ids"]
    assert body["confidence"] == 1.0
    assert body["needs_human_review"] is False


def test_unanswerable_goes_to_queue(client):
    resp = client.post(
        "/questions",
        json={"query": "What is the quarterly dividend policy for employee stock options?"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["needs_human_review"] is True
    qid = body["question_id"]
    # the queue lists it
    queue = client.get("/questions").json()
    assert any(item["question_id"] == qid for item in queue)


def test_decision_endpoint_records(client):
    qid = client.post(
        "/questions", json={"query": "Which email may be privileged legal advice from counsel?"}
    ).json()["question_id"]
    resp = client.post(
        "/decisions",
        json={
            "question_id": qid,
            "status": "accepted",
            "reviewer": "attorney",
            "notes": "confirmed",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["recorded"] is True
    audit = client.get("/audit").json()
    assert audit["valid"] is True
    assert any(e["event"] == "decision.recorded" for e in audit["entries"])


def test_validation_rejects_bad_status(client):
    resp = client.post("/decisions", json={"question_id": "x", "status": "not-a-status"})
    assert resp.status_code == 422
