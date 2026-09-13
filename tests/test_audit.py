import json
import os
import tempfile

import pytest

from ediscovery_copilot.audit import GENESIS, AuditTrail


@pytest.fixture()
def trail():
    path = os.path.join(tempfile.mkdtemp(), "chain.jsonl")
    return AuditTrail(path)


def test_append_and_verify(trail):
    trail.record("q", {"n": 1})
    trail.record("a", {"n": 2})
    trail.record("decision", {"status": "accepted"})
    ok, bad = trail.verify_chain()
    assert ok is True
    assert bad is None


def test_tamper_detection_modifies_payload(trail):
    trail.record("q", {"secret": "value"})
    trail.record("a", {"result": "ok"})
    lines = [
        json.loads(l) for l in trail.path.read_text(encoding="utf-8").splitlines() if l.strip()
    ]
    lines[0]["payload"]["secret"] = "tampered"
    with open(trail.path, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(l, sort_keys=True) + "\n" for l in lines)
    fresh = AuditTrail(trail.path)
    ok, bad = fresh.verify_chain()
    assert ok is False
    assert bad == 1


def test_tamper_detection_deletes_entry(trail):
    trail.record("q", {"n": 1})
    trail.record("a", {"n": 2})
    trail.record("d", {"n": 3})
    lines = trail.path.read_text(encoding="utf-8").splitlines()
    with open(trail.path, "w", encoding="utf-8") as fh:
        fh.writelines(l + "\n" for l in lines[1:] if l.strip())
    fresh = AuditTrail(trail.path)
    ok, bad = fresh.verify_chain()
    assert ok is False
    assert bad == 2


def test_entries_round_trip(trail):
    trail.record("evt", {"x": 42})
    entries = trail.entries()
    assert len(entries) == 1
    assert entries[0]["event"] == "evt"
    assert entries[0]["payload"]["x"] == 42


def test_empty_trail_verifies(trail):
    ok, _bad = trail.verify_chain()
    assert ok is True


def test_prev_hash_chain_is_linked(trail):
    r1 = trail.record("a", {})
    r2 = trail.record("b", {})
    assert r2["prev_hash"] == r1["entry_hash"]
    assert r1["prev_hash"] == GENESIS
