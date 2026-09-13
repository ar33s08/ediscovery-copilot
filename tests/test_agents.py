from ediscovery_copilot.agents import INSUFFICIENT, Verification, verify_answer
from ediscovery_copilot.llm import RawAnswer
from ediscovery_copilot.models import Chunk, ReviewDecision, ReviewStatus


def _chunk(cid, text, doc_id="d1"):
    return Chunk(
        chunk_id=cid,
        doc_id=doc_id,
        text=text,
        sent_start=0,
        sent_end=2,
        char_start=0,
        char_end=len(text),
        content_hash="x",
        sentence_count=2,
    )


def _index(*chunks):
    return {c.chunk_id: c for c in chunks}


def test_verifier_accepts_direct_quotes():
    ch = _chunk("c1", "The witness was present at the board meeting in March.")
    raw = RawAnswer("The witness was present at the board meeting in March [c1]", ["c1"], "t")
    ver = verify_answer(raw, _index(ch))
    assert ver.ok
    assert ver.support_ratio == 1.0


def test_verifier_flags_hallucinated_content():
    ch = _chunk("c1", "The witness was present at the board meeting in March.")
    raw = RawAnswer(
        "The CFO embezzled forty million dollars from the pension fund last decade [c1]",
        ["c1"],
        "t",
    )
    ver = verify_answer(raw, _index(ch))
    assert not ver.ok
    assert ver.support_ratio < 0.6


def test_verifier_treats_refusal_as_safe():
    ch = _chunk("c1", "Some irrelevant text about coffee and pastries delivered Friday.")
    raw = RawAnswer(INSUFFICIENT_EVIDENCE_PLACEHOLDER, ["c1"], "t")
    ver = verify_answer(raw, _index(ch))
    assert ver.ok  # refusal is never a hallucination


def test_verifier_punishes_uncited_paragraph():
    ch = _chunk(
        "c1", "Document retention obligations apply to all custodians listed in the schedule."
    )
    raw = RawAnswer(
        "Document retention obligations apply to all custodians listed in the schedule [c1]. "
        "The CEO personally approved offshore wire transfers at midnight.",
        ["c1"],
        "t",
    )
    ver = verify_answer(raw, _index(ch))
    assert not ver.ok  # the second sentence has zero grounding
    assert any("offshore" in s.casefold() for s in ver.ungrounded)


def test_agent_routes_grounded_answer_without_review(agent, corpus):
    q = agent.ask("Are the deal terms confidential under the NDA?")
    assert q.answer != INSUFFICIENT
    assert q.confidence == 1.0
    assert q.needs_human_review is False
    assert q.citation_ids


def test_agent_refuses_unanswerable_query(agent):
    q = agent.ask("What is the quarterly dividend policy for employee stock options?")
    assert q.answer.startswith(INSUFFICIENT) or q.needs_human_review is True


def test_agent_audits_every_question(agent):
    agent.ask("Is there a document about document preservation obligations?")
    entries = agent.audit.entries()
    events = [e["event"] for e in entries]
    assert "question.answered" in events
    assert agent.audit.verify_chain() == (True, None)


def test_agent_records_human_decision(agent):
    q = agent.ask("Are the deal terms confidential under the NDA?")
    agent.record_decision(
        q,
        ReviewDecision(
            question_id=q.question_id, status=ReviewStatus.ACCEPTED, reviewer="attorney", notes="ok"
        ),
    )
    events = [e["event"] for e in agent.audit.entries()]
    assert "decision.recorded" in events


def test_verification_dict_shape():
    ver = Verification(ok=True, support_ratio=0.95, grounded=["a"], ungrounded=[])
    d = ver.asdict()
    assert d["ok"] is True and d["support_ratio"] == 0.95


INSUFFICIENT_EVIDENCE_PLACEHOLDER = "INSUFFICIENT_EVIDENCE"
