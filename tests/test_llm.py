from ediscovery_copilot.llm import ExtractiveFallbackProvider, OpenAICompatibleProvider, RawAnswer
from ediscovery_copilot.models import Chunk
from ediscovery_copilot.retrieval import Scored


def _mk_scored(cid, text, score=0.5):
    c = Chunk(
        chunk_id=cid,
        doc_id="d",
        text=text,
        sent_start=0,
        sent_end=2,
        char_start=0,
        char_end=len(text),
        content_hash="x",
        sentence_count=2,
    )
    return Scored(chunk=c, score=score, source="fused")


def test_fallback_produces_grounded_sentences():
    ev = [
        _mk_scored(
            "c1",
            "The settlement amount was five million dollars. It was paid last week.",
            score=0.9,
        )
    ]
    p = ExtractiveFallbackProvider(max_sentences=2)
    raw = p.synthesize("what was the settlement amount", ev)
    assert raw.text
    assert "settlement" in raw.text.casefold()
    assert "c1" in raw.support_ids


def test_fallback_refuses_on_single_coincidental_token():
    # "beacon" is the ONLY token the question shares with this sentence. Quoting
    # it would answer a question the evidence does not actually address.
    ev = [
        _mk_scored(
            "c1",
            "Effective immediately all employees must preserve documents related to the "
            "Beacon matter including emails spreads and chat records from every device.",
            score=0.9,
        )
    ]
    raw = ExtractiveFallbackProvider().synthesize("When was Beacon notified about the audit?", ev)
    assert raw.text.startswith("INSUFFICIENT")
    assert raw.support_ids == []


def test_fallback_refuses_when_empty_evidence():
    p = ExtractiveFallbackProvider()
    raw = p.synthesize("anything?", [])
    assert raw.text.startswith("INSUFFICIENT")
    assert raw.support_ids == []


def test_fallback_deduplicates_sentences():
    ev = [
        _mk_scored(
            "c1",
            "The litigation hold covers every custodian named in the schedule. The litigation hold covers every custodian named in the schedule.",
            score=0.9,
        ),
        _mk_scored(
            "c2",
            "The litigation hold covers every custodian named in the schedule. Destruction may result in sanctions.",
            score=0.5,
        ),
    ]
    provider = ExtractiveFallbackProvider(max_sentences=4)
    raw = provider.synthesize("who does the litigation hold cover", ev)
    assert raw.text.count("litigation hold covers") == 1


def test_raw_answer_asdict_shape():
    raw = RawAnswer("text", ["c1", "c2"], "test")
    d = raw.asdict()
    assert d["provider"] == "test"
    assert set(d["support_ids"]) == {"c1", "c2"}


def test_openai_provider_construction():
    p = OpenAICompatibleProvider("https://example.com/v1", "key", "model")
    assert p.name == "openai-compatible"
    assert p.base_url == "https://example.com/v1"
