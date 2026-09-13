from ediscovery_copilot.models import Chunk, Document, content_hash, normalize_text, split_sentences


def test_normalize_and_hash_stable():
    assert normalize_text("  The   Deal  TERMS ") == "the deal terms"
    assert content_hash("A b") == content_hash("a  B")


def test_split_sentences_basic():
    parts = split_sentences("One. Two! Three?")
    assert parts == ["One.", "Two!", "Three?"]


def test_chunk_addresses_are_complete_and_ordered():
    doc = Document(
        doc_id="d", title="t", text="First sentence here. Second sentence here. Third one follows."
    )
    chunks = Chunk.build(doc, max_sentences=2)
    assert chunks, "chunker produced nothing"
    first, last = chunks[0], chunks[-1]
    assert first.char_start == doc.text.find("First")
    joined = doc.text[first.char_start : last.char_end]
    assert "Third one follows." in joined
    # ids encode the sentence window
    assert first.chunk_id.startswith("d:")


def test_chunk_determinism():
    doc = Document(
        doc_id="d", title="t", text="Alpha beta gamma. Delta epsilon zeta. Eta theta iota."
    )
    a = Chunk.build(doc)
    b = Chunk.build(doc)
    assert [(c.chunk_id, c.content_hash, c.char_start, c.char_end) for c in a] == [
        (c.chunk_id, c.content_hash, c.char_start, c.char_end) for c in b
    ]
