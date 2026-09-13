from ediscovery_copilot.retrieval import Retriever, stem, tokenize


def test_tokenize_stems_and_filters():
    toks = tokenize("The preservation obligations under the document retention policy")
    assert "policy" in toks or "polici" in "".join(toks)
    assert "the" not in toks


def test_stem_is_aggressive_but_safe():
    assert stem("preservation") == stem("preservations")
    assert stem("is") == "is"  # too short to strip


def test_bm25_prefers_exact_terms(tiny_corpus):
    r = Retriever(tiny_corpus)
    hits = r.bm25("confidentiality agreement")
    assert hits
    assert "confidentiality" in hits[0].chunk.text.casefold()


def test_fused_search_deterministic(tiny_corpus):
    r = Retriever(tiny_corpus)
    one = [h.chunk.chunk_id for h in r.search("confidentiality agreement secret", top_k=3)]
    two = [h.chunk.chunk_id for h in r.search("confidentiality agreement secret", top_k=3)]
    assert one == two and one


def test_empty_query_returns_no_results(tiny_corpus):
    r = Retriever(tiny_corpus)
    assert r.search("  ", top_k=3) == []
