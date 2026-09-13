from ediscovery_copilot.evalkit import DEFAULT_CORPUS, DEFAULT_GOLD, evaluate


def test_eval_meets_production_thresholds():
    rep = evaluate(corpus_path=DEFAULT_CORPUS, gold_path=DEFAULT_GOLD)
    s = rep.as_dict()
    assert s["retrieval_hit_rate"] >= 0.80, s
    assert s["hallucination_rate"] <= 0.10, s
    assert s["refusal_correctness"] == 1.0, s
    assert s["grounded_answer_rate"] == 1.0, s


def test_eval_reports_per_item_details():
    rep = evaluate(corpus_path=DEFAULT_CORPUS, gold_path=DEFAULT_GOLD)
    assert len(rep.details) == 10
    assert all("id" in row and "answered" in row for row in rep.details)
