"""Offline evaluation harness: measures retrieval recall and answer honesty.

Run: python -m ediscovery_copilot.evalkit

Metrics reported for the gold set (data/gold.json):
  retrieval_hit_rate   fraction of queries whose expected document appears in
                       the fused top-k chunks
  answer_rate          fraction of answerable items that produced an answer
  hallucination_rate   answerable items answered WITHOUT verifiable citation
                       support (verifier rejected) -- lower is better
  refusal_correctness  unanswerable items the system correctly refused
  needs_human_review   how much work is routed to humans
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from ediscovery_copilot.agents import INSUFFICIENT, ReviewAgent, verify_answer
from ediscovery_copilot.audit import AuditTrail
from ediscovery_copilot.corpus import Corpus
from ediscovery_copilot.llm import ExtractiveFallbackProvider, provider_from_env

_ROOT = os.path.dirname(os.path.dirname(__file__))
DEFAULT_CORPUS = os.path.join(_ROOT, "data", "corpus.json")
DEFAULT_GOLD = os.path.join(_ROOT, "data", "gold.json")


@dataclass
class EvalReport:
    total: int = 0
    retrieval_hits: int = 0
    answered: int = 0
    grounded_ok: int = 0
    hallucinations: int = 0
    refusals_expected: int = 0
    refusals_correct: int = 0
    routed_to_human: int = 0
    details: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        answerable = self.total - self.refusals_expected
        return {
            "total": self.total,
            "retrieval_hit_rate": round(self.retrieval_hits / max(self.total, 1), 3),
            "answer_rate": round(self.answered / max(answerable, 1), 3),
            "grounded_answer_rate": round(self.grounded_ok / max(self.answered, 1), 3),
            "hallucination_rate": round(self.hallucinations / max(self.answered, 1), 3),
            "refusal_correctness": round(self.refusals_correct / max(self.refusals_expected, 1), 3),
            "needs_human_review_rate": round(self.routed_to_human / max(self.total, 1), 3),
        }


def evaluate(
    corpus_path: str = DEFAULT_CORPUS, gold_path: str = DEFAULT_GOLD, top_k: int = 6
) -> EvalReport:
    corpus = Corpus.from_json(corpus_path)
    agent = ReviewAgent(
        corpus=corpus,
        provider=provider_from_env(dict(os.environ))
        if os.environ
        else ExtractiveFallbackProvider(),
        audit=AuditTrail(os.path.join(os.getcwd(), "var", "eval-audit.jsonl")),
        top_k=top_k,
    )
    with open(gold_path, encoding="utf-8") as gh:
        gold = json.loads(gh.read())
    rep = EvalReport(total=len(gold))
    for item in gold:
        query = item["query"]
        expected = item.get("expected_doc")
        hits = agent.retriever.search(query, top_k=top_k)
        doc_ids = [h.chunk.doc_id for h in hits]
        hit = expected is not None and expected in doc_ids
        if expected is None:
            rep.refusals_expected += 1
        else:
            if hit:
                rep.retrieval_hits += 1
        result = agent.ask(query, category=item.get("expected_category"))
        answered = not result.answer.strip().casefold().startswith(INSUFFICIENT.casefold())
        if answered:
            rep.answered += 1
            ver = verify_answer(_RawFromQuestion(result), corpus.chunk_index)
            if ver.ok:
                rep.grounded_ok += 1
            else:
                rep.hallucinations += 1
        elif expected is None:
            rep.refusals_correct += 1
        if result.needs_human_review:
            rep.routed_to_human += 1
        rep.details.append(
            {
                "id": item["id"],
                "query": query[:60],
                "retrieval_hit": hit if expected else None,
                "answered": answered,
                "confidence": result.confidence,
                "needs_human_review": result.needs_human_review,
                "citations": len(result.citation_ids),
            }
        )
    return rep


def _RawFromQuestion(result):
    from ediscovery_copilot.llm import RawAnswer

    return RawAnswer(result.answer, result.citation_ids, "eval")


def main() -> int:
    report = evaluate()
    summary = report.as_dict()
    print(json.dumps(summary, indent=2))
    for row in report.details:
        print(
            "  {id:<4} hit={retrieval_hit!s:<5} ans={answered!s:<5} conf={confidence} cites={citations}".format(
                **row
            )
        )
    ok = summary["retrieval_hit_rate"] >= 0.8 and summary["hallucination_rate"] <= 0.1
    print("EVAL:", "PASS" if ok else "BELOW-THRESHOLD")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
