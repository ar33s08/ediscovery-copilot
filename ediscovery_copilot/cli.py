"""Command line entrypoint: ask questions against a corpus without the server."""

from __future__ import annotations

import argparse
import os

from ediscovery_copilot.agents import ReviewAgent
from ediscovery_copilot.audit import AuditTrail
from ediscovery_copilot.corpus import Corpus
from ediscovery_copilot.llm import provider_from_env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ediscovery")
    parser.add_argument("--corpus", default=os.path.join("data", "corpus.json"))
    parser.add_argument("--audit", default=os.path.join("var", "audit.jsonl"))
    parser.add_argument("question", nargs="?", default=None)
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args(argv)

    corpus = Corpus.from_json(args.corpus)
    agent = ReviewAgent(
        corpus=corpus,
        provider=provider_from_env(dict(os.environ)),
        audit=AuditTrail(args.audit),
        top_k=args.top_k,
    )
    if args.question is None:
        print('give a question: ediscovery "are the deal terms confidential?"')
        return 2
    result = agent.ask(args.question)
    print(result.answer)
    print("-" * 60)
    print(f"confidence={result.confidence} needs_human_review={result.needs_human_review}")
    for cid in result.citation_ids:
        chunk = corpus.chunk_index.get(cid)
        if chunk:
            print(f"\n[{cid}] ({chunk.doc_id}) {chunk.text[:160]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
