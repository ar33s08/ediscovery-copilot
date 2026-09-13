"""Corpus loading: JSON documents -> validated Document models -> chunks."""

from __future__ import annotations

import json
from pathlib import Path

from ediscovery_copilot.models import Chunk, Document, DocumentType


class Corpus:
    """An in-memory produced corpus with a stable chunk index."""

    def __init__(self, documents: list[Document]) -> None:
        self.documents = {doc.doc_id: doc for doc in documents}
        self.chunks: list[Chunk] = []
        for doc in documents:
            self.chunks.extend(Chunk.build(doc))
        self.chunk_index = {chunk.chunk_id: chunk for chunk in self.chunks}

    @classmethod
    def from_json(cls, path: Path | str) -> Corpus:
        """Load a corpus from a JSON array file (see data/corpus.json)."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        docs: list[Document] = []
        for item in raw:
            docs.append(
                Document(
                    doc_id=item["doc_id"],
                    title=item["title"],
                    author=item.get("author"),
                    recipients=list(item.get("recipients", [])),
                    created_at=item.get("created_at"),
                    doc_type=DocumentType(item.get("doc_type", "other")),
                    custodian=item.get("custodian"),
                    native_metadata=dict(item.get("native_metadata", {})),
                    text=item["text"],
                )
            )
        return cls(docs)

    def get(self, doc_id: str) -> Document:
        return self.documents[doc_id]
