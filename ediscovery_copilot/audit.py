"""Tamper-evident audit trail: hash-chained JSONL.

Each entry embeds the digest of the previous entry, so editing or deleting a
past record breaks the chain and is detectable by verify_chain().
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

GENESIS = "0" * 64


def _entry_hash(payload: dict[str, Any], prev_hash: str, seq: int) -> str:
    blob = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"))
        + "|"
        + prev_hash
        + "|"
        + str(seq)
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class AuditTrail:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._last_hash = GENESIS
        self._seq = 0
        self._load_tail()

    def _load_tail(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            self._last_hash = rec["entry_hash"]
            self._seq = rec["seq"]

    def record(self, event: str, payload: dict[str, Any]) -> dict[str, Any]:
        rec = {
            "seq": self._seq + 1,
            "at": datetime.now(UTC).isoformat(),
            "event": event,
            "payload": payload,
            "prev_hash": self._last_hash,
        }
        rec["entry_hash"] = _entry_hash(payload, self._last_hash, self._seq + 1)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
        self._last_hash = rec["entry_hash"]
        self._seq = rec["seq"]
        return rec

    def verify_chain(self) -> tuple[bool, int | None]:
        """Returns (valid, first_bad_seq)."""
        prev = GENESIS
        seq = 0
        if not self.path.exists():
            return True, None
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            seq = rec["seq"]
            if rec["prev_hash"] != prev:
                return False, seq
            want = _entry_hash(rec["payload"], prev, seq)
            if rec["entry_hash"] != want:
                return False, seq
            prev = rec["entry_hash"]
        return True, None

    def entries(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [
            json.loads(l) for l in self.path.read_text(encoding="utf-8").splitlines() if l.strip()
        ]
