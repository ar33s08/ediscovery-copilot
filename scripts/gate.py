"""One-command quality gate: lint -> tests -> eval. CI runs the same thing."""

from __future__ import annotations

import subprocess
import sys

STEPS = [
    ("lint", ["ruff", "check", "ediscovery_copilot", "tests", "scripts"]),
    ("format", ["ruff", "format", "--check", "ediscovery_copilot", "tests"]),
    ("tests", ["pytest", "-q"]),
    ("eval", [sys.executable, "-m", "ediscovery_copilot.evalkit"]),
]


def main() -> int:
    failed = []
    for label, argv in STEPS:
        print(f"== {label}: {' '.join(argv)}")
        code = subprocess.call(argv)
        if code != 0:
            failed.append(label)
        print()
    if failed:
        print(f"GATE: RED -> {failed}")
        return 1
    print("GATE: GREEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
