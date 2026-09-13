import os
import tempfile

from ediscovery_copilot import cli

_ROOT = os.path.dirname(os.path.dirname(__file__))


def test_cli_answers_question(capsys):
    audit = os.path.join(tempfile.mkdtemp(), "cli.jsonl")
    code = cli.main(
        [
            "--corpus",
            os.path.join(_ROOT, "data", "corpus.json"),
            "--audit",
            audit,
            "Are the deal terms confidential under the NDA?",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "confidence=" in out
    assert "PROD-0003" in out or "c1" in out or "[" in out


def test_cli_requires_question(capsys):
    audit = os.path.join(tempfile.mkdtemp(), "cli.jsonl")
    code = cli.main(
        [
            "--corpus",
            os.path.join(_ROOT, "data", "corpus.json"),
            "--audit",
            audit,
        ]
    )
    assert code == 2
