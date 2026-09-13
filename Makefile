.PHONY: install test lint fmt gate eval demo

install:
	uv sync --extra dev

lint:
	uv run ruff check ediscovery_copilot tests scripts

fmt:
	uv run ruff format ediscovery_copilot tests scripts

test:
	uv run pytest -q

eval:
	uv run python -m ediscovery_copilot.evalkit

gate: lint test eval
	@echo "all checks green"

demo:
	uv run ediscovery "Are the deal terms confidential under the NDA?"
