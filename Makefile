.PHONY: help install install-dev install-modal install-langfuse run-all \
        quickstart docker-repl lm-repl modal-repl detective langfuse-example \
        lint format test check

help:
	@echo "RLM Examples Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make install        - Install base dependencies with uv"
	@echo "  make install-dev    - Install dev dependencies with uv"
	@echo "  make install-modal  - Install modal dependencies with uv"
	@echo "  make install-langfuse - Install langfuse dependencies with uv"
	@echo "  make run-all        - Run all examples (requires all deps and API keys)"
	@echo ""
	@echo "Examples:"
	@echo "  make quickstart     - Run quickstart.py (needs OPENAI_API_KEY)"
	@echo "  make docker-repl    - Run docker_repl_example.py (needs Docker)"
	@echo "  make lm-repl        - Run lm_in_repl.py (needs PORTKEY_API_KEY)"
	@echo "  make modal-repl     - Run modal_repl_example.py (needs Modal)"
	@echo "  make detective      - Run detective_example.py (needs PROXY_API_KEY)"
	@echo "  make langfuse-example - Run langfuse_example.py (needs Langfuse + proxy keys)"
	@echo ""
	@echo "Development:"
	@echo "  make lint           - Run ruff linter"
	@echo "  make format         - Run ruff formatter"
	@echo "  make test           - Run tests"
	@echo "  make check          - Run lint + format + tests"

install:
	uv sync

install-dev:
	uv sync --group dev --group test

install-modal:
	uv pip install -e ".[modal]"

install-langfuse:
	uv pip install -e ".[langfuse]"

run-all: quickstart docker-repl lm-repl modal-repl

quickstart: install
	uv run python -m examples.quickstart

docker-repl: install
	uv run python -m examples.docker_repl_example

lm-repl: install
	uv run python -m examples.lm_in_repl

modal-repl: install-modal
	uv run python -m examples.modal_repl_example

detective: install
	uv run python -m examples.detective_example

langfuse-example: install-langfuse
	uv run python -m examples.langfuse_example

lint: install-dev
	uv run ruff check .

format: install-dev
	uv run ruff format .

test: install-dev
	uv run pytest

check: lint format test
