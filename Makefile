.PHONY: dev fmt lint test test-fast build

dev:
	python -m venv .venv
	./.venv/bin/pip install --upgrade pip
	./.venv/bin/pip install -e .[dev]
	./.venv/bin/pip install -e ./psannlm
	./.venv/bin/pre-commit install

fmt:
	python -m ruff format src tests scripts examples psannlm
	python -m black src tests scripts examples psannlm

lint:
	python -m ruff check src tests scripts examples psannlm --select F,E9
	python -m black --check src tests scripts examples psannlm
	python -m mypy src psannlm

test:
	python -m pytest

test-fast:
	python -m pytest -m "not slow and not gpu"

build:
	python -m build
	python -m build psannlm
