PYTHON ?= python

ifeq ($(OS),Windows_NT)
VENV_PYTHON := .venv/Scripts/python.exe
else
VENV_PYTHON := .venv/bin/python
endif

.PHONY: dev fmt lint test test-fast build

dev:
	$(PYTHON) -m venv .venv
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -e .[dev]
	$(VENV_PYTHON) -m pip install -e ./psannlm
	$(VENV_PYTHON) -m pre_commit install

fmt:
	$(PYTHON) -m ruff format src tests scripts examples psannlm
	$(PYTHON) -m black src tests scripts examples psannlm

lint:
	$(PYTHON) -m ruff check src tests scripts examples psannlm --select F,E9
	$(PYTHON) -m black --check src tests scripts examples psannlm
	$(PYTHON) -m mypy src psannlm

test:
	$(PYTHON) -m pytest

test-fast:
	$(PYTHON) -m pytest -m "not slow and not gpu"

build:
	$(PYTHON) -m build
	$(PYTHON) -m build ./psannlm
