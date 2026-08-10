PYTHON ?= python

ifeq ($(OS),Windows_NT)
VENV_PYTHON := .venv/Scripts/python.exe
else
VENV_PYTHON := .venv/bin/python
endif

.PHONY: dev fmt lint test test-fast coverage hygiene build package-smoke

dev:
	$(PYTHON) -m venv .venv
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -e .[dev]
	$(VENV_PYTHON) -m pip install -e ./psannlm
	$(VENV_PYTHON) -m pre_commit install

fmt:
	$(PYTHON) tools/quality.py format

lint:
	$(PYTHON) tools/quality.py lint

test:
	$(PYTHON) -m pytest

test-fast:
	$(PYTHON) -m pytest -m "not slow and not gpu"

coverage:
	$(PYTHON) tools/run_coverage.py

hygiene:
	$(PYTHON) tools/repo_hygiene_audit.py --strict-long-files

build:
	$(PYTHON) -m build
	$(PYTHON) -m build ./psannlm

package-smoke: build
	$(PYTHON) tools/package_smoke.py --system-site-packages --no-deps
