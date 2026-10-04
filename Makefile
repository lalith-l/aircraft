# Nirnay Prototype — Makefile
# Python 3.11 virtual environment assumed at .venv/

PYTHON = .venv/bin/python
PIP    = .venv/bin/pip
PYTEST = .venv/bin/pytest

.PHONY: test lint format run clean install

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

test:
	PYTHONPATH=. $(PYTHON) -m pytest tests/ -v --tb=short

test-cov:
	PYTHONPATH=. $(PYTHON) -m pytest tests/ -v --tb=short --cov=nirnay --cov-report=term-missing

lint:
	$(PYTHON) -m py_compile nirnay/contracts.py
	@echo "Lint passed (basic syntax check)"

format:
	@echo "No formatter configured yet"

run:
	$(PYTHON) -m uvicorn nirnay.api.main:app --reload --port 8000

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov
