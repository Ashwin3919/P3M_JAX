.PHONY: setup clean

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Environment ready. Activate with: source $(VENV)/bin/activate"

clean:
	rm -rf $(VENV)
	rm -rf .pytest_cache
	rm -rf .jax_cache
	rm -rf .nv
	find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} +
	find . -type f -name "*.pyc" -not -path "./.venv/*" -delete
	find . -type f -name "*.pyo" -not -path "./.venv/*" -delete
	find . -name ".DS_Store" -delete
	@echo "Clean done."
