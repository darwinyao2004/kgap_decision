PYTHON ?= .venv/bin/python

.PHONY: install test quick full report smoke

.venv/bin/python:
	python3 -m venv .venv

install: .venv/bin/python
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest -q

quick:
	$(PYTHON) -m knowledge_gap_decision.run_experiment --quick

full:
	$(PYTHON) -m knowledge_gap_decision.run_experiment --target-size 800

report:
	$(PYTHON) -m knowledge_gap_decision.report

smoke:
	$(PYTHON) scripts/smoke_test.py
