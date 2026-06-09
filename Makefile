.PHONY: install test quick full report smoke

install:
	python -m pip install -r requirements.txt

test:
	python -m pytest -q

quick:
	python -m knowledge_gap_decision.run_experiment --quick

full:
	python -m knowledge_gap_decision.run_experiment --target-size 800

report:
	python -m knowledge_gap_decision.report

smoke:
	python scripts/smoke_test.py
