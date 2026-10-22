PYTHON ?= .venv/bin/python

.PHONY: setup test pilot full stats figures paper clean

setup:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest -q

pilot:
	$(PYTHON) scripts/run_pilot.py

full:
	$(PYTHON) scripts/run_full.py

stats:
	$(PYTHON) scripts/analyze_results.py

figures:
	$(PYTHON) scripts/make_figures.py

paper:
	$(PYTHON) scripts/build_paper.py

clean:
	rm -f results/raw/fold_metrics.csv results/raw/stability_metrics.csv
	rm -rf results/raw/checkpoints/*
