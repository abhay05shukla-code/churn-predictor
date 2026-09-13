PYTHON ?= python3
PY := .venv/bin/python

.PHONY: setup data train api dashboard test

## Create a virtualenv and install pinned dependencies (needs Python 3.12+; override with `make setup PYTHON=python3.12`)
setup:
	@$(PYTHON) -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)' \
		|| { echo "Python 3.12+ required, found $$($(PYTHON) --version 2>&1). Try: make setup PYTHON=python3.12"; exit 1; }
	$(PYTHON) -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

## Download the IBM Telco customer churn dataset
data:
	mkdir -p data
	curl -sL -o data/telco_churn.csv https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

## Train candidate models and export the best pipeline + metrics
train:
	$(PY) train.py

## Serve the model (docs at http://127.0.0.1:8000/docs)
api:
	.venv/bin/uvicorn api.main:app --reload --port 8000

## Launch the Streamlit dashboard (expects the API to be running)
dashboard:
	.venv/bin/streamlit run dashboard/app.py

## Run unit + API + dashboard integration tests
test:
	$(PY) -m pytest
