PYTHON ?= python3
PIP ?= $(PYTHON) -m pip
API_HOST ?= 0.0.0.0
API_PORT ?= 8000
STATIC_PORT ?= 8080

.PHONY: install lint test verify api static-ui smoke-api

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

lint:
	PYTHONPATH=src $(PYTHON) -m ruff check src/scouting_ml/api src/scouting_ml/core src/scouting_ml/services src/scouting_ml/tests

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q

verify: lint smoke-api test

api:
	PYTHONPATH=src $(PYTHON) -m uvicorn scouting_ml.api.main:app --reload --host $(API_HOST) --port $(API_PORT)

static-ui:
	$(PYTHON) -m http.server $(STATIC_PORT)

smoke-api:
	PYTHONPATH=src $(PYTHON) -c "from scouting_ml.api.main import app; paths = {route.path for route in app.routes}; required = {'/', '/health', '/market-value/health'}; missing = sorted(required - paths); assert not missing, missing; print('API smoke ok')"
