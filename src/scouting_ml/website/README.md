# Scouting ML Frontend

`src/scouting_ml/website/static/` is the canonical backend-connected frontend for ScoutML.

Current frontend paths:

- `static/index.html` - canonical recruitment UI backed by the FastAPI market-value API
- `static/assets/app.js` - API client, state management, watchlist actions, exports, and detail workflows
- `static/assets/styles.css` - canonical dashboard styling
- `frontend/` - separate Next.js mock-data demo kept for portfolio purposes, not the canonical backend UI
- `legacy/` - archived generated frontend and build scripts kept only as reference

## Main views

- `Overview` - readiness, artifact routing, metrics, and benchmark context
- `Recruitment Board` - valuation and shortlist workflows with age, role, budget, contract, and non-Big5 filters
- `Target Funnel` - candidate funnel for non-Big5 scouting prioritization

## Run locally

### 1) Start the backend API

PowerShell:

```powershell
$env:PYTHONPATH = "src"
$env:SCOUTING_API_CORS_ORIGINS = "http://localhost:8080,http://127.0.0.1:8080,http://localhost:5500,http://127.0.0.1:5500"
uvicorn scouting_ml.api.main:app --reload --host 0.0.0.0 --port 8000
```

Bash:

```bash
export PYTHONPATH=src
export SCOUTING_API_CORS_ORIGINS="http://localhost:8080,http://127.0.0.1:8080,http://localhost:5500,http://127.0.0.1:5500"
python3 -m uvicorn scouting_ml.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 2) Start a static server

From the repository root:

```bash
python3 -m http.server 8080
```

Or:

```bash
make static-ui
```

Open:

- `http://localhost:8080/src/scouting_ml/website/static/index.html`

In the UI, set API base to:

- `http://localhost:8000`

## Degraded mode

The UI treats `/market-value/health` as a readiness gate.

- If artifacts are ready, the board, funnel, and watchlist load normally.
- If artifacts are missing or strict startup validation fails, the UI stops after health and shows the degraded status instead of cascading into failing market-value requests.

## Artifact expectations

The canonical frontend expects the backend to resolve these artifact paths by default:

- `data/model/big5_predictions_full_v2.csv`
- `data/model/big5_predictions_full_v2_val.csv`
- `data/model/big5_predictions_full_v2.metrics.json`

If artifacts live elsewhere, set the usual env vars before starting `uvicorn`:

- `SCOUTING_TEST_PREDICTIONS_PATH`
- `SCOUTING_VAL_PREDICTIONS_PATH`
- `SCOUTING_METRICS_PATH`
- `SCOUTING_MODEL_MANIFEST_PATH`
- `SCOUTING_BENCHMARK_REPORT_PATH`

## Frontend status

- Canonical backend-connected frontend: `src/scouting_ml/website/static/`
- Optional mock-data demo: `frontend/`
- Legacy generated frontend archive: `src/scouting_ml/website/legacy/`; kept only as reference, not the active product path
