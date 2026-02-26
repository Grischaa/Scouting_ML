# Scouting ML Frontend

This folder contains the static frontend for the market-value backend API:

- `static/index.html` - multi-view scouting application shell
- `static/assets/app.js` - API client + state + valuation/funnel logic
- `static/assets/styles.css` - responsive dashboard styling

## Main views

- `Overview` - model trust card, val/test KPIs, value-segment reliability, league coverage
- `Valuation Workbench` - filter/sort players and inspect over/undervalued signals
- `Talent Funnel` - build scouting shortlists with a lower-league-only toggle

## Run locally

### 1) Start backend API

PowerShell:

```powershell
$env:PYTHONPATH = "src"
$env:SCOUTING_API_CORS_ORIGINS = "http://localhost:8080,http://127.0.0.1:8080,http://localhost:5500"
uvicorn scouting_ml.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 2) Start a static server

From repository root:

```bash
python3 -m http.server 8080
```

Open:

- `http://localhost:8080/src/scouting_ml/website/static/index.html`

In the UI, set API base to:

- `http://localhost:8000`

## Required artifacts

The frontend uses backend endpoints that read these files by default:

- `data/model/big5_predictions_full_v2.csv`
- `data/model/big5_predictions_full_v2_val.csv`
- `data/model/big5_predictions_full_v2.metrics.json`

If you store artifacts elsewhere, set:

- `SCOUTING_TEST_PREDICTIONS_PATH`
- `SCOUTING_VAL_PREDICTIONS_PATH`
- `SCOUTING_METRICS_PATH`

before starting `uvicorn`.
