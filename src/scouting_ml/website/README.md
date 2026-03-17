# Scouting ML Frontend

This folder contains the static frontend for the recruitment-intelligence backend API aimed at smaller clubs and consultants hunting undervalued non-Big5 players:

- `static/index.html` - multi-view scouting application shell
- `static/assets/app.js` - API client + state + valuation/funnel logic
- `static/assets/styles.css` - responsive dashboard styling

## Main views

- `Overview` - model trust card, valuation-vs-shortlist champion routing, val/test KPIs, value-segment reliability, league coverage
- `Recruitment Board` - move between valuation and shortlist modes, then filter by age corridor, role need, budget band, contract years left, and outside-Big-5 focus
- `Target Funnel` - build non-Big5 scouting shortlists with the same smaller-club filters used on the board

## Recruitment exports

- `Export Club CSV` - compact shortlist export for club discussion or analyst review
- `Export Window Pack JSON` - consultant-style pack with active filters, champion routing, and current board rows
- `Export Memo JSON/CSV` - player-level memo export from the detail panel
- `Export Memo Pack JSON` - watchlist-oriented bulk pack for consultant delivery

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

Recommended first workflow:

1. connect the API
2. review `Overview`
3. use `Recruitment Board` in shortlist mode and tighten the age / budget / contract / role filters
4. move strongest targets into the watchlist
5. export a `Window Pack JSON` or `Memo Pack JSON` for club / consultant delivery
6. use `Target Funnel` for outside-Big-5 sourcing

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
