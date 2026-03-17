# Market Value Benchmark Report

This report is the single benchmark surface for current valuation model readiness across leagues.

It aggregates:

- current model metrics (`overall`, `segments`)
- league holdout benchmark outputs (`*.holdout_*.metrics.json`)
- non-Big5 onboarding readiness
- latest ablation bundle, if present
- current prediction coverage by league

## Build The Report

```powershell
$env:PYTHONPATH = "src"

python -m scouting_ml.scripts.build_market_value_benchmark_report `
  --metrics "data/model/champion_predictions_2024-25.metrics.json" `
  --predictions "data/model/champion_predictions_2024-25.csv" `
  --holdout-metrics-glob "data/model/**/*.holdout_*.metrics.json" `
  --onboarding-json "data/model/onboarding/non_big5_onboarding_report.json" `
  --out-json "data/model/reports/market_value_benchmark_report.json" `
  --out-md "data/model/reports/market_value_benchmark_report.md"
```

## What It Gives You

- one machine-readable payload for `/market-value/benchmarks`
- one Markdown summary for review in git or the editor
- top and weakest holdout leagues
- onboarding status counts
- current ablation winners
- current coverage snapshot by league

## UI/API Exposure

- API: `GET /market-value/benchmarks`
- Frontend: `Overview` now shows:
  - `League Benchmarks`
  - `Experiment Snapshot`

## Transfermarkt Context Upgrades

The latest dataset build now derives additional scouting context from Transfermarkt:

- contract join year -> `club_tenure_years`, `recent_arrival_flag`
- agent / loan presence -> `contract_agent_known_flag`, `contract_loan_context_flag`
- injury-type mix -> `injury_soft_tissue_share`, `injury_structural_share`, `injury_surgery_flag`
- transfer stability -> `transfer_recent_paid_share_3y`, `transfer_recent_loan_share_3y`, `transfer_last_move_paid_flag`

Use these as context features, not as stand-alone scouting decisions.
