# ScoutML: Artifact-Driven Recruitment Intelligence

ScoutML is a football scouting and market-value ML platform built around an artifact-driven valuation pipeline, a FastAPI serving layer, and a backend-connected static recruitment UI. It supports both a local-first analyst workflow and a database-backed Team Edition for shared scouting workspaces.

What is materially implemented today:

- position-aware player valuation artifacts and confidence-aware undervaluation ranking
- shortlist, scout-target, and backend-owned system-fit workflows
- detail-view payloads for scouting memos, archetype fit, formation fit, radar context, history strength, similar players, and trajectory
- PDF memo export plus consultant-ready CSV / JSON export flows
- local-first watchlist / decision logging and Team Edition shared workspaces with auth, comments, assignments, compare lists, and scout preference profiles
- explicit degraded-mode/readiness behavior when required artifacts are missing
- Docker / Compose plus Railway / Render deployment config

This repo is best understood as a serious recruitment-workflow product foundation: artifact-driven ML and evaluation on the backend, a usable scouting workbench on the frontend, and a first-pass collaborative team layer for multi-scout workflows.

---

## 1) What This Project Does

ScoutML builds and serves player valuation predictions from season-level data and external enrichment, then turns them into concrete recruitment workflows for affordable non-Big-5 opportunity mapping.

Core capabilities:

- Position-aware valuation modeling (`GK`, `DF`, `MF`, `FW`)
- Conservative valuation guardrails (capped gap logic)
- League-adjusted valuation correction and trust-aware discovery weighting
- Backend-first system fit with named tactical templates and slot-level rankings
- Player report API with:
  - strengths / weaknesses / development levers
  - risk flags
  - history-strength score
  - archetype + formation fit + radar payload
  - similar players
  - trajectory
  - proxy estimates for sparse secondary metrics
- Frontend console for:
  - model trust + reliability overview
  - recruitment board with budget / contract / age / role filters
  - target funnel + watchlist management
  - system-fit slot inspector
  - scout decisions, compare tray, and preference-aware reranking
  - consultant-ready exports (club CSV, window pack JSON, player memo exports)
- Team Edition:
  - email/password auth
  - shared workspaces and invites
  - role-based access (`admin`, `scout`, `viewer`)
  - shared watchlist, decisions, comments, assignments, activity, compare lists
- One-command artifact pipeline + one-command weekly scout ops

---

## 2) Repository Layout

Key folders:

- `src/scouting_ml/models` -> dataset build, cleaning, model training
- `src/scouting_ml/scripts` -> orchestration and utility scripts
- `src/scouting_ml/services` -> business logic (valuation, artifacts, reports, watchlist, memo, similarity, trajectory)
- `src/scouting_ml/api` -> FastAPI routes
- `src/scouting_ml/team` -> Team Edition SQLAlchemy models, DB helpers, and workspace services
- `src/scouting_ml/website/static` -> canonical backend-connected frontend (HTML/CSS/JS)
- `frontend` -> optional Next.js mock-data demo, kept for portfolio use and not the canonical backend UI
- `src/scouting_ml/website` -> canonical static frontend sources plus archived legacy generated frontend reference
- `data/model` -> model artifacts, backtests, reports
- `data/processed` -> source player-season CSV organization
- `data/external` -> enrichment tables (injury, contract, transfer, national, context)

Operational notes:

- startup readiness is artifact-driven, and market-value routes degrade explicitly when artifacts are unavailable
- local mode keeps JSONL-backed watchlist / decisions
- team mode switches shared workflow state to SQLAlchemy-backed workspace persistence without replacing the underlying player/model endpoints

---

## 3) Setup

### 3.1 Create and activate virtual environment

PowerShell:

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
```

Bash:

```bash
python3 -m venv .venv
. .venv/bin/activate
```

### 3.2 Install dependencies

PowerShell:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Bash:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

If dependency resolution fails, confirm you are in `.venv` and retry.
Current `requirements.txt` expects `pyyaml==6.0.2` (needed with `scraperfc==4.1.0`).

### 3.3 Runtime configuration

- Review `.env.example` for the canonical API, artifact, watchlist, provider, similarity, and experimental NLP environment variables.
- Minimum backend env for local work is usually just:
  - `PYTHONPATH=src`
  - `SCOUTING_API_CORS_ORIGINS=...`
- Artifact path overrides remain backwards-compatible with `data/model/model_artifacts.env`.
- Team Edition is enabled when:
  - `SCOUTING_DATABASE_URL` is set
  - `SCOUTING_TEAM_MODE=1` (or left unset and inferred from the DB URL)
- Team auth/session env:
  - `SCOUTING_SESSION_COOKIE_NAME`
  - `SCOUTING_SESSION_SECRET`
  - `SCOUTING_INVITE_TOKEN_TTL_HOURS`
  - `SCOUTING_SESSION_SECURE_COOKIE`

### 3.4 Hosted deployment

The repo includes:

- `Dockerfile`
- `docker-compose.yml`
- `railway.toml`
- `render.yaml`

Typical hosted env:

- `PYTHONPATH=src`
- `SCOUTING_API_CORS_ORIGINS`
- artifact env from `data/model/model_artifacts.env` or equivalent secrets
- `MODEL_ARTIFACTS_DIR` when mounting artifact bundles
- Team Edition envs when running shared workspace mode

The frontend reads `window.SCOUTING_API_BASE` when injected by the host page and otherwise falls back to `http://127.0.0.1:8000`.

---

## 4) Data Prerequisite: Organize Processed CSVs

If your non-Big5 data is not structured like Big5, run:

```powershell
$env:PYTHONPATH = "src"
python -m scouting_ml.scripts.organize_processed_csvs `
  --source-dir "data/processed" `
  --combined-dir "data/processed/Clubs combined" `
  --country-root "data/processed/by_country" `
  --season-root "data/processed/by_season" `
  --manifest "data/processed/organization_manifest.csv"
```

This is critical for league/season completeness logic and onboarding checks.

---

## 5) One-Command Production Pipeline

Runs:

- enrichment + dataset + clean + train,
- rolling backtest,
- artifact lock (manifest + env),
- weekly scout ops (KPI + onboarding + workflow).

```powershell
$env:PYTHONPATH = "src"

python -m scouting_ml.scripts.run_production_pipeline `
  --players-source "data/processed/Clubs combined" `
  --data-dir "data/processed/Clubs combined" `
  --external-dir "data/external" `
  --dataset-output "data/model/champion_players.parquet" `
  --clean-output "data/model/champion_players_clean.parquet" `
  --predictions-output "data/model/champion_predictions_2024-25.csv" `
  --val-season "2023/24" `
  --test-season "2024/25" `
  --trials 40 `
  --optimize-metric "lowmid_wmape" `
  --band-min-samples 160 `
  --band-blend-alpha 0.35 `
  --with-backtest `
  --backtest-test-seasons "2022/23,2023/24,2024/25,2025/26" `
  --backtest-skip-incomplete-test-seasons `
  --run-weekly-ops `
  --weekly-watchlist-tag "u23_non_big5"
```

Fast rerun (reuse existing enrichment/dataset):

```powershell
$env:PYTHONPATH = "src"

python -m scouting_ml.scripts.run_production_pipeline `
  --predictions-output "data/model/champion_predictions_2024-25.csv" `
  --skip-injuries --skip-contracts --skip-transfers --skip-national --skip-context --skip-dataset-build --skip-clean
```

---

## 6) Weekly Scout Ops Only

If you already have fresh predictions and just want scouting outputs:

```powershell
$env:PYTHONPATH = "src"

python -m scouting_ml.scripts.run_weekly_scout_ops `
  --predictions "data/model/champion_predictions_2024-25.csv" `
  --split test `
  --k-values "10,25,50" `
  --label-col "interval_contains_truth" `
  --min-minutes 900 `
  --max-age 23 `
  --non-big5-only `
  --workflow-write-watchlist `
  --workflow-watchlist-tag "u23_non_big5"
```

Generated outputs include:

- `data/model/reports/weekly_scout_kpi_*.csv|json`
- `data/model/reports/future_value_benchmark_report.json|md`
- `data/model/reports/future_target_coverage_audit.json|csv`
- `data/model/reports/future_scored/*`
- `data/model/onboarding/non_big5_onboarding_report.csv|json`
- `data/model/scout_workflow/scout_workflow_*`
- `data/model/scout_workflow/weekly_ops_summary_*.json`

---

## 7) Manual Core Commands (When Needed)

### 7.1 Full pipeline (without production wrapper)

```powershell
$env:PYTHONPATH = "src"
python -m scouting_ml.scripts.run_full_pipeline --help
```

Useful option:

- `--summary-json "data/model/full_pipeline_summary.json"`
  - writes a structured manifest of the run inputs, flags, validated artifact paths, and key metric/backtest snapshots
  - this is now also emitted automatically by the production and provider-promotion wrappers

### 7.2 Lock artifact bundle

```powershell
$env:PYTHONPATH = "src"

python -m scouting_ml.scripts.lock_market_value_artifacts `
  --test-predictions "data/model/champion_predictions_2024-25.csv" `
  --val-predictions "data/model/champion_predictions_2024-25_val.csv" `
  --metrics "data/model/champion_predictions_2024-25.metrics.json" `
  --manifest-out "data/model/model_manifest.json" `
  --env-out "data/model/model_artifacts.env" `
  --label "champion_2024_25"
```

Load lock env before API:

```powershell
Get-Content data/model/model_artifacts.env | ForEach-Object {
  if ($_ -match '=') {
    $k, $v = $_ -split '=', 2
    [System.Environment]::SetEnvironmentVariable($k, $v, 'Process')
  }
}
```

### 7.3 Rolling backtest

```powershell
$env:PYTHONPATH = "src"
python -m scouting_ml.scripts.run_rolling_backtest --help
```

### 7.4 Band hyperparameter sweep

```powershell
$env:PYTHONPATH = "src"
python -m scouting_ml.scripts.run_band_hyper_sweep --help
```

### 7.5 Collect Provider Snapshots

Use this when you have live provider credentials and want to materialize raw `2024/25` snapshots plus a generated provider pipeline config:

```powershell
$env:PYTHONPATH = "src"
$env:SPORTMONKS_API_TOKEN = "..."
$env:ODDS_API_KEY = "..."

python -m scouting_ml.scripts.collect_provider_snapshots `
  --config-json "docs/provider_snapshot_collection.example.json"
```

Notes:

- This writes raw JSON payloads under `data/raw/providers/`.
- It also emits a generated provider config, ready for the promotion pipeline.
- Dry-run support is available with `--dry-run`.
- Full workflow details: `docs/provider_snapshot_collection.md`

### 7.6 Provider Candidate Promotion Workflow

Use this when you want to test provider-enriched features against the current champion before replacing active artifacts:

```powershell
$env:PYTHONPATH = "src"

python -m scouting_ml.scripts.run_provider_promotion_pipeline `
  --provider-config-json "docs/provider_pipeline_config.example.json" `
  --candidate-tag "provider_candidate" `
  --out-dir "data/model/provider_promotion" `
  --trials 40 `
  --with-backtest `
  --promote-on-pass
```

Notes:

- The run stages a candidate-only `external/` directory and does not need to overwrite your main champion artifacts.
- Missing provider snapshot files are pruned automatically from the effective config.
- `fixture_context`, `player_availability`, and `market_context` can use either local `input_json` files or `api_url` values in the provider config.
- A generated config from the snapshot collector can be passed directly to `--provider-config-json`.
- Full workflow details: `docs/provider_promotion_pipeline.md`

### 7.7 Collect SofaScore Website Snapshots

Use this when you want to build `fixture_context` and `player_availability` directly from SofaScore website data instead of Sportmonks/API-Football:

```powershell
$env:PYTHONPATH = "src"

python -m scouting_ml.scripts.collect_sofascore_snapshots `
  --config-json "docs/sofascore_snapshot_collection.example.json"
```

Notes:

- This writes raw season-fixture and lineup snapshots under `data/raw/providers/`.
- The generated provider config includes `provider="sofascore"` and requires `players_source` so Sofa IDs resolve back to your internal player/club keys.
- You can pass the generated config directly into `run_provider_promotion_pipeline`.
- Full workflow details: `docs/sofascore_snapshot_collection.md`

### 7.8 Run Market Value Ablations

Use this when you want a reproducible feature-block comparison with slice diagnostics instead of only overall metrics:

```powershell
$env:PYTHONPATH = "src"

python -m scouting_ml.scripts.run_market_value_ablation `
  --dataset "data/model/big5_players_clean.parquet" `
  --val-season "2023/24" `
  --test-season "2024/25" `
  --configs "full,no_provider,no_context,no_profile_context,baseline_stats_only" `
  --out-dir "data/model/ablation" `
  --trials 40
```

Artifacts:

- `ablation_summary_<season>.csv`: overall ranking plus low/mid-market slice metrics
- `ablation_slices_<season>.csv`: slice matrix by position, league, value segment, and position x value segment
- `ablation_bundle_<season>.json`: machine-readable winners and weak-slice summary
- `ablation_report_<season>.md`: readable report highlighting best configs and weakest full-model slices

### 7.9 Build Multi-League Benchmark Report

Use this after training or holdout runs to aggregate current metrics, league-holdout outputs, onboarding readiness, and ablation winners into one report consumed by the API/UI overview:

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

Artifacts:

- `market_value_benchmark_report.json`: machine-readable benchmark payload for API/UI
- `market_value_benchmark_report.md`: readable benchmark snapshot for review

### 7.10 Build Future Value Benchmark Report

Use this after training when you want to answer the scouting question directly: does the current ranking actually surface players who gain market value next season?

```powershell
$env:PYTHONPATH = "src"

python -m scouting_ml.scripts.build_future_value_benchmark_report `
  --test-predictions "data/model/reports/low_value_strategy_focused/cheap_aggressive_2024-25.csv" `
  --val-predictions "data/model/reports/low_value_strategy_focused/cheap_aggressive_2024-25_val.csv" `
  --dataset "data/model/tm_context_candidate_clean.parquet" `
  --min-next-minutes 450 `
  --min-minutes 900 `
  --k-values "10,25,50" `
  --cohort-min-labeled 25 `
  --out-json "data/model/reports/future_value_benchmark_report.json" `
  --out-md "data/model/reports/future_value_benchmark_report.md"
```

Artifacts:

- `future_value_benchmark_report.json`: machine-readable split benchmark with label coverage, score-vs-growth correlation, and precision@k by cohort
- `future_value_benchmark_report.md`: readable snapshot for scouting review

Notes:

- If `--future-targets` is not provided, the script builds next-season labels in memory from `--dataset`.
- This report is only as strong as next-season data coverage. For `2024/25` today, `2025/26` coverage is still sparse, so the `test` split will warn about low label coverage.

### 7.11 Build Future Target Coverage Audit

Use this to build a durable future-target parquet and audit how much next-season label coverage actually exists in the current dataset:

```powershell
$env:PYTHONPATH = "src"

python -m scouting_ml.scripts.build_future_target_coverage_audit `
  --dataset "data/model/tm_context_candidate_clean.parquet" `
  --future-targets-output "data/model/big5_players_future_targets.parquet" `
  --out-json "data/model/reports/future_target_coverage_audit.json" `
  --out-csv "data/model/reports/future_target_coverage_audit.csv"
```

Artifacts:

- `future_target_coverage_audit.json`: source-file inventory plus season / season+league label coverage
- `future_target_coverage_audit.csv`: flat coverage table for quick inspection
- `big5_players_future_targets.parquet`: next-season target dataset built from the current clean data

### 7.12 Build A Future-Target-Tuned Scout Score

Use this when you want a scouting score trained directly on future value-growth outcomes instead of reusing only `undervaluation_score`:

```powershell
$env:PYTHONPATH = "src"

python -m scouting_ml.scripts.build_future_scout_score `
  --val-predictions "data/model/reports/low_value_strategy_focused/cheap_aggressive_2024-25_val.csv" `
  --test-predictions "data/model/reports/low_value_strategy_focused/cheap_aggressive_2024-25.csv" `
  --out-val "data/model/reports/future_scored/cheap_aggressive_2024-25_val_future_scored.csv" `
  --out-test "data/model/reports/future_scored/cheap_aggressive_2024-25_future_scored.csv" `
  --diagnostics-out "data/model/reports/future_scored/cheap_aggressive_future_score_diagnostics.json" `
  --dataset "data/model/tm_context_candidate_clean.parquet" `
  --min-next-minutes 450 `
  --min-minutes 900 `
  --label-mode positive_growth `
  --k-eval 25
```

Artifacts:

- scored prediction CSVs with `future_growth_probability` and `future_scout_blend_score`
- diagnostics JSON with training rows, AUC / AP, precision@k vs the base ranking, and top logistic coefficients

Notes:

- If `future_scout_blend_score` exists in a prediction artifact, shortlist selection will now prefer it automatically.
- This score still depends on next-season label coverage, so it is strongest on `val` right now and weak on the current `test` split.

### 7.13 Refresh Future-Season Data In One Command

Use this when you receive new `2025/26` `*_with_sofa.csv` league files and want to:

- import them into `data/processed`,
- rebuild the clean dataset,
- rebuild future targets + coverage audit,
- rescore the active cheap-player predictions with future labels,
- refresh future benchmark reports,
- emit league/position/value-segment future diagnostics,
- optionally rerun future-gated promotion.

```powershell
$env:PYTHONPATH = "src"

python -m scouting_ml.scripts.run_future_data_refresh `
  --import-dir "data/incoming_future_2025_26" `
  --run-promotion
```

Important behavior:

- `--import-dir` should contain at most one file per league-season basename, for example `dutch_eredivisie_2025-26_with_sofa.csv`
- if `--import-dir` is missing or empty, the script now skips the import step and still rebuilds the future artifacts from the files already on disk
- imported files are copied into `data/processed/_incoming_future/` and then canonicalized into:
  - `data/processed/Clubs combined`
  - `data/processed/by_country`
  - `data/processed/by_season`
- the refresh summary is written to:
  - `data/model/reports/future_scored/future_refresh_summary.json`

Default outputs refreshed by the command:

- `data/model/big5_players_future_targets.parquet`
- `data/model/reports/future_target_coverage_audit.json|csv`
- `data/model/reports/future_scored/cheap_aggressive_2024-25_val_future_scored.csv`
- `data/model/reports/future_scored/cheap_aggressive_2024-25_future_scored.csv`
- `data/model/reports/future_scored/cheap_aggressive_future_benchmark_report.json|md`
- `data/model/reports/future_scored/cheap_aggressive_future_diagnostics.json|md`
- `data/model/reports/future_scored/cheap_aggressive_future_benchmark_u23_nonbig5_report.json|md`

If you only want to regenerate the diagnostics from an already-built future benchmark:

```powershell
$env:PYTHONPATH = "src"

python -m scouting_ml.scripts.build_future_value_diagnostics_report `
  --benchmark-json "data/model/reports/future_scored/cheap_aggressive_future_benchmark_report.json" `
  --out-json "data/model/reports/future_scored/cheap_aggressive_future_diagnostics.json" `
  --out-md "data/model/reports/future_scored/cheap_aggressive_future_diagnostics.md"
```

### 7.14 Backfill Missing Future League Files

Use this when the missing piece is the actual `2025/26` league file itself and you want the repo to generate it from the existing Transfermarkt + Sofascore pipeline.

This runner will:

- scrape the league season from Transfermarkt,
- pull aggregated Sofa league stats,
- merge them into a canonical `*_with_sofa.csv`,
- copy the result into `data/incoming_future_2025_26/` for `run_future_data_refresh`.

```powershell
$env:PYTHONPATH = "src"

python -m scouting_ml.scripts.backfill_future_league_files `
  --season "2025/26" `
  --leagues "dutch_eredivisie,portuguese_primeira_liga,belgian_pro_league,turkish_super_lig,greek_super_league,scottish_premiership" `
  --import-dir "data/incoming_future_2025_26" `
  --summary-json "data/model/reports/future_scored/future_backfill_summary.json"
```

Then run the normal future refresh:

```powershell
$env:PYTHONPATH = "src"

python -m scouting_ml.scripts.run_future_data_refresh `
  --import-dir "data/incoming_future_2025_26" `
  --run-promotion
```

Important behavior:

- If the requested season is not present in the league registry, the runner infers:
  - Transfermarkt season id from the start year, for example `2025/26 -> 2025`
  - Sofa season label from the compact split-year form, for example `2025/26 -> 25/26`
- Use `--force` to rebuild cached TM / Sofa / merged artifacts.
- Use `--skip-transfermarkt` or `--skip-sofascore` when one side is already cached locally.

### 7.15 Promote A Candidate Model

Use this when you already have candidate prediction artifacts and want one repeatable compare/report/promote step instead of ad hoc shell commands:

```powershell
$env:PYTHONPATH = "src"

python -m scouting_ml.scripts.run_market_value_candidate_promotion `
  --champion-metrics "data/model/champion_predictions_2024-25.metrics.json" `
  --candidate-predictions "data/model/reports/low_value_strategy_focused/cheap_aggressive_2024-25.csv" `
  --candidate-val-predictions "data/model/reports/low_value_strategy_focused/cheap_aggressive_2024-25_val.csv" `
  --candidate-metrics "data/model/reports/low_value_strategy_focused/cheap_aggressive_2024-25.metrics.json" `
  --candidate-holdout-glob "data/model/reports/low_value_contract_holdout/cheap_aggressive/*.holdout_*.metrics.json" `
  --reference-holdout-glob "data/model/reports/holdout_compare/full_*/*.holdout_*.metrics.json" `
  --candidate-label "cheap_aggressive" `
  --champion-label "champion" `
  --promote-on-pass
```

Notes:

- The script always writes a candidate-vs-champion comparison report first.
- With `--promote-on-pass`, it refreshes the benchmark report and rewrites the lock bundle only if the configured gates pass.
- Optional future-benchmark gates are available via:
  - `--candidate-future-benchmark-json`
  - `--champion-future-benchmark-json`
  - `--require-future-benchmark`
  - `--require-future-precision-vs-champion`
- Promotion gates default to:
  - candidate test `WMAPE <= champion`
  - candidate under-`€20m` test `WMAPE <= champion`
  - candidate weighted holdout `WMAPE <= reference holdout`

For faster promotion-oriented training runs, disable SHAP PNG generation:

```powershell
python -m scouting_ml.models.train_market_value_full `
  --dataset "data/model/tm_context_candidate_clean.parquet" `
  --val-season "2023/24" `
  --test-season "2024/25" `
  --output "data/model/candidates/cheap_aggressive.csv" `
  --trials 60 `
  --under-5m-weight 2.5 `
  --mid-5m-20m-weight 1.75 `
  --over-20m-weight 0.6 `
  --optimize-metric "lowmid_wmape" `
  --no-save-shap-artifacts
```

Transfermarkt context additions now included in dataset build:

- contract tenure (`club_tenure_years`, `recent_arrival_flag`)
- agent / loan context flags
- injury profile mix (`injury_soft_tissue_share`, `injury_structural_share`, `injury_surgery_flag`)
- richer transfer stability signals (`transfer_recent_paid_share_3y`, `transfer_recent_loan_share_3y`, `transfer_last_move_paid_flag`)

---

## 8) Run Backend API

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

Or:

```bash
make api
```

Optional Team Edition env:

```powershell
$env:SCOUTING_DATABASE_URL = "postgresql+psycopg://user:pass@host:5432/scoutml"
$env:SCOUTING_TEAM_MODE = "1"
$env:SCOUTING_SESSION_SECRET = "replace-me"
```

Health / docs:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/market-value/health`
- `http://127.0.0.1:8000/market-value/benchmarks`
- `http://127.0.0.1:8000/app/index.html`

Startup behavior:

- `/health` and `/market-value/health` always report readiness
- market-value routes return structured `503` JSON when strict startup validation fails or required artifacts are not ready

---

## 9) Run Frontend

Canonical backend-connected UI:

- `src/scouting_ml/website/static/index.html`

Recommended path:

- run the API
- open the backend-mounted frontend directly:
  - `http://127.0.0.1:8000/app/index.html`

Optional separate static serving path:

Terminal 1: run API.
Terminal 2 (repo root):

PowerShell:

```powershell
python -m http.server 8080
```

Bash:

```bash
python3 -m http.server 8080
```

Or:

```bash
make static-ui
```

Open:

- `http://localhost:8080/src/scouting_ml/website/static/index.html`

Set API base in UI to:

- `http://127.0.0.1:8000`

If you inject `window.SCOUTING_API_BASE` in hosted/static deployments, the UI will use that automatically.

Views:

- `Overview`
- `Recruitment Board`
- `Target Funnel`
- team/workspace controls appear in the existing workbench and detail rail when team mode is active

Other frontend paths:

- `frontend/` is a separate mock-data Next.js demo, not the canonical backend UI
- `src/scouting_ml/website/legacy/` contains the archived generated frontend reference and is not an active product path

---

## 10) Main Market Value API Endpoints

- `GET /market-value/health`
- `GET /market-value/ui-bootstrap`
- `GET /market-value/metrics`
- `GET /market-value/model-manifest`
- `GET /market-value/active-artifacts`
- `GET /market-value/benchmarks`
- `GET /market-value/predictions`
- `GET /market-value/shortlist`
- `GET /market-value/scout-targets`
- `GET /market-value/system-fit/templates`
- `POST /market-value/system-fit/query`
- `GET /market-value/player/{player_id}`
- `GET /market-value/player/{player_id}/profile`
- `GET /market-value/player/{player_id}/report`
- `GET /market-value/player/{player_id}/similar`
- `GET /market-value/player/{player_id}/trajectory`
- `GET /market-value/player/{player_id}/memo.pdf`
- `GET /market-value/player/{player_id}/advanced-profile`
- `GET /market-value/player/{player_id}/history-strength`
- `GET /market-value/player-reports`
- `GET /market-value/watchlist`
- `POST /market-value/watchlist/items`
- `DELETE /market-value/watchlist/items/{watch_id}`
- `POST /market-value/decisions`
- `GET /market-value/player/{player_id}/decisions`

Experimental NLP routes:

- `/players/{player_id}/scouting-report`
- `/players/{player_id}/similar`
- `/players/{player_id}/role`

These are disabled by default. Set `SCOUTING_ENABLE_EXPERIMENTAL_NLP_ROUTES=1` to enable them.

Team Edition APIs:

- `POST /auth/bootstrap-admin`
- `POST /auth/login`
- `POST /auth/logout`
- `GET /auth/me`
- `GET /workspaces/me`
- `POST /workspaces`
- `POST /workspaces/{workspace_id}/invites`
- `POST /invites/{token}/accept`
- `GET /team/watchlist`
- `POST /team/watchlist/items`
- `PATCH /team/watchlist/items/{item_id}`
- `DELETE /team/watchlist/items/{item_id}`
- `POST /team/decisions`
- `GET /team/player/{player_id}/decisions`
- `POST /team/assignments`
- `PATCH /team/assignments/{assignment_id}`
- `GET /team/assignments`
- `GET /team/player/{player_id}/comments`
- `POST /team/player/{player_id}/comments`
- `GET /team/activity`
- `GET /team/compare-lists`
- `POST /team/compare-lists`
- `PATCH /team/compare-lists/{compare_id}`
- `DELETE /team/compare-lists/{compare_id}`
- `POST /team/compare-lists/{compare_id}/players`
- `DELETE /team/compare-lists/{compare_id}/players/{player_id}`
- `GET /team/preferences/me`
- `PUT /team/preferences/me`

---

## 11) Current Artifact Outputs

Typical outputs after successful production run:

- Predictions:
  - `data/model/champion_predictions_2024-25.csv`
  - `data/model/champion_predictions_2024-25_val.csv`
  - `data/model/champion_predictions_2024-25.metrics.json`
  - `data/model/champion_predictions_2024-25.quality.json`
  - `data/model/reports/market_value_benchmark_report.json|md`
- Backtests:
  - `data/model/backtests/rolling_backtest_summary.csv|json`
- Lock bundle:
  - `data/model/model_manifest.json`
  - `data/model/model_artifacts.env`
- Scouting workflow:
  - `data/model/scout_workflow/*`

---

## 12) Troubleshooting

### `ModuleNotFoundError` (e.g. `sklearn`, `optuna`)

- Ensure `.venv` is activated.
- Run `python -m pip install -r requirements.txt`.
- Verify with:

```powershell
python -c "import sklearn,optuna,lightgbm,xgboost,catboost,shap; print('ok')"
```

### Frontend says "failed to fetch"

- API not running on expected host/port.
- Wrong API base in UI (use `http://127.0.0.1:8000`).
- CORS env missing (`SCOUTING_API_CORS_ORIGINS`).
- Backend readiness is degraded; check `/market-value/health`.

### PowerShell parser error on multiline command

- Every continued line must end with backtick `` ` ``.
- No stray newline before next `--flag`.

### Player report endpoint returns no prediction

- `player_id` not in selected split/season artifact.
- Check with `/market-value/predictions` and the same split/season filter first.

### Too few non-Big5 candidates

- Confirm CSV organization via `organize_processed_csvs`.
- Check league-season completeness thresholds.
- Lower strict filters (`min_minutes`, confidence, gap) for exploration.

---

## 13) Recommended Next Iteration

For the next product step after the current Team Edition baseline:

- harden auth / migration / production Postgres operations
- improve compare-workspace ergonomics and manager-facing shared shortlist views
- continue performance work before a broader hosted rollout
- keep weekly KPI + onboarding + decision-feedback reporting as the tuning loop for ranking quality
