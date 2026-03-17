# Provider Promotion Pipeline

This workflow is for evaluating a provider-enriched candidate model against the current champion before replacing active artifacts.

It runs five steps in one command:

1. Stage a candidate-only external directory from the baseline external tables
2. Build provider tables from available snapshots and/or live API URLs
3. Bootstrap provider links and rebuild the provider tables with resolved IDs
4. Retrain and optionally backtest a provider-enriched candidate model
5. Compare the candidate against the champion and promote only if the gates pass

## Entry Point

```powershell
$env:PYTHONPATH = "src"

python -m scouting_ml.scripts.run_provider_promotion_pipeline `
  --provider-config-json "docs/provider_pipeline_config.example.json" `
  --candidate-tag "statsbomb_candidate" `
  --out-dir "data/model/provider_promotion" `
  --trials 40 `
  --with-backtest `
  --promote-on-pass
```

If you need to fetch live provider snapshots first, use:

```powershell
python -m scouting_ml.scripts.collect_provider_snapshots `
  --config-json "docs/provider_snapshot_collection.example.json"
```

If you want to build website-derived fixture and availability snapshots from SofaScore instead, use:

```powershell
python -m scouting_ml.scripts.collect_sofascore_snapshots `
  --config-json "docs/sofascore_snapshot_collection.example.json"
```

## Important Defaults

- The candidate run stages its own external tables under `data/model/provider_promotion/{candidate_tag}/external`
- Existing enrichment tables (`contracts`, `injuries`, `transfers`, `national`, `club`, `league`, `uefa`) are copied into that staged directory
- Expensive legacy enrichments are skipped by default:
  - `--skip-injuries`
  - `--skip-contracts`
  - `--skip-transfers`
  - `--skip-national`
  - `--skip-context`
- Missing provider snapshot files are pruned from the effective config instead of failing the whole run

## Live Fetch Support

The provider config can now use either:

- `input_json`
- `api_url`

for:

- `fixture_context`
- `player_availability`
- `market_context`

For `provider="sofascore"`, use `input_json` snapshots plus `players_source`. The builder resolves SofaScore IDs back to internal player and club keys from your existing `*_with_sofa.csv` data.

If `api_url` is used, the existing provider clients read credentials from environment variables:

- `SPORTMONKS_API_TOKEN`
- `APIFOOTBALL_API_KEY`
- `ODDS_API_KEY`

## Promotion Gates

The script evaluates the candidate against the baseline using:

- minimum test-season provider coverage
- maximum allowed test WMAPE delta
- minimum allowed test R2 delta
- maximum allowed low/mid-band weighted WMAPE delta
- optional backtest non-degradation checks

Current defaults are conservative:

- `--min-test-provider-coverage 0.05`
- `--max-test-wmape-delta 0.0`
- `--min-test-r2-delta 0.0`
- `--max-test-lowmid-wmape-delta 0.0`
- `--max-backtest-test-wmape-delta 0.0`
- `--min-backtest-test-r2-delta 0.0`

That means the candidate must both add meaningful provider coverage and avoid degrading the current champion.

## Outputs

The run writes a candidate bundle under:

- `data/model/provider_promotion/{candidate_tag}/`

Key artifacts:

- `provider_config.effective.json`
- `provider_build_initial.json`
- `provider_link_bootstrap.json`
- `provider_build_linked.json`
- `provider_link_audit.json`
- `candidate_model_manifest.json`
- `candidate_model_artifacts.env`
- `provider_promotion_summary.json`

If `--promote-on-pass` is set and the gates pass, the active manifest/env outputs are also rewritten.
