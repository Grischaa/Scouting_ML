# Provider Snapshot Collection

This step fetches raw provider payloads into `data/raw/providers/` and emits a generated provider pipeline config that points at those snapshots.

## Entry Point

```powershell
$env:PYTHONPATH = "src"
$env:SPORTMONKS_API_TOKEN = "..."
$env:ODDS_API_KEY = "..."

python -m scouting_ml.scripts.collect_provider_snapshots `
  --config-json "docs/provider_snapshot_collection.example.json"
```

## What It Does

1. Reads a config describing one or more live provider requests
2. Fetches each request with the existing provider clients
3. Writes raw JSON payloads under `data/raw/providers/`
4. Emits a generated provider pipeline config with `input_json` paths
5. Writes a summary JSON describing fetched, skipped, or planned requests

## Config Notes

- `fixture_context.provider` and `player_availability.provider` must be `sportmonks` or `api-football`
- `market_context` always uses the odds client
- Each section can define multiple `requests`
- Each request can use:
  - `api_url`
  - or `endpoint` plus optional `params`
- `output_json` is optional; when omitted, the script auto-generates a file path

## Credentials

The script reads provider secrets from environment variables:

- `SPORTMONKS_API_TOKEN`
- `APIFOOTBALL_API_KEY`
- `ODDS_API_KEY`

If a secret is missing:

- default behavior: fail fast
- optional behavior: use `--allow-missing-secrets` to skip that section and continue

## Dry Run

Use this to validate the config without making live requests:

```powershell
python -m scouting_ml.scripts.collect_provider_snapshots `
  --config-json "docs/provider_snapshot_collection.example.json" `
  --dry-run
```

## Next Command

After the snapshots are fetched, run the provider promotion pipeline against the generated config:

```powershell
python -m scouting_ml.scripts.run_provider_promotion_pipeline `
  --provider-config-json "data/raw/providers/provider_pipeline_2024-25.generated.json" `
  --candidate-tag "provider_live_2024_25" `
  --out-dir "data/model/provider_promotion" `
  --trials 40 `
  --with-backtest
```
