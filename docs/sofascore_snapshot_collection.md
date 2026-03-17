# SofaScore Snapshot Collection

This workflow fetches season fixtures and per-match lineups directly from SofaScore website endpoints, writes raw JSON snapshots under `data/raw/providers/`, and emits a generated provider pipeline config you can pass to the promotion workflow.

## Entry Point

```powershell
$env:PYTHONPATH = "src"

python -m scouting_ml.scripts.collect_sofascore_snapshots `
  --config-json "docs/sofascore_snapshot_collection.example.json"
```

## What It Does

1. Reads a list of competitions identified by `tournament_id` and either `season_id` or a season label the collector can resolve automatically
2. Fetches season match pages from SofaScore
3. Fetches lineups for every collected match
4. Writes raw JSON snapshots under `data/raw/providers/`
5. Emits a generated provider config with `provider="sofascore"` and `input_json` paths

## Required Config

- `players_source`
  - path to a file or directory containing your `*_with_sofa.csv` data
  - this is required because the builder resolves SofaScore IDs back to `player_id`, `transfermarkt_id`, and `club` from your existing merged data
- `fixture_context.competitions`
- `player_availability.competitions`

Each competition entry supports:

- `name`
- `league`
- `season`
- `tournament_id`
- optional `season_id`
- optional `season_lookup`
- optional `segments`
- optional `max_pages`
- optional `seasons_endpoint_template`
- optional `events_endpoint_template`
- optional `lineups_endpoint_template`

If `season_id` is omitted, the collector fetches the tournament seasons first and tries to match the configured `season` or `season_lookup`.

Default events endpoint template:

- `/unique-tournament/{tournamentId}/season/{seasonId}/events/{segment}/{page}`

Default lineups endpoint template:

- `/event/{matchId}/lineups`

Default seasons endpoint template:

- `/unique-tournament/{tournamentId}/seasons`

## Output

The generated provider config can be used directly:

```powershell
python -m scouting_ml.scripts.run_provider_promotion_pipeline `
  --provider-config-json "data/raw/providers/sofascore_provider_pipeline_2024-25.generated.json" `
  --candidate-tag "sofascore_2024_25" `
  --out-dir "data/model/provider_promotion" `
  --trials 40 `
  --with-backtest
```

## Notes

- This path builds `fixture_context` and `player_availability` from website data.
- It does not replace StatsBomb event features.
- `market_context` odds remain a separate source.
- For RapidAPI-backed runs, set `base_url` to `https://sofascore.p.rapidapi.com` and provide `RAPIDAPI_KEY` in the shell environment.
