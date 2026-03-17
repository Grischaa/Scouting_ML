# Provider Real-Data Run - 2026-03-10

## Scope

This run moved the provider pipeline from scaffolding to a real-data pass using a local StatsBomb Open Data slice plus generated provider link tables.

Artifacts produced in this run:

- `data/raw/statsbomb/bundesliga_2023_24`
- `data/external/statsbomb_player_season_features_raw.csv`
- `data/external/statsbomb_player_season_features.csv`
- `data/external/player_provider_links.csv`
- `data/external/club_provider_links.csv`
- `data/external/provider_link_audit.json`
- `data/external/provider_link_audit.csv`
- `data/model/champion_players_statsbomb.parquet`
- `data/model/champion_players_statsbomb_clean.parquet`
- `data/model/champion_predictions_statsbomb_2024-25.csv`
- `data/model/champion_predictions_statsbomb_2024-25.metrics.json`

## Coverage Summary

Provider link audit:

- `player_links:statsbomb.rows = 373`
- `player_links:statsbomb.matched_players = 171`
- `player_links:statsbomb.coverage = 0.0117` over all processed players
- `statsbomb_player_season_features.rows = 380`
- `statsbomb_player_season_features.matched_rows = 171`
- `statsbomb_player_season_features.coverage = 0.45` over the external table itself
- `player_availability`, `fixture_context`, and `market_context` remain empty because no local Sportmonks / API-Football / Odds snapshots or credentials were available in this environment

Model-dataset coverage:

- StatsBomb columns added to the model dataset: `59`
- Validation season `2023/24`: `294 / 7564` rows had at least one `sb_` value (`3.89%`)
- Test season `2024/25`: `0 / 3881` rows had any `sb_` value (`0.00%`)

This is the decisive limitation of the current real-data pass: the model was trained with a small amount of historical StatsBomb context in validation, but the held-out `2024/25` test season had no StatsBomb coverage at all.

## Model Comparison

Baseline champion:

- Dataset: `data/model/champion_players_clean.parquet`
- Metrics: `data/model/champion_predictions_2024-25.metrics.json`

StatsBomb-enriched run:

- Dataset: `data/model/champion_players_statsbomb_clean.parquet`
- Metrics: `data/model/champion_predictions_statsbomb_2024-25.metrics.json`

Overall deltas (`new - baseline`):

Validation:

- `R2: -0.0125`
- `WMAPE: +0.0060`
- `MAE: +EUR 47,942`
- `MAPE: +0.0023`

Test:

- `R2: -0.0255`
- `WMAPE: +0.0156`
- `MAE: +EUR 122,637`
- `MAPE: -0.0034`

Test-segment deltas:

- `under_5m`: `R2 +0.0688`, `WMAPE -0.0088`, `MAE -EUR 12,579`
- `5m_to_20m`: `R2 +0.0581`, `WMAPE -0.0031`, `MAE -EUR 29,495`
- `over_20m`: `R2 -0.1212`, `WMAPE +0.0290`, `MAE +EUR 1,045,739`

Interpretation:

- The partial StatsBomb features helped the lower-value scouting bands slightly.
- The overall model still regressed because the high-value segment worsened materially.
- Since the `2024/25` test split had zero StatsBomb coverage, this run does not justify promoting StatsBomb features into the champion path yet.

## Pipeline Hardening Applied

The real run exposed Windows console and preprocessing issues, and the code was tightened accordingly:

- Replaced Unicode arrows / symbols in CLI output with ASCII in:
  - `src/scouting_ml/models/build_dataset.py`
  - `src/scouting_ml/models/clean_dataset.py`
  - `src/scouting_ml/models/train_market_value_full.py`
- Added train-split empty-feature dropping in `src/scouting_ml/models/train_market_value_full.py`
- Made imputers keep empty features explicitly in `src/scouting_ml/models/train_market_value_full.py`
- Increased the routing band classifier `max_iter` from `1500` to `3000`

## Operational Conclusion

What worked:

- Real StatsBomb Open Data was downloaded and aggregated successfully.
- Provider links were generated and auditable.
- The enriched dataset and retrain artifacts were produced end to end.

What is still missing:

- No current-season open-data coverage for the `2024/25` test split in this run
- No live or snapshot inputs for availability, fixture context, or odds context
- No basis yet to replace the existing champion model

## Recommended Next Move

Do not ship the StatsBomb-enriched model as champion.

Instead:

1. Add provider data that covers the actual scoring season, especially `2024/25`
2. Populate one operational provider path fully:
   - Sportmonks or API-Football for availability and fixtures
   - Odds snapshots for market context
3. Re-run the same comparison only after the test split has non-zero provider coverage
