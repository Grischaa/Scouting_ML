# Data Source Integration Roadmap

Date: 2026-03-09

This roadmap is for the current `Scout_Pred` architecture, not a generic football-data stack.

## Current State

The project already has three clear data lanes:

1. Core league refresh
   - Transfermarkt ingestion via `src/scouting_ml/tm/`
   - Sofascore ingestion via `src/scouting_ml/sofa/`
   - League orchestration via `src/scouting_ml/refresh.py`
   - Per-league merged season files via `src/scouting_ml/pipeline/merge.py`

2. External feature enrichment
   - Optional external CSV merges in `src/scouting_ml/models/build_dataset.py`
   - Existing external tables:
     - `player_contracts.csv`
     - `player_injuries.csv`
     - `player_transfers.csv`
     - `national_team_caps.csv`
     - `club_context.csv`
     - `league_context.csv`

3. Product and model consumption
   - Market-value training in `src/scouting_ml/models/train_market_value_full.py`
   - Weekly/scouting workflows in `src/scouting_ml/scripts/run_weekly_scout_ops.py`
   - API/profile surfaces in `src/scouting_ml/services/market_value_service.py`

That means new data sources should usually enter through `data/external/*.csv` first. They should not be bolted directly into the core `refresh` loop unless they are replacing or extending league-season ingestion at the same abstraction level as Sofascore.

## Recommended Order

1. StatsBomb Open Data
2. One operational API vendor: Sportmonks or API-Football
3. The Odds API

Rationale:

- StatsBomb gives the highest feature-quality upside for formation fit, player type, and similarity.
- A vendor API is best for operational context such as fixtures, injuries, lineups, and availability.
- Odds are useful, but they are secondary context features rather than core player-quality data.

## Phase 1: StatsBomb Open Data

Official source checked on 2026-03-09:
- https://github.com/statsbomb/open-data

### Goal

Add event-derived, tactic-aware player-season features that are not currently available in the Transfermarkt + Sofascore aggregate stack.

### Why It Fits This Repo

Your current model and UI already care about:

- player role and archetype
- formation fit
- similar players
- grouped technical strengths and weaknesses

StatsBomb helps most in those exact areas because it can be aggregated into stable player-season features.

### Features To Build

Create per-player, per-season outputs such as:

- `sb_progressive_passes_per90`
- `sb_progressive_carries_per90`
- `sb_passes_into_box_per90`
- `sb_shot_assists_per90`
- `sb_pressures_per90`
- `sb_counterpressures_per90`
- `sb_def_actions_high_regains_per90`
- `sb_touch_central_final_third_share`
- `sb_receipts_between_lines_per90`
- `sb_duel_win_rate`
- `sb_aerial_win_rate`
- `sb_role_exposure_in_possession`
- `sb_role_exposure_out_of_possession`
- `sb_minutes_in_433`
- `sb_minutes_in_4231`
- `sb_minutes_in_3421`

Do not start with raw event delivery to the UI. Start with season-level aggregates for modeling and scouting summaries.

### Concrete Repo Changes

Add a new provider area:

- `src/scouting_ml/providers/statsbomb/__init__.py`
- `src/scouting_ml/providers/statsbomb/open_data.py`
- `src/scouting_ml/providers/statsbomb/aggregate.py`

Add a build script:

- `src/scouting_ml/scripts/build_statsbomb_player_events.py`

Output files:

- `data/external/statsbomb_player_season_features.csv`
- `data/external/statsbomb_match_tactical_context.csv`

Merge into dataset builder by extending `specs` in `src/scouting_ml/models/build_dataset.py` with:

- `statsbomb_player_season_features.csv` with prefix `sb_`

### Join Strategy

Current joins in `build_dataset.py` rely on:

- `player_id + season`
- `transfermarkt_id + season`
- `name + dob + season`

That is not strong enough for StatsBomb on its own. Add a mapping table first:

- `data/external/player_provider_links.csv`

Columns:

- `player_id`
- `transfermarkt_id`
- `statsbomb_player_id`
- `statsbomb_player_name`
- `dob`
- `club`
- `season`
- `match_confidence`
- `match_method`

Short term:

- build a heuristic linker using `name`, `dob`, `club`, `season`
- persist results
- review low-confidence links manually

Long term:

- promote this into a shared identity layer for all providers

### Main Risks

- Open-data coverage is incomplete and not uniform across leagues/seasons
- Player identity resolution will be the first failure point
- Formation labels need normalization before they become stable features

### Success Criteria

- New `sb_` feature family appears in the clean training dataset
- Similar-player results become more tactically coherent
- Formation-fit explanations improve without frontend logic changes

## Phase 2: Operational API Vendor

Official sources checked on 2026-03-09:
- Sportmonks docs: https://docs.sportmonks.com/football/
- API-Football product/pricing: https://www.api-football.com/

Pick one vendor. Do not integrate both at the start.

### Recommendation

If the main goal is model quality and richer scouting workflows, choose Sportmonks first.

If the main goal is lower-cost operational coverage and faster setup, choose API-Football first.

### Goal

Add current operational context that scraped season tables do not cover well:

- upcoming fixtures
- lineup expectations
- injuries and availability
- squad churn
- opponent difficulty
- short-horizon scouting opportunity signals

### What Not To Do

Do not pipe live API data directly into the historical market-value training table without snapshotting it by date. That will create leakage and reproducibility problems.

### Concrete Repo Changes

Add a provider package:

- `src/scouting_ml/providers/football_api/__init__.py`
- `src/scouting_ml/providers/football_api/client.py`
- `src/scouting_ml/providers/football_api/normalize.py`

Add scripts:

- `src/scouting_ml/scripts/build_fixture_context.py`
- `src/scouting_ml/scripts/build_player_availability.py`

Output files:

- `data/external/fixture_context.csv`
- `data/external/player_availability.csv`

Extend `src/scouting_ml/models/build_dataset.py` with:

- `fixture_context.csv` prefix `fixture_`
- `player_availability.csv` prefix `avail_`

Extend scouting ops:

- `src/scouting_ml/scripts/run_weekly_scout_ops.py`
- `src/scouting_ml/scripts/make_scout_shortlist.py`

Possible features:

- `fixture_days_rest`
- `fixture_opponent_strength`
- `fixture_next4_difficulty`
- `fixture_home_share_next4`
- `avail_injury_flag`
- `avail_expected_start_probability`
- `avail_recent_squad_inclusion_rate`
- `avail_transfer_window_instability`

### Join Strategy

This integration needs both player and club mapping tables:

- `data/external/player_provider_links.csv`
- `data/external/club_provider_links.csv`

Columns for clubs:

- `club`
- `league`
- `season`
- `vendor_team_id`
- `vendor_team_name`
- `match_confidence`
- `match_method`

### Where It Should Surface

Not just in modeling. This provider is most valuable in:

- scout shortlist ranking
- player profile risk section
- watchlist alerts
- weekly workflow outputs

### Main Risks

- Historical backfill can be expensive or incomplete depending on plan
- Vendor schemas change more often than your current scrapers
- Live data needs timestamped snapshots to stay reproducible

### Success Criteria

- Weekly scout output can rank players by opportunity, not just quality
- Player profile can explain short-term availability/risk with structured data
- External operational features remain optional and do not break offline training

## Phase 3: The Odds API

Official source checked on 2026-03-09:
- https://the-odds-api.com/

### Goal

Add market-implied team strength and match difficulty context.

This is useful for:

- contextualizing player output
- identifying overperformance in weak teams
- assessing expected environment changes for transfers

### Concrete Repo Changes

Add provider package:

- `src/scouting_ml/providers/odds/__init__.py`
- `src/scouting_ml/providers/odds/client.py`
- `src/scouting_ml/providers/odds/normalize.py`

Add build script:

- `src/scouting_ml/scripts/build_market_context.py`

Output file:

- `data/external/market_context.csv`

Extend `src/scouting_ml/models/build_dataset.py` with:

- `market_context.csv` prefix `odds_`

Candidate features:

- `odds_implied_team_strength`
- `odds_implied_opponent_strength`
- `odds_match_upset_probability`
- `odds_relegation_pressure_flag`
- `odds_title_pressure_flag`
- `odds_expected_points_next5`

### Join Strategy

This is club- and fixture-centric, not player-centric. Merge through:

- `club`
- `league`
- `season`

For future match-level products, add:

- `vendor_match_id`
- `match_date`
- `home_club`
- `away_club`

### Main Risks

- Odds are time-sensitive and must be snapshotted by retrieval time
- Coverage differs by competition and bookmaker set
- These features are more useful for ops and ranking than for core valuation

### Success Criteria

- Scout workflows can separate difficult environments from easy ones
- Club-context features become less naive than pure table position or past results

## Cross-Cutting Schema Work

This is the most important enabling work. Without it, every new source becomes a custom join hack.

### Add A Canonical Identity Layer

Create stable mapping files:

- `data/external/player_provider_links.csv`
- `data/external/club_provider_links.csv`
- `data/external/match_provider_links.csv`

And a small shared module:

- `src/scouting_ml/providers/identity.py`

Responsibilities:

- normalize names
- normalize seasons
- normalize clubs and competitions
- store confidence-scored cross-provider matches

### Add Raw / Curated Separation

Create a consistent layout:

- `data/raw/{provider}/...`
- `data/staging/{provider}/...`
- `data/external/...`

Use:

- raw: original API or JSON payloads
- staging: normalized provider tables
- external: modeling-ready tables

### Add Snapshot Metadata

Every generated external file should carry:

- `source_provider`
- `source_version`
- `retrieved_at`
- `snapshot_date`
- `coverage_note`

This is mandatory for any live API integration.

## Implementation Order In This Repo

### Quick Wins

1. Add the shared provider identity layer
2. Implement StatsBomb aggregation to `statsbomb_player_season_features.csv`
3. Merge `sb_` features in `build_dataset.py`
4. Expose a few `sb_` values in player profile payloads

### Second Wave

1. Pick Sportmonks or API-Football
2. Build `player_availability.csv`
3. Add availability/risk sections to weekly scout ops and player profiles

### Third Wave

1. Add `market_context.csv` from The Odds API
2. Feed it into club and fixture context
3. Use it in opportunity ranking, not as a first-order valuation target

## Tests To Add

Add provider-focused tests before expanding coverage:

- `src/scouting_ml/tests/test_provider_identity.py`
- `src/scouting_ml/tests/test_statsbomb_aggregation.py`
- `src/scouting_ml/tests/test_external_feature_merges.py`
- `src/scouting_ml/tests/test_player_profile_provider_payloads.py`

Test for:

- stable join keys
- no value-leakage columns entering model features
- graceful missing-coverage behavior
- deterministic outputs from cached raw payloads

## Recommended First Build

Build StatsBomb first.

It is the best match for the product direction already visible in this repo:

- tactical fit
- player profile depth
- similar players
- richer scout explanations

It also fits your current architecture cleanly because it can be introduced as an offline external feature job without destabilizing the existing Transfermarkt + Sofascore refresh flow.
