from __future__ import annotations

import json

import pandas as pd

from scouting_ml.models.build_dataset import add_model_features
from scouting_ml.providers.football_api.normalize import build_fixture_context, build_player_availability, normalize_fixtures, normalize_player_availability
from scouting_ml.providers.odds.normalize import build_market_context, normalize_odds_events
from scouting_ml.scripts import build_provider_external_data as provider_build_module


def test_provider_context_builders_and_dataset_merge(tmp_path) -> None:
    club_links = pd.DataFrame(
        [
            {
                "provider": "sportmonks",
                "provider_team_id": "1",
                "provider_team_name": "Example United",
                "club": "Example United",
                "league": "Premier League",
                "season": "2024/25",
            },
            {
                "provider": "sportmonks",
                "provider_team_id": "2",
                "provider_team_name": "Example City",
                "club": "Example City",
                "league": "Premier League",
                "season": "2024/25",
            },
            {
                "provider": "odds",
                "provider_team_name": "Example United",
                "club": "Example United",
                "league": "Premier League",
                "season": "2024/25",
            },
            {
                "provider": "odds",
                "provider_team_name": "Example City",
                "club": "Example City",
                "league": "Premier League",
                "season": "2024/25",
            },
        ]
    )
    club_links_path = tmp_path / "club_provider_links.csv"
    club_links.to_csv(club_links_path, index=False)

    player_links = pd.DataFrame(
        [
            {
                "provider": "sportmonks",
                "provider_player_id": "10",
                "provider_player_name": "Jose Maria",
                "provider_team_name": "Example United",
                "season": "2024/25",
                "player_id": "tm_10",
                "transfermarkt_id": "10",
            }
        ]
    )
    player_links_path = tmp_path / "player_provider_links.csv"
    player_links.to_csv(player_links_path, index=False)

    sportmonks_fixtures = {
        "data": [
            {
                "id": 101,
                "season": {"name": "2024/25"},
                "league": {"name": "Premier League"},
                "starting_at": "2024-08-10T15:00:00Z",
                "participants": [
                    {"id": 1, "name": "Example United", "meta": {"location": "home"}},
                    {"id": 2, "name": "Example City", "meta": {"location": "away"}},
                ],
                "scores": [
                    {"participant_id": 1, "description": "current", "score": {"goals": 2}},
                    {"participant_id": 2, "description": "current", "score": {"goals": 1}},
                ],
            }
        ]
    }
    fixtures = normalize_fixtures(sportmonks_fixtures, provider="sportmonks")
    fixture_context = build_fixture_context(fixtures, provider="sportmonks", club_links_path=str(club_links_path))
    assert fixture_context.loc[fixture_context["club"] == "Example United", "fixture_matches"].iloc[0] == 1

    sportmonks_availability = {
        "data": [
            {
                "player": {"id": 10, "name": "Jose Maria"},
                "team": {"id": 1, "name": "Example United"},
                "league": {"name": "Premier League"},
                "season": {"name": "2024/25"},
                "fixture": {"id": 101, "starting_at": "2024-08-10T15:00:00Z"},
                "minutes": 90,
                "starting": True,
                "expected_starting": True,
            }
        ]
    }
    availability = normalize_player_availability(sportmonks_availability, provider="sportmonks")
    availability_context = build_player_availability(
        availability,
        provider="sportmonks",
        player_links_path=str(player_links_path),
        club_links_path=str(club_links_path),
    )
    assert availability_context.loc[0, "player_id"] == "tm_10"
    assert availability_context.loc[0, "avail_start_share"] == 1.0

    odds_payload = [
        {
            "home_team": "Example United",
            "away_team": "Example City",
            "commence_time": "2024-08-10T15:00:00Z",
            "bookmakers": [
                {
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Example United", "price": 2.0},
                                {"name": "Example City", "price": 4.0},
                                {"name": "Draw", "price": 3.0},
                            ],
                        },
                        {
                            "key": "totals",
                            "outcomes": [{"name": "Over", "price": 1.9, "point": 2.5}],
                        },
                    ]
                }
            ],
        }
    ]
    odds_events = normalize_odds_events(odds_payload, season="2024/25", league="Premier League")
    market_context = build_market_context(odds_events, club_links_path=str(club_links_path))
    assert market_context.loc[market_context["club"] == "Example United", "odds_matches"].iloc[0] == 1

    external_dir = tmp_path / "external"
    external_dir.mkdir()
    pd.DataFrame([{"player_id": "tm_10", "season": "2024/25", "completed_passes_per90": 1.5}]).to_csv(
        external_dir / "statsbomb_player_season_features.csv",
        index=False,
    )
    availability_context.to_csv(external_dir / "player_availability.csv", index=False)
    fixture_context.to_csv(external_dir / "fixture_context.csv", index=False)
    market_context.to_csv(external_dir / "market_context.csv", index=False)

    base = pd.DataFrame(
        [
            {
                "player_id": "tm_10",
                "transfermarkt_id": "10",
                "name": "Jose Maria",
                "season": "2024/25",
                "league": "Premier League",
                "club": "Example United",
                "position_group": "FW",
                "age": 21,
                "sofa_minutesPlayed": 1000,
                "sofa_goals": 8,
                "sofa_assists": 3,
                "sofa_expectedGoals": 7,
                "sofa_totalShots": 20,
            }
        ]
    )
    enriched = add_model_features(base, external_dir=external_dir)
    row = enriched.iloc[0]
    assert row["sb_completed_passes_per90"] == 1.5
    assert row["avail_start_share"] == 1.0
    assert row["fixture_matches"] == 1
    assert row["odds_matches"] == 1


def test_build_provider_external_data_supports_api_urls(monkeypatch, tmp_path) -> None:
    external_dir = tmp_path / "external"
    external_dir.mkdir()

    pd.DataFrame(
        [
            {
                "provider": "sportmonks",
                "provider_team_id": "1",
                "provider_team_name": "Example United",
                "club": "Example United",
                "league": "Premier League",
                "season": "2024/25",
            },
            {
                "provider": "sportmonks",
                "provider_team_id": "2",
                "provider_team_name": "Example City",
                "club": "Example City",
                "league": "Premier League",
                "season": "2024/25",
            },
            {
                "provider": "odds",
                "provider_team_name": "Example United",
                "club": "Example United",
                "league": "Premier League",
                "season": "2024/25",
            },
            {
                "provider": "odds",
                "provider_team_name": "Example City",
                "club": "Example City",
                "league": "Premier League",
                "season": "2024/25",
            },
        ]
    ).to_csv(external_dir / "club_provider_links.csv", index=False)
    pd.DataFrame(
        [
            {
                "provider": "sportmonks",
                "provider_player_id": "10",
                "provider_player_name": "Jose Maria",
                "provider_team_name": "Example United",
                "season": "2024/25",
                "player_id": "tm_10",
                "transfermarkt_id": "10",
            }
        ]
    ).to_csv(external_dir / "player_provider_links.csv", index=False)

    sportmonks_fixtures = {
        "data": [
            {
                "id": 101,
                "season": {"name": "2024/25"},
                "league": {"name": "Premier League"},
                "starting_at": "2024-08-10T15:00:00Z",
                "participants": [
                    {"id": 1, "name": "Example United", "meta": {"location": "home"}},
                    {"id": 2, "name": "Example City", "meta": {"location": "away"}},
                ],
                "scores": [
                    {"participant_id": 1, "description": "current", "score": {"goals": 2}},
                    {"participant_id": 2, "description": "current", "score": {"goals": 1}},
                ],
            }
        ]
    }
    sportmonks_availability = {
        "data": [
            {
                "player": {"id": 10, "name": "Jose Maria"},
                "team": {"id": 1, "name": "Example United"},
                "league": {"name": "Premier League"},
                "season": {"name": "2024/25"},
                "fixture": {"id": 101, "starting_at": "2024-08-10T15:00:00Z"},
                "minutes": 90,
                "starting": True,
                "expected_starting": True,
            }
        ]
    }
    odds_payload = [
        {
            "home_team": "Example United",
            "away_team": "Example City",
            "commence_time": "2024-08-10T15:00:00Z",
            "bookmakers": [
                {
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Example United", "price": 2.0},
                                {"name": "Example City", "price": 4.0},
                                {"name": "Draw", "price": 3.0},
                            ],
                        }
                    ]
                }
            ],
        }
    ]

    monkeypatch.setattr(provider_build_module, "_fetch_fixture_payload", lambda provider, api_url: sportmonks_fixtures)
    monkeypatch.setattr(provider_build_module, "_fetch_availability_payload", lambda provider, api_url: sportmonks_availability)
    monkeypatch.setattr(provider_build_module, "_fetch_odds_payload", lambda api_url: odds_payload)

    config_path = tmp_path / "provider_config.json"
    config_path.write_text(
        json.dumps(
            {
                "snapshot_date": "2026-03-11",
                "player_links": str(external_dir / "player_provider_links.csv"),
                "club_links": str(external_dir / "club_provider_links.csv"),
                "fixture_context": {"provider": "sportmonks", "api_url": ["fixtures/live"]},
                "player_availability": {"provider": "sportmonks", "api_url": ["availability/live"]},
                "market_context": {
                    "league": "Premier League",
                    "season": "2024/25",
                    "api_url": ["odds/live"],
                },
            }
        ),
        encoding="utf-8",
    )

    payload = provider_build_module.build_provider_external_data(
        config_path=config_path,
        external_dir=external_dir,
    )

    assert payload["outputs"]["fixture_context"]["rows"] == 2
    assert payload["outputs"]["player_availability"]["rows"] == 1
    assert payload["outputs"]["market_context"]["rows"] == 2
