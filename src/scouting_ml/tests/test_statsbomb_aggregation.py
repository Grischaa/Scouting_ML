from __future__ import annotations

import json

import pandas as pd

from scouting_ml.providers.statsbomb import aggregate_player_season_features


def test_statsbomb_aggregation_builds_player_season_features(tmp_path) -> None:
    root = tmp_path / "open-data"
    (root / "matches" / "1").mkdir(parents=True)
    (root / "events").mkdir(parents=True)

    matches = [
        {
            "match_id": 1001,
            "competition": {"competition_name": "Test League"},
            "season": {"season_name": "2024/25"},
            "match_date": "2024-08-01",
        }
    ]
    events = [
        {
            "type": {"name": "Starting XI"},
            "team": {"id": 1, "name": "Example United"},
            "minute": 0,
            "tactics": {
                "formation": 433,
                "lineup": [{"player": {"id": 10, "name": "Jose Maria"}}],
            },
        },
        {
            "type": {"name": "Starting XI"},
            "team": {"id": 2, "name": "Example City"},
            "minute": 0,
            "tactics": {
                "formation": 4231,
                "lineup": [{"player": {"id": 99, "name": "Opponent"}}],
            },
        },
        {
            "type": {"name": "Pass"},
            "team": {"id": 1, "name": "Example United"},
            "player": {"id": 10, "name": "Jose Maria"},
            "minute": 5,
            "second": 0,
            "location": [60, 40],
            "pass": {"end_location": [106, 40], "shot_assist": True},
        },
        {
            "type": {"name": "Carry"},
            "team": {"id": 1, "name": "Example United"},
            "player": {"id": 10, "name": "Jose Maria"},
            "minute": 15,
            "location": [50, 40],
            "carry": {"end_location": [75, 40]},
        },
        {
            "type": {"name": "Pressure"},
            "team": {"id": 1, "name": "Example United"},
            "player": {"id": 10, "name": "Jose Maria"},
            "minute": 20,
            "counterpress": True,
        },
        {
            "type": {"name": "Ball Recovery"},
            "team": {"id": 1, "name": "Example United"},
            "player": {"id": 10, "name": "Jose Maria"},
            "minute": 22,
            "location": [70, 40],
        },
        {
            "type": {"name": "Ball Receipt*"},
            "team": {"id": 1, "name": "Example United"},
            "player": {"id": 10, "name": "Jose Maria"},
            "minute": 25,
            "location": [70, 40],
        },
        {
            "type": {"name": "Duel"},
            "team": {"id": 1, "name": "Example United"},
            "player": {"id": 10, "name": "Jose Maria"},
            "minute": 30,
            "duel": {"outcome": {"name": "Won"}},
        },
        {
            "type": {"name": "Substitution"},
            "team": {"id": 1, "name": "Example United"},
            "player": {"id": 10, "name": "Jose Maria"},
            "minute": 60,
            "substitution": {"replacement": {"id": 11, "name": "Replacement"}},
        },
    ]
    (root / "matches" / "1" / "2024.json").write_text(json.dumps(matches), encoding="utf-8")
    (root / "events" / "1001.json").write_text(json.dumps(events), encoding="utf-8")

    links = pd.DataFrame(
        [
            {
                "provider": "statsbomb",
                "provider_player_id": "10",
                "provider_player_name": "Jose Maria",
                "provider_team_name": "Example United",
                "season": "2024/25",
                "player_id": "tm_10",
                "transfermarkt_id": "10",
            }
        ]
    )
    links_path = tmp_path / "player_provider_links.csv"
    links.to_csv(links_path, index=False)

    out = aggregate_player_season_features(root, player_links_path=links_path, snapshot_date="2026-03-10", retrieved_at="2026-03-10T00:00:00Z")

    row = out.loc[out["provider_player_id"] == "10"].iloc[0]
    assert row["player_id"] == "tm_10"
    assert row["completed_passes"] == 1
    assert row["progressive_passes"] == 1
    assert row["passes_into_box"] == 1
    assert row["shot_assists"] == 1
    assert row["pressures"] == 1
    assert row["counterpressures"] == 1
    assert row["high_regains"] == 1
    assert row["between_lines_receipts"] == 1
    assert row["duel_wins"] == 1
    assert row["minutes_in_433"] == 60
