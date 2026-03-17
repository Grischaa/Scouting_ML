from __future__ import annotations

from pathlib import Path

import pandas as pd

from scouting_ml.scripts.build_provider_link_audit import build_provider_link_audit


def test_provider_link_audit_reports_link_and_external_coverage(tmp_path: Path) -> None:
    players = pd.DataFrame(
        [
            {"player_id": "tm_10", "transfermarkt_id": "10", "name": "Jose Maria", "club": "Example United", "league": "Premier League", "season": "2024/25"},
            {"player_id": "tm_11", "transfermarkt_id": "11", "name": "Another Player", "club": "Example City", "league": "Premier League", "season": "2024/25"},
        ]
    )
    players_path = tmp_path / "players.csv"
    players.to_csv(players_path, index=False)

    ext_dir = tmp_path / "external"
    ext_dir.mkdir()
    pd.DataFrame(
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
    ).to_csv(ext_dir / "player_provider_links.csv", index=False)
    pd.DataFrame(
        [
            {
                "provider": "odds",
                "provider_team_name": "Example United",
                "club": "Example United",
                "league": "Premier League",
                "season": "2024/25",
            }
        ]
    ).to_csv(ext_dir / "club_provider_links.csv", index=False)
    pd.DataFrame([{"player_id": "tm_10", "season": "2024/25", "source_provider": "statsbomb"}]).to_csv(
        ext_dir / "statsbomb_player_season_features.csv",
        index=False,
    )

    out_json = tmp_path / "audit.json"
    out_csv = tmp_path / "audit.csv"
    payload = build_provider_link_audit(
        players_source=str(players_path),
        external_dir=str(ext_dir),
        player_links=str(ext_dir / "player_provider_links.csv"),
        club_links=str(ext_dir / "club_provider_links.csv"),
        out_json=str(out_json),
        out_csv=str(out_csv),
    )

    assert payload["link_summary"]["player_links:statsbomb"]["matched_players"] == 1
    assert payload["external_summary"]["statsbomb_player_season_features"]["matched_rows"] == 1
    assert payload["season_coverage"]["links"]["player_links:statsbomb"][0]["season"] == "2024/25"
    assert payload["season_coverage"]["links"]["player_links:statsbomb"][0]["matched_rows"] == 1
    assert payload["season_coverage"]["external"]["statsbomb_player_season_features"][0]["coverage_0_to_1"] == 0.5
    assert out_json.exists()
    assert out_csv.exists()
