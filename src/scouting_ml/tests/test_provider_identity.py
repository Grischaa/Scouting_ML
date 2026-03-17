from __future__ import annotations

import pandas as pd

from scouting_ml.providers.identity import merge_club_links, merge_player_links, normalize_club_name, normalize_person_name, normalize_season_label


def test_identity_helpers_normalize_common_values() -> None:
    assert normalize_season_label("2024-25") == "2024/25"
    assert normalize_person_name("Jos\u00e9 Mar\u00eda") == "jose maria"
    assert normalize_club_name("FC Example United") == "example united"


def test_identity_merges_player_and_club_links() -> None:
    player_rows = pd.DataFrame(
        [
            {
                "provider_player_id": "sb-10",
                "player_name": "Jose Maria",
                "team_name": "FC Example United",
                "season": "2024/25",
            }
        ]
    )
    player_links = pd.DataFrame(
        [
            {
                "provider": "statsbomb",
                "provider_player_id": "sb-10",
                "provider_player_name": "Jose Maria",
                "provider_team_name": "Example United",
                "season": "2024/25",
                "player_id": "tm_10",
                "transfermarkt_id": "10",
            }
        ]
    )
    out = merge_player_links(player_rows, player_links, provider="statsbomb", provider_id_col="provider_player_id", player_name_col="player_name", club_col="team_name")
    assert out.loc[0, "player_id"] == "tm_10"
    assert out.loc[0, "transfermarkt_id"] == "10"

    club_rows = pd.DataFrame(
        [
            {
                "provider_team_id": "",
                "team_name": "Example United",
                "league": "Premier League",
                "season": "2024/25",
            }
        ]
    )
    club_links = pd.DataFrame(
        [
            {
                "provider": "odds",
                "provider_team_name": "FC Example United",
                "club": "Example United",
                "league": "Premier League",
                "season": "2024/25",
            }
        ]
    )
    club_out = merge_club_links(club_rows, club_links, provider="odds", provider_team_id_col="provider_team_id", team_name_col="team_name")
    assert club_out.loc[0, "club"] == "Example United"
