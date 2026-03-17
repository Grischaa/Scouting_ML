from __future__ import annotations

from pathlib import Path

import pandas as pd

from scouting_ml.scripts.bootstrap_provider_links import bootstrap_provider_links


def test_bootstrap_provider_links_emits_review_queue_for_fallback_matches(tmp_path: Path) -> None:
    players = pd.DataFrame(
        [
            {
                "player_id": "tm_10",
                "transfermarkt_id": "10",
                "name": "Jose Maria",
                "club": "Example United",
                "league": "Premier League",
                "dob": "2002-01-01",
                "season": "2024/25",
            }
        ]
    )
    players_path = tmp_path / "players.csv"
    players.to_csv(players_path, index=False)

    external_dir = tmp_path / "external"
    external_dir.mkdir()
    pd.DataFrame(
        [
            {
                "provider_player_id": "sb_10",
                "player_name": "Jose Maria",
                "team_name": "",
                "league": "",
                "season": "2024/25",
            }
        ]
    ).to_csv(external_dir / "statsbomb_player_season_features.csv", index=False)

    payload = bootstrap_provider_links(
        players_source=str(players_path),
        external_dir=str(external_dir),
        player_links_out=str(external_dir / "player_provider_links.csv"),
        club_links_out=str(external_dir / "club_provider_links.csv"),
        review_confidence_threshold=0.80,
    )

    review_path = external_dir / "player_provider_link_review_queue.csv"
    review = pd.read_csv(review_path)

    assert payload["player_links_rows"] == 1
    assert payload["player_review_queue_rows"] == 1
    assert review_path.exists()
    assert review.loc[0, "match_method"] == "name_season_unique"
