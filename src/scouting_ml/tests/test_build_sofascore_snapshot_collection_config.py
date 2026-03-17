from __future__ import annotations

import json
from pathlib import Path

from scouting_ml.league_registry import LeagueConfig
from scouting_ml.scripts import build_sofascore_snapshot_collection_config as cfg_module


def test_build_sofascore_snapshot_collection_config_uses_registry_and_processed_presence(
    monkeypatch, tmp_path: Path
) -> None:
    processed_root = tmp_path / "processed"
    processed_root.mkdir()
    league = LeagueConfig(
        slug="belgian_pro_league",
        name="Belgian Pro League",
        tm_league_url="https://example.com",
        seasons=["2024/25"],
        tm_season_ids={"2024/25": 2024},
        sofa_league_key="Belgian Pro League",
        sofa_tournament_id=38,
        sofa_season_map={"2024/25": "24/25"},
        processed_dataset_pattern=str(processed_root / "belgian_pro_league_{season_slug}_with_sofa.csv"),
    )
    league.guess_processed_dataset("2024/25").write_text("player_id\n1\n", encoding="utf-8")

    monkeypatch.setattr(cfg_module, "LEAGUES", {league.slug: league})
    monkeypatch.setattr(cfg_module, "get_league", lambda slug: league)

    output_path = tmp_path / "snapshot_config.json"
    payload = cfg_module.build_sofascore_snapshot_collection_config(
        season="2024/25",
        league_slugs=[league.slug],
        output_path=output_path,
        players_source=processed_root,
        raw_output_dir=tmp_path / "raw",
        provider_config_out=tmp_path / "provider.generated.json",
        summary_out=tmp_path / "summary.json",
        use_team_schedule=True,
    )

    assert payload["players_source"] == str(processed_root)
    comps = payload["fixture_context"]["competitions"]
    assert len(comps) == 1
    assert comps[0]["league"] == "Belgian Pro League"
    assert comps[0]["tournament_id"] == 38
    assert comps[0]["events_mode"] == "team_schedule"

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["player_availability"]["competitions"][0]["name"] == "belgian_pro_league_2024_25"


def test_build_sofascore_snapshot_collection_config_rejects_missing_processed_dataset(
    monkeypatch, tmp_path: Path
) -> None:
    league = LeagueConfig(
        slug="greek_super_league",
        name="Greek Super League",
        tm_league_url="https://example.com",
        seasons=["2024/25"],
        tm_season_ids={"2024/25": 2024},
        sofa_league_key="Greek Super League",
        sofa_tournament_id=185,
        sofa_season_map={"2024/25": "24/25"},
        processed_dataset_pattern=str(tmp_path / "processed" / "greek_super_league_{season_slug}_with_sofa.csv"),
    )
    monkeypatch.setattr(cfg_module, "LEAGUES", {league.slug: league})
    monkeypatch.setattr(cfg_module, "get_league", lambda slug: league)

    try:
        cfg_module.build_sofascore_snapshot_collection_config(
            season="2024/25",
            league_slugs=[league.slug],
            output_path=tmp_path / "snapshot_config.json",
        )
    except ValueError as exc:
        assert "processed_missing" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing processed dataset")
