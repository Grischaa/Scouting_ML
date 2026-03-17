from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scouting_ml.scripts import build_provider_external_data as provider_build_module
from scouting_ml.scripts.collect_sofascore_snapshots import collect_sofascore_snapshots


class _RapidApiClient:
    class Cfg:
        base_url = "https://sofascore.p.rapidapi.com"

    cfg = Cfg()


def _fixture_event() -> dict:
    return {
        "id": 101,
        "startTimestamp": 1723302000,
        "tournament": {
            "uniqueTournament": {"name": "Premier League"},
            "season": {"name": "2024/25"},
        },
        "homeTeam": {"id": 1, "name": "Example United"},
        "awayTeam": {"id": 2, "name": "Example City"},
        "homeScore": {"current": 2},
        "awayScore": {"current": 1},
    }


def _lineups_payload() -> dict:
    return {
        "home": {
            "players": [
                {
                    "player": {"id": 10, "name": "Jose Maria"},
                    "substitute": False,
                    "captain": True,
                    "statistics": {
                        "minutesPlayed": 90,
                        "ratingVersions": {"original": 7.8},
                        "goals": 1,
                        "assists": 1,
                        "totalShots": 4,
                        "shotsOnTarget": 2,
                        "keyPasses": 3,
                        "touches": 56,
                        "totalDuelsWon": 8,
                        "groundDuelsWon": 5,
                        "aerialDuelsWon": 2,
                        "successfulDribbles": 4,
                        "tackles": 2,
                        "interceptions": 1,
                        "clearances": 1,
                    },
                },
                {
                    "player": {"id": 20, "name": "Bench Guy"},
                    "substitute": True,
                    "statistics": {
                        "minutesPlayed": 20,
                        "rating": 6.7,
                        "totalShots": 1,
                        "shotsOnTarget": 1,
                    },
                },
            ],
            "missingPlayers": [
                {
                    "player": {"id": 40, "name": "Injured Guy"},
                    "reason": "Injury",
                }
            ],
        },
        "away": {
            "players": [
                {
                    "player": {"id": 30, "name": "Away Starter"},
                    "substitute": False,
                    "statistics": {
                        "minutesPlayed": 90,
                        "rating": 7.1,
                        "totalDuelsWon": 6,
                        "interceptions": 2,
                        "clearances": 4,
                    },
                }
            ]
        },
    }


def _players_source_path(tmp_path) -> str:
    players_source = tmp_path / "players_with_sofa.csv"
    pd.DataFrame(
        [
            {
                "player_id": "tm_10",
                "transfermarkt_id": "10",
                "name": "Jose Maria",
                "dob": "2003-01-01",
                "club": "Example United",
                "league": "Premier League",
                "season": "2024/25",
                "sofa_player_id": "10",
                "sofa_team_id": "1",
                "sofa_team_name": "Example United",
            },
            {
                "player_id": "tm_20",
                "transfermarkt_id": "20",
                "name": "Bench Guy",
                "dob": "2002-02-02",
                "club": "Example United",
                "league": "Premier League",
                "season": "2024/25",
                "sofa_player_id": "20",
                "sofa_team_id": "1",
                "sofa_team_name": "Example United",
            },
            {
                "player_id": "tm_30",
                "transfermarkt_id": "30",
                "name": "Away Starter",
                "dob": "2001-03-03",
                "club": "Example City",
                "league": "Premier League",
                "season": "2024/25",
                "sofa_player_id": "30",
                "sofa_team_id": "2",
                "sofa_team_name": "Example City",
            },
        ]
    ).to_csv(players_source, index=False)
    return str(players_source)


def test_build_provider_external_data_supports_sofascore_snapshots(tmp_path) -> None:
    players_source = _players_source_path(tmp_path)
    fixture_snapshot = tmp_path / "sofascore_fixtures.json"
    availability_snapshot = tmp_path / "sofascore_lineups.json"
    fixture_snapshot.write_text(
        json.dumps({"provider": "sofascore", "competition": {"league": "Premier League", "season": "2024/25"}, "events": [_fixture_event()]}),
        encoding="utf-8",
    )
    availability_snapshot.write_text(
        json.dumps(
            {
                "provider": "sofascore",
                "competition": {"league": "Premier League", "season": "2024/25"},
                "matches": [{"event": _fixture_event(), "lineups": _lineups_payload()}],
            }
        ),
        encoding="utf-8",
    )

    config_path = tmp_path / "provider_config.json"
    config_path.write_text(
        json.dumps(
            {
                "snapshot_date": "2026-03-12",
                "raw_output_dir": str(tmp_path / "raw"),
                "provider_config_out": str(tmp_path / "provider_config.generated.json"),
                "summary_out": str(tmp_path / "summary.json"),
                "players_source": players_source,
                "fixture_context": {
                    "provider": "sofascore",
                    "input_json": [str(fixture_snapshot)],
                },
                "player_availability": {
                    "provider": "sofascore",
                    "input_json": [str(availability_snapshot)],
                },
            }
        ),
        encoding="utf-8",
    )

    external_dir = tmp_path / "external"
    payload = provider_build_module.build_provider_external_data(
        config_path=str(config_path),
        external_dir=str(external_dir),
    )

    fixture_context = pd.read_csv(external_dir / "fixture_context.csv")
    availability = pd.read_csv(external_dir / "player_availability.csv")

    assert payload["outputs"]["fixture_context"]["rows"] == 2
    assert payload["outputs"]["player_availability"]["rows"] == 4
    assert set(fixture_context["club"]) == {"Example United", "Example City"}
    united = fixture_context.loc[fixture_context["club"] == "Example United"].iloc[0]
    assert united["fixture_matches"] == 1
    assert united["fixture_points_per_match"] == 3.0
    assert united["fixture_goal_diff_per_match"] == 1.0
    assert united["fixture_win_share"] == 1.0
    assert united["fixture_scoring_environment"] == 3.0
    jose = availability.loc[availability["player_id"] == "tm_10"].iloc[0]
    assert jose["avail_minutes"] == 90
    assert jose["avail_start_share"] == 1.0
    assert jose["avail_minutes_share"] == 1.0
    assert jose["avail_full_match_share"] == 1.0
    assert jose["avail_captain_share"] == 1.0
    assert jose["avail_goal_contrib"] == 2.0
    assert jose["avail_goal_contrib_per_report"] == 2.0
    assert jose["avail_goal_contrib_per90"] == 2.0
    assert jose["avail_key_passes_per_report"] == 3.0
    assert jose["avail_def_actions_per_report"] == 4.0
    assert round(float(jose["avail_rating_mean"]), 1) == 7.8
    assert jose["club"] == "Example United"
    assert availability["avail_injury_count"].max() == 1


def test_collect_sofascore_snapshots_writes_generated_config_and_raw_payloads(monkeypatch, tmp_path) -> None:
    players_source = _players_source_path(tmp_path)
    config_path = tmp_path / "sofascore_snapshot_collection.json"
    config_path.write_text(
        json.dumps(
            {
                "snapshot_date": "2026-03-12",
                "players_source": players_source,
                "fixture_context": {
                    "provider": "sofascore",
                    "competitions": [
                        {
                            "name": "premier_league_2024_25",
                            "league": "Premier League",
                            "season": "2024/25",
                            "tournament_id": 17,
                            "season_id": 61627,
                            "max_pages": 2,
                        }
                    ],
                },
                "player_availability": {
                    "provider": "sofascore",
                    "competitions": [
                        {
                            "name": "premier_league_2024_25",
                            "league": "Premier League",
                            "season": "2024/25",
                            "tournament_id": 17,
                            "season_id": 61627,
                            "max_pages": 2,
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("scouting_ml.scripts.collect_sofascore_snapshots.create_client", lambda **_: object())

    def fake_fetch(*, endpoint: str, segment: str | None = None, page: int | None = None, matchId: int | None = None, **kwargs):
        if "unique-tournament" in endpoint:
            if page == 0 and segment == "last":
                return {"events": [_fixture_event()]}
            return {"events": []}
        if "/event/" in endpoint:
            assert matchId == 101
            return _lineups_payload()
        raise AssertionError(f"Unexpected endpoint: {endpoint}")

    monkeypatch.setattr(
        "scouting_ml.scripts.collect_sofascore_snapshots._sofa_fetch",
        lambda client, endpoint, params=None, **tpl_vars: fake_fetch(endpoint=endpoint, **tpl_vars),
    )

    summary = collect_sofascore_snapshots(config_path=str(config_path))

    provider_cfg = summary["generated_provider_config"]
    assert provider_cfg["players_source"] == players_source
    assert provider_cfg["fixture_context"]["provider"] == "sofascore"
    assert provider_cfg["player_availability"]["provider"] == "sofascore"
    fixture_path = Path(provider_cfg["fixture_context"]["input_json"][0])
    availability_path = Path(provider_cfg["player_availability"]["input_json"][0])
    assert fixture_path.exists()
    assert availability_path.exists()
    assert summary["sections"]["fixture_context"]["status"] == "ready"
    assert summary["sections"]["fixture_context"]["competitions"][0]["events"] == 1
    assert summary["sections"]["player_availability"]["competitions"][0]["matches"] == 1


def test_collect_sofascore_snapshots_resolves_season_id_from_season_label(monkeypatch, tmp_path) -> None:
    players_source = _players_source_path(tmp_path)
    config_path = tmp_path / "estonia_sofascore_snapshot_collection.json"
    config_path.write_text(
        json.dumps(
            {
                "snapshot_date": "2026-03-12",
                "base_url": "https://sofascore.p.rapidapi.com",
                "raw_output_dir": str(tmp_path / "raw"),
                "provider_config_out": str(tmp_path / "provider_config.generated.json"),
                "summary_out": str(tmp_path / "summary.json"),
                "players_source": players_source,
                "fixture_context": {
                    "provider": "sofascore",
                    "competitions": [
                        {
                            "name": "estonian_meistriliiga_2025",
                            "league": "Estonian Meistriliiga",
                            "season": "2025",
                            "tournament_id": 178,
                            "max_pages": 2,
                        }
                    ],
                },
                "player_availability": {
                    "provider": "sofascore",
                    "competitions": [
                        {
                            "name": "estonian_meistriliiga_2025",
                            "league": "Estonian Meistriliiga",
                            "season": "2025",
                            "tournament_id": 178,
                            "max_pages": 2,
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("scouting_ml.scripts.collect_sofascore_snapshots.create_client", lambda **_: object())

    def fake_fetch(*, endpoint: str, tournamentId: int | None = None, segment: str | None = None, page: int | None = None, matchId: int | None = None, **kwargs):
        if endpoint == "/unique-tournament/{tournamentId}/seasons":
            assert tournamentId == 178
            return {
                "seasons": [
                    {"id": 70000, "name": "2024"},
                    {"id": 70001, "name": "2025"},
                ]
            }
        if endpoint == "/unique-tournament/{tournamentId}/season/{seasonId}/events/{segment}/{page}":
            assert kwargs["seasonId"] == 70001
            if page == 0 and segment == "last":
                event = _fixture_event()
                event["tournament"]["uniqueTournament"]["name"] = "Premium Liiga"
                event["tournament"]["season"]["name"] = "2025"
                return {"events": [event]}
            return {"events": []}
        if endpoint == "/event/{matchId}/lineups":
            assert matchId == 101
            return _lineups_payload()
        raise AssertionError(f"Unexpected endpoint: {endpoint}")

    monkeypatch.setattr(
        "scouting_ml.scripts.collect_sofascore_snapshots._sofa_fetch",
        lambda client, endpoint, params=None, **tpl_vars: fake_fetch(endpoint=endpoint, **tpl_vars),
    )

    summary = collect_sofascore_snapshots(config_path=str(config_path))

    fixture_comp = summary["sections"]["fixture_context"]["competitions"][0]
    availability_comp = summary["sections"]["player_availability"]["competitions"][0]
    assert fixture_comp["season_id"] == 70001
    assert availability_comp["season_id"] == 70001
    assert fixture_comp["resolved_from_seasons"] == 2
    assert availability_comp["resolved_from_seasons"] == 2


def test_collect_sofascore_snapshots_uses_rapidapi_defaults(monkeypatch, tmp_path) -> None:
    players_source = _players_source_path(tmp_path)
    config_path = tmp_path / "estonia_sofascore_rapidapi.json"
    config_path.write_text(
        json.dumps(
            {
                "snapshot_date": "2026-03-12",
                "base_url": "https://sofascore.p.rapidapi.com",
                "raw_output_dir": str(tmp_path / "raw"),
                "provider_config_out": str(tmp_path / "provider_config.generated.json"),
                "summary_out": str(tmp_path / "summary.json"),
                "players_source": players_source,
                "fixture_context": {
                    "provider": "sofascore",
                    "competitions": [
                        {
                            "name": "estonian_meistriliiga_2025",
                            "league": "Estonian Meistriliiga",
                            "season": "2025",
                            "tournament_id": 178,
                            "season_id": 71438,
                            "max_pages": 2,
                        }
                    ],
                },
                "player_availability": {
                    "provider": "sofascore",
                    "competitions": [
                        {
                            "name": "estonian_meistriliiga_2025",
                            "league": "Estonian Meistriliiga",
                            "season": "2025",
                            "tournament_id": 178,
                            "season_id": 71438,
                            "max_pages": 2,
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "scouting_ml.scripts.collect_sofascore_snapshots.create_client",
        lambda **_: _RapidApiClient(),
    )

    def fake_fetch(*, endpoint: str, params: dict | None = None, matchId: int | None = None, **kwargs):
        if endpoint == "/tournaments/get-last-matches":
            assert params is not None
            assert params["tournamentId"] == "178"
            assert params["seasonId"] == "71438"
            if params["page"] == "0":
                event = _fixture_event()
                event["tournament"]["uniqueTournament"]["name"] = "Premium Liiga"
                event["tournament"]["season"]["name"] = "2025"
                return {"events": [event]}
            return {"events": []}
        if endpoint == "/matches/get-lineups":
            assert matchId == 101
            assert params == {"matchId": "101"}
            return _lineups_payload()
        raise AssertionError(f"Unexpected endpoint: {endpoint}")

    monkeypatch.setattr(
        "scouting_ml.scripts.collect_sofascore_snapshots._sofa_fetch",
        lambda client, endpoint, params=None, **tpl_vars: fake_fetch(endpoint=endpoint, params=params, **tpl_vars),
    )

    summary = collect_sofascore_snapshots(config_path=str(config_path))

    fixture_comp = summary["sections"]["fixture_context"]["competitions"][0]
    availability_comp = summary["sections"]["player_availability"]["competitions"][0]
    assert fixture_comp["events"] == 1
    assert availability_comp["matches"] == 1


def test_collect_sofascore_snapshots_team_schedule_mode_dedupes_team_match_feeds(monkeypatch, tmp_path) -> None:
    players_source = _players_source_path(tmp_path)
    config_path = tmp_path / "team_schedule_mode.json"
    config_path.write_text(
        json.dumps(
            {
                "snapshot_date": "2026-03-12",
                "base_url": "https://sofascore.p.rapidapi.com",
                "raw_output_dir": str(tmp_path / "raw"),
                "provider_config_out": str(tmp_path / "provider_config.generated.json"),
                "summary_out": str(tmp_path / "summary.json"),
                "players_source": players_source,
                "fixture_context": {
                    "provider": "sofascore",
                    "competitions": [
                        {
                            "name": "example_team_schedule_mode",
                            "league": "Premier League",
                            "season": "2024/25",
                            "tournament_id": 17,
                            "season_id": 61627,
                            "events_mode": "team_schedule",
                            "team_schedule_max_pages": 2,
                        }
                    ],
                },
                "player_availability": {
                    "provider": "sofascore",
                    "competitions": [
                        {
                            "name": "example_team_schedule_mode",
                            "league": "Premier League",
                            "season": "2024/25",
                            "tournament_id": 17,
                            "season_id": 61627,
                            "events_mode": "team_schedule",
                            "team_schedule_max_pages": 2,
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "scouting_ml.scripts.collect_sofascore_snapshots.create_client",
        lambda **_: _RapidApiClient(),
    )

    def fake_fetch(*, endpoint: str, params: dict | None = None, matchId: int | None = None, **kwargs):
        if endpoint == "/teams/get-matches":
            assert params is not None
            if params["pageIndex"] == "0":
                event = _fixture_event()
                event["tournament"]["uniqueTournament"]["id"] = 17
                event["season"] = {"id": 61627, "name": "2024/25"}
                return {"events": [event], "hasNextPage": True}
            return {"events": [], "hasNextPage": False}
        if endpoint == "/matches/get-lineups":
            assert matchId == 101
            return _lineups_payload()
        raise AssertionError(f"Unexpected endpoint: {endpoint}")

    monkeypatch.setattr(
        "scouting_ml.scripts.collect_sofascore_snapshots._sofa_fetch",
        lambda client, endpoint, params=None, **tpl_vars: fake_fetch(endpoint=endpoint, params=params, **tpl_vars),
    )

    summary = collect_sofascore_snapshots(config_path=str(config_path))

    fixture_comp = summary["sections"]["fixture_context"]["competitions"][0]
    availability_comp = summary["sections"]["player_availability"]["competitions"][0]
    assert fixture_comp["events"] == 1
    assert availability_comp["matches"] == 1


def test_collect_sofascore_snapshots_team_schedule_mode_skips_season_lookup(monkeypatch, tmp_path) -> None:
    players_source = _players_source_path(tmp_path)
    config_path = tmp_path / "team_schedule_without_season_id.json"
    config_path.write_text(
        json.dumps(
            {
                "snapshot_date": "2026-03-12",
                "base_url": "https://sofascore.p.rapidapi.com",
                "raw_output_dir": str(tmp_path / "raw"),
                "provider_config_out": str(tmp_path / "provider_config.generated.json"),
                "summary_out": str(tmp_path / "summary.json"),
                "players_source": players_source,
                "fixture_context": {
                    "provider": "sofascore",
                    "competitions": [
                        {
                            "name": "example_team_schedule_mode",
                            "league": "Premier League",
                            "season": "2024/25",
                            "tournament_id": 17,
                            "events_mode": "team_schedule",
                            "team_schedule_max_pages": 2,
                        }
                    ],
                },
                "player_availability": {
                    "provider": "sofascore",
                    "competitions": [
                        {
                            "name": "example_team_schedule_mode",
                            "league": "Premier League",
                            "season": "2024/25",
                            "tournament_id": 17,
                            "events_mode": "team_schedule",
                            "team_schedule_max_pages": 2,
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "scouting_ml.scripts.collect_sofascore_snapshots.create_client",
        lambda **_: _RapidApiClient(),
    )

    def fake_fetch(*, endpoint: str, params: dict | None = None, matchId: int | None = None, **kwargs):
        if endpoint == "/tournaments/get-seasons":
            raise AssertionError("team_schedule mode should not resolve season_id")
        if endpoint == "/teams/get-matches":
            assert params is not None
            if params["pageIndex"] == "0":
                event = _fixture_event()
                event["tournament"]["uniqueTournament"]["id"] = 17
                event["season"] = {"id": 61627, "name": "2024/25"}
                return {"events": [event], "hasNextPage": False}
            return {"events": [], "hasNextPage": False}
        if endpoint == "/matches/get-lineups":
            assert matchId == 101
            return _lineups_payload()
        raise AssertionError(f"Unexpected endpoint: {endpoint}")

    monkeypatch.setattr(
        "scouting_ml.scripts.collect_sofascore_snapshots._sofa_fetch",
        lambda client, endpoint, params=None, **tpl_vars: fake_fetch(endpoint=endpoint, params=params, **tpl_vars),
    )

    summary = collect_sofascore_snapshots(config_path=str(config_path))

    fixture_comp = summary["sections"]["fixture_context"]["competitions"][0]
    availability_comp = summary["sections"]["player_availability"]["competitions"][0]
    assert fixture_comp["season_id"] is None
    assert availability_comp["season_id"] is None
    assert fixture_comp["events"] == 1
    assert availability_comp["matches"] == 1


def test_collect_sofascore_snapshots_filters_selected_competitions(monkeypatch, tmp_path) -> None:
    players_source = _players_source_path(tmp_path)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "sofascore_fixtures_should_not_run_2026-03-12.json").write_text(
        json.dumps({"competition": {"league": "Greek Super League", "season": "2024/25"}, "events": []}),
        encoding="utf-8",
    )
    (raw_dir / "sofascore_lineups_should_not_run_2026-03-12.json").write_text(
        json.dumps({"competition": {"league": "Greek Super League", "season": "2024/25"}, "matches": []}),
        encoding="utf-8",
    )
    config_path = tmp_path / "filtered_competitions.json"
    config_path.write_text(
        json.dumps(
            {
                "snapshot_date": "2026-03-12",
                "base_url": "https://sofascore.p.rapidapi.com",
                "raw_output_dir": str(raw_dir),
                "provider_config_out": str(tmp_path / "provider_config.generated.json"),
                "summary_out": str(tmp_path / "summary.json"),
                "players_source": players_source,
                "fixture_context": {
                    "provider": "sofascore",
                    "competitions": [
                        {
                            "name": "example_team_schedule_mode",
                            "league": "Premier League",
                            "season": "2024/25",
                            "tournament_id": 17,
                            "events_mode": "team_schedule",
                            "team_schedule_max_pages": 1,
                        },
                        {
                            "name": "should_not_run",
                            "league": "Greek Super League",
                            "season": "2024/25",
                            "tournament_id": 185,
                            "events_mode": "team_schedule",
                            "team_schedule_max_pages": 1,
                        },
                    ],
                },
                "player_availability": {
                    "provider": "sofascore",
                    "competitions": [
                        {
                            "name": "example_team_schedule_mode",
                            "league": "Premier League",
                            "season": "2024/25",
                            "tournament_id": 17,
                            "events_mode": "team_schedule",
                            "team_schedule_max_pages": 1,
                        },
                        {
                            "name": "should_not_run",
                            "league": "Greek Super League",
                            "season": "2024/25",
                            "tournament_id": 185,
                            "events_mode": "team_schedule",
                            "team_schedule_max_pages": 1,
                        },
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "scouting_ml.scripts.collect_sofascore_snapshots.create_client",
        lambda **_: _RapidApiClient(),
    )

    def fake_fetch(*, endpoint: str, params: dict | None = None, matchId: int | None = None, **kwargs):
        if endpoint == "/teams/get-matches":
            assert params is not None
            event = _fixture_event()
            event["tournament"]["uniqueTournament"]["id"] = 17
            event["season"] = {"id": 61627, "name": "2024/25"}
            return {"events": [event], "hasNextPage": False}
        if endpoint == "/matches/get-lineups":
            assert matchId == 101
            return _lineups_payload()
        raise AssertionError(f"Unexpected endpoint: {endpoint}")

    monkeypatch.setattr(
        "scouting_ml.scripts.collect_sofascore_snapshots._sofa_fetch",
        lambda client, endpoint, params=None, **tpl_vars: fake_fetch(endpoint=endpoint, params=params, **tpl_vars),
    )

    summary = collect_sofascore_snapshots(
        config_path=str(config_path),
        competitions=["example_team_schedule_mode"],
    )

    fixture_names = [item["name"] for item in summary["sections"]["fixture_context"]["competitions"]]
    availability_names = [item["name"] for item in summary["sections"]["player_availability"]["competitions"]]
    assert fixture_names == ["example_team_schedule_mode"]
    assert availability_names == ["example_team_schedule_mode"]
    assert len(summary["generated_provider_config"]["fixture_context"]["input_json"]) == 2
    assert len(summary["generated_provider_config"]["player_availability"]["input_json"]) == 2
