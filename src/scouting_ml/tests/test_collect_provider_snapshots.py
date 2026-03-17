from __future__ import annotations

import json
from pathlib import Path

from scouting_ml.scripts.collect_provider_snapshots import collect_provider_snapshots


def test_collect_provider_snapshots_writes_generated_config_and_raw_payloads(
    tmp_path: Path,
    monkeypatch,
) -> None:
    raw_dir = tmp_path / "raw"
    provider_cfg_out = tmp_path / "provider.generated.json"
    summary_out = tmp_path / "provider.summary.json"
    existing_fixture = tmp_path / "existing_fixture.json"
    existing_fixture.write_text(json.dumps({"data": []}), encoding="utf-8")

    config_path = tmp_path / "snapshot_config.json"
    config_path.write_text(
        json.dumps(
            {
                "snapshot_date": "2026-03-12",
                "raw_output_dir": str(raw_dir),
                "provider_config_out": str(provider_cfg_out),
                "summary_out": str(summary_out),
                "player_links": "data/external/player_provider_links.csv",
                "club_links": "data/external/club_provider_links.csv",
                "statsbomb": {
                    "open_data_root": "data/raw/statsbomb/bundesliga_2023_24",
                    "competition_ids": [9],
                    "season_ids": [281],
                },
                "fixture_context": {
                    "provider": "sportmonks",
                    "input_json": [str(existing_fixture)],
                    "requests": [
                        {
                            "name": "epl_fixtures",
                            "endpoint": "fixtures",
                            "params": {"season": "2024/25"},
                        }
                    ],
                },
                "market_context": {
                    "league": "Premier League",
                    "season": "2024/25",
                    "requests": [
                        {
                            "name": "epl_odds",
                            "endpoint": "sports/soccer_epl/odds",
                            "params": {"markets": "h2h,totals"},
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("SPORTMONKS_API_TOKEN", "token")
    monkeypatch.setenv("ODDS_API_KEY", "token")

    def fake_fetch(*, provider: str, endpoint: str, params=None, headers=None):
        return {
            "provider": provider,
            "endpoint": endpoint,
            "params": params or {},
            "headers": headers or {},
        }

    monkeypatch.setattr(
        "scouting_ml.scripts.collect_provider_snapshots._client_fetch",
        fake_fetch,
    )

    summary = collect_provider_snapshots(config_path=str(config_path))

    assert provider_cfg_out.exists()
    assert summary_out.exists()

    provider_cfg = json.loads(provider_cfg_out.read_text(encoding="utf-8"))
    assert provider_cfg["statsbomb"]["competition_ids"] == [9]
    assert provider_cfg["fixture_context"]["input_json"][0] == str(existing_fixture)
    assert len(provider_cfg["fixture_context"]["input_json"]) == 2
    assert len(provider_cfg["market_context"]["input_json"]) == 1

    fetched_fixture = Path(provider_cfg["fixture_context"]["input_json"][1])
    fetched_market = Path(provider_cfg["market_context"]["input_json"][0])
    assert fetched_fixture.exists()
    assert fetched_market.exists()

    fixture_payload = json.loads(fetched_fixture.read_text(encoding="utf-8"))
    market_payload = json.loads(fetched_market.read_text(encoding="utf-8"))
    assert fixture_payload["provider"] == "sportmonks"
    assert market_payload["provider"] == "odds"
    assert summary["sections"]["fixture_context"]["status"] == "ready"
    assert summary["sections"]["market_context"]["status"] == "ready"


def test_collect_provider_snapshots_dry_run_plans_requests_without_creds(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "snapshot_config.json"
    provider_cfg_out = tmp_path / "provider.generated.json"
    summary_out = tmp_path / "provider.summary.json"
    config_path.write_text(
        json.dumps(
            {
                "snapshot_date": "2026-03-12",
                "provider_config_out": str(provider_cfg_out),
                "summary_out": str(summary_out),
                "fixture_context": {
                    "provider": "sportmonks",
                    "requests": [
                        {
                            "name": "fixtures",
                            "endpoint": "fixtures",
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    summary = collect_provider_snapshots(
        config_path=str(config_path),
        dry_run=True,
    )

    assert provider_cfg_out.exists() is False
    assert summary_out.exists()
    assert summary["sections"]["fixture_context"]["status"] == "planned"
    request_summary = summary["sections"]["fixture_context"]["requests"][0]
    assert request_summary["required_env_var"] == "SPORTMONKS_API_TOKEN"
    assert request_summary["has_secret"] is False


def test_collect_provider_snapshots_can_skip_missing_secrets(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "snapshot_config.json"
    provider_cfg_out = tmp_path / "provider.generated.json"
    config_path.write_text(
        json.dumps(
            {
                "snapshot_date": "2026-03-12",
                "provider_config_out": str(provider_cfg_out),
                "player_availability": {
                    "provider": "api-football",
                    "requests": [
                        {
                            "name": "availability",
                            "endpoint": "fixtures/players",
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    summary = collect_provider_snapshots(
        config_path=str(config_path),
        allow_missing_secrets=True,
    )

    assert provider_cfg_out.exists()
    generated = json.loads(provider_cfg_out.read_text(encoding="utf-8"))
    assert "player_availability" not in generated
    assert summary["sections"]["player_availability"]["status"] == "skipped_missing_secret"
    assert summary["warnings"]
