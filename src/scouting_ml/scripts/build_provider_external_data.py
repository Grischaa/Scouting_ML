from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from scouting_ml.providers.football_api.client import ApiFootballClient, SportmonksClient
from scouting_ml.providers.odds.client import OddsApiClient
from scouting_ml.providers.football_api.normalize import (
    build_fixture_context,
    build_player_availability,
    normalize_fixtures,
    normalize_player_availability,
)
from scouting_ml.providers.odds.normalize import build_market_context, normalize_odds_events
from scouting_ml.providers.sofascore.normalize import (
    build_fixture_context as build_sofascore_fixture_context,
    build_player_availability as build_sofascore_player_availability,
    normalize_fixtures as normalize_sofascore_fixtures,
    normalize_player_availability as normalize_sofascore_player_availability,
)
from scouting_ml.providers.statsbomb import aggregate_player_season_features


def _read_json_payloads(paths: list[str]) -> list[object]:
    payloads: list[object] = []
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            raise FileNotFoundError(f"Provider snapshot missing: {path}")
        payloads.append(json.loads(path.read_text(encoding="utf-8")))
    return payloads


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        item = value.strip()
        return [item] if item else []
    out: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            out.append(text)
    return out


def _parse_int_list(values: Any) -> list[int] | None:
    if not values:
        return None
    out: list[int] = []
    for value in values:
        out.append(int(value))
    return out or None


def _fetch_fixture_payload(provider: str, api_url: str) -> object:
    if provider == "sofascore":
        raise ValueError("SofaScore fixture scraping should be collected via snapshot JSON, not api_url in this builder.")
    if provider == "sportmonks":
        return SportmonksClient().get_json(api_url)
    return ApiFootballClient().get_json(api_url)


def _fetch_availability_payload(provider: str, api_url: str) -> object:
    if provider == "sofascore":
        raise ValueError("SofaScore availability scraping should be collected via snapshot JSON, not api_url in this builder.")
    if provider == "sportmonks":
        return SportmonksClient().get_json(api_url)
    return ApiFootballClient().get_json(api_url)


def _fetch_odds_payload(api_url: str) -> object:
    return OddsApiClient().get_json(api_url)


def build_provider_external_data(
    *,
    config_path: str | Path,
    external_dir: str | Path = "data/external",
) -> dict[str, Any]:
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Provider config not found: {config_file}")

    config = json.loads(config_file.read_text(encoding="utf-8"))
    ext_dir = Path(external_dir)
    ext_dir.mkdir(parents=True, exist_ok=True)

    snapshot_date = str(config.get("snapshot_date") or datetime.now(timezone.utc).date().isoformat())
    retrieved_at = datetime.now(timezone.utc).isoformat()
    player_links = str(config.get("player_links") or (ext_dir / "player_provider_links.csv"))
    club_links = str(config.get("club_links") or (ext_dir / "club_provider_links.csv"))
    players_source = str(config.get("players_source") or "").strip()

    outputs: dict[str, Any] = {
        "config_path": str(config_file),
        "snapshot_date": snapshot_date,
        "retrieved_at": retrieved_at,
        "outputs": {},
    }

    statsbomb_cfg = config.get("statsbomb") or {}
    if statsbomb_cfg.get("open_data_root"):
        out_path = Path(statsbomb_cfg.get("output") or (ext_dir / "statsbomb_player_season_features.csv"))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frame = aggregate_player_season_features(
            statsbomb_cfg["open_data_root"],
            competition_ids=_parse_int_list(statsbomb_cfg.get("competition_ids")),
            season_ids=_parse_int_list(statsbomb_cfg.get("season_ids")),
            player_links_path=statsbomb_cfg.get("player_links") or player_links,
            snapshot_date=snapshot_date,
            retrieved_at=retrieved_at,
        )
        frame.to_csv(out_path, index=False)
        outputs["outputs"]["statsbomb_player_season_features"] = {
            "path": str(out_path),
            "rows": int(len(frame)),
        }

    fixture_cfg = config.get("fixture_context") or {}
    if fixture_cfg.get("provider") and (
        fixture_cfg.get("input_json") or fixture_cfg.get("input_payload") or fixture_cfg.get("api_url")
    ):
        payloads = _read_json_payloads(_as_str_list(fixture_cfg.get("input_json")))
        provider_name = str(fixture_cfg["provider"]).strip().lower()
        for api_url in _as_str_list(fixture_cfg.get("api_url")):
            payloads.append(_fetch_fixture_payload(provider_name, api_url))
        if provider_name == "sofascore":
            sofa_players_source = str(fixture_cfg.get("players_source") or players_source).strip()
            if not sofa_players_source:
                raise ValueError("SofaScore fixture_context requires players_source in the top-level config or section.")
            frames = [normalize_sofascore_fixtures(payload) for payload in payloads]
            fixtures = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True, sort=False) if frames else pd.DataFrame()
            out = build_sofascore_fixture_context(
                fixtures,
                players_source=sofa_players_source,
                snapshot_date=snapshot_date,
                retrieved_at=retrieved_at,
            )
        else:
            frames = [normalize_fixtures(payload, provider=provider_name) for payload in payloads]
            fixtures = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True, sort=False) if frames else pd.DataFrame()
            out = build_fixture_context(
                fixtures,
                provider=provider_name,
                club_links_path=str(fixture_cfg.get("club_links") or club_links),
                snapshot_date=snapshot_date,
                retrieved_at=retrieved_at,
            )
        out_path = Path(fixture_cfg.get("output") or (ext_dir / "fixture_context.csv"))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)
        outputs["outputs"]["fixture_context"] = {"path": str(out_path), "rows": int(len(out))}

    availability_cfg = config.get("player_availability") or {}
    if availability_cfg.get("provider") and (
        availability_cfg.get("input_json") or availability_cfg.get("input_payload") or availability_cfg.get("api_url")
    ):
        payloads = _read_json_payloads(_as_str_list(availability_cfg.get("input_json")))
        provider_name = str(availability_cfg["provider"]).strip().lower()
        for api_url in _as_str_list(availability_cfg.get("api_url")):
            payloads.append(_fetch_availability_payload(provider_name, api_url))
        if provider_name == "sofascore":
            sofa_players_source = str(availability_cfg.get("players_source") or players_source).strip()
            if not sofa_players_source:
                raise ValueError("SofaScore player_availability requires players_source in the top-level config or section.")
            frames = [normalize_sofascore_player_availability(payload) for payload in payloads]
            availability = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True, sort=False) if frames else pd.DataFrame()
            out = build_sofascore_player_availability(
                availability,
                players_source=sofa_players_source,
                snapshot_date=snapshot_date,
                retrieved_at=retrieved_at,
            )
        else:
            frames = [normalize_player_availability(payload, provider=provider_name) for payload in payloads]
            availability = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True, sort=False) if frames else pd.DataFrame()
            out = build_player_availability(
                availability,
                provider=provider_name,
                player_links_path=str(availability_cfg.get("player_links") or player_links),
                club_links_path=str(availability_cfg.get("club_links") or club_links),
                snapshot_date=snapshot_date,
                retrieved_at=retrieved_at,
            )
        out_path = Path(availability_cfg.get("output") or (ext_dir / "player_availability.csv"))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)
        outputs["outputs"]["player_availability"] = {"path": str(out_path), "rows": int(len(out))}

    odds_cfg = config.get("market_context") or {}
    if (odds_cfg.get("input_json") or odds_cfg.get("input_payload") or odds_cfg.get("api_url")) and odds_cfg.get("league") and odds_cfg.get("season"):
        payloads = _read_json_payloads(_as_str_list(odds_cfg.get("input_json")))
        for api_url in _as_str_list(odds_cfg.get("api_url")):
            payloads.append(_fetch_odds_payload(api_url))
        frames = [
            normalize_odds_events(payload, season=str(odds_cfg["season"]), league=str(odds_cfg["league"]))
            for payload in payloads
        ]
        odds_events = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True, sort=False) if frames else pd.DataFrame()
        out = build_market_context(
            odds_events,
            club_links_path=str(odds_cfg.get("club_links") or club_links),
            snapshot_date=snapshot_date,
            retrieved_at=retrieved_at,
        )
        out_path = Path(odds_cfg.get("output") or (ext_dir / "market_context.csv"))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)
        outputs["outputs"]["market_context"] = {"path": str(out_path), "rows": int(len(out))}

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build optional provider external tables from a JSON config.")
    parser.add_argument("--config", required=True, help="Path to provider config JSON.")
    parser.add_argument("--external-dir", default="data/external")
    parser.add_argument("--summary-out", default="", help="Optional JSON summary output path.")
    args = parser.parse_args()

    payload = build_provider_external_data(config_path=args.config, external_dir=args.external_dir)
    if args.summary_out:
        out_path = Path(args.summary_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[provider-build] wrote summary -> {out_path}")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
